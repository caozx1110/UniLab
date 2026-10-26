"""Cold numerical control carriers shared by production and CPU acceptance.

These carriers exist only across initialization/restoration. Runtime loops retain
ownership of their existing locals; no event, scheduling or RNG action occurs here.
"""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import asdict

from unilab.rl.logging.metrics_drain import RewardComponentWindow
from unilab.rl.runtime.offpolicy.metrics import MetricFlush


class CollectorControlState:
    SCALARS = (
        "inference_tick",
        "total_steps",
        "done_count_window",
        "timeout_count_window",
        "max_action_backlog_since_metric",
        "max_in_flight_since_metric",
        "max_publication_lag_since_metric",
    )

    def __init__(self):
        for name in self.SCALARS:
            setattr(self, name, 0)
        self.ep_rewards = deque(maxlen=100)
        self.ep_lengths = deque(maxlen=100)
        self.ep_reward_components = defaultdict(list)
        self.timing_accum_ms = defaultdict(float)
        self.timing_counts = defaultdict(int)
        self.pending_metric_flush = None

    def import_values(self, saved, *, num_envs, prev_dones):
        if saved["total_steps"] != saved["inference_tick"] * num_envs:
            raise ValueError("Collector continuation step/tick counters disagree")
        source_dones = saved["prev_dones"]
        if source_dones.shape != prev_dones.shape or source_dones.dtype != prev_dones.dtype:
            raise ValueError("Collector continuation previous-done shape/dtype changed")
        prev_dones.copy_(source_dones.to(prev_dones.device))
        for name in self.SCALARS:
            setattr(self, name, saved[name])
        self.ep_rewards = deque(saved["ep_rewards"], maxlen=100)
        self.ep_lengths = deque(saved["ep_lengths"], maxlen=100)
        self.ep_reward_components = defaultdict(list, deepcopy(saved["ep_reward_components"]))
        self.timing_accum_ms = defaultdict(float, saved["timing_accum_ms"])
        self.timing_counts = defaultdict(int, saved["timing_counts"])
        self.pending_metric_flush = (
            None
            if saved["pending_metric_flush"] is None
            else MetricFlush(**deepcopy(saved["pending_metric_flush"]))
        )

    def value_views(self):
        return {
            **{name: getattr(self, name) for name in self.SCALARS},
            "ep_rewards": list(self.ep_rewards),
            "ep_lengths": list(self.ep_lengths),
            "ep_reward_components": dict(self.ep_reward_components),
            "timing_accum_ms": dict(self.timing_accum_ms),
            "timing_counts": dict(self.timing_counts),
            "pending_metric_flush": None
            if self.pending_metric_flush is None
            else asdict(self.pending_metric_flush),
        }


class RunnerControlState:
    SCALARS = (
        "reward_stats_ptr",
        "prepared_tick",
        "has_logged_reward",
        "last_buf_log",
        "write_read_ema",
        "best_mean_reward",
        "last_mean_reward",
    )
    LOGGER_FIELDS = (
        "_total_steps",
        "_buffer_size",
        "_buffer_target",
        "_mean_ep_length",
        "_timeout_rate",
        "_buffer_utilization",
    )

    def __init__(self, scheduler, logger, *, best_mean_reward, last_mean_reward):
        self.inference_scheduler = scheduler
        self.logger = logger
        self.reward_stats_ptr = 0
        self.prepared_tick = None
        self.reward_history = deque(maxlen=10)
        self.reward_components = RewardComponentWindow()
        self.has_logged_reward = False
        self.last_buf_log = 0
        self.write_read_ema = 0.0
        self.best_mean_reward = best_mean_reward
        self.last_mean_reward = last_mean_reward

    def import_values(self, saved):
        scheduler = self.inference_scheduler
        scheduler_state = saved["inference_scheduler"]
        if (
            set(scheduler_state) != set(vars(scheduler))
            or scheduler_state["env_steps_per_sync"] != scheduler.env_steps_per_sync
            or scheduler_state["_pending_tick"] is not None
            or scheduler_state["_ticks_since_update"] != 0
        ):
            raise ValueError("Invalid inference scheduler continuation boundary")
        if set(saved["logger"]) != set(self.LOGGER_FIELDS):
            raise ValueError("Invalid logger continuation fields")
        vars(scheduler).update(scheduler_state)
        for name in self.SCALARS:
            setattr(self, name, saved[name])
        self.reward_history.extend(saved["reward_history"])
        self.reward_components = RewardComponentWindow(**deepcopy(saved["reward_components"]))
        for name, value in saved["logger"].items():
            setattr(self.logger, name, value)

    def value_views(self):
        return {
            **{name: getattr(self, name) for name in self.SCALARS},
            "inference_scheduler": dict(vars(self.inference_scheduler)),
            "reward_history": list(self.reward_history),
            "reward_components": asdict(self.reward_components),
            "logger": {name: getattr(self.logger, name) for name in self.LOGGER_FIELDS},
        }
