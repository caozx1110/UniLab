from copy import deepcopy

import pytest
import torch

from unilab.rl.logging.offpolicy import OffPolicyLogger
from unilab.rl.runtime.offpolicy.control_continuation import (
    CollectorControlState,
    RunnerControlState,
)
from unilab.rl.runtime.offpolicy.double_buffer_runner import _LearnerInferenceScheduler


def runner_target():
    return RunnerControlState(
        _LearnerInferenceScheduler(env_steps_per_sync=1),
        OffPolicyLogger(log_backend="none"),
        best_mean_reward=float("-inf"),
        last_mean_reward=0.0,
    )


def test_runner_restores_held_tick_without_consuming_or_rng():
    target = runner_target()
    saved = target.value_views()
    saved["inference_scheduler"].update(next_tick=2146, policy_version=2048)
    saved.update(prepared_tick=2049, reward_stats_ptr=6589440, reward_history=[0.1, 0.3])
    saved["reward_components"] = {"_sums": {"term": 1.7}, "_counts": {"term": 3}}
    saved["logger"]["_total_steps"] = 6592512
    rng = torch.get_rng_state().clone()
    target.import_values(saved)
    assert target.value_views() == saved
    assert torch.equal(rng, torch.get_rng_state())
    saved["reward_components"]["_sums"]["term"] = -99
    assert target.reward_components._sums["term"] == 1.7
    scheduler = target.inference_scheduler
    scheduler.record_inference(2146)
    assert scheduler.release_pending() == 2146
    scheduler.finish_update()
    assert scheduler.next_tick == 2147 and scheduler.policy_version == 2049


@pytest.mark.parametrize(
    "name,value", [("_pending_tick", 2146), ("_ticks_since_update", 1), ("env_steps_per_sync", 2)]
)
def test_runner_rejects_nonheld_boundary(name, value):
    target = runner_target()
    saved = target.value_views()
    saved["inference_scheduler"][name] = value
    with pytest.raises(ValueError, match="scheduler"):
        target.import_values(saved)
    assert target.inference_scheduler.next_tick == 0


def test_collector_restores_pending_metric_payload_and_windows_without_alias():
    target = CollectorControlState()
    saved = target.value_views()
    saved.update(
        inference_tick=2146, total_steps=4292, ep_rewards=[0.1, 1.2], ep_lengths=[4, 20]
    )
    saved["pending_metric_flush"] = {
        "rewards": [1.5],
        "lengths": [17],
        "done_count": 1,
        "timeout_count": 0,
        "reward_component_means": {"term": 1.1},
        "reward_component_updates": 3,
    }
    saved["timing_accum_ms"] = {"env_step_ms": 1.03}
    saved["timing_counts"] = {"env_step_ms": 1}
    saved["prev_dones"] = torch.tensor([0.0, -0.0])
    target_dones = torch.zeros(2)
    rng = torch.get_rng_state().clone()
    target.import_values(saved, num_envs=2, prev_dones=target_dones)
    assert target.value_views() == {k: v for k, v in saved.items() if k != "prev_dones"}
    assert torch.equal(target_dones.view(torch.uint8), saved["prev_dones"].view(torch.uint8))
    assert target_dones.data_ptr() != saved["prev_dones"].data_ptr()
    assert torch.equal(rng, torch.get_rng_state())
    saved["pending_metric_flush"]["rewards"][0] = 99
    assert target.pending_metric_flush.rewards == [1.5]


def test_collector_rejects_counter_mismatch_before_copy():
    target = CollectorControlState()
    saved = deepcopy(target.value_views())
    saved.update(inference_tick=2, total_steps=3, prev_dones=torch.ones(2))
    target_dones = torch.zeros(2)
    with pytest.raises(ValueError, match="counters"):
        target.import_values(saved, num_envs=2, prev_dones=target_dones)
    assert target_dones.tolist() == [0, 0]
