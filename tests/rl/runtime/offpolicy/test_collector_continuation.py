import queue
import random
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import unilab.rl.runtime.offpolicy.worker as worker_module
from unilab.rl.runtime.ipc.inference_ring import SharedInferenceRing
from unilab.rl.runtime.offpolicy.collector_continuation import (
    cpu_tree,
    restore_metric_state,
    service_checkpoint_request,
)
from unilab.rl.runtime.offpolicy.continuation import load_state
from unilab.rl.runtime.offpolicy.metrics import CollectorMetrics
from unilab.rl.runtime.offpolicy.worker import _wait_for_inference_tick


def test_partial_episode_metrics_continue_through_new_collector():
    original = CollectorMetrics(2, interval=3, device="cpu")
    original.update(
        torch.tensor([1.0, 2.0]), torch.tensor([False, True]), torch.tensor([False, True])
    )
    snapshot = cpu_tree(vars(original))
    restored = CollectorMetrics(2, interval=3, device="cpu")
    restore_metric_state(restored, snapshot)
    for metrics in (original, restored):
        metrics.update(
            torch.tensor([3.0, 4.0]), torch.tensor([False, False]), torch.tensor([False, False])
        )
        metrics.update(
            torch.tensor([5.0, 6.0]), torch.tensor([True, True]), torch.tensor([False, False])
        )
    assert original.flush() == restored.flush()
    assert torch.equal(original.current_rewards, restored.current_rewards)


def test_held_inference_poll_saves_without_consuming_action(tmp_path):
    requests, responses, actions = queue.Queue(), queue.Queue(), queue.Queue()
    requests.put(
        {
            "operation": "snapshot",
            "request_id": "50",
            "tick": 53,
            "path": str(tmp_path / "collector.pt"),
        }
    )
    calls = []

    def snapshot():
        calls.append("snapshot")
        return {"value": torch.tensor([7]), "inference_tick": 53}

    def poll():
        service_checkpoint_request(requests, responses, tick=53, snapshot=snapshot)
        actions.put(53)

    assert _wait_for_inference_tick(actions, 53, threading.Event(), checkpoint_poll=poll)
    reply = responses.get_nowait()
    assert reply["tick"] == 53 and reply["request_id"] == "50"
    state = load_state({key: reply[key] for key in ("path", "sha256", "bytes")})
    assert state["inference_tick"] == 53 and calls == ["snapshot"]


def test_wrong_tick_snapshot_fails_and_returns_error(tmp_path):
    requests, responses = queue.Queue(), queue.Queue()
    requests.put(
        {"operation": "snapshot", "request_id": "bad", "tick": 1, "path": str(tmp_path / "bad.pt")}
    )
    with pytest.raises(ValueError, match="held tick"):
        service_checkpoint_request(requests, responses, tick=2, snapshot=lambda: {})
    assert "error" in responses.get_nowait()
    assert not (tmp_path / "bad.pt").exists()


def test_worker_continues_held_tick_rng_and_metric_barrier_with_fresh_owners(tmp_path, monkeypatch):
    """Exercise the worker wiring, including a partial episode at the hold."""
    torch.manual_seed(47)
    np.random.seed(53)
    random.seed(59)
    saved_binding = None

    def run(resume=None):
        nonlocal saved_binding
        stop = threading.Event()
        requests, responses, metrics = queue.Queue(), queue.Queue(), queue.Queue()
        ring = SharedInferenceRing(2, 2, 2, capacity=4)
        if resume is not None:
            ring.initialize_continuation_tick(3)
        published_inputs, transitions, events = [], [], []

        class Env:
            def __init__(self):
                self.steps = 0
                # Cold initialization deliberately consumes each RNG. Resume
                # must restore their saved positions after this constructor.
                initial = torch.rand((2, 2)) + np.random.random() + random.random()
                self.state = SimpleNamespace(
                    obs={"obs": initial},
                    reward=torch.zeros(2),
                    terminated=torch.zeros(2, dtype=torch.bool),
                    truncated=torch.zeros(2, dtype=torch.bool),
                    final_observation=None,
                    info={},
                )

            def step(self, actions):
                self.steps += 1
                obs = actions + torch.rand((2, 2)) + np.random.random() + random.random()
                self.state = SimpleNamespace(
                    obs={"obs": obs},
                    reward=obs.sum(-1),
                    terminated=torch.tensor([self.steps % 3 == 0, self.steps % 5 == 0]),
                    truncated=torch.zeros(2, dtype=torch.bool),
                    final_observation=None,
                    info={},
                )
                if self.steps == 6:
                    stop.set()
                return self.state

            def export_continuation_state(self):
                assert self.steps == 3
                # This is the env export point: all earlier metrics have
                # already been enqueued before the FIFO boundary marker.
                pending = list(metrics.queue)
                assert pending[-1] == {"continuation_barrier": 3}
                assert any(item.get("total_steps") == 4 for item in pending[:-1])
                events.append("snapshot")
                return {"steps": self.steps, "state": cpu_tree(vars(self.state))}

            def import_continuation_state(self, state):
                self.steps = state["steps"]
                self.state = SimpleNamespace(**cpu_tree(state["state"]))
                events.append("restore")

            def close(self):
                events.append("close")

        class Replay:
            size = torch.zeros(1, dtype=torch.int64)

            def attach_stop_event(self, event):
                assert event is stop

            def add_batch(self, *args, **kwargs):
                transitions.append((cpu_tree(args), cpu_tree(kwargs)))
                return True

            def release_ipc(self):
                pass

        def serve_action(_, tick, __, *, checkpoint_poll, **kwargs):
            nonlocal saved_binding
            if resume is None and tick == 3:
                requests.put(
                    {
                        "operation": "snapshot",
                        "request_id": "3",
                        "tick": 3,
                        "path": str(tmp_path / "worker.pt"),
                    }
                )
                checkpoint_poll()
                reply = responses.get_nowait()
                assert "error" not in reply
                saved_binding = {key: reply[key] for key in ("path", "sha256", "bytes")}
            obs, dones = torch.empty((2, 2)), torch.empty(2)
            ring.copy_observation_to(tick_id=tick, observations=obs, dones=dones, epoch=0)
            published_inputs.append((tick, obs.clone(), dones.clone()))
            ring.publish_action(tick_id=tick, policy_version=tick, actions=obs * 0.25, epoch=0)
            return True

        monkeypatch.setattr(worker_module, "_wait_for_inference_tick", serve_action)
        worker_module._run_collector(
            stop_event=stop,
            env_factory=lambda *args: Env(),
            num_envs=2,
            replay_buffer=Replay(),
            inference_slot=ring,
            inference_request_queue=queue.Queue(),
            inference_response_queue=queue.Queue(),
            algo_type="sac",
            actor_adapter_modules=None,
            metrics_queue=metrics,
            inference_transport="host",
            sim_backend="mujoco",
            backend_device=None,
            env_cfg_override=None,
            seed=61,
            trace_enabled=False,
            trace_thread_time=False,
            collector_metrics_interval=2,
            checkpoint_request_queue=requests,
            checkpoint_response_queue=responses,
            resume_collector_state=resume,
        )
        return published_inputs, transitions, events

    original_inputs, original_transitions, original_events = run()
    assert original_events == ["snapshot", "close"]
    assert saved_binding is not None
    saved = load_state(saved_binding)
    assert saved["inference_tick"] == 3 and saved["total_steps"] == 6
    assert saved["collector_metrics"]["_updates"] == 1
    resumed_inputs, resumed_transitions, resumed_events = run(saved_binding)
    assert resumed_events == ["restore", "close"]
    assert [item[0] for item in resumed_inputs] == [3, 4, 5]
    for before, after in zip(original_inputs[3:], resumed_inputs, strict=True):
        assert before[0] == after[0]
        assert torch.equal(before[1], after[1]) and torch.equal(before[2], after[2])
    for before, after in zip(original_transitions[3:], resumed_transitions, strict=True):
        for left, right in zip(before[0], after[0], strict=True):
            assert torch.equal(left, right)
        for name, value in before[1].items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, after[1][name])
            else:
                assert value == after[1][name]
