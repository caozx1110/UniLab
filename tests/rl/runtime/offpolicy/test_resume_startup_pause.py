"""CPU regression for keeping restored replay parked through collector startup."""

import json
import queue
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from unilab.rl.runtime.offpolicy import continuation, lifecycle
from unilab.rl.runtime.offpolicy import double_buffer_runner as runner_module

from .test_runner_unit import _FakeLogger, _FakePipeline, _FakeReplayBuffer, _make_device_runner


class _StartupError(RuntimeError):
    pass


@pytest.mark.parametrize(
    "failure_stage",
    [
        None,
        "restore",
        "collector_start",
        "collector_ready",
        "capture",
        "hook",
        "died",
        "cold_ready",
        "cold_stall",
        "checkpoint_stall",
    ],
)
def test_restore_pause_blocks_replay_until_snapshot_and_hook_then_releases_on_all_exits(
    monkeypatch, tmp_path, failure_stage
):
    events = []
    pipeline_holder = []
    wait_calls = []
    queue_reads = []
    restored_rng = []
    output = tmp_path / "resumed"
    watcher = None
    clock = [1000.0]
    if failure_stage in {"cold_ready", "cold_stall", "checkpoint_stall"}:
        # Downstream integration is optional for the generic UniLab suite. The
        # SONIC acceptance overlay supplies these actual production guards.
        goal_progress = pytest.importorskip("sonic_unilab.goal_progress")
        lifecycle_health = pytest.importorskip("sonic_unilab.lifecycle_health")
        fake_time = SimpleNamespace(time=lambda: clock[0], monotonic=lambda: clock[0])
        for module in (goal_progress, lifecycle_health, lifecycle):
            monkeypatch.setattr(module, "time", fake_time)
        watcher = goal_progress.TrainingProgress(
            {
                "world_size": 1,
                "num_envs_per_rank": 2,
                "startup_stale_seconds": 1800,
                "running_stale_seconds": 300,
                "lifecycle_pause": True,
            },
            tmp_path,
        )
        watcher.training = output
        watcher.root.mkdir()

    class Pipeline(_FakePipeline):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.lock = threading.RLock()
            self.depth = 0
            self.pending_pointer = 19
            pipeline_holder.append(self)

        @contextmanager
        def paused(self):
            with self.lock:
                self.depth += 1
                try:
                    yield
                finally:
                    self.depth -= 1

        def retained_state(self):
            return {"pending_pointer": self.pending_pointer}

        def progress(self):
            assert self.depth == 0, "Cold resume must not drive replay while it is paused"
            return True

        def probe_daemon(self, stage, *, paused):
            observed = []

            def attempt_pending_commit():
                acquired = self.lock.acquire(blocking=False)
                observed.append(acquired)
                if acquired:
                    try:
                        self.pending_pointer += 1
                    finally:
                        self.lock.release()

            worker = threading.Thread(target=attempt_pending_commit)
            worker.start()
            worker.join(timeout=5)
            assert not worker.is_alive(), "replay probe thread deadlocked"
            assert observed == [not paused], (stage, self.depth, observed)
            events.append((stage, self.depth))
            if stage == failure_stage:
                raise _StartupError(stage)

    monkeypatch.setattr(runner_module, "ReplayBuffer", _FakeReplayBuffer)
    monkeypatch.setattr(runner_module, "DeviceReplayPipeline", Pipeline)
    monkeypatch.setattr(runner_module, "OffPolicyLogger", _FakeLogger)
    # No child process is launched here; avoid allocating unrelated OS queue
    # semaphores while testing the real runner's startup/cleanup ordering.
    monkeypatch.setattr(runner_module, "_SPAWN_CTX", SimpleNamespace(Queue=queue.Queue))
    monkeypatch.setattr(runner_module.time, "sleep", lambda _: None)
    runner = _make_device_runner(monkeypatch, device="cpu")
    runner.resume_start_iteration = 65
    runner.learner.update_count = 64
    runner._resume_continuation = {"collector": {"path": "not-used-by-cpu-fixture"}}
    runner._save_continuation_state = True
    runner._checkpoint_iterations = frozenset()
    runner._lifecycle_pause_seconds = 0.0
    runner._lifecycle_stop_reason = None
    native = tmp_path / "original.pt"
    native.write_bytes(b"byte-identical native checkpoint")
    runner._resume_checkpoint_path = str(native)

    def prepare(**kwargs):
        assert kwargs["replay_pipeline"].depth == 0  # cold preparation precedes restoration

    def restore(pipeline):
        # Match the real restore's nested pause. A new outer startup pause must
        # stay acquired after this inner context exits.
        assert pipeline.depth == 1
        with pipeline.paused():
            pipeline.probe_daemon("restore", paused=True)
        assert pipeline.depth == 1
        restored_rng.append(continuation.capture_rng_state())
        return {
            "held_tick": 162,
            "rng": restored_rng[-1],
            "runner": {
                "inference_scheduler": {
                    "env_steps_per_sync": 1,
                    "next_tick": 162,
                    "policy_version": 64,
                    "_ticks_since_update": 0,
                    "_pending_tick": None,
                },
                "reward_stats_ptr": 19,
                "prepared_tick": 65,
                "reward_history": [],
                "reward_components": {"_sums": {}, "_counts": {}},
                "has_logged_reward": False,
                "last_buf_log": 0,
                "write_read_ema": 0.0,
                "best_mean_reward": -float("inf"),
                "last_mean_reward": 0.0,
                "logger": {
                    "_total_steps": 324,
                    "_buffer_size": 19,
                    "_buffer_target": 16,
                    "_mean_ep_length": 0.0,
                    "_timeout_rate": 0.0,
                    "_buffer_utilization": 1.0,
                },
            },
        }

    def start_collector(**kwargs):
        pipeline_holder[0].probe_daemon("collector_start", paused=True)

    original_wait = runner._wait_for_inference_request

    def wait_for_request(*args, expected_tick, **kwargs):
        wait_calls.append(expected_tick)
        assert wait_calls == [162], "Saved held tick must be consumed exactly once"
        assert kwargs["progress_replay"] is False
        assert not (output / "lifecycle.rank0.json").exists()
        assert not (output / "model_64.pt").exists()
        pipeline = pipeline_holder[0]
        pipeline.probe_daemon("collector_ready", paused=True)
        if failure_stage == "died":

            class EmptyQueue:
                def get(self, **options):
                    raise queue.Empty

            assert kwargs["defer_collector_failure_cleanup"] is True
            return original_wait(EmptyQueue(), expected_tick=expected_tick, **kwargs)
        assert expected_tick == 162
        if watcher is not None:

            class ColdQueue:
                def get(self, **options):
                    assert not (output / "lifecycle.rank0.json").exists()
                    assert not (output / "model_64.pt").exists()
                    queue_reads.append(len(queue_reads))
                    if len(queue_reads) <= 2:
                        clock[0] += 60
                        (watcher.root / "init.json").write_text(
                            json.dumps(
                                {
                                    "schema": "sonic.env-init-health.v1",
                                    "pid": 123,
                                    "rank": 0,
                                    "world_size": 1,
                                    "frames_done": len(queue_reads) * 65536,
                                    "updated_at": clock[0],
                                }
                            )
                        )
                        assert watcher.check(force=True)["lifecycle_pause"] is None
                        pipeline.probe_daemon("cold_wait", paused=True)
                        if failure_stage == "cold_stall":
                            clock[0] += 1801
                            with pytest.raises(TimeoutError, match="init counters stopped"):
                                watcher.check(force=True)
                            raise _StartupError("startup stall correctly rejected")
                        raise queue.Empty
                    if len(queue_reads) == 3:
                        return runner_module.COLLECTOR_READY_TICK
                    assert len(queue_reads) == 4
                    return expected_tick

            runner._collector_ready = False
            return original_wait(ColdQueue(), expected_tick=expected_tick, **kwargs)
        return expected_tick

    def capture(**kwargs):
        pipeline = pipeline_holder[0]
        assert pipeline.depth == 2  # startup pause plus lifecycle's reentrant pause
        pipeline.probe_daemon("capture", paused=True)
        assert pipeline.pending_pointer == 19
        state = kwargs["runner_state"]()
        assert state["prepared_tick"] == 65
        assert state["inference_scheduler"]["next_tick"] == 162
        assert state["inference_scheduler"]["_pending_tick"] is None
        assert runner.learner.update_count == 64
        assert kwargs["held_tick"] == 162
        assert Path(kwargs["log_dir"], "model_64.pt").read_bytes() == native.read_bytes()
        actual_rng = continuation.capture_rng_state()
        assert actual_rng["python"] == restored_rng[-1]["python"]
        assert (actual_rng["numpy"][1] == restored_rng[-1]["numpy"][1]).all()
        assert actual_rng["numpy"][2:] == restored_rng[-1]["numpy"][2:]
        assert torch.equal(actual_rng["torch_cpu"], restored_rng[-1]["torch_cpu"])
        if watcher is not None:
            assert len(queue_reads) == 4
            assert watcher.check(force=True)["lifecycle_pause"]["iteration"] == 64
            if failure_stage == "checkpoint_stall":
                clock[0] += 46
                with pytest.raises(TimeoutError, match="heartbeat stale"):
                    watcher.check(force=True)
                raise _StartupError("checkpoint stall correctly rejected")

    def hook(context):
        pipeline = pipeline_holder[0]
        if context.phase == "checkpoint":
            assert pipeline.depth == 2
            pipeline.probe_daemon("hook", paused=True)
            assert pipeline.pending_pointer == 19
        else:
            assert context.iteration == 65 and pipeline.depth == 0
            pipeline.probe_daemon("next_iteration", paused=False)
            raise _StartupError("probe reached next iteration")

    def shutdown():
        pipeline = pipeline_holder[0]
        assert pipeline.depth == 0
        pipeline.probe_daemon("shutdown", paused=False)

    def collector_died(*args):
        pipeline = pipeline_holder[0]
        assert pipeline.depth == 0  # release must precede this branch's pipeline.close()
        pipeline.probe_daemon("collector_failure_cleanup", paused=False)
        raise _StartupError("collector failure reported")

    runner._lifecycle_hook = hook
    monkeypatch.setattr(runner, "_prepare_learner", prepare)
    monkeypatch.setattr(runner, "_restore_continuation", restore)
    monkeypatch.setattr(runner, "_start_collector", start_collector)
    monkeypatch.setattr(runner, "_check_collector_alive", lambda: failure_stage != "died")
    monkeypatch.setattr(runner, "_wait_for_inference_request", wait_for_request)
    monkeypatch.setattr(runner, "_save_continuation", capture)
    monkeypatch.setattr(runner, "_shutdown_collector", shutdown)
    monkeypatch.setattr(runner, "_fail_collector_died", collector_died)
    try:
        with pytest.raises(_StartupError):
            runner._learn_impl(max_iterations=65, save_interval=0, log_dir=str(output))
        pipeline = pipeline_holder[0]
        assert pipeline.depth == 0
        assert events[-1] == ("shutdown", 0)
        if failure_stage is None:
            assert events[:-1] == [
                ("restore", 2),
                ("collector_start", 1),
                ("collector_ready", 1),
                ("capture", 2),
                ("hook", 2),
                ("next_iteration", 0),
            ]
        if failure_stage == "died":
            assert ("collector_failure_cleanup", 0) in events
        if failure_stage == "cold_ready":
            assert len(queue_reads) == 4 and wait_calls == [162]
            assert events.count(("cold_wait", 1)) == 2
            assert ("capture", 2) in events and ("hook", 2) in events
            assert ("next_iteration", 0) in events
        if failure_stage == "cold_stall":
            assert not (output / "model_64.pt").exists()
            assert not (output / "lifecycle.rank0.json").exists()
    finally:
        # Restore the real cleanup method before closing test-owned IPC resources.
        monkeypatch.undo()
        runner.close()
