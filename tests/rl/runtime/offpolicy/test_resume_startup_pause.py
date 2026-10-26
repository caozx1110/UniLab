"""CPU regression for keeping restored replay parked through collector startup."""

import queue
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from unilab.rl.runtime.offpolicy import continuation
from unilab.rl.runtime.offpolicy import double_buffer_runner as runner_module

from .test_runner_unit import _FakeLogger, _FakePipeline, _FakeReplayBuffer, _make_device_runner


class _StartupError(RuntimeError):
    pass


@pytest.mark.parametrize(
    "failure_stage",
    [None, "restore", "collector_start", "collector_ready", "capture", "hook", "died"],
)
def test_restore_pause_blocks_replay_until_snapshot_and_hook_then_releases_on_all_exits(
    monkeypatch, tmp_path, failure_stage
):
    events = []
    pipeline_holder = []

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
        return {
            "held_tick": 162,
            "rng": continuation.capture_rng_state(),
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
        pipeline = pipeline_holder[0]
        pipeline.probe_daemon("collector_ready", paused=True)
        if failure_stage == "died":

            class EmptyQueue:
                def get(self, **options):
                    raise queue.Empty

            assert kwargs["defer_collector_failure_cleanup"] is True
            return original_wait(EmptyQueue(), expected_tick=expected_tick, **kwargs)
        assert expected_tick == 162
        return expected_tick

    def capture(**kwargs):
        pipeline = pipeline_holder[0]
        assert pipeline.depth == 2  # startup pause plus lifecycle's reentrant pause
        pipeline.probe_daemon("capture", paused=True)
        assert pipeline.pending_pointer == 19
        assert kwargs["runner_state"]()["prepared_tick"] == 65
        assert Path(kwargs["log_dir"], "model_64.pt").read_bytes() == native.read_bytes()

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
    output = tmp_path / "resumed"
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
    finally:
        # Restore the real cleanup method before closing test-owned IPC resources.
        monkeypatch.undo()
        runner.close()
