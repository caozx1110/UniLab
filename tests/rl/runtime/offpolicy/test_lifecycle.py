"""CPU contracts for owner hooks and retained milestone state."""

import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from unilab.rl.runtime.distributed.sync import DpParameterSync
from unilab.rl.runtime.ipc.replay_pipelines.base import ReplayTickMetadata
from unilab.rl.runtime.ipc.replay_pipelines.device import DeviceReplayPipeline
from unilab.rl.runtime.offpolicy.coordination import LearnerCoordinationState, LearnerPhase
from unilab.rl.runtime.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner
from unilab.rl.runtime.offpolicy.lifecycle import (
    LifecycleContext,
    checkpoint_hook_state,
    publish_lifecycle_state,
    restore_hook_state,
    validate_checkpoint_iterations,
    validate_hook,
)


class StatefulHook:
    def __init__(self):
        self.phase = 7

    def __call__(self, context):
        pass

    def state_dict(self):
        return {"protocol": "fixed-owner-protocol", "phase": self.phase}

    def load_state_dict(self, state, *, iteration):
        if state["protocol"] != "fixed-owner-protocol" or state["phase"] != iteration:
            raise ValueError("Owner schedule identity/phase mismatch")
        self.phase = state["phase"]


def test_hook_state_roundtrip_and_owner_identity_rejection():
    hook = StatefulHook()
    validate_hook(hook)
    state = checkpoint_hook_state(hook, 7)
    restore_hook_state(hook, state, 7)
    with pytest.raises(ValueError, match="iteration mismatch"):
        restore_hook_state(hook, state, 8)
    state["hook"]["protocol"] = "another-protocol"
    with pytest.raises(ValueError, match="Owner schedule"):
        restore_hook_state(hook, state, 7)
    with pytest.raises(ValueError, match="original lifecycle"):
        restore_hook_state(None, state, 7)
    with pytest.raises(ValueError, match="original lifecycle"):
        restore_hook_state(hook, None, 7)


@pytest.mark.parametrize("values", [(True,), (0,), (-1,), (11,), (2, 2), (2.0,)])
def test_invalid_checkpoint_milestones_reject(values):
    with pytest.raises(ValueError):
        validate_checkpoint_iterations(values, 10)


def test_checkpoint_milestones_are_explicit_and_bounded():
    assert validate_checkpoint_iterations([2, 5, 10], 10) == {2, 5, 10}


def test_atomic_state_keeps_details_separate_from_identity(tmp_path):
    context = LifecycleContext(
        "checkpoint", object(), 7, 1, 2, str(tmp_path), "model_7.pt", "nonce"
    )
    publish_lifecycle_state(
        context, "checkpoint_hook", {"phase": "cannot replace", "child_pid": 99}
    )
    state = json.loads((tmp_path / "lifecycle.rank1.json").read_text())
    assert state["phase"] == "checkpoint_hook" and state["iteration"] == 7
    assert state["checkpoint_token"] == "nonce" and state["details"]["child_pid"] == 99
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(RuntimeError, match="checkpoint hook"):
        context.request_stop("unsupported without bound runner")


@pytest.mark.parametrize("rank", [0, 1])
def test_checkpoint_hook_parked_request_and_shared_checkpoint(tmp_path, rank):
    events = []
    checkpoint = tmp_path / "model_7.pt"
    torch.save({"update_count": 7}, checkpoint)
    runner = object.__new__(DoubleBufferOffPolicyRunner)
    runner.device = torch.device("cpu")
    runner.learner = SimpleNamespace(update_count=7)
    runner._learner_coordination = LearnerCoordinationState()
    runner.dp_sync = SimpleNamespace(
        rank=rank,
        world_size=2,
        barrier=lambda: events.append("entry_fence"),
        agree_stop_reason=lambda reason: events.append("exit_agreement") or reason,
    )

    @contextmanager
    def paused():
        assert events == ["held_request"]
        events.append("pause")
        yield
        events.append("resume")

    def hook(context):
        assert events == ["held_request", "pause", "entry_fence"]
        assert context.rank == rank and context.world_size == 2
        assert context.checkpoint_path == str(checkpoint)
        assert torch.load(context.checkpoint_path, weights_only=False)["update_count"] == 7
        assert runner._learner_coordination.snapshot()[0] == LearnerPhase.BUSY
        context.publish_progress({"child_phase": "checking"})
        context.request_stop("owner threshold reached")
        events.append("hook")

    runner._lifecycle_hook = hook
    before_rng = torch.random.get_rng_state().clone()
    held = runner._checkpoint_lifecycle(
        iteration=7,
        log_dir=str(tmp_path),
        replay_pipeline=SimpleNamespace(paused=paused, retained_state=lambda: {"visible_ptr": 19}),
        wait_for_request=lambda: events.append("held_request") or 19,
    )
    assert held == 19
    assert events == ["held_request", "pause", "entry_fence", "hook", "exit_agreement", "resume"]
    assert torch.equal(before_rng, torch.random.get_rng_state())
    status = json.loads((tmp_path / f"lifecycle.rank{rank}.json").read_text())
    assert status["phase"] == "stopping"
    assert runner._lifecycle_stop_reason == "owner threshold reached"
    audit = json.loads((tmp_path / f"lifecycle.retention.iter7.rank{rank}.json").read_text())
    assert audit["verified"] and audit["before"] == audit["after"]
    assert audit["before"]["collector_held_tick"] == 19


def test_failed_hook_unpauses_pipeline_and_publishes_failure(tmp_path):
    (tmp_path / "model_1.pt").write_bytes(b"complete")
    runner = object.__new__(DoubleBufferOffPolicyRunner)
    runner.device = torch.device("cpu")
    runner.learner = object()
    runner.dp_sync = None
    runner._learner_coordination = LearnerCoordinationState()
    exits = []

    @contextmanager
    def paused():
        try:
            yield
        finally:
            exits.append("unpaused")

    def fail(context):
        raise RuntimeError("evaluation failed")

    runner._lifecycle_hook = fail
    with pytest.raises(RuntimeError, match="evaluation failed"):
        runner._checkpoint_lifecycle(
            iteration=1,
            log_dir=str(tmp_path),
            replay_pipeline=SimpleNamespace(paused=paused, retained_state=lambda: {}),
            wait_for_request=lambda: 1,
        )
    assert exits == ["unpaused"]
    assert json.loads((tmp_path / "lifecycle.rank0.json").read_text())["phase"] == "failed"


def test_replay_pause_freezes_daemon_and_preserves_pending_state():
    pipeline = object.__new__(DeviceReplayPipeline)
    pipeline._work_lock = threading.RLock()
    pipeline._closed = False
    pipeline._prepare_error = None
    pipeline._sync_stream = SimpleNamespace(synchronize=lambda: None)
    pipeline._prepare_condition = threading.Condition()
    metadata = ReplayTickMetadata(12, 34, 34, 56, 2, batch_gpu_slot=0)
    pipeline._prepared_metadata = metadata
    pipeline._prepare_tick_id = 12
    pipeline._prepare_state = "gather_submitted"
    pipeline._hot, pipeline._cold = 1, 0
    pipeline._replay_buffer = SimpleNamespace(published_ptr=34, ptr=[32], size=[32])
    pipeline._gpu_storage = torch.zeros(4, 2)
    pipeline._gpu_packed = [torch.zeros(2, 2), torch.zeros(2, 2)]
    pipeline._submitted_ptr, pipeline._visible_ptr = 34, 32
    pipeline._span_events = []
    pipeline._prepare_required_ptr = 34
    pipeline._hot_metadata = None
    pipeline._has_hot_batch = False
    pipeline._base_seed = 44
    counts = []
    advanced = threading.Event()

    def pump():
        counts.append(1)
        advanced.set()
        return False

    pipeline._drain_completed_spans = pump
    pipeline._service_pending_prepare = lambda: False
    pipeline._submit_new_spans = lambda: False
    worker = threading.Thread(target=pipeline._sync_worker)
    worker.start()
    try:
        assert advanced.wait(2)
        with pytest.raises(RuntimeError, match="callback error"), pipeline.paused():
            before = pipeline.retained_state()
            count = len(counts)
            advanced.clear()
            assert not advanced.wait(0.03)
            assert len(counts) == count
            assert pipeline._prepared_metadata is metadata
            assert (
                pipeline._prepare_tick_id,
                pipeline._prepare_state,
                pipeline._hot,
                pipeline._cold,
            ) == (12, "gather_submitted", 1, 0)
            assert before == pipeline.retained_state()
            assert before["published_ptr"] == 34 and before["visible_ptr"] == 32
            assert before["prepared_metadata"]["sample_seed"] == 56
            raise RuntimeError("callback error")
        assert advanced.wait(2)
    finally:
        pipeline._closed = True
        worker.join(timeout=2)
    assert not worker.is_alive()


def test_checkpoint_rejects_replay_movement_during_hook(tmp_path):
    (tmp_path / "model_1.pt").write_bytes(b"complete")
    runner = object.__new__(DoubleBufferOffPolicyRunner)
    runner.device = torch.device("cpu")
    runner.learner = object()
    runner.dp_sync = None
    runner._learner_coordination = LearnerCoordinationState()
    replay = {"committed_ptr": 10}

    @contextmanager
    def paused():
        yield

    def corrupt(context):
        replay["committed_ptr"] += 1

    runner._lifecycle_hook = corrupt
    with pytest.raises(RuntimeError, match="changed retained"):
        runner._checkpoint_lifecycle(
            iteration=1,
            log_dir=str(tmp_path),
            replay_pipeline=SimpleNamespace(paused=paused, retained_state=lambda: dict(replay)),
            wait_for_request=lambda: 1,
        )
    audit = json.loads((tmp_path / "lifecycle.retention.iter1.rank0.json").read_text())
    assert not audit["verified"]
    assert audit["before"]["replay"]["committed_ptr"] == 10
    assert audit["after"]["replay"]["committed_ptr"] == 11


def test_lifecycle_checkpoint_contains_hook_state_and_is_atomic(tmp_path):
    runner = object.__new__(DoubleBufferOffPolicyRunner)
    runner.dp_sync = None
    runner.learner = SimpleNamespace(get_state_dict=lambda: {"update_count": 7})
    runner._lifecycle_hook = StatefulHook()
    path = runner._save_checkpoint(
        log_dir=str(tmp_path), iteration=7, logger=SimpleNamespace(log_save=lambda _: None)
    )
    state = torch.load(path, weights_only=False)
    assert state["lifecycle_state"]["hook"]["phase"] == state["update_count"] == 7
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("reasons", [[None, None], ["done", "done"], ["done", None]])
def test_distributed_stop_decision_requires_agreement(monkeypatch, reasons):
    sync = object.__new__(DpParameterSync)
    sync.world_size = 2
    monkeypatch.setattr(
        "unilab.rl.runtime.distributed.sync.dist.all_gather_object",
        lambda target, local: target.__setitem__(slice(None), reasons),
    )
    if reasons[0] != reasons[1]:
        with pytest.raises(RuntimeError, match="differs across ranks"):
            sync.agree_stop_reason(reasons[0])
    else:
        assert sync.agree_stop_reason(reasons[0]) == reasons[0]
