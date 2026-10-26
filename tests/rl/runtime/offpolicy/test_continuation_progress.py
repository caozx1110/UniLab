"""Large snapshot progress is byte-backed, never an unbounded heartbeat."""

import hashlib
import json
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from unilab.rl.runtime.offpolicy import continuation, lifecycle
from unilab.rl.runtime.offpolicy.coordination import LearnerCoordinationState
from unilab.rl.runtime.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner


def test_native_snapshot_keeps_shared_storage_rng_and_chunked_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(continuation, "_HASH_CHUNK_BYTES", 1024)
    monkeypatch.setattr(continuation, "_HASH_PROGRESS_SECONDS", 0.0)
    tensor = torch.arange(4096, dtype=torch.int64)
    rng = continuation.capture_rng_state()
    progress = []
    before_crc = torch.serialization.get_crc32_options()
    binding = continuation.save_state(
        tmp_path / "replay.pt", {"tensor": tensor, "view": tensor[3:9]}, progress=progress.append
    )
    assert torch.serialization.get_crc32_options() == before_crc
    # Snapshot I/O, its thread and progress reporting must not draw learner RNG.
    actual_rng = continuation.capture_rng_state()
    assert rng["python"] == actual_rng["python"]
    assert (rng["numpy"][1] == actual_rng["numpy"][1]).all()
    assert rng["numpy"][2:] == actual_rng["numpy"][2:]
    assert torch.equal(rng["torch_cpu"], actual_rng["torch_cpu"])
    actual = continuation.load_state(binding)
    assert torch.equal(actual["tensor"], tensor)
    assert actual["tensor"].untyped_storage().data_ptr() == (
        actual["view"].untyped_storage().data_ptr()
    )
    assert actual["view"].storage_offset() == 3
    for stage in ("write", "hash"):
        values = [item["bytes"] for item in progress if item["stage"] == stage]
        assert values and values == sorted(set(values))
        assert values[-1] == binding["bytes"]
    assert len([item for item in progress if item["stage"] == "hash"]) > 2
    assert binding["sha256"] == hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest()


def test_stalled_native_writer_gets_no_heartbeat(tmp_path, monkeypatch):
    monkeypatch.setattr(continuation, "_WRITE_POLL_SECONDS", 0.001)
    first_report = threading.Event()
    continue_write = threading.Event()
    reports = []
    errors = []

    def native(path, state):
        with path.open("wb", buffering=0) as stream:
            stream.write(b"a" * 64)
            assert continue_write.wait(3)
            stream.write(b"b" * 64)

    def progress(item):
        reports.append(item)
        if item == {"stage": "write", "bytes": 64}:
            first_report.set()

    def save():
        try:
            continuation.save_state(tmp_path / "probe.pt", {}, progress=progress)
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(continuation, "_save_native", native)
    worker = threading.Thread(target=save)
    worker.start()
    try:
        assert first_report.wait(3)
        time.sleep(0.03)
        assert reports == [{"stage": "write", "bytes": 64}]
        assert not (tmp_path / "probe.pt").exists()
    finally:
        continue_write.set()
        worker.join(timeout=3)
    assert not worker.is_alive() and not errors
    assert [item["bytes"] for item in reports if item["stage"] == "write"] == [64, 128]
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("progress", [None, lambda _: None])
def test_native_write_failure_restores_crc_and_never_commits(tmp_path, monkeypatch, progress):
    before_crc = torch.serialization.get_crc32_options()

    def fail(state, path):
        assert isinstance(path, str)  # No full-storage Python bytes adapter.
        assert torch.serialization.get_crc32_options() is False
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        continuation.save_state(tmp_path / "runtime.pt", {}, progress=progress)
    assert torch.serialization.get_crc32_options() == before_crc
    assert not (tmp_path / "runtime.pt").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_capture_has_bound_identity_and_advancing_health_before_hook(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "model_7.pt"
    checkpoint.write_bytes(b"native-checkpoint")
    clock = [1000.0]
    monkeypatch.setattr(lifecycle, "time", SimpleNamespace(time=lambda: clock[0]))
    status_path = tmp_path / "lifecycle.rank0.json"
    expected_token = f"7:{checkpoint.stat().st_mtime_ns}:{checkpoint.stat().st_size}"
    rng = torch.random.get_rng_state().clone()
    snapshots = []

    def status():
        value = json.loads(status_path.read_text())
        assert value["checkpoint_token"] == expected_token
        assert value["iteration"] == 7
        snapshots.append(value)
        return value

    def wait():
        assert status()["phase"] == "checkpoint_wait"
        return 19

    def capture(tick, progress):
        assert tick == 19
        assert status()["phase"] == "checkpoint_wait"
        for written in (8, 16, 24):
            clock[0] += 30  # Total save exceeds the production 45-second deadline.
            progress({"continuation_component": "replay", "stage": "write", "bytes": written})
            value = status()
            assert value["phase"] == "checkpoint_wait"
            assert value["updated_at"] == clock[0]
            assert value["details"]["bytes"] == written

    def hook(context):
        assert status()["phase"] == "checkpoint_hook"
        assert context.checkpoint_token == expected_token

    runner = object.__new__(DoubleBufferOffPolicyRunner)
    runner.device = "cpu"
    runner.learner = SimpleNamespace(update_count=7)
    runner.dp_sync = None
    runner._learner_coordination = LearnerCoordinationState()
    runner._lifecycle_hook = hook
    held = runner._checkpoint_lifecycle(
        iteration=7,
        log_dir=str(tmp_path),
        replay_pipeline=SimpleNamespace(paused=nullcontext, retained_state=lambda: {"ptr": 19}),
        wait_for_request=wait,
        capture_continuation=capture,
    )
    assert held == 19
    assert status()["phase"] == "running"
    assert snapshots[-1]["updated_at"] - snapshots[0]["updated_at"] == 90
    assert torch.equal(rng, torch.random.get_rng_state())
