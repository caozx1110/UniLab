"""Portable replay data must survive a new owner with pending ingress/prefetch."""

from __future__ import annotations

import threading
from collections import deque

import pytest
import torch

from unilab.rl.runtime.ipc.replay_buffer import ReplayBuffer
from unilab.rl.runtime.ipc.replay_pipelines.base import ReplayTickMetadata
from unilab.rl.runtime.ipc.replay_pipelines.device import DeviceReplayPipeline


class _CompletedEvent:
    def record(self, *_args):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass


def _pipeline() -> DeviceReplayPipeline:
    """Exercise portable serialization on CPU without pretending CPU training is supported."""
    pipeline = object.__new__(DeviceReplayPipeline)
    pipeline._replay_buffer = ReplayBuffer(
        capacity=8,
        obs_dim=2,
        action_dim=1,
        device="cpu",
        ingress_slot_rows=2,
        ingress_depth=3,
    )
    pipeline._device = torch.device("cpu")
    pipeline._capacity = 8
    pipeline._storage_width = pipeline._replay_buffer.storage_width
    pipeline._packed_width = pipeline._storage_width
    pipeline._sample_count = 4
    pipeline._base_seed = 7
    pipeline._gpu_storage = torch.zeros(8, pipeline._storage_width)
    pipeline._gpu_packed = [torch.zeros(4, pipeline._storage_width) for _ in range(2)]
    pipeline._submitted_ptr = pipeline._visible_ptr = 0
    pipeline._hot, pipeline._cold = 0, 1
    pipeline._has_hot_batch = False
    pipeline._hot_metadata = pipeline._prepared_metadata = None
    pipeline._prepare_tick_id = None
    pipeline._prepare_required_ptr = 0
    pipeline._prepare_state = "idle"
    pipeline._prepare_error = None
    pipeline._ingress_high_water = 0
    pipeline._span_events = deque()
    pipeline._slot_events = [_CompletedEvent(), _CompletedEvent()]
    pipeline._sync_stream = None
    pipeline._work_lock = threading.RLock()
    pipeline._submission_lock = threading.Lock()
    pipeline._prepare_condition = threading.Condition()
    pipeline._closed = False
    pipeline._main_thread_submission = False
    pipeline._trace_recorder = None
    pipeline._trace_cuda_events = False
    return pipeline


def _publish(rb: ReplayBuffer, start: int) -> None:
    rows = torch.arange(start, start + 2, dtype=torch.float32)
    assert rb.add_batch(
        obs=rows[:, None].expand(-1, 2),
        next_obs=(rows + 1)[:, None].expand(-1, 2),
        actions=rows[:, None],
        rewards=rows,
        dones=torch.zeros(2),
        truncated=torch.zeros(2),
    )


def _pending_state() -> DeviceReplayPipeline:
    pipeline = _pipeline()
    rb = pipeline._replay_buffer
    _publish(rb, 0)
    slot, start, count, data = rb.take_published_ingress()
    pipeline._gpu_storage[:2].copy_(data)
    rb.commit_ingress(slot=slot, start=start, count=count)
    pipeline._visible_ptr = pipeline._submitted_ptr = 2
    # Preserve an already gathered next batch, even if the ring later changes.
    pipeline._gpu_packed[1].copy_(pipeline._gpu_storage[:2].repeat(2, 1))
    pipeline._prepared_metadata = ReplayTickMetadata(11, 2, 2, 18, 4, batch_gpu_slot=1)
    pipeline._prepare_tick_id = 11
    pipeline._prepare_required_ptr = 2
    pipeline._prepare_state = "ready"
    _publish(rb, 2)
    slot, start, count, data = rb.take_published_ingress()
    pipeline._gpu_storage[2:4].copy_(data)
    pipeline._submitted_ptr = 4
    pipeline._span_events.append((4, _CompletedEvent(), slot, start, count))
    _publish(rb, 4)
    return pipeline


def test_pending_ingress_and_prefetch_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.mps, "Event", _CompletedEvent)
    original = _pending_state()
    with original.paused():
        before = original.retained_state()
        snapshot = original.export_continuation_state()
        assert original.retained_state() == before
    path = tmp_path / "replay.pt"
    torch.save(snapshot, path)
    snapshot = torch.load(path, weights_only=False)
    restored = _pipeline()
    with restored.paused():
        restored.import_continuation_state(snapshot)
    assert torch.equal(restored._gpu_storage[:4], original._gpu_storage[:4])
    assert restored._replay_buffer.published_ptr == 6
    assert restored._visible_ptr == 2
    assert restored._submitted_ptr == 4
    assert torch.equal(
        restored.sample_large_batch(11, 4)["rewards"], torch.tensor([0.0, 1.0, 0.0, 1.0])
    )
    assert restored._drain_completed_spans()
    assert int(restored._replay_buffer.ptr[0]) == 4
    assert restored._submit_new_spans()
    assert restored._drain_completed_spans()
    assert int(restored._replay_buffer.ptr[0]) == 6
    assert torch.equal(
        restored._gpu_storage[:6, restored._replay_buffer._rew_col], torch.arange(6.0)
    )
    # Restored free/ready semaphores and absolute sequence offsets must remain usable.
    _publish(restored._replay_buffer, 6)
    assert restored._submit_new_spans()
    assert restored._drain_completed_spans()
    assert restored._visible_ptr == 8
    assert torch.equal(
        restored._gpu_storage[:, restored._replay_buffer._rew_col], torch.arange(8.0)
    )


def test_replay_export_requires_pause_and_restores_owned_copy():
    pipeline = _pending_state()
    with pytest.raises(RuntimeError, match="paused"):
        pipeline.export_continuation_state()
    with pipeline.paused():
        snapshot = pipeline.export_continuation_state()
    pipeline._gpu_storage.zero_()
    assert snapshot["storage"].count_nonzero() > 0


def test_wrapped_ring_preserves_absolute_sampling_position_and_prefetch(monkeypatch):
    monkeypatch.setattr(torch.mps, "Event", _CompletedEvent)
    original = _pending_state()
    original._drain_completed_spans()
    original._submit_new_spans()
    original._drain_completed_spans()
    for start in (6, 8, 10):
        _publish(original._replay_buffer, start)
        original._submit_new_spans()
        original._drain_completed_spans()
    assert original._visible_ptr == 12
    with original.paused():
        state = original.export_continuation_state()
    assert state["storage"].shape[0] == 8
    restored = _pipeline()
    with restored.paused():
        restored.import_continuation_state(state)
    assert int(restored._replay_buffer.ptr[0]) == 12
    assert int(restored._replay_buffer.size[0]) == 8
    assert torch.equal(restored._gpu_storage, original._gpu_storage)
    # Rows 0 and 1 have been overwritten, but the already sampled batch still
    # contains their earlier values. Re-gathering from the ring would be wrong.
    assert restored._gpu_storage[0, restored._replay_buffer._rew_col] == 8
    assert torch.equal(
        restored.sample_large_batch(11, 4)["rewards"], torch.tensor([0.0, 1.0, 0.0, 1.0])
    )
    for pipeline in (original, restored):
        generator = torch.Generator(device="cpu").manual_seed(pipeline._base_seed + 12)
        pipeline._gather_rows(visible_size=8, slot=0, gen=generator)
    assert torch.equal(original._gpu_packed[0], restored._gpu_packed[0])


@pytest.mark.parametrize("damage", ["seed", "slot", "span", "sampler"])
def test_replay_continuation_rejects_incompatible_or_incomplete_data(damage):
    original = _pending_state()
    with original.paused():
        state = original.export_continuation_state()
    if damage == "seed":
        state["base_seed"] += 1
    elif damage == "slot":
        state["slots"][1] = None
    elif damage == "span":
        state["pending_spans"][0][0] += 1
    else:
        state["sampler"]["type"] = "different-sampler"
    restored = _pipeline()
    with restored.paused(), pytest.raises(ValueError):
        restored.import_continuation_state(state)
    assert restored._replay_buffer.published_ptr == 0
