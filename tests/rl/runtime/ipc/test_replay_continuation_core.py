"""Cold CPU carriers invoke the exact numeric restore core used on device."""

import copy

import pytest
import torch

from unilab.rl.runtime.ipc.replay_buffer import ReplayBuffer
from unilab.rl.runtime.ipc.replay_pipelines.continuation import HostReplayContinuation

from .test_replay_continuation import _pending_state


def host():
    ingress = ReplayBuffer(
        capacity=8,
        obs_dim=2,
        action_dim=1,
        device="cuda:0",
        ingress_slot_rows=2,
        ingress_depth=3,
        ingress_device="cpu",
    )
    return HostReplayContinuation(
        ingress,
        sample_count=4,
        base_seed=7,
        logical_device="cuda:0",
        sampler={"type": "unilab.rl.runtime.ipc.replay_pipelines.device.DeviceReplayPipeline"},
    )


def test_host_cold_constructor_restores_pending_prefetch_without_rng_draw_or_alias():
    original = _pending_state()
    with original.paused():
        saved = original.export_continuation_state()
    # Raw-bit identity includes NaN payloads and signed zero, not tolerance.
    saved["storage"].view(torch.int32)[0, 0] = 0x7FC00123
    saved["storage"].view(torch.int32)[0, 1] = -2147483648
    target = host()
    assert target.logical_device == torch.device("cuda:0")
    assert target._gpu_storage.device.type == "cpu"
    assert target._gpu_storage.shape == (8, 8)
    assert target._submitted_ptr == 0 and target._prepare_tick_id is None
    rng = torch.random.get_rng_state().clone()
    target.import_continuation_state(saved)
    actual = target.continuation_value_views()
    assert torch.equal(actual["storage"].view(torch.uint8), saved["storage"].view(torch.uint8))
    assert torch.equal(actual["slots"][1], saved["slots"][1])
    assert actual["storage"].data_ptr() == target._gpu_storage.data_ptr()
    assert actual["storage"].data_ptr() != saved["storage"].data_ptr()
    assert actual["pending_spans"] == saved["pending_spans"]
    assert actual["pending_spans"] is not saved["pending_spans"]
    assert actual["prepared_metadata"] == saved["prepared_metadata"]
    assert torch.equal(rng, torch.random.get_rng_state())
    saved["slots"][1].zero_()
    saved["storage"].zero_()
    assert actual["slots"][1].count_nonzero() > 0
    assert actual["storage"].view(torch.int32)[0, 0] == 0x7FC00123


@pytest.mark.parametrize("damage", ["layout", "storage_dtype", "slot_shape", "seed", "sampler"])
def test_numeric_core_rejects_incompatible_values_before_ingress_mutation(damage):
    original = _pending_state()
    with original.paused():
        saved = original.export_continuation_state()
    saved = copy.copy(saved)
    if damage == "layout":
        saved["layout"] = [9, 8, 4]
    elif damage == "storage_dtype":
        saved["storage"] = saved["storage"].to(torch.float64)
    elif damage == "slot_shape":
        saved["slots"] = [None, torch.zeros(3, 8)]
    elif damage == "seed":
        saved["base_seed"] = 8
    else:
        saved["sampler"] = {"type": "changed"}
    target = host()
    with pytest.raises(ValueError):
        target.import_continuation_state(saved)
    assert target._submitted_ptr == target._replay_buffer.published_ptr == 0
