"""CPU placement retains logical device identity in the shared metric core."""

import pytest
import torch

from unilab.rl.runtime.offpolicy.collector_continuation import cpu_tree, restore_metric_state
from unilab.rl.runtime.offpolicy.metrics import CollectorMetrics, CollectorMetricState


def test_metric_cold_and_first_update_shapes_are_fixed_and_restore_raw_bits():
    original = CollectorMetrics(3, interval=2, device="cpu", reward_component_names=("a",))
    initial = {
        key: (value.shape, value.dtype)
        for key, value in vars(original).items()
        if isinstance(value, torch.Tensor)
    }
    original.update(
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([False, True, False]),
        torch.tensor([False, False, False]),
        torch.ones(3, 1),
    )
    shaped = {
        key: (value.shape, value.dtype)
        for key, value in vars(original).items()
        if isinstance(value, torch.Tensor)
    }
    assert initial == shaped
    saved = cpu_tree(vars(original))
    saved["device"] = torch.device("cuda:0")  # Fixture source metadata; values remain physical CPU.
    target = CollectorMetricState(
        3, interval=2, device="cuda:0", storage_device="cpu", reward_component_names=("a",)
    )
    assert target.device == saved["device"]
    before = torch.random.get_rng_state().clone()
    restore_metric_state(target, saved)
    assert torch.equal(before, torch.random.get_rng_state())
    assert set(vars(target)) == set(saved)
    for name, expected in saved.items():
        actual = getattr(target, name)
        if isinstance(expected, torch.Tensor):
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            assert actual.data_ptr() != expected.data_ptr()
            assert torch.equal(
                actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)
            )
        else:
            assert actual == expected


@pytest.mark.parametrize("damage", ["device", "dtype", "shape", "missing"])
def test_metric_shared_core_fails_closed_on_configuration_or_buffer_mismatch(damage):
    target = CollectorMetricState(3, interval=2, device="cuda:0", storage_device="cpu")
    saved = cpu_tree(vars(target))
    if damage == "device":
        saved["device"] = torch.device("cuda:1")
    elif damage == "dtype":
        saved["current_lengths"] = saved["current_lengths"].float()
    elif damage == "shape":
        saved["current_rewards"] = torch.zeros(2)
    else:
        del saved["_updates"]
    with pytest.raises(ValueError):
        restore_metric_state(target, saved)
