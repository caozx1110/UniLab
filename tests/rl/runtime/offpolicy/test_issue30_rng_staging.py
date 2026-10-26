"""Host RNG restoration shares production core; opaque device bytes stay opaque."""

import copy
import random

import numpy as np
import pytest
import torch

from unilab.rl.runtime.offpolicy.continuation import (
    capture_rng_state,
    restore_host_rng_state,
    restore_rng_state,
    stage_rng_state,
)


def test_host_state_restores_next_draws_from_real_cold_generators():
    original = capture_rng_state()
    try:
        random.seed(91)
        np.random.seed(92)
        torch.manual_seed(93)
        state = capture_rng_state()
        python_cold = random.Random(1)
        numpy_cold = np.random.RandomState(2)
        torch_cold = torch.Generator(device="cpu").manual_seed(3)
        python_cold.setstate(state["python"])
        numpy_cold.set_state(state["numpy"])
        torch_cold.set_state(state["torch_cpu"])
        random.random()
        np.random.random_sample(9)
        torch.rand(17)
        restore_rng_state(state)
        assert [random.random() for _ in range(9)] == [python_cold.random() for _ in range(9)]
        assert np.array_equal(np.random.random_sample(9), numpy_cold.random_sample(9))
        assert torch.equal(torch.rand(9), torch.rand(9, generator=torch_cold))
    finally:
        restore_rng_state(original)


def test_staging_preserves_device_bytes_without_materializing_cuda(monkeypatch):
    original = capture_rng_state()
    state = copy.deepcopy(original)
    state["device_type"] = "cuda"
    state["cuda"] = [torch.arange(16, dtype=torch.uint8)]
    staged = stage_rng_state(state)
    assert torch.equal(staged["cuda"][0], state["cuda"][0])
    assert staged["cuda"][0].data_ptr() != state["cuda"][0].data_ptr()
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(ValueError, match="device count"):
        restore_rng_state(state)
    restore_host_rng_state(state)
    assert torch.equal(torch.random.get_rng_state(), original["torch_cpu"])


@pytest.mark.parametrize("change", ["dtype", "shape", "extra", "device", "numpy"])
def test_invalid_rng_rejected_before_host_mutation(change):
    state = capture_rng_state()
    before = torch.random.get_rng_state()
    if change == "dtype":
        state["torch_cpu"] = state["torch_cpu"].to(torch.int64)
    elif change == "shape":
        state["torch_cpu"] = state["torch_cpu"].reshape(1, -1)
    elif change == "extra":
        state["unknown"] = 1
    elif change == "device":
        state["device_type"] = "unknown"
    else:
        state["numpy"] = ("invalid",)
    with pytest.raises((ValueError, TypeError)):
        restore_host_rng_state(state)
    assert torch.equal(before, torch.random.get_rng_state())
