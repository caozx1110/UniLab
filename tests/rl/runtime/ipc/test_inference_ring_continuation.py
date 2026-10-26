"""A restarted collector must publish its held tick exactly once on new IPC."""

from __future__ import annotations

import pytest
import torch

from unilab.rl.runtime.ipc.inference_ring import SharedInferenceRing


@pytest.mark.parametrize("next_tick", [0, 5, 50097])
@pytest.mark.parametrize("capacity", [1, 2])
def test_new_ring_continues_nonzero_tick_without_replaying_action(next_tick, capacity):
    ring = SharedInferenceRing(3, 4, 2, capacity=capacity, device="cpu", epoch=9)
    try:
        ring.initialize_continuation_tick(next_tick)
        for tick in range(next_tick, next_tick + 3):
            obs = torch.full((3, 4), float(tick))
            dones = torch.tensor([0.0, 1.0, 0.0])
            ring.publish_observation(tick_id=tick, observations=obs, dones=dones, epoch=9)
            copied_obs, copied_dones = torch.empty_like(obs), torch.empty_like(dones)
            ring.copy_observation_to(
                tick_id=tick,
                observations=copied_obs,
                dones=copied_dones,
                epoch=9,
            )
            assert torch.equal(copied_obs, obs)
            assert torch.equal(copied_dones, dones)
            expected = torch.full((3, 2), float(tick + 1))
            ring.publish_action(tick_id=tick, policy_version=50000, actions=expected, epoch=9)
            actual, policy_version = ring.consume_action(tick_id=tick, epoch=9)
            assert torch.equal(actual, expected)
            assert policy_version == 50000
    finally:
        ring.close()


@pytest.mark.parametrize("tick", [-1, True, 1.5])
def test_continuation_rejects_invalid_first_tick(tick):
    ring = SharedInferenceRing(1, 2, 1, capacity=1, device="cpu")
    try:
        with pytest.raises(ValueError, match="non-negative integer"):
            ring.initialize_continuation_tick(tick)
    finally:
        ring.close()


def test_continuation_rejects_used_or_reinitialized_ring():
    ring = SharedInferenceRing(1, 2, 1, capacity=1, device="cpu")
    try:
        ring.publish_observation(tick_id=0, observations=torch.zeros(1, 2), dones=torch.zeros(1))
        with pytest.raises(RuntimeError, match="new empty"):
            ring.initialize_continuation_tick(5)
    finally:
        ring.close()
    ring = SharedInferenceRing(1, 2, 1, capacity=1, device="cpu")
    try:
        ring.initialize_continuation_tick(0)
        with pytest.raises(RuntimeError, match="already initialized"):
            ring.initialize_continuation_tick(0)
    finally:
        ring.close()


def test_continuation_rejects_noncreator(monkeypatch):
    ring = SharedInferenceRing(1, 2, 1, capacity=1, device="cpu")
    try:
        monkeypatch.setattr(ring, "_creator_pid", -123)
        with pytest.raises(RuntimeError, match="creator"):
            ring.initialize_continuation_tick(5)
    finally:
        ring.close()
