from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unilab.algos.torch.sonic_ppo.adaptive_state import (
    SonicAdaptiveStateError,
    capture_sampler_state,
    restore_sampler_state,
    sampler_layout_identity,
    sync_sampler_state,
)


class _Sampler:
    bin_size = 50
    bin_count = 3
    _clip_offsets = np.asarray([0, 100], dtype=np.int32)
    _clip_lengths = np.asarray([100, 75], dtype=np.int32)
    _bin_clip_indices = np.asarray([0, 0, 1], dtype=np.int32)
    _bin_local_starts = np.asarray([0, 50, 0], dtype=np.int32)
    _bin_local_ends = np.asarray([50, 100, 75], dtype=np.int32)

    def __init__(self) -> None:
        self.bin_episode_count = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
        self.bin_failed_count = np.asarray([1.0, 1.0, 2.0], dtype=np.float64)
        self._steps_since_probability_refresh = 4
        self._sampling_prob = np.empty(3, dtype=np.float64)
        self._sampling_cdf = np.empty(3, dtype=np.float64)

    def _refresh_sampling_probabilities(self) -> None:
        probability = self.bin_failed_count / self.bin_episode_count
        probability /= probability.sum()
        self._sampling_prob[...] = probability
        np.cumsum(probability, out=self._sampling_cdf)
        self._sampling_cdf[-1] = 1.0


def _env(*, global_mmap: str | None = "/tmp/metadata.json", shard: bool = False):
    return SimpleNamespace(
        motion_sampler=_Sampler(),
        _cfg=SimpleNamespace(
            motion_global_mmap_sidecar=global_mmap,
            motion_shard_clips=shard,
        ),
    )


def test_adaptive_state_round_trip_and_layout_identity() -> None:
    env = _env()
    state = capture_sampler_state(env)
    assert state is not None
    assert state["layout"] == sampler_layout_identity(env.motion_sampler)

    env.motion_sampler.bin_episode_count.fill(99.0)
    env.motion_sampler.bin_failed_count.fill(88.0)
    restore_sampler_state(env, state)
    np.testing.assert_allclose(env.motion_sampler.bin_episode_count, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(env.motion_sampler.bin_failed_count, [1.0, 1.0, 2.0])
    assert env.motion_sampler._steps_since_probability_refresh == 4


def test_adaptive_state_restore_reloads_active_pool_after_restoring_counters() -> None:
    class _ActivePoolSampler(_Sampler):
        def __init__(self) -> None:
            super().__init__()
            self.events: list[tuple[str, np.ndarray, np.ndarray]] = []

        def _refresh_sampling_probabilities(self) -> None:
            self.events.append(
                (
                    "refresh",
                    self.bin_episode_count.copy(),
                    self.bin_failed_count.copy(),
                )
            )
            super()._refresh_sampling_probabilities()

        def reload_active_motion_pool_after_restore(self) -> bool:
            self.events.append(
                (
                    "reload",
                    self.bin_episode_count.copy(),
                    self.bin_failed_count.copy(),
                )
            )
            return True

    sampler = _ActivePoolSampler()
    env = _env()
    env.motion_sampler = sampler
    state = capture_sampler_state(env)
    assert state is not None

    sampler.bin_episode_count.fill(99.0)
    sampler.bin_failed_count.fill(88.0)
    restore_sampler_state(env, state)

    assert [event[0] for event in sampler.events] == ["refresh", "reload"]
    for _, episode, failed in sampler.events:
        np.testing.assert_allclose(episode, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(failed, [1.0, 1.0, 2.0])


def test_adaptive_state_restore_rejects_non_callable_active_pool_hook() -> None:
    env = _env()
    state = capture_sampler_state(env)
    assert state is not None
    env.motion_sampler.reload_active_motion_pool_after_restore = None

    with pytest.raises(SonicAdaptiveStateError, match="must be callable"):
        restore_sampler_state(env, state)


def test_adaptive_state_rejects_layout_mismatch() -> None:
    env = _env()
    state = capture_sampler_state(env)
    assert state is not None
    state["layout"]["layout_sha256"] = "bad"
    with pytest.raises(SonicAdaptiveStateError, match="layout mismatch"):
        restore_sampler_state(env, state)


def test_rank_local_sampler_is_not_a_single_checkpoint_owner() -> None:
    global_env = _env()
    state = capture_sampler_state(global_env)
    assert state is not None

    rank_local_env = _env(global_mmap=None, shard=True)
    assert capture_sampler_state(rank_local_env) is None
    with pytest.raises(SonicAdaptiveStateError, match="restore requires global mmap"):
        restore_sampler_state(rank_local_env, state)


def test_adaptive_sync_is_noop_without_distributed_group() -> None:
    env = _env()
    assert sync_sampler_state(env, device=torch.device("cpu")) is False


def test_adaptive_sync_fails_closed_for_rank_local_shards(monkeypatch: pytest.MonkeyPatch) -> None:
    import unilab.algos.torch.sonic_ppo.adaptive_state as module

    class _Dist:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_initialized() -> bool:
            return True

        @staticmethod
        def get_world_size() -> int:
            return 2

        @staticmethod
        def all_gather_object(output, value) -> None:
            output[:] = [value, value]

    monkeypatch.setattr(module, "dist", _Dist)
    with pytest.raises(SonicAdaptiveStateError, match="global mmap"):
        sync_sampler_state(_env(global_mmap=None, shard=True), device=torch.device("cpu"))
