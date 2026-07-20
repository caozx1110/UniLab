"""Tests for GH termination (Phase 7): cum_error consecutive-exceed counter."""

from __future__ import annotations

import numpy as np

from unilab.envs.gh_tracking.terminations import (
    CumErrorTermination,
    apply_terminate_gate,
    compute_truncation,
    sample_reset_episode_length,
)


def test_cum_error_needs_51_consecutive_to_terminate() -> None:
    t = CumErrorTermination(num_envs=1, thres=1.0, min_steps=50)
    for _ in range(50):
        t.update(np.full((1, 3), 2.0))  # exceeded
    assert not t.terminated()[0, 0]  # count=50, 50 > 50 is False
    t.update(np.full((1, 3), 2.0))
    assert t.terminated()[0, 0]  # count=51, 51 > 50 is True (51 consecutive)


def test_cum_error_count_resets_on_non_exceed() -> None:
    t = CumErrorTermination(1, 1.0, 50)
    for _ in range(40):
        t.update(np.full((1, 3), 2.0))
    t.update(np.zeros((1, 3)))  # not exceeded -> count reset to 0
    np.testing.assert_array_equal(t.error_exceeded_count, [[0]])


def test_cum_error_any_component_exceeds() -> None:
    t = CumErrorTermination(1, 1.0, 50)
    cum = np.array([[0.0, 5.0, 0.0]])  # only rot component exceeds
    t.update(cum)
    np.testing.assert_array_equal(t.error_exceeded_count, [[1]])


def test_cum_error_reset_zeros_count() -> None:
    t = CumErrorTermination(2, 1.0, 50)
    for _ in range(10):
        t.update(np.full((2, 3), 2.0))
    t.reset(np.array([0]))
    np.testing.assert_array_equal(t.error_exceeded_count[0], [0])
    np.testing.assert_array_equal(t.error_exceeded_count[1], [10])


def test_terminate_gate_masks_first_5_steps() -> None:
    terminated = np.array([[True], [True]])
    episode_len = np.array([5, 6])  # env 0 at step 5 (masked), env 1 at 6 (allowed)
    out = apply_terminate_gate(terminated, episode_len)
    np.testing.assert_array_equal(out, [[False], [True]])  # gate: episode_length > 5


def test_truncation_timeout_or_finished() -> None:
    episode_len = np.array([1000, 3, 3])
    finished = np.array([False, True, False])
    out = compute_truncation(episode_len, max_episode_length=1000, finished=finished)
    np.testing.assert_array_equal(out.reshape(-1), [True, True, False])  # timeout OR clip finished


def test_reset_episode_length_bounds() -> None:
    rng = np.random.default_rng(0)
    # small reset batch (< 20%): randint(0, max//5)
    small = sample_reset_episode_length(n_reset=10, n_total=1000, max_episode_length=1000, rng=rng)
    assert small.shape == (10,)
    assert (small >= 0).all() and (small < 200).all()
    # large reset batch (>= 20%): randint(0, max)
    large = sample_reset_episode_length(n_reset=500, n_total=1000, max_episode_length=1000, rng=rng)
    assert (large >= 0).all() and (large < 1000).all()
    assert large.max() >= 200  # spans the full range
