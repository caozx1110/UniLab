"""Tests for the TemporalLerp numpy port (Phase 5)."""

from __future__ import annotations

import numpy as np

from unilab.envs.gh_tracking.temporal_lerp import TemporalLerp


def test_linear_lerp_midpoint_and_done() -> None:
    tl = TemporalLerp((1, 1), default=0.0)
    tl.set(np.array([0]), end=np.array([[10.0]]), total_steps=10)
    for _ in range(5):
        tl.update_time(1)
    np.testing.assert_allclose(tl.current[0, 0], 5.0, atol=1e-6)  # halfway
    for _ in range(5):
        tl.update_time(1)
    np.testing.assert_allclose(tl.current[0, 0], 10.0)
    assert tl.mask_done[0]


def test_clamp_bounds() -> None:
    tl = TemporalLerp((1, 1), clamp=(5.0, 15.0))
    tl.set(np.array([0]), end=np.array([[100.0]]), total_steps=2)
    tl.update_time(2)
    np.testing.assert_allclose(tl.current[0, 0], 15.0)  # clamped to upper bound


def test_delta_mode_sets_end_relative_to_start() -> None:
    tl = TemporalLerp((1, 1), default=3.0)
    tl.set(np.array([0]), delta=np.array([[4.0]]), total_steps=1)
    tl.update_time(1)
    np.testing.assert_allclose(tl.current[0, 0], 7.0)  # 3 + 4


def test_reset_deactivates_and_holds_value() -> None:
    tl = TemporalLerp((1, 1))
    tl.set(np.array([0]), end=np.array([[1.0]]), total_steps=5)
    tl.reset(np.array([0]))
    assert tl.mask_done[0]
    tl.update_time(10)  # inactive -> value frozen
    np.testing.assert_allclose(tl.current[0, 0], 0.0)


def test_only_active_elements_advance() -> None:
    tl = TemporalLerp((2, 1))
    tl.set(np.array([0]), end=np.array([[10.0]]), total_steps=10)  # env 0 only
    for _ in range(10):
        tl.update_time(1)
    np.testing.assert_allclose(tl.current[0, 0], 10.0)
    np.testing.assert_allclose(tl.current[1, 0], 0.0)  # env 1 never set -> unchanged
