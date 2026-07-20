"""Backend contract: 6D body wrench + per-substep wrench recompute (GH migration).

- P0-1: ``apply_body_wrench`` writes the full 6D ``xfrc_applied`` (force + torque).
- P0-2 / §4.3: per-physics-substep wrench recompute hook + post-step callback so
  compliant forces track the current body pose each substep (not frozen at
  substep 0), and the last substep is covered.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unilab.base.backend.mujoco.backend import MuJoCoBackend


def _fake_xfrc_backend(nbody: int = 3, num_envs: int = 2) -> SimpleNamespace:
    b = SimpleNamespace(
        _num_envs=num_envs,
        _pending_xfrc_applied=np.zeros((num_envs, 6 * nbody), dtype=np.float64),
    )
    # Bind the real force-slice resolver so apply_body_force works on the fake.
    b._resolve_push_body_force_slice = MuJoCoBackend._resolve_push_body_force_slice.__get__(b)
    return b


# --------------------------------------------------------------------------- #
# 1.1 apply_body_wrench (6D)                                                   #
# --------------------------------------------------------------------------- #


def test_apply_body_wrench_writes_force_and_torque_halves() -> None:
    b = _fake_xfrc_backend()
    force = np.tile(np.array([1.0, 2.0, 3.0]), (2, 1)).reshape(2, 1, 3)
    torque = np.tile(np.array([4.0, 5.0, 6.0]), (2, 1)).reshape(2, 1, 3)

    MuJoCoBackend.apply_body_wrench(b, np.array([1]), force, torque)

    base = 6 * 1
    np.testing.assert_allclose(b._pending_xfrc_applied[:, base : base + 3], force[:, 0, :])
    np.testing.assert_allclose(b._pending_xfrc_applied[:, base + 3 : base + 6], torque[:, 0, :])
    assert not b._pending_xfrc_applied[:, 0:6].any()  # body 0 untouched


def test_apply_body_wrench_force_half_matches_apply_body_force() -> None:
    b1 = _fake_xfrc_backend()
    b2 = _fake_xfrc_backend()
    force = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
    zero = np.zeros_like(force)

    MuJoCoBackend.apply_body_force(b1, np.array([1, 2]), force)
    MuJoCoBackend.apply_body_wrench(b2, np.array([1, 2]), force, zero)

    np.testing.assert_allclose(b1._pending_xfrc_applied, b2._pending_xfrc_applied)


def test_apply_body_wrench_accumulates() -> None:
    b = _fake_xfrc_backend()
    f = np.ones((2, 1, 3))
    t = np.full((2, 1, 3), 2.0)

    MuJoCoBackend.apply_body_wrench(b, np.array([1]), f, t)
    MuJoCoBackend.apply_body_wrench(b, np.array([1]), f, t)

    base = 6
    np.testing.assert_allclose(b._pending_xfrc_applied[:, base : base + 3], 2.0)
    np.testing.assert_allclose(b._pending_xfrc_applied[:, base + 3 : base + 6], 4.0)


def test_apply_body_wrench_shape_validation() -> None:
    b = _fake_xfrc_backend()
    with pytest.raises(ValueError, match="body wrench must have shape"):
        MuJoCoBackend.apply_body_wrench(
            b, np.array([1]), np.zeros((2, 1, 3)), np.zeros((2, 1, 2))
        )


# --------------------------------------------------------------------------- #
# 1.2 per-substep wrench recompute + post-step callback                       #
# --------------------------------------------------------------------------- #


class _RecordingPool:
    """Fake BatchEnvPool that records the control (incl. xfrc) passed each step."""

    def __init__(self) -> None:
        self.controls: list[np.ndarray] = []

    def step(
        self,
        state,
        *,
        nstep,
        control,
        control_spec,
        return_sensor=False,
        post_step_forward_sensor=False,
        chunk_size=None,
    ):
        self.controls.append(np.array(control, copy=True))
        state_out = np.asarray(state) + 1.0
        if return_sensor:
            return state_out, state_out[:, :1]
        return state_out


def _fake_step_backend(nbody: int = 2, num_envs: int = 1):
    # Real MuJoCoBackend instance (has methods) with __init__ bypassed + fake state.
    b = object.__new__(MuJoCoBackend)
    b._pre_step_control_fn = lambda bk, c: c  # no-op control -> routes to slow path
    b._pre_step_wrench_fn = None
    b._post_step_callback_fn = None
    b._num_envs = num_envs
    b._np_dtype = np.float32
    b._physics_state = np.zeros((num_envs, 1), dtype=np.float32)
    b._sensor_data = np.zeros((num_envs, 1), dtype=np.float32)
    b._pending_xfrc_applied = np.zeros((num_envs, 6 * nbody), dtype=np.float64)
    b._post_step_forward_sensor = False
    b._chunk_size = None
    b._pool = _RecordingPool()
    return b


def _xfrc_zidx(ctrl_width: int, body_id: int) -> int:
    # In control_traj = concat(ctrl, xfrc), body's z-force = ctrl_width + 6*body + 2.
    return ctrl_width + 6 * body_id + 2


def test_pre_step_wrench_recomputes_and_zeros_each_substep() -> None:
    b = _fake_step_backend()
    calls = {"n": 0}

    def wrench(bk) -> None:
        calls["n"] += 1
        # z-force on body 1 grows each substep -> proves per-substep recompute.
        bk.apply_body_wrench(
            np.array([1]),
            np.array([[[0.0, 0.0, float(calls["n"])]]]),
            np.zeros((1, 1, 3)),
        )

    post: list[np.ndarray] = []
    b.set_pre_step_wrench(wrench)
    b.set_post_step_callback(lambda bk: post.append(bk._sensor_data.copy()))
    ctrl = np.array([[0.5, -0.5]], dtype=np.float32)

    b.step(ctrl, nsteps=3)

    assert calls["n"] == 3  # wrench recomputed every substep incl. the last
    assert len(post) == 3  # post-step callback fired every substep incl. the last
    assert len(b._pool.controls) == 3
    zidx = _xfrc_zidx(ctrl.shape[-1], body_id=1)
    zvals = [float(c[0, 0, zidx]) for c in b._pool.controls]
    assert zvals == [1.0, 2.0, 3.0]  # zeroed + recomputed each substep (no accumulation)


def test_pre_step_wrench_absent_preserves_static_legacy_path() -> None:
    # Regression guard: with no wrench fn, a pre-staged xfrc broadcasts unchanged
    # across substeps and is cleared after the step (existing behavior).
    b = _fake_step_backend()
    b._pending_xfrc_applied[:, 6 * 1 + 2] = 5.0  # static z-force on body 1
    ctrl = np.array([[0.5, -0.5]], dtype=np.float32)

    b.step(ctrl, nsteps=2)

    zidx = _xfrc_zidx(ctrl.shape[-1], body_id=1)
    zvals = [float(c[0, 0, zidx]) for c in b._pool.controls]
    assert zvals == [5.0, 5.0]  # frozen across substeps
    np.testing.assert_allclose(b._pending_xfrc_applied, 0.0)  # cleared after step
