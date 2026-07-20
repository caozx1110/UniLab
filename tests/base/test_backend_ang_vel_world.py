"""Backend contract: world-frame base angular velocity (GH migration, DP1-b).

New method ``get_base_ang_vel_world`` rotates the free-joint LOCAL ``qvel[3:6]``
into the world frame. The pre-existing ``get_base_ang_vel`` (which documents
world-frame but returns the local view — P1-2 contract violation) is left
untouched; a global fix is tracked as a separate follow-up issue.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.utils.rotation import np_quat_apply, np_yaw_to_quat


def _fake(quat: np.ndarray, ang_local: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        _base_quat_view=np.asarray(quat, dtype=np.float64),
        _base_ang_vel_view=np.asarray(ang_local, dtype=np.float64),
        _np_dtype=np.float64,
    )


def test_ang_vel_world_90deg_yaw_matches_quat_apply() -> None:
    quat = np_yaw_to_quat(np.array([np.pi / 2]))  # (1,4) wxyz, yaw +90°
    ang_local = np.array([[1.0, 0.0, 0.0]])

    out = MuJoCoBackend.get_base_ang_vel_world(_fake(quat, ang_local))

    # yaw +90° rotates local +x -> world +y
    np.testing.assert_allclose(out, [[0.0, 1.0, 0.0]], atol=1e-6)
    np.testing.assert_allclose(out, np_quat_apply(quat, ang_local), atol=1e-9)


def test_ang_vel_world_zaxis_invariant_under_yaw() -> None:
    quat = np_yaw_to_quat(np.array([0.7]))
    ang_local = np.array([[0.0, 0.0, 2.3]])  # spin about world z is invariant under yaw

    out = MuJoCoBackend.get_base_ang_vel_world(_fake(quat, ang_local))

    np.testing.assert_allclose(out, [[0.0, 0.0, 2.3]], atol=1e-6)


def test_ang_vel_world_batched_multi_env() -> None:
    quat = np_yaw_to_quat(np.array([np.pi / 2, -np.pi / 2]))  # (2,4)
    ang_local = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    out = MuJoCoBackend.get_base_ang_vel_world(_fake(quat, ang_local))

    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], atol=1e-6)


def test_old_get_base_ang_vel_unchanged_returns_local_view() -> None:
    # Guard P1-2: the pre-existing method must keep returning the raw local view.
    b = SimpleNamespace(_base_ang_vel_view=np.array([[9.0, 8.0, 7.0]]))
    assert MuJoCoBackend.get_base_ang_vel(b) is b._base_ang_vel_view
