"""Parity and lifecycle checks for the opt-in SONIC tokenizer kernel."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from unilab.tasks.motion_tracking.g1.sonic.observation_terms import (
    SonicFutureReference,
    _tokenizer_layout,
)


def _fixture(num_envs: int = 4) -> tuple[SonicFutureReference, SimpleNamespace, SimpleNamespace]:
    rng = np.random.default_rng(19)
    shape = (num_envs, 10, 29)
    joint_pos = rng.normal(size=shape).astype(np.float32)
    joint_vel = rng.normal(size=shape).astype(np.float32)
    body_pos = rng.normal(size=(num_envs, 10, 14, 3)).astype(np.float32)
    body_quat = rng.normal(size=(num_envs, 10, 14, 4)).astype(np.float32)
    body_quat /= np.linalg.norm(body_quat, axis=-1, keepdims=True)
    smpl_joint_pos = rng.normal(size=shape).astype(np.float32)
    smpl_joints = rng.normal(size=(num_envs, 10, 24, 3)).astype(np.float32)
    smpl_root = rng.normal(size=(num_envs, 10, 4)).astype(np.float32)
    smpl_root /= np.linalg.norm(smpl_root, axis=-1, keepdims=True)
    robot_anchor = rng.normal(size=(num_envs, 4)).astype(np.float32)
    robot_anchor /= np.linalg.norm(robot_anchor, axis=-1, keepdims=True)
    reference = SonicFutureReference(
        joint_pos,
        joint_vel,
        body_pos,
        body_quat,
        smpl_joint_pos,
        smpl_joints,
        smpl_root,
    )
    cfg = SimpleNamespace(
        body_names=("left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link", *[f"b{i}" for i in range(11)]),
        params=SimpleNamespace(tokenizer_enable_corruption=False),
    )
    env = SimpleNamespace(rng=np.random.default_rng(23))
    common = dict(
        num_envs=num_envs,
        robot_anchor_quat_w=robot_anchor,
        anchor_body_idx=0,
        vr_body_indices=np.asarray((0, 1, 2), dtype=np.intp),
        cfg=cfg,
        _env=env,
        encoder_index=np.tile(np.asarray((1.0, 0.0, 0.0), dtype=np.float32), (num_envs, 1)),
    )
    return reference, SimpleNamespace(use_numba_observation_assembly=False, **common), SimpleNamespace(
        use_numba_observation_assembly=True,
        observation_update_env_ids=None,
        tokenizer_assembly_buffer=np.empty((num_envs, 1761), dtype=np.float32),
        **common,
    )


def test_numba_tokenizer_layout_matches_numpy_reference() -> None:
    reference, numpy_command, numba_command = _fixture()
    expected = _tokenizer_layout(numpy_command, reference)
    actual = _tokenizer_layout(numba_command, reference)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_numba_tokenizer_layout_scopes_reset_rows() -> None:
    reference, _, numba_command = _fixture()
    numba_command.tokenizer_assembly_buffer.fill(-77.0)
    numba_command.observation_update_env_ids = np.asarray((1, 3), dtype=np.int32)
    _tokenizer_layout(numba_command, reference)
    np.testing.assert_array_equal(numba_command.tokenizer_assembly_buffer[[0, 2]], -77.0)
    assert np.all(np.isfinite(numba_command.tokenizer_assembly_buffer[[1, 3]]))
