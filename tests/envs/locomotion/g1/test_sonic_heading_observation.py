from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unilab.envs.motion_tracking.g1.sonic import (
    SONIC_BODY_ORDER,
    SONIC_MUJOCO_TO_POLICY,
    SONIC_V1_1_OBSERVATION_PROFILE,
    SONIC_V1_1_REVISION,
    SonicG1TrackingCfg,
    SonicG1TrackingEnv,
    _sonic_heading_relative_rot6d,
)

# Pinned float32 oracle generated from the Python training path at
# GR00T-WholeBodyControl@a0732b642c0333077e127a2f56ab0014c196bca4.  The
# relevant get_heading_q/command/observation blobs are fixed in the issue
# provenance; no UniLab rotation helper is used to construct these values.
_UPSTREAM_ROBOT = np.asarray(
    [0.825136960, 0.206284240, -0.103142120, 0.515710592], dtype=np.float32
)
_UPSTREAM_TARGETS = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.707106769, -0.202030510, 0.303045779, 0.606091559],
        [0.5, 0.5, -0.5, 0.5],
        [0.207390338, 0.829561353, 0.414780676, -0.311085492],
    ],
    dtype=np.float32,
)
_UPSTREAM_ROT6D = np.asarray(
    [
        [0.438202202, 0.898876429, -0.898876429, 0.438202202, -0.0, 0.0],
        [0.696170568, -0.264159620, 0.248566926, 0.961018085, -0.673469484, 0.081632756],
        [0.0, -0.438202202, 0.0, 0.898876429, 1.0, 0.0],
        [0.705207169, -0.154162124, -0.170593247, -0.984293699, -0.688171983, 0.086021565],
    ],
    dtype=np.float32,
)
_G1_FUTURE_INDICES = np.asarray([1, 0, 2, 3, 0, 1, 2, 3, 0, 1])
_SMPL_FUTURE_INDICES = np.asarray([3, 2, 1, 0, 3, 2, 1, 0, 3, 2])


def _compute_heading_fixture(
    robot_quat: np.ndarray,
    g1_future_quat: np.ndarray,
    smpl_future_quat: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run the real three-term packing path with corruption disabled."""

    robot_quat = np.asarray(robot_quat, dtype=np.float32)
    g1_future_quat = np.asarray(g1_future_quat, dtype=np.float32)
    smpl_future_quat = np.asarray(smpl_future_quat, dtype=np.float32)
    assert robot_quat.shape == (4,)
    assert g1_future_quat.shape == (10, 4)
    assert smpl_future_quat.shape == (10, 4)

    env = object.__new__(SonicG1TrackingEnv)
    env._num_envs = 1
    env._backend_to_policy = np.asarray(SONIC_MUJOCO_TO_POLICY, dtype=np.int32)
    env._policy_default_angles = np.zeros(29, dtype=np.float32)
    env.anchor_body_idx = 0
    env._sonic_reset_ids = np.asarray([0], dtype=np.int32)
    env._history = np.zeros((1, 10, 93), dtype=np.float32)
    env._critic_history = np.zeros_like(env._history)
    env._encoder_index = np.asarray([[1.0, 0.0, 1.0]], dtype=np.float32)
    env._vr_body_rows = np.asarray([1, 2, 3], dtype=np.int32)
    env._vr_body_offsets = np.zeros((3, 3), dtype=np.float32)
    env._cfg = SonicG1TrackingCfg()
    env._cfg.noise_config.level = 0.0
    env._cfg.tokenizer_enable_corruption = False
    env.motion_sampler = SimpleNamespace(current_frames=np.asarray([0], dtype=np.int64))

    future = env._zero_future_reference(1)
    future["body_quat"][0, :, 0] = g1_future_quat
    future["smpl_root_quat"][0] = smpl_future_quat
    env._future_reference = lambda _frame_indices: future

    robot_body_quat = np.broadcast_to(
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        (1, len(SONIC_BODY_ORDER), 4),
    ).copy()
    robot_body_quat[0, 0] = robot_quat
    zeros_3 = np.zeros((1, 3), dtype=np.float32)
    zeros_29 = np.zeros((1, 29), dtype=np.float32)
    return env._compute_obs(
        {"current_actions": zeros_29},
        None,
        zeros_3,
        zeros_3,
        zeros_29,
        zeros_29,
        np.zeros((1, len(SONIC_BODY_ORDER), 3), dtype=np.float32),
        robot_body_quat,
    )


def _left_multiply_world_yaw(quat: np.ndarray, angle: float) -> np.ndarray:
    """Independent WXYZ ``qz(angle) * quat`` used only to perturb the fixture."""

    quat = np.asarray(quat, dtype=np.float32)
    cosine = np.float32(np.cos(angle * 0.5))
    sine = np.float32(np.sin(angle * 0.5))
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        [
            cosine * w - sine * z,
            cosine * x - sine * y,
            cosine * y + sine * x,
            cosine * z + sine * w,
        ],
        axis=-1,
    ).astype(np.float32)


def test_v11_heading_helper_matches_pinned_upstream_float32_oracle() -> None:
    actual = _sonic_heading_relative_rot6d(_UPSTREAM_ROBOT, _UPSTREAM_TARGETS)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, _UPSTREAM_ROT6D, rtol=1.0e-6, atol=1.0e-6)


def test_v11_three_heading_terms_and_critic_body_orientation_are_distinct() -> None:
    obs = _compute_heading_fixture(
        _UPSTREAM_ROBOT,
        _UPSTREAM_TARGETS[_G1_FUTURE_INDICES],
        _UPSTREAM_TARGETS[_SMPL_FUTURE_INDICES],
    )

    assert {name: value.shape for name, value in obs.items()} == {
        "actor_obs": (1, 930),
        "critic_obs": (1, 1645),
        "tokenizer": (1, 1761),
    }
    assert all(value.dtype == np.float32 for value in obs.values())
    np.testing.assert_allclose(
        obs["tokenizer"][:, 594:654],
        _UPSTREAM_ROT6D[_G1_FUTURE_INDICES].reshape(1, 60),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        obs["tokenizer"][:, 654:660],
        _UPSTREAM_ROT6D[1:2],
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        obs["tokenizer"][:, 1641:1701],
        _UPSTREAM_ROT6D[_SMPL_FUTURE_INDICES].reshape(1, 60),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    # The critic intentionally keeps inv(full robot body) * current target.
    np.testing.assert_allclose(
        obs["critic_obs"][:, 583:589],
        np.asarray(
            [[0.372557342, -0.257924497, 0.050803389, 0.964828491, -0.926617503, -0.050803136]],
            dtype=np.float32,
        ),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert not np.allclose(obs["critic_obs"][:, 583:589], obs["tokenizer"][:, 654:660])


def test_v11_heading_terms_are_invariant_to_common_left_world_yaw() -> None:
    g1_targets = _UPSTREAM_TARGETS[_G1_FUTURE_INDICES]
    smpl_targets = _UPSTREAM_TARGETS[_SMPL_FUTURE_INDICES]
    baseline = _compute_heading_fixture(_UPSTREAM_ROBOT, g1_targets, smpl_targets)
    shifted = _compute_heading_fixture(
        _left_multiply_world_yaw(_UPSTREAM_ROBOT, 0.73),
        _left_multiply_world_yaw(g1_targets, 0.73),
        _left_multiply_world_yaw(smpl_targets, 0.73),
    )

    for start, stop in ((594, 654), (654, 660), (1641, 1701)):
        np.testing.assert_allclose(
            shifted["tokenizer"][:, start:stop],
            baseline["tokenizer"][:, start:stop],
            rtol=2.0e-6,
            atol=2.0e-6,
        )


def test_v11_heading_preserves_relative_pitch_and_roll() -> None:
    robot_heading = np.asarray([0.921060994, 0.0, 0.0, 0.389418342], dtype=np.float32)
    target_with_pitch_roll = np.asarray(
        [0.875443766, 0.246641131, -0.080983764, 0.407686147], dtype=np.float32
    )
    expected = np.asarray(
        [0.939372713, -0.133530696, 0.0, 0.921060994, 0.342897807, 0.365808965],
        dtype=np.float32,
    )

    neutral = _sonic_heading_relative_rot6d(robot_heading, robot_heading)
    actual = _sonic_heading_relative_rot6d(robot_heading, target_with_pitch_roll)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-6)
    assert not np.allclose(actual, neutral, rtol=2.0e-6, atol=2.0e-6)


def test_v11_observation_profile_is_versioned_and_fail_closed() -> None:
    assert SONIC_V1_1_REVISION == "a0732b642c0333077e127a2f56ab0014c196bca4"
    assert SONIC_V1_1_OBSERVATION_PROFILE == "unitoken_all_noz_heading"
    with pytest.raises(ValueError, match="observation_profile"):
        SonicG1TrackingEnv(
            SonicG1TrackingCfg(observation_profile="unitoken_all_noz"),
            num_envs=1,
            backend_type="mujoco",
        )
