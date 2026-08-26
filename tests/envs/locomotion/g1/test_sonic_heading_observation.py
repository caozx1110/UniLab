from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unilab.envs.motion_tracking.g1.sonic import (
    SONIC_BODY_ORDER,
    SONIC_MUJOCO_TO_POLICY,
    SONIC_RELEASE_OBSERVATION_PROFILE,
    SONIC_RELEASE_REVISION,
    SonicG1TrackingCfg,
    SonicG1TrackingEnv,
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
    env.motion_sampler = SimpleNamespace(
        current_frames=np.asarray([0], dtype=np.int64),
        clamp_reference_indices=lambda indices, _env_ids=None: np.asarray(
            indices, dtype=np.int32
        ),
    )

    future = env._zero_future_reference(1)
    future["body_quat"][0, :, 0] = g1_future_quat
    future["smpl_root_quat"][0] = smpl_future_quat
    env._future_reference = lambda _frame_indices, *, env_ids=None: future

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


def test_release_body_relative_orientation_terms_use_current_then_future_order() -> None:
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
        obs["tokenizer"][:, 594:600],
        obs["critic_obs"][:, 583:589],
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        obs["tokenizer"][:, 600:606],
        obs["tokenizer"][:, 594:600],
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert obs["tokenizer"][:, 600:660].shape == (1, 60)
    assert obs["tokenizer"][:, 1641:1701].shape == (1, 60)
    assert np.all(np.isfinite(obs["tokenizer"][:, 594:660]))
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
    np.testing.assert_allclose(obs["critic_obs"][:, 583:589], obs["tokenizer"][:, 594:600])


def test_release_body_relative_orientation_is_invariant_to_common_left_world_yaw() -> None:
    g1_targets = _UPSTREAM_TARGETS[_G1_FUTURE_INDICES]
    smpl_targets = _UPSTREAM_TARGETS[_SMPL_FUTURE_INDICES]
    baseline = _compute_heading_fixture(_UPSTREAM_ROBOT, g1_targets, smpl_targets)
    shifted = _compute_heading_fixture(
        _left_multiply_world_yaw(_UPSTREAM_ROBOT, 0.73),
        _left_multiply_world_yaw(g1_targets, 0.73),
        _left_multiply_world_yaw(smpl_targets, 0.73),
    )

    for start, stop in ((594, 600), (600, 660), (1641, 1701)):
        np.testing.assert_allclose(
            shifted["tokenizer"][:, start:stop],
            baseline["tokenizer"][:, start:stop],
            rtol=2.0e-6,
            atol=2.0e-6,
        )


def test_release_observation_profile_is_versioned_and_fail_closed() -> None:
    assert SONIC_RELEASE_REVISION == "c374bae5b9039cd0ee71377e654d11ce1bc69e1d"
    assert SONIC_RELEASE_OBSERVATION_PROFILE == "unitoken_all_noz"
    with pytest.raises(ValueError, match="observation_profile"):
        SonicG1TrackingEnv(
            SonicG1TrackingCfg(observation_profile="unitoken_all_noz_heading"),
            num_envs=1,
            backend_type="mujoco",
        )
