from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unilab.algos.torch.sonic_ppo import SonicPPORunner
from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.envs.motion_tracking.g1.sonic import (
    SONIC_ACTION_SCALE,
    SONIC_ACTOR_OBSERVATION_TERMS,
    SONIC_BODY_ORDER,
    SONIC_CRITIC_OBSERVATION_TERMS,
    SONIC_JOINT_ORDER,
    SONIC_LOWER_BODY_POLICY_INDICES,
    SONIC_MUJOCO_TO_POLICY,
    SONIC_POLICY_JOINT_ORDER,
    SONIC_POLICY_TO_MUJOCO,
    SONIC_TOKENIZER_OBS_DIM,
    SONIC_TOKENIZER_OBSERVATION_TERMS,
    SONIC_VR_BODY_OFFSETS,
    SONIC_WRIST_JOINT_INDICES,
    SonicG1TrackingCfg,
    SonicG1TrackingEnv,
)
from unilab.training.sonic_motion import materialize_motion_store
from unilab.training.sonic_store import SonicMotionLoader

mujoco = pytest.importorskip("mujoco")

# These expected layouts are transcribed from
# GR00T-WholeBodyControl@a0732b642c0333077e127a2f56ab0014c196bca4:
# gear_sonic/config/manager_env/observations/{policy/local_dir_hist,
# critic/privileged_mf_hist,tokenizer/unitoken_all_noz_heading}.yaml and the
# PolicyCfg/PrivilegedCfg/TokenizerCfg declarations in observations.py.
# IsaacLab v2.3.2 ObservationManager iterates those configclass declarations.


def _indices(value: str) -> tuple[int, ...]:
    return tuple(map(int, value.split()))


def _write_clip(path: Path, *, body_order: list[str], joint_order: list[str]) -> None:
    frames = 4
    joints = (
        np.arange(len(joint_order), dtype=np.float32)[None, :]
        + np.arange(frames, dtype=np.float32)[:, None]
    )
    body_pos = np.zeros((frames, len(body_order), 3), dtype=np.float32)
    body_pos[:, :, 0] = np.arange(len(body_order), dtype=np.float32)[None, :] + 1.0
    body_pos[:, :, 2] = 1.0
    body_quat = np.zeros((frames, len(body_order), 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    zeros = np.zeros_like(body_pos)
    np.savez(
        path,
        fps=np.asarray(50, dtype=np.int32),
        joint_pos=joints,
        joint_vel=np.zeros_like(joints),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=zeros,
        body_ang_vel_w=zeros,
    )


def test_registry_and_scene_own_sonic_contract() -> None:
    registry.ensure_registries()
    listed = registry.list_registered_envs()
    assert set(listed["SonicG1Tracking"]["available_backends"]) == {"mujoco"}

    model = mujoco.MjModel.from_xml_path(
        str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_sonic.xml")
    )
    assert model.opt.timestep == pytest.approx(0.005)
    assert model.nu == 29
    assert model.nq == 36
    assert len(SONIC_ACTION_SCALE) == 29
    assert np.all(np.isfinite(SONIC_ACTION_SCALE))
    assert np.all(SONIC_ACTION_SCALE > 0)

    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(model.nu)
    )
    actuator_ids = {name: index for index, name in enumerate(actuator_names)}
    for name in ("left_hip_pitch_joint", "right_hip_pitch_joint"):
        actuator_id = actuator_ids[name]
        assert model.actuator_gainprm[actuator_id, 0] == pytest.approx(99.098)
        assert model.actuator_biasprm[actuator_id, 2] == pytest.approx(-6.309)
        np.testing.assert_allclose(model.actuator_forcerange[actuator_id], [-139.0, 139.0])
    # Ordinary G1 tasks retain their historical actuator contract.
    ordinary = mujoco.MjModel.from_xml_path(
        str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
    )
    ordinary_names = tuple(
        mujoco.mj_id2name(ordinary, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(ordinary.nu)
    )
    ordinary_pitch = ordinary_names.index("left_hip_pitch_joint")
    assert ordinary.actuator_gainprm[ordinary_pitch, 0] == pytest.approx(40.179)


def test_sonic_wrist_feature_uses_release_order() -> None:
    assert SONIC_WRIST_JOINT_INDICES == (23, 24, 25, 26, 27, 28)


def test_sonic_policy_joint_abi_matches_fixed_upstream_mapping() -> None:
    assert SONIC_POLICY_JOINT_ORDER[:6] == (
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "waist_roll_joint",
    )
    assert SONIC_MUJOCO_TO_POLICY == _indices(
        "0 6 12 1 7 13 2 8 14 3 9 15 22 4 10 16 23 5 11 17 24 18 25 19 26 20 27 21 28"
    )
    assert SONIC_POLICY_TO_MUJOCO == _indices(
        "0 3 6 9 13 17 1 4 7 10 14 18 2 5 8 11 15 19 21 23 25 27 12 16 20 22 24 26 28"
    )
    assert SONIC_LOWER_BODY_POLICY_INDICES == (0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18)


def test_sonic_observation_noise_matches_release_scales() -> None:
    cfg = SonicG1TrackingCfg()
    noise = cfg.noise_config
    assert noise.level == pytest.approx(1.0)
    assert noise.scale_gravity == pytest.approx(0.05)
    assert noise.scale_gyro == pytest.approx(0.2)
    assert noise.scale_joint_angle == pytest.approx(0.01)
    assert noise.scale_joint_vel == pytest.approx(0.5)
    env = object.__new__(SonicG1TrackingEnv)
    env._cfg, noise.level, noise.seed = cfg, 0.0, 7
    data = np.zeros((32, 3), dtype=np.float32)
    assert np.any(env._tokenizer_corruption(data, 0.05) != data)
    cfg.tokenizer_enable_corruption = False
    assert env._tokenizer_corruption(data, 0.05) is data


def test_sonic_observation_layouts_are_immutable_and_contiguous() -> None:
    expected = {
        SONIC_ACTOR_OBSERVATION_TERMS: (
            ("base_ang_vel", (10, 3), 0, 30),
            ("joint_pos", (10, 29), 30, 320),
            ("joint_vel", (10, 29), 320, 610),
            ("actions", (10, 29), 610, 900),
            ("gravity_dir", (10, 3), 900, 930),
        ),
        SONIC_CRITIC_OBSERVATION_TERMS: (
            ("command_multi_future", (580,), 0, 580),
            ("motion_anchor_pos_b", (3,), 580, 583),
            ("motion_anchor_ori_b", (6,), 583, 589),
            ("body_pos", (14, 3), 589, 631),
            ("body_ori", (14, 6), 631, 715),
            ("base_lin_vel", (10, 3), 715, 745),
            ("base_ang_vel", (10, 3), 745, 775),
            ("joint_pos", (10, 29), 775, 1065),
            ("joint_vel", (10, 29), 1065, 1355),
            ("actions", (10, 29), 1355, 1645),
        ),
        SONIC_TOKENIZER_OBSERVATION_TERMS: (
            ("encoder_index", (3,), 0, 3),
            ("command_multi_future_nonflat", (10, 58), 3, 583),
            ("command_z_multi_future_nonflat", (10, 1), 583, 593),
            ("command_z", (1,), 593, 594),
            ("motion_anchor_ori_heading_mf_nonflat", (10, 6), 594, 654),
            ("motion_anchor_ori_heading", (6,), 654, 660),
            ("command_multi_future_lower_body", (240,), 660, 900),
            ("vr_3point_local_target", (9,), 900, 909),
            ("vr_3point_local_orn_target", (12,), 909, 921),
            ("smpl_joints_multi_future_local_nonflat", (10, 72), 921, 1641),
            ("smpl_root_ori_heading_multi_future", (10, 6), 1641, 1701),
            ("joint_pos_multi_future_wrist_for_smpl", (10, 6), 1701, 1761),
        ),
    }
    for layout, expected_terms in expected.items():
        actual = tuple((term.name, term.shape, term.start, term.stop) for term in layout)
        assert actual == expected_terms
        assert tuple(term.stop for term in layout[:-1]) == tuple(term.start for term in layout[1:])


def _history_fixture(*, noise_level: float = 0.0, num_envs: int = 1) -> SonicG1TrackingEnv:
    instance = object.__new__(SonicG1TrackingEnv)
    instance._num_envs = num_envs
    instance._backend_to_policy = np.asarray(SONIC_MUJOCO_TO_POLICY, dtype=np.int32)
    instance.anchor_body_idx = 0
    instance._sonic_reset_ids = np.asarray([0], dtype=np.int32)
    instance._history = np.zeros((num_envs, 10, 93), dtype=np.float32)
    instance._critic_history = np.zeros_like(instance._history)
    instance._cfg = SonicG1TrackingCfg()
    instance._cfg.noise_config.level = noise_level
    return instance


def test_sonic_history_reset_step_and_clip_refresh_contract() -> None:
    instance = _history_fixture(noise_level=0.0)
    identity = np.asarray([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
    first = instance._build_history(
        np.asarray([0], dtype=np.int32),
        np.full((1, 3), 1.0, dtype=np.float32),
        np.full((1, 3), 2.0, dtype=np.float32),
        np.full((1, 29), 3.0, dtype=np.float32),
        np.full((1, 29), 4.0, dtype=np.float32),
        identity,
        np.full((1, 29), 5.0, dtype=np.float32),
        policy_default_angles=np.zeros((1, 29), dtype=np.float32),
        advance_history=True,
    )
    np.testing.assert_allclose(first[0]["base_ang_vel"], 2.0)
    np.testing.assert_allclose(first[1]["base_lin_vel"], 1.0)
    np.testing.assert_allclose(instance._history[:, 0], instance._history[:, -1])

    instance._sonic_reset_ids = None
    second = instance._build_history(
        np.asarray([0], dtype=np.int32),
        np.full((1, 3), 11.0, dtype=np.float32),
        np.full((1, 3), 12.0, dtype=np.float32),
        np.full((1, 29), 13.0, dtype=np.float32),
        np.full((1, 29), 14.0, dtype=np.float32),
        identity,
        np.full((1, 29), 15.0, dtype=np.float32),
        policy_default_angles=np.zeros((1, 29), dtype=np.float32),
        advance_history=True,
    )
    np.testing.assert_allclose(second[0]["base_ang_vel"][:, :-1], 2.0)
    np.testing.assert_allclose(second[0]["base_ang_vel"][:, -1], 12.0)
    np.testing.assert_allclose(second[1]["base_lin_vel"][:, -1], 11.0)
    refresh_instance = _history_fixture(noise_level=0.0, num_envs=2)
    refresh_instance._sonic_reset_ids = None
    refresh_instance._history[0] = 99.0
    refresh_instance._critic_history[0] = 98.0
    refreshed = refresh_instance._build_history(
        np.asarray([1], dtype=np.int32),
        np.full((1, 3), 21.0, dtype=np.float32),
        np.full((1, 3), 22.0, dtype=np.float32),
        np.full((1, 29), 23.0, dtype=np.float32),
        np.full((1, 29), 24.0, dtype=np.float32),
        identity,
        np.full((1, 29), 25.0, dtype=np.float32),
        policy_default_angles=np.zeros((1, 29), dtype=np.float32),
        advance_history=False,
    )
    np.testing.assert_allclose(refresh_instance._history[0], 99.0)
    np.testing.assert_allclose(refresh_instance._critic_history[0], 98.0)
    np.testing.assert_allclose(refreshed[0]["base_ang_vel"], 22.0)
    np.testing.assert_allclose(refreshed[1]["base_lin_vel"], 21.0)


def test_sonic_future_timing_and_encoder_sampling_contract() -> None:
    assert np.array_equal(
        SonicG1TrackingEnv._future_frame_offsets(10, 0.1, 50), np.arange(0, 50, 5)
    )
    assert np.array_equal(SonicG1TrackingEnv._future_frame_offsets(10, 0.02, 50), np.arange(10))
    with pytest.raises(ValueError, match="positive integer step"):
        SonicG1TrackingEnv._future_frame_offsets(10, 0.03, 50)

    instance = object.__new__(SonicG1TrackingEnv)
    instance._sonic_has_smpl = True
    instance._num_envs = 4
    instance._encoder_index = np.zeros((4, 3), dtype=np.float32)
    instance._cfg = SonicG1TrackingCfg(
        encoder_sample_probs=(0.0, 0.0, 1.0), teleop_sample_prob_when_smpl=1.0
    )
    instance._sample_encoder_indices(np.arange(4, dtype=np.int32))
    np.testing.assert_array_equal(instance._encoder_index, np.ones((4, 3), dtype=np.float32))

    instance._sonic_has_smpl = False
    instance._encoder_index.fill(0.0)
    instance._cfg = SonicG1TrackingCfg(encoder_sample_probs=(1.0, 1.0, 1.0))
    instance._sample_encoder_indices(np.arange(4, dtype=np.int32))
    assert np.all(instance._encoder_index[:, 2] == 0.0)


def test_sonic_vr_offsets_and_training_deploy_width_provenance() -> None:
    np.testing.assert_allclose(
        SONIC_VR_BODY_OFFSETS,
        ((0.18, -0.025, 0.0), (0.18, 0.025, 0.0), (0.0, 0.0, 0.35)),
    )
    # unitoken_all_noz_heading.yaml is the 1761-wide v1.1 training group.  The release
    # encoder export observation_config_sonic_v1_1.yaml selects 1750 active
    # tokenizer dimensions plus one scalar mode, omitting the two command-z
    # terms (11 dimensions total).
    assert SONIC_TOKENIZER_OBS_DIM == 1761
    assert SONIC_TOKENIZER_OBS_DIM - (10 + 1) + 1 == 1751


def _golden_qz(angle: np.ndarray) -> np.ndarray:
    """Independent w-first quaternion oracle for the synthetic upstream fixture."""

    half = np.asarray(angle) * 0.5
    return np.stack([np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)], -1)


def _golden_rotate_z(angle: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Hand-written Z rotation; intentionally does not use UniLab rotation helpers."""

    angle, value = np.asarray(angle), np.asarray(value)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.stack(
        [
            cosine * value[..., 0] - sine * value[..., 1],
            sine * value[..., 0] + cosine * value[..., 1],
            value[..., 2],
        ],
        axis=-1,
    )


def _golden_rot6d_z(angle: np.ndarray) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    zeros = np.zeros_like(cosine)
    return np.stack([cosine, -sine, sine, cosine, zeros, zeros], axis=-1)


def _synthetic_upstream_future(phase: int) -> dict[str, np.ndarray]:
    """Sentinels derived from the fixed upstream SONIC observation equations."""

    frames, bodies = np.arange(10), np.arange(14)
    joint_pos = (phase * 1000 + np.arange(290)).reshape(1, 10, 29).astype(np.float32)
    joint_vel = (10000 + phase * 1000 + np.arange(290)).reshape(1, 10, 29).astype(np.float32)
    anchor_pos = np.stack(
        [10 + phase + frames, 20 - phase + 2 * frames, 30 + phase + 3 * frames], -1
    )
    body_delta = np.stack([bodies, -2 * bodies, 0.5 * bodies], -1)
    body_pos = (anchor_pos[:, None, :] + body_delta[None, :, :])[None].astype(np.float32)
    body_angle = (
        np.pi / 4 + phase * np.pi / 6 + frames[:, None] * np.pi / 20 + bodies[None, :] * np.pi / 30
    )[None]
    smpl_angle = (-np.pi / 6 + phase * np.pi / 10 + frames * np.pi / 12)[None]
    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos": body_pos,
        "body_quat": _golden_qz(body_angle).astype(np.float32),
        "body_angle": body_angle,
        "smpl_joint_pos": (20000 + phase * 1000 + np.arange(290))
        .reshape(1, 10, 29)
        .astype(np.float32),
        "smpl_joints": (phase + np.arange(720).reshape(1, 10, 24, 3) / 100).astype(np.float32),
        "smpl_root_quat": _golden_qz(smpl_angle).astype(np.float32),
        "smpl_angle": smpl_angle,
    }


def test_sonic_compute_obs_matches_independent_upstream_reset_and_step_fixture() -> None:
    """Every slice follows GR00T-WBC@a0732b64, without a production layout oracle."""

    mujoco_to_policy = _indices(
        "0 6 12 1 7 13 2 8 14 3 9 15 22 4 10 16 23 5 11 17 24 18 25 19 26 20 27 21 28"
    )
    instance = object.__new__(SonicG1TrackingEnv)
    instance._num_envs = 1
    instance._backend_to_policy = np.asarray(mujoco_to_policy, dtype=np.int32)
    instance._policy_default_angles = np.zeros(29, dtype=np.float32)
    instance.anchor_body_idx = 0
    instance._history = np.zeros((1, 10, 93), dtype=np.float32)
    instance._critic_history = np.zeros_like(instance._history)
    instance._encoder_index = np.asarray([[1.0, 0.0, 1.0]], dtype=np.float32)
    instance._vr_body_rows = np.asarray([10, 13, 7], dtype=np.int32)
    offsets = np.asarray(((0.18, -0.025, 0), (0.18, 0.025, 0), (0, 0, 0.35)))
    instance._vr_body_offsets = offsets.astype(np.float32)
    instance._cfg = SonicG1TrackingCfg()
    instance.motion_sampler = SimpleNamespace(current_frames=np.asarray([0], dtype=np.int64))
    futures = [_synthetic_upstream_future(phase) for phase in range(2)]
    instance._future_reference = lambda frame: futures[int(frame[0])]
    # Deterministic stand-in for upstream AdditiveUniformNoiseCfg. This proves
    # which terms are corrupted without coupling the golden values to RNG code.
    instance._obs_noise = lambda data, scale: np.asarray(data) + scale
    instance._tokenizer_corruption = lambda data, scale: np.asarray(data) + scale

    actor_history = critic_history = None
    for phase in range(2):
        robot_angle = np.asarray([np.pi / 2 + phase * np.pi / 3])
        body_angles = robot_angle[:, None] + np.arange(14)[None] * np.pi / 16
        anchor_pos_w = np.asarray([[phase + 1, 2 - phase, 3 + phase]], dtype=np.float32)
        body_delta = np.stack([np.arange(14), np.arange(14) * 2, -np.arange(14)], -1)
        body_pos_w = anchor_pos_w[:, None] + body_delta[None].astype(np.float32)
        body_quat_w = _golden_qz(body_angles).astype(np.float32)
        linvel = np.asarray([[1, 2, 3]], dtype=np.float32) + phase * 100
        gyro = np.asarray([[4, 5, 6]], dtype=np.float32) + phase * 100
        dof_pos = np.arange(29, dtype=np.float32)[None] + 10 + phase * 100
        dof_vel = np.arange(29, dtype=np.float32)[None] + 50 + phase * 100
        actions = np.arange(29, dtype=np.float32)[None] + 90 + phase * 100
        gravity = np.asarray([[0, 0, -1]], dtype=np.float32)
        dof_pos_policy = dof_pos[:, mujoco_to_policy]
        dof_vel_policy = dof_vel[:, mujoco_to_policy]
        actor_current = np.concatenate(
            [gyro + 0.2, dof_pos_policy + 0.01, dof_vel_policy + 0.5, actions, gravity + 0.05], 1
        )
        critic_current = np.concatenate([linvel, gyro, dof_pos_policy, dof_vel_policy, actions], 1)
        if phase == 0:
            actor_history = np.repeat(actor_current[:, None], 10, axis=1)
            critic_history = np.repeat(critic_current[:, None], 10, axis=1)
            instance._sonic_reset_ids = np.asarray([0], dtype=np.int32)
        else:
            actor_history = np.concatenate([actor_history[:, 1:], actor_current[:, None]], 1)
            critic_history = np.concatenate([critic_history[:, 1:], critic_current[:, None]], 1)
            instance._sonic_reset_ids = None
        instance.motion_sampler.current_frames[:] = phase
        future = futures[phase]
        ref_anchor_pos = future["body_pos"][:, 0, 0]
        ref_anchor_angle = future["body_angle"][:, 0, 0]
        anchor_pos_b = _golden_rotate_z(-robot_angle, ref_anchor_pos - anchor_pos_w)
        anchor_ori_b = _golden_rot6d_z(ref_anchor_angle - robot_angle)
        body_pos_b = _golden_rotate_z(-robot_angle[:, None], body_pos_w - anchor_pos_w[:, None])
        body_ori_b = _golden_rot6d_z(body_angles - robot_angle[:, None])
        future_ori = _golden_rot6d_z(future["body_angle"][:, :, 0] - robot_angle[:, None])
        command = np.concatenate(
            [future["joint_pos"].reshape(1, -1), future["joint_vel"].reshape(1, -1)], 1
        )
        command_z = future["body_pos"][:, :, 0, 2:3]
        lower = np.concatenate(
            [
                future["joint_pos"][:, :, (0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18)].reshape(1, -1),
                future["joint_vel"][:, :, (0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18)].reshape(1, -1),
            ],
            1,
        )
        rows = np.asarray([10, 13, 7])
        vr_world = future["body_pos"][:, 0, rows] + _golden_rotate_z(
            future["body_angle"][:, 0, rows], offsets[None]
        )
        vr_pos = _golden_rotate_z(-ref_anchor_angle[:, None], vr_world - ref_anchor_pos[:, None])
        vr_quat = _golden_qz(future["body_angle"][:, 0, rows] - ref_anchor_angle[:, None])
        smpl_local = _golden_rotate_z(-future["smpl_angle"][:, :, None], future["smpl_joints"])
        smpl_ori = _golden_rot6d_z(future["smpl_angle"] - robot_angle[:, None])

        obs = instance._compute_obs(
            {"current_actions": actions},
            None,
            linvel,
            gyro,
            dof_pos,
            dof_vel,
            body_pos_w,
            body_quat_w,
        )
        expected_terms = {
            "actor_obs": (
                (0, 30, actor_history[:, :, :3]),
                (30, 320, actor_history[:, :, 3:32]),
                (320, 610, actor_history[:, :, 32:61]),
                (610, 900, actor_history[:, :, 61:90]),
                (900, 930, actor_history[:, :, 90:93]),
            ),
            "critic_obs": (
                (0, 580, command),
                (580, 583, anchor_pos_b),
                (583, 589, anchor_ori_b),
                (589, 631, body_pos_b),
                (631, 715, body_ori_b),
                (715, 745, critic_history[:, :, :3]),
                (745, 775, critic_history[:, :, 3:6]),
                (775, 1065, critic_history[:, :, 6:35]),
                (1065, 1355, critic_history[:, :, 35:64]),
                (1355, 1645, critic_history[:, :, 64:93]),
            ),
            "tokenizer": (
                (0, 3, instance._encoder_index),
                (3, 583, command.reshape(1, 10, 58)),
                (583, 593, command_z),
                (593, 594, command_z[:, 0]),
                (594, 654, future_ori + 0.05),
                (654, 660, anchor_ori_b + 0.05),
                (660, 900, lower),
                (900, 909, vr_pos),
                (909, 921, vr_quat),
                (921, 1641, smpl_local + 0.05),
                (1641, 1701, smpl_ori + 0.05),
                (1701, 1761, future["smpl_joint_pos"][:, :, (23, 24, 25, 26, 27, 28)]),
            ),
        }
        for group, terms in expected_terms.items():
            for start, stop, expected in terms:
                np.testing.assert_allclose(
                    obs[group][:, start:stop],
                    np.asarray(expected).reshape(1, -1),
                    rtol=0,
                    atol=2e-5,
                )


def test_sonic_actions_for_obs_selects_full_state_or_accepts_partial_rows() -> None:
    instance = object.__new__(SonicG1TrackingEnv)
    instance._num_envs = 3
    full = np.arange(87, dtype=np.float32).reshape(3, 29)
    instance._state = SimpleNamespace(info={"current_actions": full})
    env_ids = np.asarray([2, 0], dtype=np.int32)
    np.testing.assert_array_equal(instance._actions_for_obs({}, env_ids), full[env_ids])
    partial = full[env_ids] + 1000
    np.testing.assert_array_equal(
        instance._actions_for_obs({"current_actions": partial}, env_ids), partial
    )


def test_sonic_policy_actions_map_to_mujoco_before_scale_default_and_bias() -> None:
    instance = object.__new__(SonicG1TrackingEnv)
    instance._cfg = SonicG1TrackingCfg()
    instance._policy_to_backend = np.asarray(SONIC_POLICY_TO_MUJOCO, dtype=np.int32)
    instance.default_angles = np.arange(29, dtype=np.float32)
    instance._policy_default_angles = instance.default_angles[list(SONIC_MUJOCO_TO_POLICY)]
    actions = (100 + np.arange(29, dtype=np.float32))[None]
    bias = (1000 + np.arange(29, dtype=np.float32))[None]
    state = SimpleNamespace(info={"default_dof_pos_bias": bias})
    target = instance.apply_action(actions, state)
    policy_to_mujoco = _indices(
        "0 3 6 9 13 17 1 4 7 10 14 18 2 5 8 11 15 19 21 23 25 27 12 16 20 22 24 26 28"
    )
    np.testing.assert_array_equal(
        target,
        actions[:, policy_to_mujoco] * SONIC_ACTION_SCALE[list(policy_to_mujoco)]
        + instance.default_angles
        + bias,
    )
    np.testing.assert_array_equal(state.info["current_actions"], actions)


def test_sonic_env_uses_reordered_materialized_store(tmp_path: Path) -> None:
    source_joint_order = list(reversed(SONIC_JOINT_ORDER))
    source_body_order = ["extra_body", *SONIC_BODY_ORDER]
    clip = tmp_path / "source.npz"
    _write_clip(clip, body_order=source_body_order, joint_order=source_joint_order)
    report = materialize_motion_store(
        [clip],
        tmp_path / "store",
        fps=50,
        joint_order=source_joint_order,
        body_order=source_body_order,
    )

    registry.ensure_registries()
    env = registry.make(
        "SonicG1Tracking",
        sim_backend="mujoco",
        num_envs=2,
        env_cfg_override={
            "motion_manifest": str(report.manifest_path),
            "sampling_mode": "start",
        },
    )
    try:
        assert isinstance(env.motion_loader, SonicMotionLoader)
        joint_values = env.motion_loader.store.arrays["joint_pos"][0]
        body_values = env.motion_loader.store.arrays["body_pos_w"][0]
        assert joint_values[0] == pytest.approx(float(len(SONIC_JOINT_ORDER) - 1))
        assert body_values[0, 0] == pytest.approx(2.0)
        fixed_m_to_i = _indices(
            "0 6 12 1 7 13 2 8 14 3 9 15 22 4 10 16 23 5 11 17 24 18 25 19 26 20 27 21 28"
        )
        np.testing.assert_array_equal(
            env._future_reference(np.asarray([0]))["joint_pos"][0, 0],
            joint_values[list(fixed_m_to_i)],
        )

        obs, info = env.reset()
        assert set(obs) == {"actor_obs", "critic_obs", "tokenizer"}
        assert obs["actor_obs"].shape == (2, 930)
        assert obs["critic_obs"].shape == (2, 1645)
        assert obs["tokenizer"].shape == (2, 1761)
        assert isinstance(info, dict)
        encoder_before = env._encoder_index.copy()
        env.cfg.encoder_sample_probs = (0.0, 1.0, 0.0)
        env._refresh_observation_rows(obs, info, np.asarray([1], dtype=np.int32))
        np.testing.assert_array_equal(env._encoder_index[0], encoder_before[0])
        np.testing.assert_array_equal(env._encoder_index[1], (0.0, 1.0, 0.0))

        state = env.step(np.zeros((2, 29), dtype=np.float32))
        assert set(state.obs) == {"actor_obs", "critic_obs", "tokenizer"}
        assert state.obs["actor_obs"].shape == (2, 930)
        assert state.obs["critic_obs"].shape == (2, 1645)
        assert state.obs["tokenizer"].shape == (2, 1761)
        assert state.reward.shape == (2,)

        runner = SonicPPORunner(
            env,
            {
                "algo": {
                    "num_steps_per_env": 1,
                    "num_learning_epochs": 1,
                    "num_mini_batches": 1,
                    "save_interval": 99,
                    "learning_rate": 0.0,
                },
                "sonic": {
                    "microbatch_size": 2,
                    "model": {
                        "hidden_dims": [8],
                        "tokenizer_hidden_dim": 8,
                    },
                },
            },
            device="cpu",
            log_dir=tmp_path / "run",
        )
        metrics = runner.learn(1)
        assert runner.current_learning_iteration == 1
        assert np.isfinite(metrics["loss"])
        assert (tmp_path / "run" / "last.pt").is_file()
    finally:
        env.close()


def test_sonic_actuator_order_mismatch_fails_closed() -> None:
    instance = object.__new__(SonicG1TrackingEnv)

    class Backend:
        def get_actuator_names(self) -> tuple[str, ...]:
            return (*SONIC_JOINT_ORDER[:-1], "unexpected_joint")

    instance._backend = Backend()
    with pytest.raises(ValueError, match="29-DoF release order"):
        instance._resolve_actuator_permutation()
