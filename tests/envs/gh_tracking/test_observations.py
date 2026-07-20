"""Tests for GH observations (Phase 6): three-group per-term assembly + telemetry.

policy 450 / priv 717 / priv_critic 3, per-term offsets, post-substep telemetry
(joint 2-slot avg substep 2/3, root-linvel EMA 4x, contact 3-frame history), and
the tricky semantics (absolute applied_action, joint history minus actuator zero
offset, contact / mass*9.81 clamp, noise clamp(-3,3)*std with priv noise=0).
"""

from __future__ import annotations

import numpy as np

from unilab.envs.gh_tracking.observations import (
    ContactForceHistory,
    HistoryBuffer,
    JointPosHistory,
    RootLinVelEMA,
    applied_action_obs,
    body_height,
    boot_indicator_state_obs,
    prev_actions_obs,
    random_noise,
)


# --- 6.1 random_noise + history buffers ------------------------------------ #


def test_random_noise_clamped_to_3sigma() -> None:
    rng = np.random.default_rng(0)
    y = random_noise(np.zeros((100000, 3)), 0.05, rng)
    assert np.abs(y).max() <= 3 * 0.05 + 1e-9  # randn clamped to +-3 then * std


def test_random_noise_zero_std_identity() -> None:
    rng = np.random.default_rng(0)
    x = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(random_noise(x, 0.0, rng), x)


def test_history_roll_newest_at_slot0_and_select() -> None:
    h = HistoryBuffer(num_envs=1, history_steps=[0, 1, 2, 3, 4, 5, 6, 7, 8], dim=3, noise_std=0.0)
    for i in range(3):
        h.update(np.full((1, 3), float(i)))
    out = h.compute()  # (1, 9*3)
    assert out.shape == (1, 27)
    grid = out.reshape(9, 3)
    np.testing.assert_allclose(grid[0], 2.0)  # newest at history step 0
    np.testing.assert_allclose(grid[1], 1.0)
    np.testing.assert_allclose(grid[2], 0.0)


def test_history_reset_fills_buffer() -> None:
    h = HistoryBuffer(1, [0, 1], dim=3, noise_std=0.0)
    h.reset(np.array([0]), np.full((1, 3), 5.0))
    np.testing.assert_allclose(h.compute().reshape(2, 3), 5.0)


def test_history_single_step_policy_dim() -> None:
    h = HistoryBuffer(1, [0], dim=3, noise_std=0.05)  # policy root_ang_vel_history[0]
    h.update(np.zeros((1, 3)))
    assert h.compute().shape == (1, 3)


# --- 6.2 joint_pos_history (2-slot substep 2/3 avg, minus actuator offset) --- #


def test_joint_history_final_average_is_substep_2_3() -> None:
    h = JointPosHistory(1, num_joints=29, history_steps=[0], noise_std=0.0, offset=np.zeros((1, 29)))
    for s in range(4):  # substeps 0,1,2,3 -> slots 0,1,0,1
        h.post_step(s, np.full((1, 29), float(s)))
    h.update()  # buffer[:,0] = mean(slot0=substep2, slot1=substep3) = (2+3)/2 = 2.5
    np.testing.assert_allclose(h.compute().reshape(-1)[0], 2.5)


def test_joint_history_subtracts_actuator_zero_offset_not_default() -> None:
    off = np.full((1, 29), 0.1)  # random actuator zero offset
    h = JointPosHistory(1, 29, [0], 0.0, offset=off)
    for s in range(4):
        h.post_step(s, np.full((1, 29), 1.0))
    h.update()
    np.testing.assert_allclose(h.compute().reshape(-1)[0], 1.0 - 0.1)  # minus offset, not default pose


def test_joint_history_multi_step_dim() -> None:
    h = JointPosHistory(1, 29, [0, 1, 2, 3, 4, 8], 0.0, offset=np.zeros((1, 29)))
    for _ in range(9):
        for s in range(4):
            h.post_step(s, np.zeros((1, 29)))
        h.update()
    assert h.compute().shape == (1, 29 * 6)  # 174 (policy joint_pos_history)


def test_joint_history_reset_fills_with_current() -> None:
    h = JointPosHistory(1, 29, [0, 1], 0.0, offset=np.zeros((1, 29)))
    h.reset(np.array([0]), np.full((1, 29), 0.3))
    np.testing.assert_allclose(h.compute().reshape(2, 29)[0], 0.3)


# --- 6.3 root-linvel EMA + contact history + body_height ------------------- #


def test_root_linvel_ema_iterates_each_substep() -> None:
    e = RootLinVelEMA(1, ema=0.2)
    for _ in range(4):
        e.post_step(np.array([[1.0, 0.0, 0.0]]))
    v = 0.0
    for _ in range(4):
        v = 0.8 * v + 0.2 * 1.0
    np.testing.assert_allclose(e.linvel_w[0, 0], v, atol=1e-9)
    out = e.compute(np.array([[1.0, 0.0, 0.0, 0.0]]))  # identity root -> world == body
    np.testing.assert_allclose(out[0], [v, 0, 0], atol=1e-9)


def test_root_linvel_ema_reset_zeros() -> None:
    e = RootLinVelEMA(2, ema=0.2)
    e.post_step(np.ones((2, 3)))
    e.reset(np.array([0]))
    np.testing.assert_allclose(e.linvel_w[0], 0.0)


def test_contact_history_mean_over_3_and_divide() -> None:
    c = ContactForceHistory(1, n_bodies=2, mass_total=10.0)  # denom = 10*9.81
    for _ in range(3):
        c.post_step(np.tile([0.0, 0.0, 98.1], (1, 2, 1)))
    out = c.compute()  # 98.1 / (10*9.81) = 1.0
    assert out.shape == (1, 6)
    np.testing.assert_allclose(out.reshape(2, 3)[0], [0, 0, 1.0], atol=1e-3)


def test_contact_history_clamps_to_10() -> None:
    c = ContactForceHistory(1, n_bodies=2, mass_total=0.1)  # tiny denom -> huge -> clamp
    for _ in range(3):
        c.post_step(np.tile([1000.0, 0.0, 0.0], (1, 2, 1)))
    np.testing.assert_allclose(c.compute().reshape(2, 3)[0, 0], 10.0)


def test_body_height_selects_z() -> None:
    bp = np.zeros((1, 4, 3))
    bp[0, :, 2] = [0.79, 0.95, 0.03, 0.03]
    np.testing.assert_allclose(body_height(bp), [[0.79, 0.95, 0.03, 0.03]])


# --- 6.4 scalar obs terms -------------------------------------------------- #


def test_applied_action_is_absolute_target_not_residual() -> None:
    tgt = np.arange(29, dtype=float)[None]  # joint_pos_target (default+offset+scale+boot)
    np.testing.assert_array_equal(applied_action_obs(tgt), tgt)  # passed through, NOT residual


def test_prev_actions_flatten_raw() -> None:
    buf = np.arange(1 * 3 * 29, dtype=float).reshape(1, 3, 29)  # action_buf[:, :3]
    out = prev_actions_obs(buf)
    assert out.shape == (1, 87)
    np.testing.assert_array_equal(out, buf.reshape(1, -1))


def test_boot_indicator_state_normalized() -> None:
    np.testing.assert_allclose(boot_indicator_state_obs(np.array([[25.0]]), 25), [[1.0]])
    np.testing.assert_allclose(boot_indicator_state_obs(np.array([[0.0]]), 25), [[0.0]])
    np.testing.assert_allclose(boot_indicator_state_obs(np.array([[5.0]]), 25), [[0.2]])


# --- 6.5 command target / keypoint observations (future frames S=5) --------- #

from unilab.envs.gh_tracking import observations as O  # noqa: E402

IDENT_Q = np.array([[1.0, 0.0, 0.0, 0.0]])


def test_target_joint_pos_obs_flatten() -> None:
    mjp = np.arange(1 * 5 * 29, dtype=float).reshape(1, 5, 29)
    out = O.target_joint_pos_obs(mjp)
    assert out.shape == (1, 145)
    np.testing.assert_array_equal(out, mjp.reshape(1, -1))


def test_target_pos_b_obs_identity_frame() -> None:
    # identity root at origin, env origin 0 -> target_pos_b == motion root pos
    m = np.zeros((1, 5, 3))
    m[0, :, 0] = [0.1, 0.2, 0.3, 0.4, 0.5]
    out = O.target_pos_b_obs(m, np.zeros((1, 3)), IDENT_Q, np.zeros((1, 3)))
    assert out.shape == (1, 15)
    np.testing.assert_allclose(out.reshape(5, 3)[:, 0], [0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-6)


def test_target_linvel_b_obs_identity() -> None:
    m = np.ones((1, 5, 3)) * 0.7
    out = O.target_linvel_b_obs(m, IDENT_Q)
    assert out.shape == (1, 15)
    np.testing.assert_allclose(out, m.reshape(1, -1), atol=1e-6)


def test_target_projected_gravity_b_identity() -> None:
    mq = np.tile([1.0, 0.0, 0.0, 0.0], (1, 5, 1))  # identity motion quat
    out = O.target_projected_gravity_b(mq)
    assert out.shape == (1, 15)
    np.testing.assert_allclose(out.reshape(5, 3), np.tile([0, 0, -1.0], (5, 1)), atol=1e-6)


def test_current_keypoint_b_dim_and_identity() -> None:
    kp = np.zeros((1, 11, 3))
    kp[0, :, 2] = np.linspace(0.1, 1.1, 11)
    out = O.current_keypoint_b(kp, np.zeros((1, 3)), IDENT_Q)
    assert out.shape == (1, 33)
    np.testing.assert_allclose(out.reshape(11, 3)[:, 2], np.linspace(0.1, 1.1, 11), atol=1e-6)


def test_current_keypoint_vel_b_dim() -> None:
    kv = np.ones((1, 11, 3)) * 0.2
    out = O.current_keypoint_vel_b(kv, IDENT_Q)
    assert out.shape == (1, 33)
    np.testing.assert_allclose(out, kv.reshape(1, -1), atol=1e-6)


def test_target_keypoints_diff_reads_body_pos_w_and_dim() -> None:  # D4
    motion_kp_w = np.ones((1, 5, 11, 3)) * 0.5  # target keypoints (future x keypoint)
    actual_kp_w = np.ones((1, 11, 3)) * 0.2
    out = O.target_keypoints_diff_b_obs(motion_kp_w, actual_kp_w, np.zeros((1, 3)), IDENT_Q, np.zeros((1, 3)))
    assert out.shape == (1, 165)  # 11 * 5 * 3
    np.testing.assert_allclose(out.reshape(5, 11, 3), 0.3, atol=1e-6)  # 0.5 - 0.2


def test_relative_quat_obs_identity_is_zero() -> None:
    mq = np.tile([1.0, 0.0, 0.0, 0.0], (1, 5, 1))
    out = O.relative_quat_obs(mq, IDENT_Q)
    assert out.shape == (1, 15)
    np.testing.assert_allclose(out, 0.0, atol=1e-6)


# --- 6.6 command(22) + force_priv(55) -------------------------------------- #


def test_command_dim_and_composition() -> None:
    mpos = np.zeros((1, 5, 3))
    mpos[0, :, 2] = [0.7, 0.71, 0.72, 0.73, 0.74]  # future root heights
    mpos[0, 1:, 0] = [0.1, 0.2, 0.3, 0.4]  # future x displacement
    mquat = np.tile([1.0, 0.0, 0.0, 0.0], (1, 5, 1))  # identity -> yaw quat identity
    out = O.command_obs(mpos, mquat, np.array([[10.0]]))
    assert out.shape == (1, 22)
    np.testing.assert_allclose(out[0, :5], [0.7, 0.71, 0.72, 0.73, 0.74], atol=1e-6)  # heights
    # pos_diff_b for future[1:5], xy (identity yaw -> world diff = future - first)
    pos_diff = out[0, 5:13].reshape(4, 2)
    np.testing.assert_allclose(pos_diff[:, 0], [0.1, 0.2, 0.3, 0.4], atol=1e-6)  # x diffs
    np.testing.assert_allclose(out[0, 21], 10.0)  # force_safe_limit last


def test_force_priv_dim_and_reads_force_state() -> None:
    kb = np.ones((1, 6, 3)) * 0.1
    ab = np.ones((1, 6, 3)) * 0.2
    eb = np.ones((1, 6, 3)) * 0.3
    timer = np.array([[42.0]])
    out = O.force_priv_obs(kb, ab, eb, timer)
    assert out.shape == (1, 55)  # 18 + 18 + 18 + 1
    np.testing.assert_allclose(out[0, :18].reshape(6, 3), 0.1)
    np.testing.assert_allclose(out[0, 18:36].reshape(6, 3), 0.2)
    np.testing.assert_allclose(out[0, 36:54].reshape(6, 3), 0.3)
    np.testing.assert_allclose(out[0, 54], 42.0)


# --- 6.7 ObservationManager three-group offset assembly -------------------- #

from unilab.envs.gh_tracking.observations import ObservationManager, ObsState  # noqa: E402


def _make_state(n: int = 2) -> ObsState:
    ident = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    return ObsState(
        root_pos_w=np.zeros((n, 3)),
        root_quat_w=ident,
        root_ang_vel_b=np.full((n, 3), 0.11),
        projected_gravity_b=np.tile([0.0, 0.0, -1.0], (n, 1)),
        env_origins=np.zeros((n, 3)),
        joint_pos_target=np.tile(np.arange(29.0), (n, 1)),
        applied_torque=np.full((n, 29), 0.7),
        action_buf=np.full((n, 3, 29), 0.3),
        body_pos_w_height=np.tile(np.array([[0.0, 0.0, 0.79]]), (n, 4, 1)),
        body_pos_w_kp=np.full((n, 11, 3), 0.4),
        body_lin_vel_w_kp=np.full((n, 11, 3), 0.5),
        motion_root_pos_w=np.zeros((n, 5, 3)),
        motion_root_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (n, 5, 1)),
        motion_root_lin_vel_w=np.full((n, 5, 3), 0.6),
        motion_joint_pos=np.full((n, 5, 29), 0.8),
        motion_body_pos_w_kp=np.full((n, 5, 11, 3), 0.9),
        force_keypoint_b=np.full((n, 6, 3), 0.12),
        force_applied_b=np.full((n, 6, 3), 0.13),
        force_expected_b=np.full((n, 6, 3), 0.14),
        force_sample_timer=np.full((n, 1), 33.0),
        force_safe_limit=np.full((n, 1), 10.0),
        boot_indicator=np.full((n, 1), 25.0),
        cum_error=np.full((n, 3), 0.9),
        joint_pos=np.full((n, 29), 1.0),
        root_lin_vel_w=np.full((n, 3), 0.5),
        net_contact_force=np.zeros((n, 2, 3)),
    )


def _drive(om: ObservationManager, state: ObsState) -> None:
    om.reset(np.arange(state.root_pos_w.shape[0]), state)
    for s in range(4):
        om.post_step(s, state)
    om.update(state)


def test_obs_groups_spec() -> None:
    om = ObservationManager(num_envs=2, mass_total=40.0, actuator_offset=np.zeros((2, 29)))
    assert om.obs_groups_spec == {"policy": 450, "priv": 717, "priv_critic": 3}


def test_policy_group_dim_and_representative_offsets() -> None:
    om = ObservationManager(num_envs=2, mass_total=40.0, actuator_offset=np.zeros((2, 29)))
    state = _make_state(2)
    _drive(om, state)
    obs = om.compute(state)
    p = obs["policy"]
    assert p.shape == (2, 450)
    np.testing.assert_allclose(p[:, 0:1], O.boot_indicator_state_obs(state.boot_indicator, 25))
    np.testing.assert_allclose(
        p[:, 1:23], O.command_obs(state.motion_root_pos_w, state.motion_root_quat_w, state.force_safe_limit)
    )
    np.testing.assert_allclose(p[:, 23:168], O.target_joint_pos_obs(state.motion_joint_pos))
    np.testing.assert_allclose(p[:, 363:450], O.prev_actions_obs(state.action_buf))


def test_priv_group_dim_and_representative_offsets() -> None:
    om = ObservationManager(num_envs=2, mass_total=40.0, actuator_offset=np.zeros((2, 29)))
    state = _make_state(2)
    _drive(om, state)
    obs = om.compute(state)
    v = obs["priv"]
    assert v.shape == (2, 717)
    np.testing.assert_allclose(
        v[:, 45:100],
        O.force_priv_obs(state.force_keypoint_b, state.force_applied_b, state.force_expected_b, state.force_sample_timer),
    )
    np.testing.assert_allclose(v[:, 659:688], O.applied_action_obs(state.joint_pos_target))
    np.testing.assert_allclose(v[:, 688:717], state.applied_torque)  # priv noise = 0


def test_priv_critic_is_cum_error() -> None:
    om = ObservationManager(num_envs=2, mass_total=40.0, actuator_offset=np.zeros((2, 29)))
    state = _make_state(2)
    _drive(om, state)
    obs = om.compute(state)
    assert obs["priv_critic"].shape == (2, 3)
    np.testing.assert_allclose(obs["priv_critic"], state.cum_error)


def test_three_groups_not_flattened() -> None:
    om = ObservationManager(num_envs=2, mass_total=40.0, actuator_offset=np.zeros((2, 29)))
    state = _make_state(2)
    _drive(om, state)
    obs = om.compute(state)
    assert set(obs.keys()) == {"policy", "priv", "priv_critic"}


# --- 6.8 post-substep telemetry through the per-substep hook + real backend --- #

from dataclasses import replace  # noqa: E402

from unilab.assets import ASSETS_ROOT_PATH  # noqa: E402
from unilab.base.backend.mujoco.backend import MuJoCoBackend  # noqa: E402
from unilab.base.scene import SceneCfg  # noqa: E402
from unilab.utils.rotation import np_quat_apply_inverse  # noqa: E402


def test_post_step_hook_joint_telemetry_is_substep_2_3_average() -> None:
    # driving post_step for substeps 0..3 with distinct joint_pos must yield the
    # substep 2/3 average at the priv joint-history offset (proves the hook path,
    # not a single control-step-end read).
    om = ObservationManager(num_envs=1, mass_total=40.0, actuator_offset=np.zeros((1, 29)))
    state = _make_state(1)
    om.reset(np.arange(1), state)
    for s in range(4):
        om.post_step(s, replace(state, joint_pos=np.full((1, 29), float(s))))
    om.update(state)
    obs = om.compute(state)
    priv_joint_step0 = obs["priv"][:, 167:196]  # priv joint_pos_history, history step 0
    np.testing.assert_allclose(priv_joint_step0[0], 2.5)  # (substep2 + substep3)/2


def test_post_step_hook_root_ema_iterates_4x() -> None:
    om = ObservationManager(num_envs=1, mass_total=40.0, actuator_offset=np.zeros((1, 29)))
    state = _make_state(1)
    om.reset(np.arange(1), state)
    for _ in range(4):
        om.post_step(0, replace(state, root_lin_vel_w=np.array([[1.0, 0.0, 0.0]])))
    om.update(state)
    obs = om.compute(state)  # identity root -> body == world
    v = 0.0
    for _ in range(4):
        v = 0.8 * v + 0.2 * 1.0
    np.testing.assert_allclose(obs["priv"][0, 110:113], [v, 0, 0], atol=1e-9)  # root_linvel_b


_KEYPOINTS = [
    "head_mimic", "left_hand_mimic", "right_hand_mimic",
    "left_wrist_roll_link", "right_wrist_roll_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_knee_link", "right_knee_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
]
_HEIGHT_BODIES = ["pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link"]
_FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]


def test_real_g1gh_backend_obs_dims() -> None:
    scene = str(ASSETS_ROOT_PATH / "robots" / "g1_gh" / "scene_flat.xml")
    n = 2
    bk = MuJoCoBackend(SceneCfg(model_file=scene), n, 0.005, base_name="pelvis", add_body_sensors=True)
    bk.materialize()

    kp = bk.get_body_ids(_KEYPOINTS)
    assert bk.get_body_pos_w(kp).shape == (n, 11, 3)  # keypoint poses via add_body_sensors

    root_quat = bk.get_base_quat()
    grav = np_quat_apply_inverse(root_quat, np.tile([0.0, 0.0, -1.0], (n, 1)))
    om = ObservationManager(num_envs=n, mass_total=float(bk.model.body_mass.sum()),
                            actuator_offset=np.zeros((n, 29)))
    state = ObsState(
        root_pos_w=bk.get_base_pos(), root_quat_w=root_quat,
        root_ang_vel_b=bk.get_base_ang_vel(), projected_gravity_b=grav,
        env_origins=np.zeros((n, 3)),
        joint_pos_target=np.zeros((n, 29)), applied_torque=bk.get_actuator_effort(),
        action_buf=np.zeros((n, 3, 29)),
        body_pos_w_height=bk.get_body_pos_w(bk.get_body_ids(_HEIGHT_BODIES)),
        body_pos_w_kp=bk.get_body_pos_w(kp), body_lin_vel_w_kp=bk.get_body_lin_vel_w(kp),
        motion_root_pos_w=np.zeros((n, 5, 3)), motion_root_quat_w=np.tile([1.0, 0, 0, 0], (n, 5, 1)),
        motion_root_lin_vel_w=np.zeros((n, 5, 3)), motion_joint_pos=np.zeros((n, 5, 29)),
        motion_body_pos_w_kp=np.zeros((n, 5, 11, 3)),
        force_keypoint_b=np.zeros((n, 6, 3)), force_applied_b=np.zeros((n, 6, 3)),
        force_expected_b=np.zeros((n, 6, 3)), force_sample_timer=np.zeros((n, 1)),
        force_safe_limit=np.full((n, 1), 10.0), boot_indicator=np.zeros((n, 1)),
        cum_error=np.zeros((n, 3)), joint_pos=bk.get_dof_pos(),
        root_lin_vel_w=bk.get_base_lin_vel(),
        net_contact_force=bk.get_body_net_contact_force_w(bk.get_body_ids(_FEET)),
    )
    om.reset(np.arange(n), state)
    for s in range(4):
        om.post_step(s, state)
    om.update(state)
    obs = om.compute(state)
    assert obs["policy"].shape == (n, 450)
    assert obs["priv"].shape == (n, 717)
    assert obs["priv_critic"].shape == (n, 3)
