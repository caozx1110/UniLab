"""Tests for GH rewards (Phase 7): 3-group vector reward + per-term formulas.

impedance / tracking / loco groups; each group = Sum(weight * term) * current_factor
* step_dt -> a 3-vector (GAE sums it). Covers the quirk-1 action_rate_l2, the
17-joint tracking subset, the instantaneous _cum_error, and the student root cache.
"""

from __future__ import annotations

import numpy as np

from unilab.envs.gh_tracking.rewards import (
    JointVel2Slot,
    RewardManager,
    action_rate_l2,
    calc_exp_sigma,
    feet_slip,
    impact_force_l2,
    joint_pos_limits,
    joint_pos_tracking,
    joint_vel_l2,
    keypoint_tracking,
    resolve_tracking_joints,
    root_pos_tracking,
    root_rot_tracking,
    survival,
)
from unilab.envs.gh_tracking.motion_dataset import JOINT_NAMES


# --- 7.1 reward aggregation + calc_exp_sigma ------------------------------- #


def test_calc_exp_sigma_positive_and_averaged() -> None:
    np.testing.assert_allclose(calc_exp_sigma(np.array([[0.0]]), [0.3]), 1.0)  # exp(0)=1
    np.testing.assert_allclose(
        calc_exp_sigma(np.array([[1.0]]), [1.0, 0.5]), (np.exp(-1) + np.exp(-2)) / 2
    )
    assert (calc_exp_sigma(np.array([[5.0]]), [0.3]) > 0).all()  # always positive


def test_reward_manager_group_vector_weight_and_step_dt() -> None:
    fns = {"survival": lambda s: np.ones((s["n"], 1))}
    rm = RewardManager(
        groups={"impedance": [], "tracking": [], "loco": [("survival", 5.0)]},
        term_fns=fns,
        step_dt=0.02,
    )
    r = rm.compute({"n": 3})
    assert r.shape == (3, 3)  # one column per group (impedance/tracking/loco)
    np.testing.assert_allclose(r[:, 2], 5.0 * 1.0 * 0.02)  # loco = weight * survival * step_dt
    np.testing.assert_allclose(r[:, 0], 0.0)  # empty impedance group


def test_reward_not_clipped() -> None:
    fns = {"big": lambda s: np.full((s["n"], 1), 1000.0)}
    rm = RewardManager(groups={"impedance": [], "tracking": [], "loco": [("big", 100.0)]},
                       term_fns=fns, step_dt=1.0)
    r = rm.compute({"n": 1})
    np.testing.assert_allclose(r[:, 2], 100000.0)  # clip_rewards is inert


def test_student_train_sets_progress_factor_one() -> None:
    fns = {"survival": lambda s: np.ones((s["n"], 1))}
    rm = RewardManager(groups={"impedance": [], "tracking": [], "loco": [("survival", 1.0)]},
                       term_fns=fns, step_dt=1.0, student_train=True)
    rm.step_schedule(0.0)  # student -> progress forced to 1.0
    np.testing.assert_allclose(rm.current_factor, 1.0)
    np.testing.assert_allclose(rm.compute({"n": 1})[:, 2], 1.0)


# --- 7.2 loco rewards (quirk 1) -------------------------------------------- #


def test_action_rate_l2_quirk1_compares_joint01_over_3_slots() -> None:
    buf = np.zeros((1, 3, 29))  # action_buf (N, hist=3, 29)
    buf[:, :, 0] = [1.0, 2.0, 3.0]  # joint dim 0 across the 3 history slots
    buf[:, :, 1] = 0.0  # joint dim 1
    # -(buf[:,:,0] - buf[:,:,1])^2 summed over the 3 slots = -(1+4+9) = -14
    np.testing.assert_allclose(action_rate_l2(buf), [[-14.0]])


def test_action_rate_l2_is_not_adjacent_timestep_diff() -> None:
    # a pure adjacent-time diff would be zero here (all slots equal per joint);
    # the quirk compares joint 0 vs joint 1, so it is non-zero.
    buf = np.zeros((1, 3, 29))
    buf[:, :, 0] = 5.0
    buf[:, :, 1] = 1.0
    np.testing.assert_allclose(action_rate_l2(buf), [[-(4.0**2) * 3]])  # (5-1)^2 * 3 slots


def test_survival_is_one() -> None:
    np.testing.assert_array_equal(survival(2), np.ones((2, 1)))


def test_impact_force_first_contact_and_clamp20() -> None:
    # net force history (N, 3, nbody, 3); take norm mean over history
    hist = np.zeros((1, 3, 2, 3))
    hist[:, :, 0, 2] = 98.1  # body 0 vertical force
    first_contact = np.array([[1.0, 0.0]])  # only body 0 just landed
    r = impact_force_l2(hist, first_contact, mass_total=10.0)  # f=98.1/(10*9.81)=1 -> -1
    np.testing.assert_allclose(r, [[-1.0]], atol=1e-3)


def test_impact_force_clamps_to_neg20() -> None:
    hist = np.full((1, 3, 2, 3), 1e5)
    first_contact = np.ones((1, 2))
    r = impact_force_l2(hist, first_contact, mass_total=0.1)
    np.testing.assert_allclose(r, [[-20.0]])  # clamp_max(20) on the magnitude


def test_feet_slip_only_in_contact() -> None:
    in_contact = np.array([[1.0, 0.0]])
    feet_vel_xy = np.array([[[3.0, 4.0], [10.0, 10.0]]])  # body0 |v|=5, body1 not in contact
    np.testing.assert_allclose(feet_slip(in_contact, feet_vel_xy), [[-25.0]])  # -(1*25 + 0)


def test_joint_vel_l2_2slot_mean() -> None:
    b = JointVel2Slot(num_envs=1, num_joints=29)
    for s in range(4):  # substeps 0..3 -> slots 0,1,0,1
        b.post_step(s, np.full((1, 29), float(s)))
    # mean(slot0=2, slot1=3) = 2.5 -> -sum(2.5^2 * 29)
    np.testing.assert_allclose(joint_vel_l2(b.mean()), [[-(2.5**2) * 29]])


def test_joint_pos_limits_soft_violation() -> None:
    jpos = np.array([[1.0]])  # one joint, above soft upper limit
    soft_lo = np.array([[-0.5]])
    soft_hi = np.array([[0.5]])
    r = joint_pos_limits(jpos, soft_lo, soft_hi, soft_factor=0.9)
    np.testing.assert_allclose(r, [[-(0.5) / (1 - 0.9)]])  # violation 0.5, /(1-0.9)


# --- 7.3 tracking rewards + _cum_error (17-joint) -------------------------- #

TRACKING_PATTERNS = ["waist_*", ".*_hip_.*", ".*_knee.*", ".*wrist.*"]


def test_resolve_tracking_joints_is_17() -> None:
    ids = resolve_tracking_joints(list(JOINT_NAMES), TRACKING_PATTERNS)
    assert len(ids) == 17  # waist3 + hip6 + knee2 + wrist6
    names = [JOINT_NAMES[i] for i in ids]
    assert "left_shoulder_pitch_joint" not in names  # shoulder excluded
    assert "left_elbow_joint" not in names  # elbow excluded
    assert "left_ankle_roll_joint" not in names  # ankle excluded
    assert "waist_yaw_joint" in names and "left_hip_pitch_joint" in names
    assert "left_knee_joint" in names and "left_wrist_yaw_joint" in names


def test_root_pos_tracking_reward_and_cum_error_0p3() -> None:
    cur = np.zeros((1, 3))
    tgt = np.array([[0.3, 0.0, 0.0]])  # error = 0.3
    r, cum = root_pos_tracking(cur, tgt, [0.3])
    np.testing.assert_allclose(cum, [[1.0]])  # error / _cum_root_pos_scale = 0.3/0.3
    np.testing.assert_allclose(r, calc_exp_sigma(np.array([[0.3]]), [0.3]))


def test_root_rot_tracking_cum_error_0p7() -> None:
    ident = np.array([[1.0, 0.0, 0.0, 0.0]])
    r, cum = root_rot_tracking(ident, ident, [1.0, 0.5])  # zero rotation error
    np.testing.assert_allclose(cum, [[0.0]], atol=1e-6)
    np.testing.assert_allclose(r, 1.0, atol=1e-6)  # exp(0)


def test_keypoint_tracking_cum_error_0p25_and_mean() -> None:
    actual = np.zeros((1, 11, 3))
    target = np.zeros((1, 11, 3))
    target[0, :, 0] = 0.25  # each keypoint error 0.25
    r, cum = keypoint_tracking(actual, target, [0.3])
    np.testing.assert_allclose(cum, [[1.0]], atol=1e-6)  # mean error 0.25 / 0.25
    np.testing.assert_allclose(r, calc_exp_sigma(np.array([[0.25]]), [0.3]), atol=1e-6)


def test_joint_pos_tracking_abs_mean_over_17() -> None:
    actual = np.zeros((1, 17))
    target = np.full((1, 17), 0.1)
    r = joint_pos_tracking(actual, target, [0.4, 0.2])
    np.testing.assert_allclose(r, calc_exp_sigma(np.array([[0.1]]), [0.4, 0.2]), atol=1e-6)


# --- 7.4 impedance rewards ------------------------------------------------- #

from unilab.envs.gh_tracking.rewards import (  # noqa: E402
    force_exd_penalty,
    force_reward,
    force_target_tracking,
)


def test_force_reward_matches_when_applied_equals_expected() -> None:
    applied = np.full((1, 6, 3), 1.0)  # |.|=sqrt3 < safe+10
    expected = applied.copy()
    r = force_reward(applied, expected, np.full((1, 1), 10.0), [8.0, 4.0])
    np.testing.assert_allclose(r, 1.0, atol=1e-6)  # diff 0 -> exp_sigma(0)=1, no exceed


def test_force_reward_zeroed_when_force_exceeds_limit() -> None:
    applied = np.zeros((1, 6, 3))
    applied[0, 0, 0] = 100.0  # body 0 exceeds safe+10
    expected = np.zeros((1, 6, 3))
    r = force_reward(applied, expected, np.full((1, 1), 10.0), [8.0, 4.0])
    np.testing.assert_allclose(r, 0.0)  # exceed -> reward masked to 0


def test_force_exd_penalty_when_over_both_thresholds() -> None:
    applied = np.zeros((1, 6, 3))
    applied[0, 0, 0] = 100.0  # > safe+10 and > exp+5
    expected = np.zeros((1, 6, 3))
    p = force_exd_penalty(applied, expected, np.full((1, 1), 10.0))
    np.testing.assert_allclose(p, [[-1.0 / 6]])  # 1 of 6 bodies exceeds -> mean = 1/6


def test_force_target_tracking_mean_distance() -> None:
    actual = np.zeros((1, 6, 3))
    target = np.zeros((1, 6, 3))
    target[0, :, 0] = 0.3
    r = force_target_tracking(actual, target, [0.3])
    np.testing.assert_allclose(r, calc_exp_sigma(np.array([[0.3]]), [0.3]), atol=1e-6)


# --- 7.5 student root-target cache ----------------------------------------- #

from unilab.envs.gh_tracking.rewards import (  # noqa: E402
    StudentRootCache,
    teacher_reward_target,
)

_IDENT = np.array([[1.0, 0.0, 0.0, 0.0]])


def test_teacher_reward_target_is_current_frame_plus_origin() -> None:
    motion_pos0 = np.array([[0.1, 0.2, 0.7]])
    origin = np.array([[5.0, 0.0, 0.0]])
    pos, quat = teacher_reward_target(motion_pos0, _IDENT, origin)
    np.testing.assert_allclose(pos, [[5.1, 0.2, 0.7]])  # motion[:,0] + env_origins
    np.testing.assert_allclose(quat, _IDENT)


def test_student_cache_consumes_fill_frames_in_order() -> None:
    c = StudentRootCache(num_envs=1, steps=50)
    fill_pos = np.zeros((1, 50, 3))
    fill_pos[0, :, 0] = np.arange(50)  # frame i has x = i
    fill_quat = np.tile([1.0, 0.0, 0.0, 0.0], (1, 50, 1))
    c.reset(np.array([0]), fill_pos, fill_quat, np.zeros((1, 3)))

    def _step():
        return c.step(
            root_pos_w=np.zeros((1, 3)), root_quat_w=_IDENT,
            ref_pos_t=np.zeros((1, 3)), ref_quat_t=_IDENT,
            ref_pos_t_plus=np.zeros((1, 3)), ref_quat_t_plus=_IDENT,
        )

    p1, _ = _step()
    p2, _ = _step()
    np.testing.assert_allclose(p1[0, 0], 0.0)  # head = frame 0
    np.testing.assert_allclose(p2[0, 0], 1.0)  # head = frame 1 (shifted)


def test_student_cache_tail_is_robot_anchored_at_step_51() -> None:
    c = StudentRootCache(num_envs=1, steps=50)
    fill_pos = np.zeros((1, 50, 3))
    fill_quat = np.tile([1.0, 0.0, 0.0, 0.0], (1, 50, 1))
    c.reset(np.array([0]), fill_pos, fill_quat, np.zeros((1, 3)))

    # step 1: robot at x=100, ref delta (t+50 - t) = [10,0,0] -> tail x = 110 (identity yaw)
    c.step(root_pos_w=np.array([[100.0, 0.0, 0.0]]), root_quat_w=_IDENT,
           ref_pos_t=np.zeros((1, 3)), ref_quat_t=_IDENT,
           ref_pos_t_plus=np.array([[10.0, 0.0, 0.0]]), ref_quat_t_plus=_IDENT)
    # steps 2..50 (no anchoring contribution; robot at origin, zero ref delta)
    for _ in range(49):
        c.step(root_pos_w=np.zeros((1, 3)), root_quat_w=_IDENT,
               ref_pos_t=np.zeros((1, 3)), ref_quat_t=_IDENT,
               ref_pos_t_plus=np.zeros((1, 3)), ref_quat_t_plus=_IDENT)
    # step 51 consumes the robot-anchored tail written at step 1
    p51, _ = c.step(root_pos_w=np.zeros((1, 3)), root_quat_w=_IDENT,
                    ref_pos_t=np.zeros((1, 3)), ref_quat_t=_IDENT,
                    ref_pos_t_plus=np.zeros((1, 3)), ref_quat_t_plus=_IDENT)
    np.testing.assert_allclose(p51[0, 0], 110.0, atol=1e-4)  # robot(100) + ref delta(10)
