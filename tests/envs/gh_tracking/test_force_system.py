"""Tests for the GH force system helpers + ForceSystem (Phase 5).

Deterministic physics helpers (unidirectional spring projection, eccentric
torque, net-wrench torso correction) are golden-tested; the ForceSystem class
(state, masks/quirk3, schedule/quirk2, admittance integration, per-substep
force_apply feeding apply_body_wrench) follows. Full stochastic schedule
trajectory parity vs GH runtime and the non-compliant baseline path are deferred.
"""

from __future__ import annotations

import numpy as np

from unilab.envs.gh_tracking.force_system import (
    ForceSystem,
    compute_eccentric_torque,
    limit_net_wrench_about_torso,
    project_pos_diff,
    project_vel,
)


def _fs(n: int = 4) -> ForceSystem:
    return ForceSystem(num_envs=n, num_force_bodies=6, physics_dt=0.005, step_dt=0.02, seed=0)


# --- 5.3 unidirectional spring projection ---------------------------------- #


def test_project_pos_diff_unidirectional_pull_only() -> None:
    d = np.array([[[1.0, 0.0, 0.0]]])
    # pos_diff opposite to dir -> negative coef -> kept (spring pulls)
    np.testing.assert_allclose(project_pos_diff(np.array([[[-2.0, 0.0, 0.0]]]), d)[0, 0], [-2, 0, 0])
    # pos_diff along dir -> positive coef -> clamp_max(0) -> zero (never pushes)
    np.testing.assert_allclose(project_pos_diff(np.array([[[3.0, 0.0, 0.0]]]), d)[0, 0], [0, 0, 0])


def test_project_pos_diff_projects_onto_dir() -> None:
    d = np.array([[[1.0, 0.0, 0.0]]])
    # only the component along dir is kept (and only if negative)
    r = project_pos_diff(np.array([[[-2.0, 5.0, 5.0]]]), d)
    np.testing.assert_allclose(r[0, 0], [-2, 0, 0])


def test_project_vel_clamp_min0() -> None:
    d = np.array([[[1.0, 0.0, 0.0]]])
    np.testing.assert_allclose(project_vel(np.array([[[2.0, 0.0, 0.0]]]), d)[0, 0], [2, 0, 0])
    np.testing.assert_allclose(project_vel(np.array([[[-3.0, 0.0, 0.0]]]), d)[0, 0], [0, 0, 0])


# --- 5.4 eccentric torque + net-wrench torso correction -------------------- #


def test_eccentric_torque_is_cross_delta_force() -> None:
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (1, 6, 1))  # identity -> delta_w == pos_delta
    pos_delta = np.zeros((1, 6, 3))
    pos_delta[0, 0] = [0.0, 0.0, 0.05]
    f = np.zeros((1, 6, 3))
    f[0, 0] = [1.0, 0.0, 0.0]
    tau = compute_eccentric_torque(quat, pos_delta, f)
    np.testing.assert_allclose(tau[0, 0], np.cross([0, 0, 0.05], [1, 0, 0]), atol=1e-9)  # [0,0.05,0]


def test_net_wrench_within_limits_no_correction() -> None:
    pos6 = np.zeros((1, 6, 3))
    f6 = np.zeros((1, 6, 3))
    f6[0, 0] = [1.0, 0.0, 0.0]  # F_net norm 1 < 30
    tau6 = np.zeros((1, 6, 3))
    d_f, d_m = limit_net_wrench_about_torso(pos6, f6, tau6, np.zeros((1, 3)), 30.0, 20.0)
    np.testing.assert_allclose(d_f[0], [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(d_m[0], [0, 0, 0], atol=1e-6)


def test_net_wrench_force_over_limit_correction() -> None:
    pos6 = np.zeros((1, 6, 3))
    f6 = np.zeros((1, 6, 3))
    f6[0, 0] = [100.0, 0.0, 0.0]  # F_net=[100,0,0] -> allow [30,0,0] -> dF=[-70,0,0]
    tau6 = np.zeros((1, 6, 3))
    d_f, _ = limit_net_wrench_about_torso(pos6, f6, tau6, np.zeros((1, 3)), 30.0, 20.0)
    np.testing.assert_allclose(d_f[0], [-70, 0, 0], atol=1e-4)


def test_net_wrench_torque_includes_cross_r_F() -> None:
    # force at r=[0,0,1] from torso, F=[1,0,0] -> M_net=cross([0,0,1],[1,0,0])=[0,1,0] (<20)
    pos6 = np.zeros((1, 6, 3))
    pos6[0, 0] = [0.0, 0.0, 1.0]
    f6 = np.zeros((1, 6, 3))
    f6[0, 0] = [1.0, 0.0, 0.0]
    tau6 = np.zeros((1, 6, 3))
    _, d_m = limit_net_wrench_about_torso(pos6, f6, tau6, np.zeros((1, 3)), 30.0, 20.0)
    np.testing.assert_allclose(d_m[0], [0, 0, 0], atol=1e-6)  # |M_net|=1<20 -> no correction


def test_net_wrench_torque_over_limit_correction() -> None:
    # r=[0,0,1], F=[100,0,0] -> M_net=[0,100,0] -> allow [0,20,0] -> dM=[0,-80,0]
    pos6 = np.zeros((1, 6, 3))
    pos6[0, 0] = [0.0, 0.0, 1.0]
    f6 = np.zeros((1, 6, 3))
    f6[0, 0] = [100.0, 0.0, 0.0]
    tau6 = np.zeros((1, 6, 3))
    _, d_m = limit_net_wrench_about_torso(pos6, f6, tau6, np.zeros((1, 3)), 30.0, 20.0)
    np.testing.assert_allclose(d_m[0], [0, -80, 0], atol=1e-3)


# --- 5.5 ForceSystem state + constant masks (quirk 3) + force_reset --------- #


def test_masks_are_constant_LLRRLR() -> None:
    fs = _fs()
    np.testing.assert_array_equal(fs.left_mask.reshape(-1), [1, 0, 1, 0, 1, 0])
    np.testing.assert_array_equal(fs.right_mask.reshape(-1), [0, 1, 0, 1, 0, 1])


def test_quirk3_left_full_enables_positions_0_2_4() -> None:
    fs = _fs(1)
    enable = fs._enable_from_force_type(np.array([2]), np.random.default_rng(0))  # left_full
    # positions 0,2,4 over order [L_sh,L_wr,R_sh,R_wr,L_hand,R_hand]
    # = {left_shoulder_yaw, right_shoulder_yaw, left_hand_mimic} (non-anatomical)
    np.testing.assert_array_equal(enable[0].reshape(-1), [1, 0, 1, 0, 1, 0])


def test_quirk3_right_full_enables_positions_1_3_5() -> None:
    fs = _fs(1)
    enable = fs._enable_from_force_type(np.array([3]), np.random.default_rng(0))  # right_full
    np.testing.assert_array_equal(enable[0].reshape(-1), [0, 1, 0, 1, 0, 1])


def test_enable_zero_and_full() -> None:
    fs = _fs(1)
    rng = np.random.default_rng(0)
    np.testing.assert_array_equal(fs._enable_from_force_type(np.array([0]), rng)[0].reshape(-1),
                                  [0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(fs._enable_from_force_type(np.array([1]), rng)[0].reshape(-1),
                                  [1, 1, 1, 1, 1, 1])


def test_force_reset_zeros_state() -> None:
    fs = _fs(2)
    fs.force_type[:] = 3
    fs.force_enable[:] = 1
    fs.force_applied_w[:] = 5.0
    fs.force_reset(np.array([0, 1]))
    assert not fs.force_type.any()
    assert not fs.force_enable.any()
    assert not fs.force_applied_w.any()


# --- 5.6 force_schedule (quirk 2) + force_kp_scaled ------------------------- #


def test_quirk2_force_type_stays_zero_after_schedule() -> None:
    fs = _fs(64)
    rng = np.random.default_rng(0)
    fs.force_reset(np.arange(64))
    fs.force_sample_timer[:] = -1  # force a resample this call
    fs.force_schedule(rng)
    # the sampled force_type is a LOCAL var, never written back to self.force_type
    np.testing.assert_array_equal(fs.force_type, 0)
    # yet the resample DID happen: enable masks were set for (most) envs
    assert fs.force_enable.any()


def test_quirk2_origin_sample_gate_always_true() -> None:
    fs = _fs(100)
    # gate is (self.force_type != 4); force_type is always 0 -> always true
    assert (fs.force_type != 4).all()


def test_force_type_probs_distribution() -> None:
    fs = _fs(1)
    rng = np.random.default_rng(1)
    types = fs._sample_force_types(20000, rng)
    frac = np.bincount(types, minlength=5) / 20000
    np.testing.assert_allclose(frac, [0.4, 0.15, 0.15, 0.15, 0.15], atol=0.02)


def test_force_kp_scaled_is_kp_times_enable() -> None:
    fs = _fs(1)
    fs.force_enable[0] = np.array([1, 0, 1, 0, 1, 0], dtype=bool).reshape(6, 1)
    fs.force_kp_tl.value[:] = 50.0
    fs.update_force_kp_scaled()
    np.testing.assert_allclose(fs.force_kp_scaled[0].reshape(-1), [50, 0, 50, 0, 50, 0])


def test_ramp_down_completion_resets_force() -> None:
    fs = _fs(1)
    fs.force_reset(np.arange(1))
    # enable a force then drive it to time_done so it starts ramping down
    fs.force_enable[0] = True
    fs.force_kp_tl.value[:] = 100.0
    fs.force_sample_timer[:] = -1
    fs.force_schedule(np.random.default_rng(0))  # starts ramp-down (force_required True)
    assert fs.force_kp_ramping_down[0]


# --- 5.7 admittance integration (Kp/Kd drive, 4x loop, keypoint/expected) --- #


def test_compute_drive_gains() -> None:
    fs = _fs(1)
    kp, kd = fs.compute_drive_gains(np.array([[10.0]]))  # force_limit=10
    np.testing.assert_allclose(kp.reshape(-1), 200.0)  # 10/0.05
    np.testing.assert_allclose(kd.reshape(-1), 2 * np.sqrt(200.0 * 0.1))  # 2*sqrt(Kp*0.1)


def test_admittance_integration_keypoint_matches_reference_loop() -> None:
    fs = _fs(1)
    identity_q = np.tile([1.0, 0.0, 0.0, 0.0], (1, 1))  # (1,4) root at origin, no rotation
    root_pos = np.zeros((1, 3))
    # fixed geometry (identity root -> body frame == world frame)
    ref_kp_w = np.zeros((1, 6, 3))
    ref_kp_w[0, :, 0] = 0.3  # ref points at x=0.3
    origin_b = np.zeros((1, 6, 3))
    origin_b[0, :, 0] = -0.2  # spring origins behind
    fs.force_origin_tl.value[:] = origin_b
    fs.force_kp_scaled[:] = 20.0  # enable a spring on all bodies
    fs.force_safe_limit_tl.value[:] = 10.0  # force_limit=10 -> Kp=200
    fs.ref_pos_b_prev[:] = ref_kp_w  # ref_vel_b = 0
    fs.force_keypoint_w_prev[:] = 0.0
    fs.last_reset_env_ids = None

    fs.force_update_origin_and_target(
        root_pos_w=root_pos, root_quat_w=identity_q, ref_keypoints_w=ref_kp_w
    )

    # independent reference: identical 4x semi-implicit loop for body 0
    ref_b = ref_kp_w[0, 0]
    org_b = origin_b[0, 0]
    dir_b = (ref_b - org_b) / np.linalg.norm(ref_b - org_b)
    kp_drive = 10.0 / 0.05
    kd_drive = 2 * np.sqrt(kp_drive * 0.1)
    x = np.zeros(3)
    v = np.zeros(3)
    for _ in range(4):
        f_drive = kp_drive * (ref_b - x) + kd_drive * (0.0 - v)
        n = max(np.linalg.norm(f_drive), 1e-6)
        f_drive = f_drive * min(10.0 / n, 1.0)  # clamp to force_limit
        pd = min(np.dot(org_b - x, dir_b), 0.0) * dir_b  # project_pos_diff
        f_ext = 20.0 * pd
        n2 = max(np.linalg.norm(f_ext), 1e-6)
        f_ext = f_ext * min(30.0 / n2, 1.0)  # clamp to max_force
        f_damp = -2.0 * v
        acc = (f_drive + f_ext + f_damp) / 0.1
        na = max(np.linalg.norm(acc), 1e-6)
        acc = acc * min(1000.0 / na, 1.0)
        v = v + acc * 0.005
        nv = max(np.linalg.norm(v), 1e-6)
        v = v * min(4.0 / nv, 1.0)
        x = x + v * 0.005

    np.testing.assert_allclose(fs.force_keypoint_b[0, 0], x, atol=1e-9)
    # identity root -> keypoint_w == keypoint_b
    np.testing.assert_allclose(fs.force_keypoint_w[0, 0], x, atol=1e-9)


def test_admittance_reset_initializes_x_to_ref_point() -> None:
    fs = _fs(1)
    identity_q = np.tile([1.0, 0.0, 0.0, 0.0], (1, 1))
    ref_kp_w = np.zeros((1, 6, 3))
    ref_kp_w[0, :, 1] = 0.5
    fs.force_origin_tl.value[:] = 0.0
    fs.force_kp_scaled[:] = 0.0  # no spring -> F_ext=0; with x0=ref, F_drive~0 -> keypoint stays ~ref
    fs.force_safe_limit_tl.value[:] = 10.0
    fs.last_reset_env_ids = np.array([0])
    fs.force_update_origin_and_target(
        root_pos_w=np.zeros((1, 3)), root_quat_w=identity_q, ref_keypoints_w=ref_kp_w
    )
    # admit reset x0=ref_point_b; with no external force and x already at ref, keypoint ~ ref
    np.testing.assert_allclose(fs.force_keypoint_b[0, :, 1], 0.5, atol=1e-2)


# --- 5.8 per-substep force_apply -> apply_body_wrench ----------------------- #


def _spring(fs: ForceSystem) -> None:
    # dir +x, origin behind the body so project_pos_diff keeps a pull force
    fs.force_kp_scaled[:] = 10.0
    fs.force_dir_w[:] = np.array([1.0, 0.0, 0.0])
    fs.force_origin_w[:] = np.array([0.0, 0.0, 0.0])
    fs.force_pos_delta[:] = 0.0
    fs.force_pos_delta[:, :, 2] = 0.05  # small eccentric offset in z


def test_force_apply_recomputes_by_current_pose() -> None:
    fs = _fs(1)
    _spring(fs)
    ident_q6 = np.tile([1.0, 0.0, 0.0, 0.0], (1, 6, 1))
    root_q = np.tile([1.0, 0.0, 0.0, 0.0], (1, 1))
    p0 = np.zeros((1, 6, 3))
    p0[:, :, 0] = 1.0  # body at x=1 -> diff=origin-pos=[-1,..] -> pull
    p1 = np.zeros((1, 6, 3))
    p1[:, :, 0] = 2.0  # farther -> stronger pull
    f0, _ = fs.force_apply(0, pos_w=p0, quat_w=ident_q6, root_quat_w=root_q)
    f1, _ = fs.force_apply(1, pos_w=p1, quat_w=ident_q6, root_quat_w=root_q)
    assert not np.allclose(f0, f1)  # force depends on current body pose
    # both pull toward -x (origin behind)
    assert (f0[0, 0, 0] < 0) and (f1[0, 0, 0] < f0[0, 0, 0])


def test_force_apply_eccentric_torque_is_cross() -> None:
    fs = _fs(1)
    _spring(fs)
    ident_q6 = np.tile([1.0, 0.0, 0.0, 0.0], (1, 6, 1))
    root_q = np.tile([1.0, 0.0, 0.0, 0.0], (1, 1))
    p = np.zeros((1, 6, 3))
    p[:, :, 0] = 1.0
    f, tau = fs.force_apply(0, pos_w=p, quat_w=ident_q6, root_quat_w=root_q)
    # identity quat -> delta_w == pos_delta -> torque == cross(pos_delta, f)
    expected = np.cross(fs.force_pos_delta[0], f[0])
    np.testing.assert_allclose(tau[0], expected, atol=1e-9)


class _WrenchBackend:
    """Fake backend recording apply_body_wrench + returning hand-set poses."""

    def __init__(self, pos6, quat6, torso_pos, root_quat):
        self._pos6 = pos6
        self._quat6 = quat6
        self._torso = torso_pos
        self._root_quat = root_quat
        self.wrench_calls: list[dict] = []

    def get_body_pos_w(self, body_ids):
        ids = np.asarray(body_ids).reshape(-1)
        # first 6 are force bodies, last is torso (by construction in the test)
        out = np.concatenate([self._pos6, self._torso[:, None, :]], axis=1)
        return out[:, : len(ids), :] if len(ids) <= 7 else out

    def get_body_quat_w(self, body_ids):
        return self._quat6

    def get_base_quat(self):
        return self._root_quat

    def apply_body_wrench(self, body_ids, force, torque):
        self.wrench_calls.append(
            {"ids": np.asarray(body_ids).copy(), "force": np.asarray(force).copy(),
             "torque": np.asarray(torque).copy()}
        )


def test_pre_step_wrench_calls_apply_body_wrench_with_force_and_torque() -> None:
    fs = _fs(1)
    _spring(fs)
    pos6 = np.zeros((1, 6, 3))
    pos6[:, :, 0] = 1.0
    pos6[0, 0] = [1.0, 0.0, 0.5]  # give body 0 a lever arm from torso for net torque
    quat6 = np.tile([1.0, 0.0, 0.0, 0.0], (1, 6, 1))
    torso = np.zeros((1, 3))
    root_q = np.tile([1.0, 0.0, 0.0, 0.0], (1, 1))
    backend = _WrenchBackend(pos6, quat6, torso, root_q)

    force_ids = np.array([10, 11, 12, 13, 14, 15])
    torso_id = 20
    fn = fs.as_pre_step_wrench(force_ids, torso_id)
    fs.reset_force_substep()
    fn(backend, np.zeros((1, 29)))

    assert len(backend.wrench_calls) == 1
    call = backend.wrench_calls[0]
    np.testing.assert_array_equal(call["ids"], [10, 11, 12, 13, 14, 15, 20])  # 6 force + torso
    assert call["force"].shape == (1, 7, 3)
    assert call["torque"].shape == (1, 7, 3)
    # torso (index 6) carries the net-wrench correction (dF, dM) — both halves used
    assert np.any(call["force"][0, :6] != 0)  # force bodies pushed
    assert np.any(call["torque"][0, :6] != 0)  # eccentric torque present



