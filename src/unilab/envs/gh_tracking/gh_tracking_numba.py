"""Optional numba (prange over env axis) acceleration for GHTrackingEnv.update_state.

Mirrors motion_tracking/g1/motion_tracking_numba.py. Path A: float32 + fastmath;
parity is rtol=1e-4/atol=1e-5, not bit-identical.

Task 1 delegated to numpy to prove the wiring. Task 2 moves the reward aggregation
into a fused ``@njit(parallel=True) for i in prange(n)`` kernel — every reward term
of ``rewards.py`` is ported to an inlined scalar ``*_i`` device fn. Obs + termination
still delegate to numpy (Tasks 3/4). ``_cum_error`` production stays in numpy
(``env._build_reward_context``); only the exp-reward term math + group aggregation
run in the kernel, so the priv_critic/termination consumer is untouched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from numba import njit, prange, set_num_threads  # noqa: F401
    _NUMBA = True
except ImportError:  # pragma: no cover
    njit = prange = set_num_threads = None  # type: ignore[assignment]
    _NUMBA = False


# Reward-term sigma lists (EXACT copy of env._REWARD_SIGMA; kept here so the kernel
# module has no import cycle with env). Asserted equal to env._REWARD_SIGMA at
# accelerator construction so the two never silently drift.
_SIGMA = {
    "root_pos": [0.3],
    "root_rot": [1.0, 0.5],
    "root_vel": [1.0, 0.5],
    "root_ang_vel": [3.0],
    "keypoint": [0.3],
    "lower_keypoint": [0.3],
    "keypoint_imp": [0.3],
    "joint_pos": [0.4, 0.2],
    "joint_vel": [2.0, 1.0],
    "force": [8.0, 4.0],
    "force_target": [0.3],
    "force_vel": [1.0, 0.5],
}
_FORCE_PENALTY_OFFSET = 10.0  # rewards.FORCE_PENALTY_OFFSET


def is_available() -> bool:
    return _NUMBA


def unsupported_terms(groups: dict) -> frozenset[str]:
    # Task 2: every reward term has a kernel translation. Obs/termination still
    # delegate to numpy but that is orthogonal to term support.
    return frozenset()


@dataclass(frozen=True)
class GHNumbaResult:
    reward_vec: np.ndarray          # (N, 3) [impedance, tracking, loco]
    obs: dict[str, np.ndarray]      # policy (N,450) / priv (N,717) / priv_critic (N,3)
    terminated: np.ndarray          # (N,) bool


if _NUMBA:

    def _dev(fn):
        return njit(inline="always", fastmath=True, cache=True, nogil=True)(fn)

    @_dev
    def _exp_sigma(err, sig):
        # calc_exp_sigma: sum_j exp(-err / sig_j) / len(sig). Always positive.
        acc = 0.0
        m = sig.shape[0]
        for j in range(m):
            acc += math.exp(-err / sig[j])
        return acc / m

    # --- tracking device fns ------------------------------------------------ #
    @_dev
    def root_pos_tracking_i(root_pos, reward_root_pos, sig, i):
        dx = reward_root_pos[i, 0] - root_pos[i, 0]
        dy = reward_root_pos[i, 1] - root_pos[i, 1]
        dz = reward_root_pos[i, 2] - root_pos[i, 2]
        return _exp_sigma(math.sqrt(dx * dx + dy * dy + dz * dz), sig)

    @_dev
    def root_rot_tracking_i(root_quat, reward_root_quat, sig, i):
        # error = || axis_angle(quat_mul(tgt, conj(cur))) ||, replicating
        # np_quat_to_axis_angle EXACTLY (incl. non-unit-quat magnitude scaling):
        # canonicalize (w>=0), then norm = |xyz| / sin_half_over_angle.
        cw = root_quat[i, 0]
        cx = -root_quat[i, 1]
        cy = -root_quat[i, 2]
        cz = -root_quat[i, 3]
        tw = reward_root_quat[i, 0]
        tx = reward_root_quat[i, 1]
        ty = reward_root_quat[i, 2]
        tz = reward_root_quat[i, 3]
        w = tw * cw - tx * cx - ty * cy - tz * cz
        x = tw * cx + tx * cw + ty * cz - tz * cy
        y = tw * cy - tx * cz + ty * cw + tz * cx
        z = tw * cz + tx * cy - ty * cx + tz * cw
        norms = math.sqrt(x * x + y * y + z * z)
        half_angle = math.atan2(norms, abs(w))  # canonicalized: w>=0
        angle = 2.0 * half_angle
        if abs(angle) < 1e-6:
            shao = 0.5 - angle * angle / 48.0
        else:
            shao = math.sin(half_angle) / angle
        return _exp_sigma(norms / shao, sig)

    @_dev
    def _mean_kp_dist_i(actual, target, n_kp, i):
        # mean over keypoints of the per-keypoint euclidean distance
        acc = 0.0
        for k in range(n_kp):
            dx = target[i, k, 0] - actual[i, k, 0]
            dy = target[i, k, 1] - actual[i, k, 1]
            dz = target[i, k, 2] - actual[i, k, 2]
            acc += math.sqrt(dx * dx + dy * dy + dz * dz)
        return acc / n_kp

    @_dev
    def lower_keypoint_tracking_i(actual, target, sig, i):
        return _exp_sigma(_mean_kp_dist_i(actual, target, actual.shape[1], i), sig)

    @_dev
    def root_vel_tracking_i(err, sig, i):
        return _exp_sigma(err[i, 0], sig)

    @_dev
    def _mean_abs_diff_i(a, b, n, i):
        acc = 0.0
        for j in range(n):
            acc += abs(b[i, j] - a[i, j])
        return acc / n

    @_dev
    def joint_pos_tracking_i(actual, target, sig, i):
        return _exp_sigma(_mean_abs_diff_i(actual, target, actual.shape[1], i), sig)

    @_dev
    def joint_vel_tracking_i(vel_diff, target_vel, sig, i):
        return _exp_sigma(_mean_abs_diff_i(vel_diff, target_vel, vel_diff.shape[1], i), sig)

    # --- impedance device fns ---------------------------------------------- #
    @_dev
    def force_reward_i(applied, expected, safe_limit, sig, offset, i):
        m = applied.shape[1]
        diff = 0.0
        exceed = False
        limit = safe_limit[i, 0] + offset
        for b in range(m):
            ax = applied[i, b, 0]
            ay = applied[i, b, 1]
            az = applied[i, b, 2]
            ex = ax - expected[i, b, 0]
            ey = ay - expected[i, b, 1]
            ez = az - expected[i, b, 2]
            diff += math.sqrt(ex * ex + ey * ey + ez * ez)
            anorm = math.sqrt(ax * ax + ay * ay + az * az)
            if anorm > limit:
                exceed = True
        reward = _exp_sigma(diff / m, sig)
        if exceed:
            return 0.0
        return reward

    @_dev
    def force_exd_penalty_i(applied, expected, safe_limit, offset, i):
        m = applied.shape[1]
        limit = safe_limit[i, 0] + offset
        half = offset * 0.5
        exd = 0.0
        for b in range(m):
            ax = applied[i, b, 0]
            ay = applied[i, b, 1]
            az = applied[i, b, 2]
            anorm = math.sqrt(ax * ax + ay * ay + az * az)
            ex = expected[i, b, 0]
            ey = expected[i, b, 1]
            ez = expected[i, b, 2]
            enorm = math.sqrt(ex * ex + ey * ey + ez * ez)
            if anorm > limit and anorm > enorm + half:
                exd += 1.0
        return -(exd / m)

    @_dev
    def force_target_tracking_i(body, keypoint, sig, i):
        return _exp_sigma(_mean_kp_dist_i(body, keypoint, body.shape[1], i), sig)

    @_dev
    def force_target_vel_tracking_i(body_vel, keypoint_vel, sig, i):
        return _exp_sigma(_mean_kp_dist_i(body_vel, keypoint_vel, body_vel.shape[1], i), sig)

    @_dev
    def keypoint_tracking_imp_i(actual, target, force_keypoint, kp_force_map, sig, i):
        # keypoint_tracking where force-body slots take the compliant keypoint target
        n_kp = actual.shape[1]
        acc = 0.0
        for k in range(n_kp):
            fk = kp_force_map[k]
            if fk >= 0:
                tx = force_keypoint[i, fk, 0]
                ty = force_keypoint[i, fk, 1]
                tz = force_keypoint[i, fk, 2]
            else:
                tx = target[i, k, 0]
                ty = target[i, k, 1]
                tz = target[i, k, 2]
            dx = tx - actual[i, k, 0]
            dy = ty - actual[i, k, 1]
            dz = tz - actual[i, k, 2]
            acc += math.sqrt(dx * dx + dy * dy + dz * dz)
        return _exp_sigma(acc / n_kp, sig)

    # --- loco device fns ---------------------------------------------------- #
    @_dev
    def action_rate_l2_i(action_buf, i):
        # -(ab[:,:,0] - ab[:,:,1])^2 summed over history slots (QUIRK 1: joint dims 0/1)
        h = action_buf.shape[1]
        acc = 0.0
        for t in range(h):
            d = action_buf[i, t, 0] - action_buf[i, t, 1]
            acc += d * d
        return -acc

    @_dev
    def impact_force_l2_i(net_force_hist, first_contact, mass_total, i):
        denom = mass_total * 9.81
        hlen = net_force_hist.shape[1]
        nbody = net_force_hist.shape[2]
        r = 0.0
        for b in range(nbody):
            fmean = 0.0
            for t in range(hlen):
                fx = net_force_hist[i, t, b, 0]
                fy = net_force_hist[i, t, b, 1]
                fz = net_force_hist[i, t, b, 2]
                fmean += math.sqrt(fx * fx + fy * fy + fz * fz)
            fmean = (fmean / hlen) / denom
            r += fmean * fmean * first_contact[i, b]
        if r > 20.0:
            r = 20.0
        return -r

    @_dev
    def feet_slip_i(in_contact, feet_vel_xy, i):
        nbody = in_contact.shape[1]
        acc = 0.0
        for b in range(nbody):
            vx = feet_vel_xy[i, b, 0]
            vy = feet_vel_xy[i, b, 1]
            acc += in_contact[i, b] * (vx * vx + vy * vy)
        return -acc

    @_dev
    def joint_vel_l2_i(joint_vel_mean, i):
        n = joint_vel_mean.shape[1]
        acc = 0.0
        for j in range(n):
            v = joint_vel_mean[i, j]
            acc += v * v
        return -acc

    @_dev
    def joint_pos_limits_i(joint_pos, soft_lo, soft_hi, soft_factor, i):
        n = joint_pos.shape[1]
        acc = 0.0
        for j in range(n):
            jp = joint_pos[i, j]
            vlo = soft_lo[i, j] - jp
            if vlo < 0.0:
                vlo = 0.0
            vhi = jp - soft_hi[i, j]
            if vhi < 0.0:
                vhi = 0.0
            acc += vlo + vhi
        return -acc / (1.0 - soft_factor)

    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _reward_kernel(
        # tracking inputs
        root_pos, reward_root_pos, root_quat, reward_root_quat,
        lower_actual_kp, lower_target_kp, root_vel_err, root_ang_vel_err,
        track_actual, track_target, track_vel_diff, track_vel_target,
        # impedance inputs
        force_applied, force_expected, force_safe_limit,
        force_body, force_keypoint, force_body_vel, force_keypoint_vel,
        actual_kp, target_kp, kp_force_map,
        # loco inputs
        action_buf, net_force_hist, first_contact, mass_total,
        in_contact, feet_vel_xy, joint_vel_mean,
        joint_pos, soft_lo, soft_hi, soft_factor, feet_air_reward,
        # sigma arrays
        sig_lower_kp, sig_root_pos, sig_root_rot, sig_root_vel, sig_root_ang_vel,
        sig_joint_pos, sig_joint_vel, sig_force, sig_force_target, sig_force_vel,
        sig_keypoint_imp,
        offset, step_dt,
        # output
        reward_vec,
    ):
        n = reward_vec.shape[0]
        for i in prange(n):
            imp = (
                2.0 * force_reward_i(force_applied, force_expected, force_safe_limit, sig_force, offset, i)
                + 6.0 * force_exd_penalty_i(force_applied, force_expected, force_safe_limit, offset, i)
                + 2.0 * force_target_tracking_i(force_body, force_keypoint, sig_force_target, i)
                + 1.0 * force_target_vel_tracking_i(force_body_vel, force_keypoint_vel, sig_force_vel, i)
                + 2.0 * keypoint_tracking_imp_i(actual_kp, target_kp, force_keypoint, kp_force_map, sig_keypoint_imp, i)
            )
            trk = (
                2.0 * lower_keypoint_tracking_i(lower_actual_kp, lower_target_kp, sig_lower_kp, i)
                + 0.5 * root_pos_tracking_i(root_pos, reward_root_pos, sig_root_pos, i)
                + 0.5 * root_rot_tracking_i(root_quat, reward_root_quat, sig_root_rot, i)
                + 1.0 * root_vel_tracking_i(root_vel_err, sig_root_vel, i)
                + 1.0 * root_vel_tracking_i(root_ang_vel_err, sig_root_ang_vel, i)
                + 1.0 * joint_pos_tracking_i(track_actual, track_target, sig_joint_pos, i)
                + 0.5 * joint_vel_tracking_i(track_vel_diff, track_vel_target, sig_joint_vel, i)
            )
            loco = (
                5.0 * 1.0
                + 4.0 * impact_force_l2_i(net_force_hist, first_contact, mass_total, i)
                + 2.0 * feet_slip_i(in_contact, feet_vel_xy, i)
                + 5e-4 * joint_vel_l2_i(joint_vel_mean, i)
                + 0.1 * action_rate_l2_i(action_buf, i)
                + 10.0 * feet_air_reward[i, 0]
                + 1.0 * joint_pos_limits_i(joint_pos, soft_lo, soft_hi, soft_factor, i)
            )
            reward_vec[i, 0] = imp * step_dt
            reward_vec[i, 1] = trk * step_dt
            reward_vec[i, 2] = loco * step_dt

    # ---------------------------------------------------------------------- #
    # Task 3: observation device fns + fused prange obs kernel.               #
    # The stateful telemetry roll + rng noise stays in numpy                  #
    # (obs_manager.update, called once before the kernel); the buffers        #
    # arrive already-rolled/noised and the kernel does only per-env-          #
    # independent assembly: quat-inverse-apply, keypoint diffs, axis-angle,   #
    # gather-and-flatten. Slice layout replicates ObservationManager.compute  #
    # exactly (policy 450 / priv 717 / priv_critic 3).                        #
    # ---------------------------------------------------------------------- #
    @_dev
    def _qapply(w, x, y, z, vx, vy, vz):
        # Rotate vector v by quaternion (w,x,y,z); ports np_quat_apply_batched.
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)
        rx = vx + w * tx + y * tz - z * ty
        ry = vy + w * ty + z * tx - x * tz
        rz = vz + w * tz + x * ty - y * tx
        return rx, ry, rz

    @_dev
    def _qinv_apply(w, x, y, z, vx, vy, vz):
        # Rotate v by conj(quat); ports _qinv_apply (apply conjugate).
        return _qapply(w, -x, -y, -z, vx, vy, vz)

    @_dev
    def _yaw_cw_sz(qw, qx, qy, qz):
        # yaw-only quaternion (w,0,0,z) components; ports np_yaw_quat.
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        half = 0.5 * yaw
        return math.cos(half), math.sin(half)

    @_dev
    def _axis_angle(w, x, y, z):
        # Axis-angle vector of a (possibly non-unit) quaternion; ports
        # np_quat_to_axis_angle EXACTLY (canonicalize w>=0, atan2, Taylor near 0).
        if w < 0.0:
            w = -w
            x = -x
            y = -y
            z = -z
        norms = math.sqrt(x * x + y * y + z * z)
        half = math.atan2(norms, w)
        angle = 2.0 * half
        if abs(angle) < 1e-6:
            shao = 0.5 - angle * angle / 48.0
        else:
            shao = math.sin(half) / angle
        return x / shao, y / shao, z / shao

    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _obs_kernel(
        # --- policy inputs ---
        boot_indicator, boot_max,
        motion_root_pos_w, motion_root_quat_w, force_safe_limit, motion_joint_pos,
        pol_angvel_buf, pol_grav_buf, pol_joint_buf, pol_joint_steps,
        offset, action_buf,
        # --- priv inputs ---
        root_pos_w, root_quat_w, env_origins, motion_root_lin_vel_w,
        force_keypoint_b, force_applied_b, force_expected_b, force_sample_timer,
        body_pos_w_height, contact_hist, contact_denom, root_ema_linvel,
        priv_angvel_buf, priv_grav_buf, priv_joint_buf, priv_steps,
        body_pos_w_kp, body_lin_vel_w_kp, motion_body_pos_w_kp,
        joint_pos_target, applied_torque,
        # --- priv_critic input ---
        cum_error,
        # --- outputs ---
        policy, priv, priv_critic,
    ):
        n = policy.shape[0]
        S = motion_root_pos_w.shape[1]
        NJ = motion_joint_pos.shape[2]
        n_pol_steps = pol_joint_steps.shape[0]
        n_act = action_buf.shape[1]
        na = action_buf.shape[2]
        nf = force_keypoint_b.shape[1]
        nh = body_pos_w_height.shape[1]
        hlen = contact_hist.shape[1]
        nfe = contact_hist.shape[2]
        nps = priv_steps.shape[0]
        nkp = body_pos_w_kp.shape[1]
        for i in prange(n):
            # ===================== POLICY (450) ===================== #
            # [0:1] boot_indicator_state_obs
            policy[i, 0] = boot_indicator[i, 0] / boot_max
            # [1:23] command_obs (heights 5, pos_diff_b xy 8, heading_b xy 8, limit 1)
            cw0, sz0 = _yaw_cw_sz(
                motion_root_quat_w[i, 0, 0], motion_root_quat_w[i, 0, 1],
                motion_root_quat_w[i, 0, 2], motion_root_quat_w[i, 0, 3],
            )
            for s in range(S):
                policy[i, 1 + s] = motion_root_pos_w[i, s, 2]
            c_pos = 1 + S
            for s in range(1, S):
                dx = motion_root_pos_w[i, s, 0] - motion_root_pos_w[i, 0, 0]
                dy = motion_root_pos_w[i, s, 1] - motion_root_pos_w[i, 0, 1]
                dz = motion_root_pos_w[i, s, 2] - motion_root_pos_w[i, 0, 2]
                px, py, pz = _qinv_apply(cw0, 0.0, 0.0, sz0, dx, dy, dz)
                policy[i, c_pos + (s - 1) * 2 + 0] = px
                policy[i, c_pos + (s - 1) * 2 + 1] = py
            c_head = c_pos + (S - 1) * 2
            for s in range(1, S):
                cwf, szf = _yaw_cw_sz(
                    motion_root_quat_w[i, s, 0], motion_root_quat_w[i, s, 1],
                    motion_root_quat_w[i, s, 2], motion_root_quat_w[i, s, 3],
                )
                hx, hy, hz = _qapply(cwf, 0.0, 0.0, szf, 1.0, 0.0, 0.0)
                bx, by, bz = _qinv_apply(cw0, 0.0, 0.0, sz0, hx, hy, hz)
                policy[i, c_head + (s - 1) * 2 + 0] = bx
                policy[i, c_head + (s - 1) * 2 + 1] = by
            c_lim = c_head + (S - 1) * 2
            policy[i, c_lim] = force_safe_limit[i, 0]
            b = c_lim + 1  # = 23
            # [23:168] target_joint_pos_obs
            for s in range(S):
                for j in range(NJ):
                    policy[i, b + s * NJ + j] = motion_joint_pos[i, s, j]
            b += S * NJ
            # [168:183] target_projected_gravity_b
            for s in range(S):
                gx, gy, gz = _qinv_apply(
                    motion_root_quat_w[i, s, 0], motion_root_quat_w[i, s, 1],
                    motion_root_quat_w[i, s, 2], motion_root_quat_w[i, s, 3],
                    0.0, 0.0, -1.0,
                )
                policy[i, b + s * 3 + 0] = gx
                policy[i, b + s * 3 + 1] = gy
                policy[i, b + s * 3 + 2] = gz
            b += S * 3
            # [183:186] pol_angvel (history step 0)
            policy[i, b + 0] = pol_angvel_buf[i, 0, 0]
            policy[i, b + 1] = pol_angvel_buf[i, 0, 1]
            policy[i, b + 2] = pol_angvel_buf[i, 0, 2]
            b += 3
            # [186:189] pol_grav (history step 0)
            policy[i, b + 0] = pol_grav_buf[i, 0, 0]
            policy[i, b + 1] = pol_grav_buf[i, 0, 1]
            policy[i, b + 2] = pol_grav_buf[i, 0, 2]
            b += 3
            # [189:363] pol_joint (buffer - actuator offset, gathered history steps)
            for hi in range(n_pol_steps):
                h = pol_joint_steps[hi]
                for j in range(NJ):
                    policy[i, b + hi * NJ + j] = pol_joint_buf[i, h, j] - offset[i, j]
            b += n_pol_steps * NJ
            # [363:450] prev_actions_obs
            for t in range(n_act):
                for j in range(na):
                    policy[i, b + t * na + j] = action_buf[i, t, j]

            # ===================== PRIV (717) ====================== #
            qw = root_quat_w[i, 0]
            qx = root_quat_w[i, 1]
            qy = root_quat_w[i, 2]
            qz = root_quat_w[i, 3]
            rpx = root_pos_w[i, 0]
            rpy = root_pos_w[i, 1]
            rpz = root_pos_w[i, 2]
            ox = env_origins[i, 0]
            oy = env_origins[i, 1]
            oz = env_origins[i, 2]
            # [0:15] target_pos_b_obs
            cpx = rpx - ox
            cpy = rpy - oy
            cpz = rpz - oz
            for s in range(S):
                vx = motion_root_pos_w[i, s, 0] - cpx
                vy = motion_root_pos_w[i, s, 1] - cpy
                vz = motion_root_pos_w[i, s, 2] - cpz
                tx, ty, tz = _qinv_apply(qw, qx, qy, qz, vx, vy, vz)
                priv[i, s * 3 + 0] = tx
                priv[i, s * 3 + 1] = ty
                priv[i, s * 3 + 2] = tz
            pb = S * 3
            # [15:30] target_linvel_b_obs
            for s in range(S):
                tx, ty, tz = _qinv_apply(
                    qw, qx, qy, qz,
                    motion_root_lin_vel_w[i, s, 0], motion_root_lin_vel_w[i, s, 1],
                    motion_root_lin_vel_w[i, s, 2],
                )
                priv[i, pb + s * 3 + 0] = tx
                priv[i, pb + s * 3 + 1] = ty
                priv[i, pb + s * 3 + 2] = tz
            pb += S * 3
            # [30:45] relative_quat_obs (axis-angle of mquat * conj(root_quat))
            for s in range(S):
                mw = motion_root_quat_w[i, s, 0]
                mx = motion_root_quat_w[i, s, 1]
                my = motion_root_quat_w[i, s, 2]
                mz = motion_root_quat_w[i, s, 3]
                c_w = qw
                c_x = -qx
                c_y = -qy
                c_z = -qz
                rw = mw * c_w - mx * c_x - my * c_y - mz * c_z
                rx = mw * c_x + mx * c_w + my * c_z - mz * c_y
                ry = mw * c_y - mx * c_z + my * c_w + mz * c_x
                rz = mw * c_z + mx * c_y - my * c_x + mz * c_w
                ax, ay, az = _axis_angle(rw, rx, ry, rz)
                priv[i, pb + s * 3 + 0] = ax
                priv[i, pb + s * 3 + 1] = ay
                priv[i, pb + s * 3 + 2] = az
            pb += S * 3
            # [45:100] force_priv_obs (keypoint 18, applied 18, expected 18, timer 1)
            for k in range(nf):
                for d in range(3):
                    priv[i, pb + k * 3 + d] = force_keypoint_b[i, k, d]
            pb += nf * 3
            for k in range(nf):
                for d in range(3):
                    priv[i, pb + k * 3 + d] = force_applied_b[i, k, d]
            pb += nf * 3
            for k in range(nf):
                for d in range(3):
                    priv[i, pb + k * 3 + d] = force_expected_b[i, k, d]
            pb += nf * 3
            priv[i, pb] = force_sample_timer[i, 0]
            pb += 1
            # [100:104] body_height (z of selected bodies)
            for k in range(nh):
                priv[i, pb + k] = body_pos_w_height[i, k, 2]
            pb += nh
            # [104:110] contact force history mean / denom, clipped [-10, 10]
            for bd in range(nfe):
                for d in range(3):
                    acc = 0.0
                    for t in range(hlen):
                        acc += contact_hist[i, t, bd, d]
                    v = (acc / hlen) / contact_denom
                    if v > 10.0:
                        v = 10.0
                    elif v < -10.0:
                        v = -10.0
                    priv[i, pb + bd * 3 + d] = v
            pb += nfe * 3
            # [110:113] root_ema (world EMA rotated into root frame)
            tx, ty, tz = _qinv_apply(
                qw, qx, qy, qz,
                root_ema_linvel[i, 0], root_ema_linvel[i, 1], root_ema_linvel[i, 2],
            )
            priv[i, pb + 0] = tx
            priv[i, pb + 1] = ty
            priv[i, pb + 2] = tz
            pb += 3
            # [113:140] priv_angvel (gathered history steps)
            for hi in range(nps):
                h = priv_steps[hi]
                for d in range(3):
                    priv[i, pb + hi * 3 + d] = priv_angvel_buf[i, h, d]
            pb += nps * 3
            # [140:167] priv_grav
            for hi in range(nps):
                h = priv_steps[hi]
                for d in range(3):
                    priv[i, pb + hi * 3 + d] = priv_grav_buf[i, h, d]
            pb += nps * 3
            # [167:428] priv_joint (buffer - actuator offset)
            for hi in range(nps):
                h = priv_steps[hi]
                for j in range(NJ):
                    priv[i, pb + hi * NJ + j] = priv_joint_buf[i, h, j] - offset[i, j]
            pb += nps * NJ
            # [428:461] current_keypoint_b
            for k in range(nkp):
                vx = body_pos_w_kp[i, k, 0] - rpx
                vy = body_pos_w_kp[i, k, 1] - rpy
                vz = body_pos_w_kp[i, k, 2] - rpz
                tx, ty, tz = _qinv_apply(qw, qx, qy, qz, vx, vy, vz)
                priv[i, pb + k * 3 + 0] = tx
                priv[i, pb + k * 3 + 1] = ty
                priv[i, pb + k * 3 + 2] = tz
            pb += nkp * 3
            # [461:494] current_keypoint_vel_b
            for k in range(nkp):
                tx, ty, tz = _qinv_apply(
                    qw, qx, qy, qz,
                    body_lin_vel_w_kp[i, k, 0], body_lin_vel_w_kp[i, k, 1],
                    body_lin_vel_w_kp[i, k, 2],
                )
                priv[i, pb + k * 3 + 0] = tx
                priv[i, pb + k * 3 + 1] = ty
                priv[i, pb + k * 3 + 2] = tz
            pb += nkp * 3
            # [494:659] target_keypoints_diff_b_obs
            for s in range(S):
                for k in range(nkp):
                    aw_x = body_pos_w_kp[i, k, 0] - ox
                    aw_y = body_pos_w_kp[i, k, 1] - oy
                    aw_z = body_pos_w_kp[i, k, 2] - oz
                    vx = motion_body_pos_w_kp[i, s, k, 0] - aw_x
                    vy = motion_body_pos_w_kp[i, s, k, 1] - aw_y
                    vz = motion_body_pos_w_kp[i, s, k, 2] - aw_z
                    tx, ty, tz = _qinv_apply(qw, qx, qy, qz, vx, vy, vz)
                    priv[i, pb + s * nkp * 3 + k * 3 + 0] = tx
                    priv[i, pb + s * nkp * 3 + k * 3 + 1] = ty
                    priv[i, pb + s * nkp * 3 + k * 3 + 2] = tz
            pb += S * nkp * 3
            # [659:688] applied_action_obs
            for j in range(NJ):
                priv[i, pb + j] = joint_pos_target[i, j]
            pb += NJ
            # [688:717] applied_torque (priv noise = 0)
            for j in range(NJ):
                priv[i, pb + j] = applied_torque[i, j]

            # ================== PRIV_CRITIC (3) ==================== #
            priv_critic[i, 0] = cum_error[i, 0]
            priv_critic[i, 1] = cum_error[i, 1]
            priv_critic[i, 2] = cum_error[i, 2]


class GHTrackingNumbaAccelerator:
    def __init__(self, num_threads: int | None) -> None:
        self.num_threads = num_threads
        # Task 2: proof-of-execution flag, not a capability flag. Only set to
        # True inside _compute_reward_vec, right where the kernel is actually
        # invoked, so it proves the kernel ran rather than just that numba is
        # importable.
        self._reward_from_kernel = False
        # Task 3: obs proof-of-execution flag (mirror _reward_from_kernel); only
        # set True inside _compute_obs_dict where the obs kernel actually runs.
        self._obs_from_kernel = False
        # float32 sigma constants (built once; reused every step)
        self._sig = {k: np.asarray(v, dtype=np.float32) for k, v in _SIGMA.items()}
        self._kp_force_map: np.ndarray | None = None
        # obs history-step index arrays (built once from obs_manager config)
        self._pol_joint_steps: np.ndarray | None = None
        self._priv_steps: np.ndarray | None = None

    @classmethod
    def from_env(cls, env, num_threads: int | None) -> "GHTrackingNumbaAccelerator":
        if not _NUMBA:
            raise RuntimeError(
                "numba_acceleration=True but numba is not importable; "
                "install numba or set numba_acceleration=False"
            )
        # guard against sigma drift between env and this module
        from unilab.envs.gh_tracking.env import _REWARD_SIGMA
        for name, vals in _REWARD_SIGMA.items():
            if [float(x) for x in _SIGMA[name]] != [float(x) for x in vals]:
                raise RuntimeError(
                    f"reward sigma drift for '{name}': env={vals} kernel={_SIGMA[name]}"
                )
        return cls(num_threads=num_threads)

    @staticmethod
    def _f32(x: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(x, dtype=np.float32)

    def _compute_reward_vec(self, env) -> np.ndarray:
        """Build float32 inputs from env._rc and run the fused prange kernel.

        env._build_reward_context() produces _cum_error (numpy, untouched) + fills
        env._rc; we only replace the numpy group aggregation with the kernel.
        """
        env._build_reward_context()
        rc = env._rc
        f32 = self._f32

        # keypoint->force-slot map (target_kp slot -> index into force_keypoint_w, else -1)
        force_in_kp_idx = np.asarray(rc["force_in_kp_idx"], dtype=np.int64)
        n_kp = int(rc["actual_kp_w"].shape[1])
        if self._kp_force_map is None or self._kp_force_map.shape[0] != n_kp:
            kp_map = np.full((n_kp,), -1, dtype=np.int64)
            kp_map[force_in_kp_idx] = np.arange(force_in_kp_idx.shape[0], dtype=np.int64)
            self._kp_force_map = kp_map

        n = int(rc["n"])
        reward_vec = np.empty((n, 3), dtype=np.float32)
        sig = self._sig

        _reward_kernel(
            f32(rc["root_pos_w"]), f32(rc["reward_root_pos_w"]),
            f32(rc["root_quat_w"]), f32(rc["reward_root_quat_w"]),
            f32(rc["lower_actual_kp_w"]), f32(rc["lower_target_kp_w"]),
            f32(rc["root_vel_err"]), f32(rc["root_ang_vel_err"]),
            f32(rc["track_joint_actual"]), f32(rc["track_joint_target"]),
            f32(rc["track_joint_vel_diff"]), f32(rc["track_joint_vel_target"]),
            f32(rc["force_applied_w"]), f32(rc["force_expected_w"]), f32(rc["force_safe_limit"]),
            f32(rc["force_body_w"]), f32(rc["force_keypoint_w"]),
            f32(rc["force_body_vel_w"]), f32(rc["force_keypoint_vel_w"]),
            f32(rc["actual_kp_w"]), f32(rc["target_kp_w"]), self._kp_force_map,
            f32(rc["action_buf"]), f32(rc["net_force_hist"]), f32(rc["first_contact"]),
            np.float32(rc["mass_total"]),
            f32(rc["in_contact"]), f32(rc["feet_vel_xy"]), f32(rc["joint_vel_mean"]),
            f32(rc["joint_pos"]), f32(rc["soft_lo"]), f32(rc["soft_hi"]),
            np.float32(rc["soft_factor"]), f32(rc["feet_air_time_reward"]),
            sig["lower_keypoint"], sig["root_pos"], sig["root_rot"], sig["root_vel"],
            sig["root_ang_vel"], sig["joint_pos"], sig["joint_vel"], sig["force"],
            sig["force_target"], sig["force_vel"], sig["keypoint_imp"],
            np.float32(_FORCE_PENALTY_OFFSET), np.float32(env._cfg.ctrl_dt),
            reward_vec,
        )
        self._reward_from_kernel = True
        return reward_vec

    def _compute_obs_dict(self, env) -> dict[str, np.ndarray]:
        """Build float32 inputs and run the fused prange obs kernel.

        The stateful telemetry roll + rng noise (obs_manager.update) runs in
        numpy exactly once here — same call-site as the numpy _compute_obs path —
        so the injected noise stream matches. The kernel only does per-env
        stateless assembly from the already-rolled buffers + ObsState fields.
        """
        state = env._build_obs_state()
        env.obs_manager.update(state)  # numpy roll + rng noise, unchanged, once
        om = env.obs_manager
        f32 = self._f32

        if self._pol_joint_steps is None:
            self._pol_joint_steps = np.asarray(om.pol_joint.history_steps, dtype=np.int64)
        if self._priv_steps is None:
            # priv_angvel / priv_grav / priv_joint share history_steps == range(9)
            self._priv_steps = np.asarray(om.priv_joint.history_steps, dtype=np.int64)

        n = int(state.root_pos_w.shape[0])
        policy = np.empty((n, 450), dtype=np.float32)
        priv = np.empty((n, 717), dtype=np.float32)
        priv_critic = np.empty((n, 3), dtype=np.float32)

        _obs_kernel(
            f32(state.boot_indicator), np.float32(om.boot_max),
            f32(state.motion_root_pos_w), f32(state.motion_root_quat_w),
            f32(state.force_safe_limit), f32(state.motion_joint_pos),
            f32(om.pol_angvel.buffer), f32(om.pol_grav.buffer),
            f32(om.pol_joint.buffer), self._pol_joint_steps,
            f32(om.pol_joint.offset), f32(state.action_buf),
            f32(state.root_pos_w), f32(state.root_quat_w), f32(state.env_origins),
            f32(state.motion_root_lin_vel_w),
            f32(state.force_keypoint_b), f32(state.force_applied_b),
            f32(state.force_expected_b), f32(state.force_sample_timer),
            f32(state.body_pos_w_height),
            f32(om.contact.history), np.float32(om.contact.denom),
            f32(om.root_ema.linvel_w),
            f32(om.priv_angvel.buffer), f32(om.priv_grav.buffer),
            f32(om.priv_joint.buffer), self._priv_steps,
            f32(state.body_pos_w_kp), f32(state.body_lin_vel_w_kp),
            f32(state.motion_body_pos_w_kp),
            f32(state.joint_pos_target), f32(state.applied_torque),
            f32(state.cum_error),
            policy, priv, priv_critic,
        )
        self._obs_from_kernel = True
        return {"policy": policy, "priv": priv, "priv_critic": priv_critic}

    def compute_update_state(self, env) -> GHNumbaResult:
        # Task 2: reward via fused kernel. Task 3: obs via fused kernel.
        # Termination still delegates to numpy.
        if self.num_threads is not None and _NUMBA:
            set_num_threads(self.num_threads)
        reward_vec = self._compute_reward_vec(env)   # (N,3) fp32; writes _cum_error via numpy
        obs = self._compute_obs_dict(env)             # dict of 3 groups (fused kernel)
        from unilab.envs.gh_tracking.terminations import apply_terminate_gate
        terminated = apply_terminate_gate(
            env.termination.terminated(), env._episode_length)[:, 0]
        return GHNumbaResult(reward_vec=reward_vec, obs=obs, terminated=terminated)
