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


class GHTrackingNumbaAccelerator:
    def __init__(self, num_threads: int | None) -> None:
        self.num_threads = num_threads
        # Task 2: proof-of-execution flag, not a capability flag. Only set to
        # True inside _compute_reward_vec, right where the kernel is actually
        # invoked, so it proves the kernel ran rather than just that numba is
        # importable.
        self._reward_from_kernel = False
        # float32 sigma constants (built once; reused every step)
        self._sig = {k: np.asarray(v, dtype=np.float32) for k, v in _SIGMA.items()}
        self._kp_force_map: np.ndarray | None = None

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

    def compute_update_state(self, env) -> GHNumbaResult:
        # Task 2: reward via fused kernel; obs/termination still delegate to numpy.
        if self.num_threads is not None and _NUMBA:
            set_num_threads(self.num_threads)
        reward_vec = self._compute_reward_vec(env)   # (N,3) fp32; writes _cum_error via numpy
        obs = env._compute_obs()                      # dict of 3 groups
        from unilab.envs.gh_tracking.terminations import apply_terminate_gate
        terminated = apply_terminate_gate(
            env.termination.terminated(), env._episode_length)[:, 0]
        return GHNumbaResult(reward_vec=reward_vec, obs=obs, terminated=terminated)
