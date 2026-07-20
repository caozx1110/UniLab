"""GH rewards for the MuJoCo migration (Phase 7).

Three reward groups (impedance / tracking / loco), each aggregated as
``sum(weight * term) * current_factor * step_dt`` into a per-group scalar; the
three are concatenated into a 3-vector (GAE sums it). Pure-numpy ports of GH
``rewards/locomotion.py`` + the reward terms in ``motion_tracking.py``.

Reward clipping is intentionally inert (GH reads ``_clip_`` but never uses it).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def calc_exp_sigma(error: np.ndarray, sigma_list: Sequence[float]) -> np.ndarray:
    """``sum_i exp(-error / sigma_i) / len`` (GH ``_calc_exp_sigma``), always positive."""
    rewards = [np.exp(-np.asarray(error, dtype=np.float64) / s) for s in sigma_list]
    return sum(rewards) / len(sigma_list)


class RewardManager:
    """Aggregates the three reward groups into a per-group 3-vector.

    ``groups`` maps group name -> list of (term_name, weight); ``term_fns`` maps
    term name -> callable(state) -> (N, 1). Group reward = sum(weight * term) *
    current_factor, then each group is scaled by step_dt (GH ``_compute_reward``
    with ``mult_dt`` default True). No reward clipping.
    """

    def __init__(
        self,
        groups: dict[str, list[tuple[str, float]]],
        term_fns: dict[str, Callable],
        step_dt: float,
        student_train: bool = False,
    ) -> None:
        self.groups = groups
        self.term_fns = term_fns
        self.step_dt = float(step_dt)
        self.student_train = student_train
        self.current_factor = 1.0  # no reward-group scale configured for this task

    def step_schedule(self, progress: float) -> None:
        """Advance the reward curriculum. Student training forces progress to 1.0
        (GH RewardGroup.step_schedule); with no scale, current_factor stays 1."""
        if self.student_train:
            progress = 1.0
        # no `scale` -> current_factor remains 1.0 (only progress plumbing matters here)

    def compute(self, state) -> np.ndarray:
        cols = []
        for _gname, terms in self.groups.items():
            if terms:
                group = sum(w * self.term_fns[name](state) for name, w in terms)
            else:
                # infer batch size from any term output shape; fall back to state["n"]
                n = state["n"] if isinstance(state, dict) and "n" in state else self._infer_n(state)
                group = np.zeros((n, 1), dtype=np.float64)
            cols.append(np.asarray(group, dtype=np.float64) * self.current_factor * self.step_dt)
        return np.concatenate(cols, axis=-1)

    @staticmethod
    def _infer_n(state) -> int:
        if isinstance(state, dict) and "n" in state:
            return int(state["n"])
        raise ValueError("cannot infer batch size for an empty reward group")


# --- loco rewards (rewards/locomotion.py) ---------------------------------- #


def survival(num_envs: int) -> np.ndarray:
    """Constant survival reward of 1 (GH survival)."""
    return np.ones((int(num_envs), 1), dtype=np.float64)


def action_rate_l2(action_buf: np.ndarray) -> np.ndarray:
    """QUIRK 1 (source-compatible): ``-(action_buf[:,:,0] - action_buf[:,:,1])^2``
    summed over the history slots (GH ``action_rate_l2`` rewards/locomotion.py:283).

    ``action_buf`` is (N, hist, 29); this compares JOINT dims 0 and 1 across all
    history slots (NOT an adjacent-timestep difference) and sums over the slots.
    Reproduced verbatim.
    """
    ab = np.asarray(action_buf, dtype=np.float64)
    diff = ab[:, :, 0] - ab[:, :, 1]
    return -(diff**2).sum(axis=-1, keepdims=True)


def impact_force_l2(
    net_force_history: np.ndarray, first_contact: np.ndarray, mass_total: float
) -> np.ndarray:
    """``-(f^2 * first_contact).sum`` clamped to a magnitude of 20 (GH impact_force_l2).

    ``net_force_history`` (N, H, nbody, 3); f = mean over history of the force norm,
    divided by ``mass_total * 9.81``.
    """
    denom = float(mass_total) * 9.81
    force = np.linalg.norm(np.asarray(net_force_history, dtype=np.float64), axis=-1).mean(axis=1)
    force = force / denom  # (N, nbody)
    r = (force**2 * np.asarray(first_contact, dtype=np.float64)).sum(axis=1, keepdims=True)
    return -np.minimum(r, 20.0)


def feet_slip(in_contact: np.ndarray, feet_vel_xy: np.ndarray) -> np.ndarray:
    """``-sum(in_contact * |vel_xy|^2)`` over feet (GH feet_slip)."""
    speed2 = np.linalg.norm(np.asarray(feet_vel_xy, dtype=np.float64), axis=-1) ** 2
    return -(np.asarray(in_contact, dtype=np.float64) * speed2).sum(axis=1, keepdims=True)


class JointVel2Slot:
    """2-slot per-substep joint-velocity sub-sampler for joint_vel_l2 (GH joint_vel_l2)."""

    def __init__(self, num_envs: int, num_joints: int) -> None:
        self.joint_vel = np.zeros((int(num_envs), 2, int(num_joints)), dtype=np.float64)

    def post_step(self, substep: int, joint_vel: np.ndarray) -> None:
        self.joint_vel[:, substep % 2] = np.asarray(joint_vel, dtype=np.float64)

    def mean(self) -> np.ndarray:
        return self.joint_vel.mean(axis=1)


def joint_vel_l2(joint_vel_mean: np.ndarray) -> np.ndarray:
    """``-sum(joint_vel^2)`` over the 2-slot mean velocity (GH joint_vel_l2)."""
    jv = np.asarray(joint_vel_mean, dtype=np.float64)
    return -(jv**2).sum(axis=1, keepdims=True)


class FeetAirTimeRef:
    """Height-weighted air-time reward, penalized at first contact when below ``thres``
    (GH ``feet_air_time_ref`` rewards/locomotion.py:163-213). Stateful across control
    steps: accumulates ``reward_time`` per foot (grows by ``step_dt * height_coef`` while
    the robot/reference contact agree, decays by ``step_dt`` where they disagree), emits
    ``sum((reward_time - thres).clamp_max(0) * first_contact)`` at each landing, then zeros
    ``reward_time`` on contact. GH default ``skip_ref=False`` -> the feet_standing branch.
    """

    H_LOW, H_HIGH = 0.035, 0.12  # foot-height coefficient bounds (locomotion.py:183)
    C_LOW, C_HIGH = 0.5, 2.0

    def __init__(self, num_envs: int, num_feet: int, thres: float, step_dt: float) -> None:
        self.thres = float(thres)
        self.step_dt = float(step_dt)
        self.reward_time = np.zeros((int(num_envs), int(num_feet)), dtype=np.float64)
        self.last_contact = np.zeros((int(num_envs), int(num_feet)), dtype=bool)
        self._exp_log_c_ratio = float(np.log(self.C_HIGH / self.C_LOW))

    def reset(self, env_ids: np.ndarray) -> None:
        env_ids = np.asarray(env_ids)
        self.reward_time[env_ids] = 0.0
        self.last_contact[env_ids] = False

    def step(
        self, current_contact: np.ndarray, feet_height: np.ndarray, feet_standing: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance one control step. Returns (reward (N,1), first_contact (N,num_feet) bool)."""
        current_contact = np.asarray(current_contact, dtype=bool)
        first_contact = (~self.last_contact) & current_contact
        self.last_contact[:] = current_contact

        t = np.clip(
            (np.asarray(feet_height, dtype=np.float64) - self.H_LOW) / (self.H_HIGH - self.H_LOW),
            0.0, 1.0,
        )
        feet_height_coef = self.C_LOW * np.exp(self._exp_log_c_ratio * t)
        contact_diff = np.asarray(feet_standing, dtype=bool) ^ current_contact
        self.reward_time = self.reward_time + np.where(
            contact_diff, -self.step_dt, self.step_dt * feet_height_coef
        )
        reward = (
            np.minimum(self.reward_time - self.thres, 0.0) * first_contact
        ).sum(axis=1, keepdims=True)
        self.reward_time = self.reward_time * (~current_contact)
        return reward, first_contact


def joint_pos_limits(
    joint_pos: np.ndarray, soft_lo: np.ndarray, soft_hi: np.ndarray, soft_factor: float
) -> np.ndarray:
    """``-(violation_min + violation_max).sum / (1 - soft_factor)`` (GH joint_pos_limits)."""
    jp = np.asarray(joint_pos, dtype=np.float64)
    v_min = np.maximum(np.asarray(soft_lo, dtype=np.float64) - jp, 0.0)
    v_max = np.maximum(jp - np.asarray(soft_hi, dtype=np.float64), 0.0)
    return -(v_min + v_max).sum(axis=1, keepdims=True) / (1 - soft_factor)


# --- tracking rewards + instantaneous _cum_error (motion_tracking.py) ------- #

CUM_ROOT_POS_SCALE = 0.3
CUM_ORIENTATION_SCALE = 0.7
CUM_KEYPOINT_SCALE = 0.25

import re  # noqa: E402


def resolve_tracking_joints(joint_names: list[str], patterns: list[str]) -> np.ndarray:
    """Resolve the joint-tracking subset (GH joint_patterns override = 17 joints:
    waist3 + hip6 + knee2 + wrist6; shoulder/elbow/ankle excluded). Prefix-matches
    each pattern (re.match), preserving joint order."""
    ids = [
        j for j, name in enumerate(joint_names)
        if any(re.match(p, name) for p in patterns)
    ]
    return np.asarray(ids, dtype=np.int64)


def root_pos_tracking(
    root_pos_w: np.ndarray, reward_root_pos_w: np.ndarray, sigma
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (reward, cum_error_pos = error / 0.3) (GH root_pos_tracking)."""
    error = np.linalg.norm(
        np.asarray(reward_root_pos_w, dtype=np.float64) - np.asarray(root_pos_w, dtype=np.float64),
        axis=-1, keepdims=True,
    )
    return calc_exp_sigma(error, sigma), error / CUM_ROOT_POS_SCALE


def root_rot_tracking(
    root_quat_w: np.ndarray, reward_root_quat_w: np.ndarray, sigma
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (reward, cum_error_rot = error / 0.7) (GH root_rot_tracking)."""
    from unilab.utils.rotation import (
        np_quat_conjugate,
        np_quat_mul,
        np_quat_to_axis_angle,
    )

    tgt = np.asarray(reward_root_quat_w, dtype=np.float64)
    cur = np.asarray(root_quat_w, dtype=np.float64)
    diff = np_quat_to_axis_angle(np_quat_mul(tgt, np_quat_conjugate(cur)))
    error = np.linalg.norm(diff, axis=-1, keepdims=True)
    return calc_exp_sigma(error, sigma), error / CUM_ORIENTATION_SCALE


def root_vel_tracking(diff_norm_error: np.ndarray, sigma) -> np.ndarray:
    """Body-frame linear-velocity tracking (caller supplies the error norm)."""
    return calc_exp_sigma(diff_norm_error, sigma)


def keypoint_tracking(
    actual_kp_w: np.ndarray, target_kp_w: np.ndarray, sigma
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (reward, cum_error_kp = error / 0.25); error = mean keypoint distance
    (GH keypoint_tracking / keypoint_tracking_imp)."""
    diff = np.asarray(target_kp_w, dtype=np.float64) - np.asarray(actual_kp_w, dtype=np.float64)
    error = np.linalg.norm(diff, axis=-1).mean(axis=-1, keepdims=True)
    return calc_exp_sigma(error, sigma), error / CUM_KEYPOINT_SCALE


def lower_keypoint_tracking(actual_kp_w: np.ndarray, target_kp_w: np.ndarray, sigma) -> np.ndarray:
    """Lower-body keypoint tracking, no cum_error (GH lower_keypoint_tracking)."""
    diff = np.asarray(target_kp_w, dtype=np.float64) - np.asarray(actual_kp_w, dtype=np.float64)
    error = np.linalg.norm(diff, axis=-1).mean(axis=-1, keepdims=True)
    return calc_exp_sigma(error, sigma)


def joint_pos_tracking(actual: np.ndarray, target: np.ndarray, sigma) -> np.ndarray:
    """``exp_sigma(mean|target - actual|)`` over the 17-joint subset (GH joint_pos_tracking)."""
    error = np.abs(np.asarray(target, dtype=np.float64) - np.asarray(actual, dtype=np.float64)).mean(
        axis=-1, keepdims=True
    )
    return calc_exp_sigma(error, sigma)


def joint_vel_tracking(vel_diff: np.ndarray, target_vel: np.ndarray, sigma) -> np.ndarray:
    """``exp_sigma(mean|target_vel - vel_diff|)`` (GH joint_vel_tracking); vel_diff is
    the adjacent control-step position difference over the 17-joint subset."""
    error = np.abs(np.asarray(target_vel, dtype=np.float64) - np.asarray(vel_diff, dtype=np.float64)).mean(
        axis=-1, keepdims=True
    )
    return calc_exp_sigma(error, sigma)


# --- impedance rewards (motion_tracking.py :1119-1150) ---------------------- #

FORCE_PENALTY_OFFSET = 10.0


def force_reward(
    force_applied_w: np.ndarray, force_expected_w: np.ndarray, force_safe_limit: np.ndarray, sigma,
    penalty_offset: float = FORCE_PENALTY_OFFSET,
) -> np.ndarray:
    """``exp_sigma(mean|applied - expected|) * ~exceed`` (GH force_reward). The reward
    is masked to 0 if ANY force body exceeds ``safe_limit + offset``."""
    applied = np.asarray(force_applied_w, dtype=np.float64)
    expected = np.asarray(force_expected_w, dtype=np.float64)
    force_norm = np.linalg.norm(applied, axis=-1)  # (N, M)
    diff = np.linalg.norm(applied - expected, axis=-1).mean(axis=-1, keepdims=True)  # (N,1)
    reward = calc_exp_sigma(diff, sigma)
    limit = np.asarray(force_safe_limit, dtype=np.float64) + penalty_offset  # (N,1)
    exceed = (force_norm > limit).any(axis=-1, keepdims=True)
    return reward * (~exceed)


def force_exd_penalty(
    force_applied_w: np.ndarray, force_expected_w: np.ndarray, force_safe_limit: np.ndarray,
    penalty_offset: float = FORCE_PENALTY_OFFSET,
) -> np.ndarray:
    """``-mean((|applied| > safe+offset) & (|applied| > |expected| + offset/2))`` (GH
    force_exd_penalty). offset=10 -> the two thresholds are safe+10 and exp+5."""
    applied_norm = np.linalg.norm(np.asarray(force_applied_w, dtype=np.float64), axis=-1)  # (N,M)
    exp_norm = np.linalg.norm(np.asarray(force_expected_w, dtype=np.float64), axis=-1)
    limit = np.asarray(force_safe_limit, dtype=np.float64) + penalty_offset
    exd = ((applied_norm > limit) & (applied_norm > exp_norm + penalty_offset * 0.5)).astype(
        np.float64
    ).mean(axis=-1, keepdims=True)
    return -exd


def force_target_tracking(actual_force_body_w: np.ndarray, force_keypoint_w: np.ndarray, sigma) -> np.ndarray:
    """``exp_sigma(mean|force_keypoint - body|)`` (GH force_target_tracking)."""
    diff = np.asarray(force_keypoint_w, dtype=np.float64) - np.asarray(actual_force_body_w, dtype=np.float64)
    error = np.linalg.norm(diff, axis=-1).mean(axis=-1, keepdims=True)
    return calc_exp_sigma(error, sigma)


def force_target_vel_tracking(
    actual_force_body_vel_w: np.ndarray, force_keypoint_vel_w: np.ndarray, sigma
) -> np.ndarray:
    """``exp_sigma(mean|force_keypoint_vel - body_vel|)`` (GH force_target_vel_tracking)."""
    diff = np.asarray(force_keypoint_vel_w, dtype=np.float64) - np.asarray(
        actual_force_body_vel_w, dtype=np.float64
    )
    error = np.linalg.norm(diff, axis=-1).mean(axis=-1, keepdims=True)
    return calc_exp_sigma(error, sigma)


def keypoint_tracking_imp(
    actual_kp_w: np.ndarray,
    target_kp_w: np.ndarray,
    force_keypoint_w: np.ndarray,
    force_in_keypoint_idx: np.ndarray,
    sigma,
) -> tuple[np.ndarray, np.ndarray]:
    """Keypoint tracking where the force-body targets use the compliant keypoint
    (GH keypoint_tracking_imp). Returns (reward, cum_error_kp)."""
    target = np.array(target_kp_w, dtype=np.float64)
    target[:, np.asarray(force_in_keypoint_idx)] = np.asarray(force_keypoint_w, dtype=np.float64)
    return keypoint_tracking(actual_kp_w, target, sigma)


# --- reward-target selection: teacher current-frame / student 50-cache ------ #


def teacher_reward_target(
    motion_root_pos_0: np.ndarray, motion_root_quat_0: np.ndarray, env_origins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Teacher reward root target = the current motion frame (GH update_reward_target,
    not student). ``reward_root_pos = motion[:,0] + env_origins``."""
    pos = np.asarray(motion_root_pos_0, dtype=np.float64) + np.asarray(env_origins, dtype=np.float64)
    return pos, np.asarray(motion_root_quat_0, dtype=np.float64)


class StudentRootCache:
    """Length-50 rolling root-target cache for the student (GH update_reward_target,
    student branch).

    ``reset`` fills the cache with 50 future motion frames (from the current t).
    Each ``step`` consumes the head as the reward target, shifts the cache forward,
    and appends a new tail anchored to the CURRENT robot root plus the reference
    t->t+50 displacement (z taken from the reference t+50). So the first 50 steps
    consume the initialization frames and step 51 onward consume robot-anchored
    predictions.
    """

    def __init__(self, num_envs: int, steps: int = 50) -> None:
        self.steps = int(steps)
        self.ts_root_pos_w = np.zeros((int(num_envs), self.steps, 3), dtype=np.float64)
        self.ts_root_quat_w = np.zeros((int(num_envs), self.steps, 4), dtype=np.float64)

    def reset(
        self,
        env_ids: np.ndarray,
        fill_root_pos_w: np.ndarray,
        fill_root_quat_w: np.ndarray,
        env_origins: np.ndarray,
    ) -> None:
        env_ids = np.asarray(env_ids)
        self.ts_root_pos_w[env_ids] = (
            np.asarray(fill_root_pos_w, dtype=np.float64) + np.asarray(env_origins, dtype=np.float64)[:, None, :]
        )
        self.ts_root_quat_w[env_ids] = np.asarray(fill_root_quat_w, dtype=np.float64)

    def step(
        self,
        root_pos_w: np.ndarray,
        root_quat_w: np.ndarray,
        ref_pos_t: np.ndarray,
        ref_quat_t: np.ndarray,
        ref_pos_t_plus: np.ndarray,
        ref_quat_t_plus: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        from unilab.utils.rotation import (
            np_quat_apply,
            np_quat_conjugate,
            np_quat_mul,
            np_yaw_quat,
        )

        reward_pos = self.ts_root_pos_w[:, 0].copy()  # head
        reward_quat = self.ts_root_quat_w[:, 0].copy()

        # shift the cache forward
        self.ts_root_pos_w[:, :-1] = self.ts_root_pos_w[:, 1:]
        self.ts_root_quat_w[:, :-1] = self.ts_root_quat_w[:, 1:]

        # new tail anchored to the current robot root + reference t->t+50 delta
        delta_yaw = np_quat_mul(
            np_yaw_quat(np.asarray(root_quat_w, dtype=np.float64)),
            np_quat_conjugate(np_yaw_quat(np.asarray(ref_quat_t, dtype=np.float64))),
        )
        tail = np_quat_apply(
            delta_yaw,
            np.asarray(ref_pos_t_plus, dtype=np.float64) - np.asarray(ref_pos_t, dtype=np.float64),
        ) + np.asarray(root_pos_w, dtype=np.float64)
        tail[:, 2] = np.asarray(ref_pos_t_plus, dtype=np.float64)[:, 2]  # recover z from reference
        self.ts_root_pos_w[:, -1] = tail
        self.ts_root_quat_w[:, -1] = np_quat_mul(delta_yaw, np.asarray(ref_quat_t_plus, dtype=np.float64))
        return reward_pos, reward_quat
