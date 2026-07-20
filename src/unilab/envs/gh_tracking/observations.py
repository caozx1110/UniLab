"""GH observations for the MuJoCo migration (Phase 6).

Three independent observation groups (policy 450 / priv 717 / priv_critic 3),
assembled per-term in GH's YAML insertion order. Pure-numpy ports of GH
``observations.py`` + the command-side obs terms in ``motion_tracking.py``.
Post-substep telemetry (joint 2-slot average, root-linvel EMA, contact-force
history) is sampled every physics substep via the Phase-1 post-step hook.
"""

from __future__ import annotations

import numpy as np


def random_noise(x: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise clamped to +-3 sigma then scaled (GH random_noise)."""
    if std <= 0.0:
        return x
    return x + np.clip(rng.standard_normal(x.shape), -3.0, 3.0) * std


class HistoryBuffer:
    """Rolling history buffer: newest at slot 0, selects ``history_steps`` on read.

    Ports GH ``root_ang_vel_history`` / ``projected_gravity_history``: ``update``
    rolls the buffer and writes the (optionally noised) current value at slot 0;
    ``compute`` gathers the configured history steps; ``reset`` fills the whole
    buffer with the current value. ``renorm`` re-normalizes after noise (gravity).
    """

    def __init__(
        self,
        num_envs: int,
        history_steps: list[int],
        dim: int,
        noise_std: float = 0.0,
        renorm: bool = False,
    ) -> None:
        self.history_steps = list(history_steps)
        self.dim = int(dim)
        self.noise_std = float(noise_std)
        self.renorm = renorm
        self.buffer = np.zeros((int(num_envs), max(self.history_steps) + 1, self.dim), dtype=np.float64)

    def _noise(self, source: np.ndarray, rng: np.random.Generator | None) -> np.ndarray:
        if self.noise_std > 0.0 and rng is not None:
            source = random_noise(source, self.noise_std, rng)
            if self.renorm:
                source = source / np.maximum(np.linalg.norm(source, axis=-1, keepdims=True), 1e-6)
        return source

    def update(self, source: np.ndarray, rng: np.random.Generator | None = None) -> None:
        source = self._noise(np.asarray(source, dtype=np.float64), rng)
        self.buffer = np.roll(self.buffer, 1, axis=1)
        self.buffer[:, 0] = source

    def reset(self, env_ids: np.ndarray, source: np.ndarray) -> None:
        env_ids = np.asarray(env_ids)
        self.buffer[env_ids] = np.asarray(source, dtype=np.float64)[env_ids][:, None, :]

    def compute(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        buf = self.buffer if env_ids is None else self.buffer[env_ids]
        n = buf.shape[0]
        return buf[:, self.history_steps].reshape(n, -1)


class JointPosHistory:
    """Joint-position history with a 2-slot per-substep sub-sampler (GH joint_pos_history).

    ``post_step`` writes the current joint pos into slot ``substep % 2``; with
    decimation 4 the final two slots hold substeps 2 and 3, whose mean is rolled
    into the history buffer on ``update``. ``compute`` subtracts the random
    actuator zero offset (``action_manager.offset``), NOT the default pose.
    """

    def __init__(
        self,
        num_envs: int,
        num_joints: int,
        history_steps: list[int],
        noise_std: float,
        offset: np.ndarray,
    ) -> None:
        self.history_steps = list(history_steps)
        self.num_joints = int(num_joints)
        self.noise_std = max(float(noise_std), 0.0)
        self.offset = np.asarray(offset, dtype=np.float64)  # (N, num_joints) actuator zero offset
        self.joint_pos = np.zeros((int(num_envs), 2, self.num_joints), dtype=np.float64)
        self.buffer = np.zeros(
            (int(num_envs), max(self.history_steps) + 1, self.num_joints), dtype=np.float64
        )

    def post_step(self, substep: int, joint_pos: np.ndarray) -> None:
        self.joint_pos[:, substep % 2] = np.asarray(joint_pos, dtype=np.float64)

    def update(self, rng: np.random.Generator | None = None) -> None:
        self.buffer = np.roll(self.buffer, 1, axis=1)
        joint_pos = self.joint_pos.mean(axis=1)  # avg of the last two substeps (2 and 3)
        if self.noise_std > 0.0 and rng is not None:
            joint_pos = random_noise(joint_pos, self.noise_std, rng)
        self.buffer[:, 0] = joint_pos

    def reset(self, env_ids: np.ndarray, joint_pos: np.ndarray) -> None:
        env_ids = np.asarray(env_ids)
        self.buffer[env_ids] = np.asarray(joint_pos, dtype=np.float64)[env_ids][:, None, :]

    def compute(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        buf = self.buffer if env_ids is None else self.buffer[env_ids]
        offset = self.offset if env_ids is None else self.offset[env_ids]
        n = buf.shape[0]
        joint_pos = buf - offset[:, None, :]  # minus actuator zero offset
        return joint_pos[:, self.history_steps].reshape(n, -1)


class RootLinVelEMA:
    """World-frame root linear velocity EMA, updated every physics substep (GH
    root_linvel_b). ``compute`` rotates the EMA into the root frame."""

    def __init__(self, num_envs: int, ema: float = 1.0) -> None:
        self.ema = float(ema)
        self.linvel_w = np.zeros((int(num_envs), 3), dtype=np.float64)

    def post_step(self, root_lin_vel_w: np.ndarray) -> None:
        self.linvel_w = (1 - self.ema) * self.linvel_w + self.ema * np.asarray(
            root_lin_vel_w, dtype=np.float64
        )

    def reset(self, env_ids: np.ndarray) -> None:
        self.linvel_w[np.asarray(env_ids)] = 0.0

    def compute(self, root_quat_w: np.ndarray, env_ids: np.ndarray | None = None) -> np.ndarray:
        from unilab.utils.rotation import np_quat_apply_inverse

        linvel_w = self.linvel_w if env_ids is None else self.linvel_w[env_ids]
        return np_quat_apply_inverse(np.asarray(root_quat_w, dtype=np.float64), linvel_w)


class ContactForceHistory:
    """Per-body net external contact force with a 3-frame history (GH contact_forces).

    ``post_step`` pushes the current per-body net contact force into a 3-deep ring
    (each physics substep); ``compute`` returns the history mean divided by
    ``mass_total * 9.81`` and clamped to [-10, 10].
    """

    def __init__(self, num_envs: int, n_bodies: int, mass_total: float, history_len: int = 3) -> None:
        self.denom = float(mass_total) * 9.81
        self.history = np.zeros((int(num_envs), history_len, int(n_bodies), 3), dtype=np.float64)

    def post_step(self, net_force_w: np.ndarray) -> None:
        self.history = np.roll(self.history, 1, axis=1)
        self.history[:, 0] = np.asarray(net_force_w, dtype=np.float64)

    def reset(self, env_ids: np.ndarray) -> None:
        self.history[np.asarray(env_ids)] = 0.0

    def compute(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        hist = self.history if env_ids is None else self.history[env_ids]
        n = hist.shape[0]
        force = hist.mean(axis=1) / self.denom  # (N, n_bodies, 3)
        return np.clip(force, -10.0, 10.0).reshape(n, -1)


def body_height(body_pos_w: np.ndarray) -> np.ndarray:
    """Z heights of the selected bodies (GH body_height). ``body_pos_w`` (N, k, 3)."""
    bp = np.asarray(body_pos_w, dtype=np.float64)
    return bp[:, :, 2].reshape(bp.shape[0], -1)


def applied_action_obs(joint_pos_target: np.ndarray) -> np.ndarray:
    """Absolute joint position target (default+offset+scale+boot), NOT the residual
    action (GH applied_action = asset.data.joint_pos_target)."""
    return np.asarray(joint_pos_target, dtype=np.float64)


def prev_actions_obs(action_buf: np.ndarray) -> np.ndarray:
    """Flattened raw action history ``action_buf[:, :steps]`` (GH prev_actions)."""
    ab = np.asarray(action_buf, dtype=np.float64)
    return ab.reshape(ab.shape[0], -1)


def boot_indicator_state_obs(boot_indicator: np.ndarray, boot_indicator_max: int) -> np.ndarray:
    """Boot-protection indicator normalized by its max (GH boot_indicator_state)."""
    return np.asarray(boot_indicator, dtype=np.float64) / float(boot_indicator_max)


# --- command-side observations (motion_tracking, future frames S=5) --------- #


def _qinv_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    from unilab.utils.rotation import np_quat_apply_batched, np_quat_conjugate_batched

    return np_quat_apply_batched(np_quat_conjugate_batched(q), v)


def _flatN(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def target_joint_pos_obs(motion_joint_pos: np.ndarray) -> np.ndarray:
    """Reference joint positions over the future frames (GH target_joint_pos_obs)."""
    return _flatN(np.asarray(motion_joint_pos, dtype=np.float64))


def target_pos_b_obs(
    motion_root_pos_w: np.ndarray,
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
    env_origins: np.ndarray,
) -> np.ndarray:
    """Reference root positions in the current root frame (GH target_pos_b_obs)."""
    current_pos = (root_pos_w - env_origins)[:, None, :]
    q = root_quat_w[:, None, :]
    return _flatN(_qinv_apply(q, np.asarray(motion_root_pos_w, dtype=np.float64) - current_pos))


def target_linvel_b_obs(motion_root_lin_vel_w: np.ndarray, root_quat_w: np.ndarray) -> np.ndarray:
    """Reference root linear velocities in the root frame (GH target_linvel_b_obs)."""
    q = root_quat_w[:, None, :]
    return _flatN(_qinv_apply(q, np.asarray(motion_root_lin_vel_w, dtype=np.float64)))


def target_projected_gravity_b(motion_root_quat_w: np.ndarray) -> np.ndarray:
    """Gravity in the reference root frame per future frame (GH target_projected_gravity_b)."""
    gravity = np.array([0.0, 0.0, -1.0]).reshape(1, 1, 3)
    return _flatN(_qinv_apply(np.asarray(motion_root_quat_w, dtype=np.float64), gravity))


def relative_quat_obs(motion_root_quat_w: np.ndarray, root_quat_w: np.ndarray) -> np.ndarray:
    """Axis-angle of the reference-vs-current root orientation (GH relative_quat_obs)."""
    from unilab.utils.rotation import (
        np_quat_conjugate_batched,
        np_quat_mul_batched,
        np_quat_to_axis_angle,
    )

    mq = np.asarray(motion_root_quat_w, dtype=np.float64)
    q = np.broadcast_to(root_quat_w[:, None, :], mq.shape)
    rel = np_quat_mul_batched(mq, np_quat_conjugate_batched(q))  # (N, S, 4)
    n, s = rel.shape[0], rel.shape[1]
    aa = np_quat_to_axis_angle(rel.reshape(-1, 4)).reshape(n, s, 3)
    return _flatN(aa)


def current_keypoint_b(
    body_pos_w_kp: np.ndarray, root_pos_w: np.ndarray, root_quat_w: np.ndarray
) -> np.ndarray:
    """Actual keypoint positions in the root frame (GH current_keypoint_b)."""
    q = root_quat_w[:, None, :]
    return _flatN(_qinv_apply(q, np.asarray(body_pos_w_kp, dtype=np.float64) - root_pos_w[:, None, :]))


def current_keypoint_vel_b(body_lin_vel_w_kp: np.ndarray, root_quat_w: np.ndarray) -> np.ndarray:
    """Actual keypoint velocities in the root frame (GH current_keypoint_vel_b)."""
    q = root_quat_w[:, None, :]
    return _flatN(_qinv_apply(q, np.asarray(body_lin_vel_w_kp, dtype=np.float64)))


def target_keypoints_diff_b_obs(
    motion_body_pos_w_kp: np.ndarray,
    actual_body_pos_w_kp: np.ndarray,
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
    env_origins: np.ndarray,
) -> np.ndarray:
    """Target-minus-actual keypoint offsets in the root frame (GH
    target_keypoints_diff_b_obs). Reads body_pos_w for both (D4), NOT body_pos_b."""
    actual_w = np.asarray(actual_body_pos_w_kp, dtype=np.float64) - env_origins[:, None, :]
    diff_w = np.asarray(motion_body_pos_w_kp, dtype=np.float64) - actual_w[:, None, :, :]
    q = root_quat_w[:, None, None, :]
    return _flatN(_qinv_apply(q, diff_w))


def command_obs(
    motion_root_pos_w: np.ndarray,
    motion_root_quat_w: np.ndarray,
    force_safe_limit: np.ndarray,
) -> np.ndarray:
    """Motion-tracking command (GH command, :1062-1084), dim 22.

    = future root heights (5) + pos_diff_b xy for future[1:5] (8) + heading_b xy
    for future[1:5] (8) + force_safe_limit (1). Displacements/headings are in the
    yaw frame of the first future frame.
    """
    from unilab.utils.rotation import np_quat_apply_batched, np_yaw_quat

    mpos = np.asarray(motion_root_pos_w, dtype=np.float64)
    mquat = np.asarray(motion_root_quat_w, dtype=np.float64)
    n, s = mpos.shape[0], mpos.shape[1]

    root_yaw_quat = np_yaw_quat(mquat[:, 0])[:, None, :]  # (N,1,4)
    root_yaw_future = np_yaw_quat(mquat[:, 1:].reshape(-1, 4)).reshape(n, s - 1, 4)  # (N,4,4)
    root_pos = mpos[:, 0:1]
    root_pos_future = mpos[:, 1:]

    pos_diff_b = _qinv_apply(root_yaw_quat, root_pos_future - root_pos)  # (N,4,3)

    heading = np.array([1.0, 0.0, 0.0]).reshape(1, 1, 3)
    target_heading = np_quat_apply_batched(root_yaw_future, heading)  # (N,4,3)
    target_heading_b = _qinv_apply(root_yaw_quat, target_heading)  # (N,4,3)

    return np.concatenate(
        [
            mpos[:, :, 2].reshape(n, -1),
            pos_diff_b[:, :, :2].reshape(n, -1),
            target_heading_b[:, :, :2].reshape(n, -1),
            np.asarray(force_safe_limit, dtype=np.float64),
        ],
        axis=-1,
    )


def force_priv_obs(
    force_keypoint_b: np.ndarray,
    force_applied_b: np.ndarray,
    force_expected_b: np.ndarray,
    force_sample_timer: np.ndarray,
) -> np.ndarray:
    """Privileged force observation (GH force_priv, :1094-1101), dim 55.

    = force_keypoint_b (18) + force_applied_b (18) + force_expected_b (18) +
    force_sample_timer (1). Reads the Phase-5 ForceSystem state.
    """
    return np.concatenate(
        [
            _flatN(np.asarray(force_keypoint_b, dtype=np.float64)),
            _flatN(np.asarray(force_applied_b, dtype=np.float64)),
            _flatN(np.asarray(force_expected_b, dtype=np.float64)),
            np.asarray(force_sample_timer, dtype=np.float64).reshape(force_keypoint_b.shape[0], -1),
        ],
        axis=-1,
    )


from dataclasses import dataclass  # noqa: E402


@dataclass
class ObsState:
    """Per-step inputs the observation terms read (asset data + command/motion/force state).

    Instantaneous fields feed the stateless terms directly; ``joint_pos`` /
    ``root_lin_vel_w`` / ``net_contact_force`` feed the per-substep telemetry
    buffers (via ``post_step``), and ``root_ang_vel_b`` / ``projected_gravity_b``
    feed the control-step history buffers (via ``update``).
    """

    root_pos_w: np.ndarray
    root_quat_w: np.ndarray
    root_ang_vel_b: np.ndarray
    projected_gravity_b: np.ndarray
    env_origins: np.ndarray
    joint_pos_target: np.ndarray
    applied_torque: np.ndarray
    action_buf: np.ndarray
    body_pos_w_height: np.ndarray
    body_pos_w_kp: np.ndarray
    body_lin_vel_w_kp: np.ndarray
    motion_root_pos_w: np.ndarray
    motion_root_quat_w: np.ndarray
    motion_root_lin_vel_w: np.ndarray
    motion_joint_pos: np.ndarray
    motion_body_pos_w_kp: np.ndarray
    force_keypoint_b: np.ndarray
    force_applied_b: np.ndarray
    force_expected_b: np.ndarray
    force_sample_timer: np.ndarray
    force_safe_limit: np.ndarray
    boot_indicator: np.ndarray
    cum_error: np.ndarray
    joint_pos: np.ndarray
    root_lin_vel_w: np.ndarray
    net_contact_force: np.ndarray


class ObservationManager:
    """Assembles the three observation groups (policy 450 / priv 717 / priv_critic 3)
    per-term in GH's YAML insertion order, owning the persistent telemetry buffers.
    """

    def __init__(
        self,
        num_envs: int,
        mass_total: float,
        actuator_offset: np.ndarray,
        num_joints: int = 29,
        num_keypoints: int = 11,
        num_force_bodies: int = 6,
        num_feet: int = 2,
        boot_indicator_max: int = 25,
        seed: int = 0,
    ) -> None:
        self.n = int(num_envs)
        self.boot_max = int(boot_indicator_max)
        self._rng = np.random.default_rng(seed)
        off = np.asarray(actuator_offset, dtype=np.float64)
        # policy telemetry (noisy, few history steps)
        self.pol_angvel = HistoryBuffer(self.n, [0], 3, noise_std=0.05)
        self.pol_grav = HistoryBuffer(self.n, [0], 3, noise_std=0.01, renorm=True)
        self.pol_joint = JointPosHistory(self.n, num_joints, [0, 1, 2, 3, 4, 8], 0.005, off)
        # priv telemetry (noise-free, deep history)
        self.priv_angvel = HistoryBuffer(self.n, list(range(9)), 3, noise_std=0.0)
        self.priv_grav = HistoryBuffer(self.n, list(range(9)), 3, noise_std=0.0, renorm=True)
        self.priv_joint = JointPosHistory(self.n, num_joints, list(range(9)), 0.0, off)
        self.root_ema = RootLinVelEMA(self.n, ema=0.2)
        self.contact = ContactForceHistory(self.n, num_feet, mass_total)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"policy": 450, "priv": 717, "priv_critic": 3}

    def reset(self, env_ids: np.ndarray, state: ObsState) -> None:
        self.pol_angvel.reset(env_ids, state.root_ang_vel_b)
        self.pol_grav.reset(env_ids, state.projected_gravity_b)
        self.priv_angvel.reset(env_ids, state.root_ang_vel_b)
        self.priv_grav.reset(env_ids, state.projected_gravity_b)
        self.pol_joint.reset(env_ids, state.joint_pos)
        self.priv_joint.reset(env_ids, state.joint_pos)
        self.root_ema.reset(env_ids)
        self.contact.reset(env_ids)

    def post_step(self, substep: int, state: ObsState) -> None:
        """Sample per-substep telemetry (Phase-1 hook: every substep incl. last)."""
        self.pol_joint.post_step(substep, state.joint_pos)
        self.priv_joint.post_step(substep, state.joint_pos)
        self.root_ema.post_step(state.root_lin_vel_w)
        self.contact.post_step(state.net_contact_force)

    def update(self, state: ObsState) -> None:
        """Control-step history roll (after all substeps)."""
        self.pol_angvel.update(state.root_ang_vel_b, self._rng)
        self.pol_grav.update(state.projected_gravity_b, self._rng)
        self.priv_angvel.update(state.root_ang_vel_b, self._rng)
        self.priv_grav.update(state.projected_gravity_b, self._rng)
        self.pol_joint.update(self._rng)
        self.priv_joint.update(self._rng)

    def compute(self, s: ObsState, env_ids: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Assemble the three obs groups. When ``env_ids`` is given, ``s`` must already
        hold only those rows (subset ObsState); the persistent telemetry buffers are
        indexed by ``env_ids`` so the result equals the full compute sliced to
        ``env_ids`` (every term is per-env independent). Used by the reset path to
        avoid recomputing obs for the ~98% of envs that did not reset."""
        policy = np.concatenate(
            [
                boot_indicator_state_obs(s.boot_indicator, self.boot_max),  # [0:1]
                command_obs(s.motion_root_pos_w, s.motion_root_quat_w, s.force_safe_limit),  # [1:23]
                target_joint_pos_obs(s.motion_joint_pos),  # [23:168]
                target_projected_gravity_b(s.motion_root_quat_w),  # [168:183]
                self.pol_angvel.compute(env_ids),  # [183:186]
                self.pol_grav.compute(env_ids),  # [186:189]
                self.pol_joint.compute(env_ids),  # [189:363]
                prev_actions_obs(s.action_buf),  # [363:450]
            ],
            axis=-1,
        )
        priv = np.concatenate(
            [
                target_pos_b_obs(s.motion_root_pos_w, s.root_pos_w, s.root_quat_w, s.env_origins),  # [0:15]
                target_linvel_b_obs(s.motion_root_lin_vel_w, s.root_quat_w),  # [15:30]
                relative_quat_obs(s.motion_root_quat_w, s.root_quat_w),  # [30:45]
                force_priv_obs(s.force_keypoint_b, s.force_applied_b, s.force_expected_b, s.force_sample_timer),  # [45:100]
                body_height(s.body_pos_w_height),  # [100:104]
                self.contact.compute(env_ids),  # [104:110]
                self.root_ema.compute(s.root_quat_w, env_ids),  # [110:113]
                self.priv_angvel.compute(env_ids),  # [113:140]
                self.priv_grav.compute(env_ids),  # [140:167]
                self.priv_joint.compute(env_ids),  # [167:428]
                current_keypoint_b(s.body_pos_w_kp, s.root_pos_w, s.root_quat_w),  # [428:461]
                current_keypoint_vel_b(s.body_lin_vel_w_kp, s.root_quat_w),  # [461:494]
                target_keypoints_diff_b_obs(
                    s.motion_body_pos_w_kp, s.body_pos_w_kp, s.root_pos_w, s.root_quat_w, s.env_origins
                ),  # [494:659]
                applied_action_obs(s.joint_pos_target),  # [659:688]
                np.asarray(s.applied_torque, dtype=np.float64),  # [688:717] priv noise = 0
            ],
            axis=-1,
        )
        priv_critic = np.asarray(s.cum_error, dtype=np.float64)  # [0:3]
        return {"policy": policy, "priv": priv, "priv_critic": priv_critic}
