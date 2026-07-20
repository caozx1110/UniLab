"""GH force system for the MuJoCo migration (Phase 5).

Pure-numpy port of the compliant-force machinery in GH ``motion_tracking.py``:
unidirectional spring projection, eccentric torque, net-wrench torso correction,
the force schedule (with the two source-compat quirks), the admittance
integration, and per-substep force application feeding the Phase-1
``apply_body_wrench`` (force + torque) backend contract.
"""

from __future__ import annotations

import numpy as np

from unilab.envs.gh_tracking.admittance import clamp_norm


def project_pos_diff(pos_diff: np.ndarray, force_dir: np.ndarray) -> np.ndarray:
    """Unidirectional projection: keep only the component of ``pos_diff`` along
    ``force_dir`` when it is non-positive (the spring pulls toward the origin, it
    never pushes). GH ``project_pos_diff`` (clamp_max 0)."""
    coef = np.minimum((pos_diff * force_dir).sum(axis=-1, keepdims=True), 0.0)
    return coef * force_dir


def project_vel(vel: np.ndarray, force_dir: np.ndarray) -> np.ndarray:
    """Keep only the non-negative component of ``vel`` along ``force_dir`` (GH
    ``project_vel``, clamp_min 0)."""
    coef = np.maximum((vel * force_dir).sum(axis=-1, keepdims=True), 0.0)
    return coef * force_dir


def compute_eccentric_torque(
    quat_w: np.ndarray, pos_delta: np.ndarray, force_w: np.ndarray
) -> np.ndarray:
    """Eccentric torque ``cross(delta_w, force_w)`` where the body-frame offset
    ``pos_delta`` is rotated to world by ``quat_w``. GH force_apply :1008-1009."""
    from unilab.utils.rotation import np_quat_apply_batched

    delta_w = np_quat_apply_batched(quat_w, pos_delta)
    return np.cross(delta_w, force_w, axis=-1)


def limit_net_wrench_about_torso(
    pos_w_6: np.ndarray,
    force_w_6: np.ndarray,
    torque_w_6: np.ndarray,
    torso_pos_w: np.ndarray,
    net_force_limit: float,
    net_torque_limit: float,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Net-wrench correction about the torso (GH ``_limit_net_wrench_about_torso``).

    Computes the net external force/torque of the 6 force bodies about the torso,
    caps their norms to (net_force_limit, net_torque_limit), and returns the
    correction wrench (dF, dM) to apply to the torso body so the net stays bounded.

    Args:
        pos_w_6: force-body world positions, (N, 6, 3)
        force_w_6: force-body applied forces (world), (N, 6, 3)
        torque_w_6: force-body applied (eccentric) torques (world), (N, 6, 3)
        torso_pos_w: torso world position, (N, 3)

    Returns:
        (dF, dM) each (N, 3), to be written to the torso body's wrench.
    """
    r = pos_w_6 - torso_pos_w[:, None, :]  # (N, 6, 3)
    f_net = force_w_6.sum(axis=1)  # (N, 3)
    m_net = np.cross(r, force_w_6, axis=-1).sum(axis=1) + torque_w_6.sum(axis=1)  # (N, 3)

    f_norm = np.maximum(np.linalg.norm(f_net, axis=-1, keepdims=True), eps)
    f_scale = np.minimum(net_force_limit / f_norm, 1.0)
    d_f = f_net * f_scale - f_net

    m_norm = np.maximum(np.linalg.norm(m_net, axis=-1, keepdims=True), eps)
    m_scale = np.minimum(net_torque_limit / m_norm, 1.0)
    d_m = m_net * m_scale - m_net
    return d_f, d_m


# force-body order is [L,L,R,R,L,R] (shoulder_yaw, wrist_roll x L/R, then hand_mimic
# x L/R); the masks below are constant over THIS order and are NOT anatomical
# left/right (quirk 3). left_mask selects positions 0,2,4; right_mask 1,3,5.
_LEFT_MASK = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
_RIGHT_MASK = np.array([0, 1, 0, 1, 0, 1], dtype=bool)

FORCE_TYPE_PROBS = np.array([0.4, 0.15, 0.15, 0.15, 0.15])  # zero/full/left/right/partial
KP_RANGE = (5.0, 250.0)
KP_SLOPE_RANGE = (-5.0, 5.0)
ZERO_KP_SLOPE_PROB = 0.5
KP_TIME_RANGE = (25, 100)
FORCE_TIME_RANGE = (20, 200)
RAMPING_TIME_RANGE = (25, 100)
SAFE_LIMIT_RESAMPLE_RANGE = (100, 200)
SAFE_LIMIT_TRANSIT_RANGE = (25, 100)
FORCE_SAFE_BOUNDS = (5.0, 15.0)
FORCE_SAFE_DEFAULT = 10.0
FORCE_PARTIAL_SINGLE_PROB = 0.5
FORCE_POS_DELTA_RMAX = 0.05


def rand_points_isotropic(n: int, m: int, r_max: float, rng: np.random.Generator) -> np.ndarray:
    """Uniformly sample points in a ball of radius ``r_max`` (GH ``rand_points_isotropic``)."""
    r = rng.random((n, m, 1))
    v = rng.random((n, m, 1))
    w = rng.random((n, m, 1))
    z = 1 - 2 * v
    phi = 2 * np.pi * w
    xy = np.sqrt(np.clip(1 - z * z, 0, None))
    x = xy * np.cos(phi)
    y = xy * np.sin(phi)
    return np.concatenate([x, y, z], axis=-1) * r * r_max


class ForceSystem:
    """Compliant external-force system (GH ``motion_tracking`` force machinery)."""

    def __init__(
        self,
        num_envs: int,
        num_force_bodies: int = 6,
        physics_dt: float = 0.005,
        step_dt: float = 0.02,
        max_force: float = 30.0,
        net_force_limit: float = 30.0,
        net_torque_limit: float = 20.0,
        force_alpha: float = 1.0,
        compliance: bool = True,
        seed: int = 0,
    ) -> None:
        from unilab.envs.gh_tracking.admittance import AdmittanceMassChain
        from unilab.envs.gh_tracking.temporal_lerp import TemporalLerp

        self.N = int(num_envs)
        self.M = int(num_force_bodies)
        self.physics_dt = float(physics_dt)
        self.step_dt = float(step_dt)
        self.max_force = float(max_force)
        self.net_force_limit = float(net_force_limit)
        self.net_torque_limit = float(net_torque_limit)
        self.force_alpha = float(force_alpha)
        self.compliance = bool(compliance)  # GH force variant selector (motion_tracking.py:1045)
        self._rng = np.random.default_rng(seed)

        self.left_mask = _LEFT_MASK[None, :, None].copy()  # (1, M, 1)
        self.right_mask = _RIGHT_MASK[None, :, None].copy()

        # sample control state
        self.force_type = np.zeros(self.N, dtype=np.int32)  # quirk 2: never written back
        self.force_enable = np.zeros((self.N, self.M, 1), dtype=bool)
        self.force_kp_scaled = np.zeros((self.N, self.M, 1), dtype=np.float64)

        # spring geometry / force buffers
        self.force_origin_w = np.zeros((self.N, self.M, 3), dtype=np.float64)
        self.force_dir_w = np.zeros((self.N, self.M, 3), dtype=np.float64)
        self.force_applied_w = np.zeros((self.N, self.M, 3), dtype=np.float64)
        self.force_applied_b = np.zeros((self.N, self.M, 3), dtype=np.float64)
        self.force_expected_w = np.zeros((self.N, self.M, 3), dtype=np.float64)
        self.force_expected_b = np.zeros((self.N, self.M, 3), dtype=np.float64)
        self.force_pos_delta = np.zeros((self.N, self.M, 3), dtype=np.float64)

        # admittance tracking
        self.ref_pos_b_prev = np.zeros((self.N, self.M, 3), dtype=np.float64)
        self.force_keypoint_w = np.zeros((self.N, self.M, 3), dtype=np.float64)
        self.force_keypoint_b = np.zeros((self.N, self.M, 3), dtype=np.float64)
        self.force_keypoint_w_prev = np.zeros((self.N, self.M, 3), dtype=np.float64)
        self.force_keypoint_vel_w = np.zeros((self.N, self.M, 3), dtype=np.float64)

        # timers (control-step units)
        self.force_sample_timer = np.zeros(self.N, dtype=np.int32)
        self.force_kp_sample_timer = np.zeros(self.N, dtype=np.int32)
        self.force_safe_limit_sample_timer = np.zeros(self.N, dtype=np.int32)
        self.force_kp_ramping_down = np.zeros(self.N, dtype=bool)

        # TemporalLerps
        self.force_kp_tl = TemporalLerp((self.N, self.M, 1), easing="linear", clamp=KP_RANGE)
        self.force_origin_tl = TemporalLerp((self.N, self.M, 3), easing="linear")
        self.force_safe_limit_tl = TemporalLerp(
            (self.N, 1), default=FORCE_SAFE_DEFAULT, easing="linear", clamp=FORCE_SAFE_BOUNDS
        )
        # non-compliant random perturbation force on the last 2 force bodies (GH :662)
        self.perturb_force = TemporalLerp((self.N, 2, 3), easing="linear")

        self.admit = AdmittanceMassChain(
            num_envs=self.N, num_points=self.M, dt=self.physics_dt, mixed_loop_steps=1,
            mass=0.1, damping=2.0, vel_clip=4.0, acc_clip=1000.0,
        )
        self.last_reset_env_ids: np.ndarray | None = None

    def _sample_force_types(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n force types from FORCE_TYPE_PROBS (0=zero..4=partial)."""
        return rng.choice(len(FORCE_TYPE_PROBS), size=n, p=FORCE_TYPE_PROBS).astype(np.int32)

    def _enable_from_force_type(
        self, force_type: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Map force types to per-body enable masks over the [L,L,R,R,L,R] order.

        type 0 -> none, 1 -> all, 2 -> left_mask, 3 -> right_mask, 4 -> random
        partial (each body rand <= 0.5). Constant masks are NOT anatomical (quirk 3).
        """
        ft = np.asarray(force_type)
        n = len(ft)
        enable = np.zeros((n, self.M, 1), dtype=bool)
        enable[ft == 1] = True
        enable[ft == 2] = self.left_mask
        enable[ft == 3] = self.right_mask
        partial = ft == 4
        if partial.any():
            k = int(partial.sum())
            enable[partial] = rng.random((k, self.M, 1)) <= FORCE_PARTIAL_SINGLE_PROB
        return enable

    def force_reset(self, env_ids: np.ndarray, refresh_time: bool = True) -> None:
        """Reset per-env force state (GH ``force_reset``)."""
        env_ids = np.asarray(env_ids)
        self.force_type[env_ids] = 0
        self.force_applied_w[env_ids] = 0.0
        self.force_applied_b[env_ids] = 0.0
        self.force_enable[env_ids] = False
        self.force_origin_tl.reset(env_ids)
        self.force_kp_tl.reset(env_ids)
        self.force_kp_ramping_down[env_ids] = False
        self.perturb_force.reset(env_ids, value=0.0)  # GH force_reset :697
        if refresh_time:
            self.force_sample_timer[env_ids] = self._rng.integers(10, 60, size=len(env_ids))
            self.force_safe_limit_tl.reset(env_ids)
            self.force_safe_limit_tl.set(
                env_ids, start=np.full((len(env_ids), 1), FORCE_SAFE_DEFAULT),
                end=np.full((len(env_ids), 1), FORCE_SAFE_DEFAULT), total_steps=1,
            )
            self.force_safe_limit_sample_timer[env_ids] = 0

    def update_force_kp_scaled(self) -> None:
        """force_kp_scaled = force_kp_tl.current * force_enable (GH :792)."""
        self.force_kp_scaled = self.force_kp_tl.current * self.force_enable

    def force_schedule(self, rng: np.random.Generator) -> None:
        """Advance the force curriculum one control step (GH ``force_schedule`` :706-792).

        Note quirk 2: the resampled ``force_type`` is a LOCAL variable used only to
        build ``force_enable``; it is NEVER written back to ``self.force_type``
        (which therefore stays 0), so the ``force_type != 4`` origin-sample gate is
        always true. Reproduced verbatim. Spring-origin geometry (needs body pose)
        is deferred to the env layer via ``_need_origin_resample``.
        """
        self.force_kp_sample_timer -= 1
        self.force_sample_timer -= 1
        self.force_safe_limit_sample_timer -= 1
        self._need_origin_resample = np.zeros(0, dtype=np.int64)

        # -0 resample safe limit
        need_safe = self.force_safe_limit_sample_timer < 0
        if need_safe.any():
            envs = np.nonzero(need_safe)[0]
            self.force_safe_limit_sample_timer[envs] = rng.integers(*SAFE_LIMIT_RESAMPLE_RANGE, size=len(envs))
            safe = rng.uniform(*FORCE_SAFE_BOUNDS, size=(len(envs), 1))
            steps = rng.integers(*SAFE_LIMIT_TRANSIT_RANGE, size=len(envs))
            self.force_safe_limit_tl.set(envs, end=safe, total_steps=steps)

        # -0 resample kp slope
        need_kp = (self.force_kp_sample_timer < 0) & (~self.force_kp_ramping_down)
        if need_kp.any():
            envs = np.nonzero(need_kp)[0]
            kp_next = rng.integers(*KP_TIME_RANGE, size=len(envs))
            self.force_kp_sample_timer[envs] = kp_next
            zero_slope = rng.random((len(envs), 2, 1, 1)) < ZERO_KP_SLOPE_PROB
            slope = rng.uniform(*KP_SLOPE_RANGE, size=(len(envs), 2, 1, 1)) * (~zero_slope)
            kp_delta = (slope[:, 0] * self.left_mask + slope[:, 1] * self.right_mask) * kp_next[:, None, None]
            self.force_kp_tl.set(envs, delta=kp_delta, total_steps=kp_next)

        # -1 finished ramping down -> reset force
        finished = self.force_kp_ramping_down & self.force_kp_tl.mask_done
        if finished.any():
            self.force_kp_ramping_down[np.nonzero(finished)[0]] = False
            self.force_reset(np.nonzero(finished)[0], refresh_time=False)

        # -2 time done -> start ramping down (only envs that currently have force)
        time_done = self.force_sample_timer < 0
        force_required = self.force_enable.any(axis=(1, 2))
        should_ramp = time_done & (~self.force_kp_ramping_down) & force_required
        if should_ramp.any():
            envs = np.nonzero(should_ramp)[0]
            self.force_kp_ramping_down[envs] = True
            steps = rng.integers(*RAMPING_TIME_RANGE, size=len(envs))
            self.force_kp_tl.set(envs, end=0.0, total_steps=steps)

        # -3 resample new force (time done, not ramping)
        need_resample = time_done & (~self.force_kp_ramping_down)
        if need_resample.any():
            envs = np.nonzero(need_resample)[0]
            force_type = self._sample_force_types(len(envs), rng)  # LOCAL — NOT stored (quirk 2)
            enable = self._enable_from_force_type(force_type, rng)
            self.force_enable[envs] = enable
            kp_left = rng.uniform(*KP_RANGE, size=(len(envs), 1, 1))
            kp_right = rng.uniform(*KP_RANGE, size=(len(envs), 1, 1))
            kp = (kp_left * self.left_mask + kp_right * self.right_mask) * enable
            self.force_kp_tl.set(envs, end=kp, total_steps=1)
            self.force_sample_timer[envs] = rng.integers(*FORCE_TIME_RANGE, size=len(envs))
            self.force_pos_delta[envs] = rand_points_isotropic(len(envs), self.M, FORCE_POS_DELTA_RMAX, rng)
            self._need_origin_resample = envs

        self.force_kp_tl.update_time(1)
        self.force_safe_limit_tl.update_time(1)
        self.force_origin_tl.update_time(1)
        self.update_force_kp_scaled()

    def compute_drive_gains(self, force_limit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Admittance drive PD gains (GH :874-875): Kp=force_limit/0.05, Kd=2*sqrt(Kp*0.1).

        ``force_limit`` (N,1) -> (Kp, Kd) each (N,1,1).
        """
        fl = np.asarray(force_limit, dtype=np.float64)[:, :, None]  # (N,1,1)
        kp = fl / 0.05
        kd = 2.0 * np.sqrt(kp * 0.1)  # critical damping @ mass 0.1
        return kp, kd

    def force_update_origin_and_target(
        self,
        *,
        root_pos_w: np.ndarray,
        root_quat_w: np.ndarray,
        ref_keypoints_w: np.ndarray,
    ) -> None:
        """Integrate the admittance chain to the compliant force keypoints (GH
        ``force_update_origin_and_target`` :830-922). Runs the 4x per-control-step
        semi-implicit loop. ``ref_keypoints_w`` are the reference keypoints for the
        6 force bodies (N,6,3) in world frame.
        """
        from unilab.utils.rotation import np_quat_apply_batched, np_quat_conjugate_batched

        n = root_pos_w.shape[0]
        q = np.broadcast_to(root_quat_w[:, None, :], (n, self.M, 4))
        q_inv = np_quat_conjugate_batched(q)
        root_pos = root_pos_w[:, None, :]

        origin_b = self.force_origin_tl.current  # (N,6,3)
        self.force_origin_w = np_quat_apply_batched(q, origin_b) + root_pos

        ref_point_b = np_quat_apply_batched(q_inv, ref_keypoints_w - root_pos)
        if self.last_reset_env_ids is not None:
            self.ref_pos_b_prev[self.last_reset_env_ids] = ref_point_b[self.last_reset_env_ids]
        ref_point_vel_b = (ref_point_b - self.ref_pos_b_prev) / self.step_dt
        self.ref_pos_b_prev = ref_point_b.copy()

        diff = ref_point_b - origin_b
        dir_b = diff / np.maximum(np.linalg.norm(diff, axis=-1, keepdims=True), 1e-6)
        self.force_dir_w = np_quat_apply_batched(q, dir_b)

        if self.last_reset_env_ids is not None:
            self.admit.reset(
                self.last_reset_env_ids,
                x0_b=ref_point_b[self.last_reset_env_ids],
                v0_b=ref_point_vel_b[self.last_reset_env_ids],
            )

        force_limit = self.force_safe_limit_tl.current  # (N,1)
        kp_drive, kd_drive = self.compute_drive_gains(force_limit)  # (N,1,1)
        max_ext = self.max_force * self.force_alpha

        # broadcast to admittance state shape (H=1, N, M, 3)
        ref_e = ref_point_b[None]
        refv_e = ref_point_vel_b[None]
        org_e = origin_b[None]
        dir_e = dir_b[None]
        kp_e = kp_drive[None]
        kd_e = kd_drive[None]
        kp_scaled_e = self.force_kp_scaled[None]
        fl_e = force_limit[:, :, None][None]  # (1,N,1,1)

        for _ in range(4):
            x = self.admit.x
            v = self.admit.v
            f_drive = clamp_norm(kp_e * (ref_e - x) + kd_e * (refv_e - v), fl_e)
            f_ext = clamp_norm(kp_scaled_e * project_pos_diff(org_e - x, dir_e), max_ext)
            self.admit.step(f_drive, f_ext)

        force_keypoint_b = self.admit.x[0]  # (N,6,3)
        self.force_keypoint_b = force_keypoint_b.copy()
        self.force_keypoint_w = np_quat_apply_batched(q, force_keypoint_b) + root_pos
        if self.last_reset_env_ids is not None:
            self.force_keypoint_w_prev[self.last_reset_env_ids] = \
                self.force_keypoint_w[self.last_reset_env_ids]
        self.force_keypoint_vel_w = (self.force_keypoint_w - self.force_keypoint_w_prev) / self.step_dt
        self.force_keypoint_w_prev = self.force_keypoint_w.copy()

        force_expected_b = clamp_norm(
            self.force_kp_scaled * project_pos_diff(origin_b - force_keypoint_b, dir_b), max_ext
        )
        self.force_expected_b = force_expected_b
        self.force_expected_w = np_quat_apply_batched(q, force_expected_b)

    def reset_force_substep(self) -> None:
        """Reset the per-control-step substep counter (call before each control step)."""
        self._force_substep = 0

    def force_apply(
        self,
        substep: int,
        *,
        pos_w: np.ndarray,
        quat_w: np.ndarray,
        root_quat_w: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the compliant force + eccentric torque on the 6 force bodies for
        one physics substep (GH ``force_apply`` :992-1013).

        The spring force is recomputed from the CURRENT force-body pose each substep
        (so it tracks body motion); at substep 0 the applied buffers are zeroed
        first. Returns (force_w, torque_w) each (N, 6, 3), world frame.
        """
        from unilab.utils.rotation import np_quat_apply_batched, np_quat_conjugate_batched

        if substep == 0:
            self.force_applied_w[:] = 0.0
            self.force_applied_b[:] = 0.0

        diff = self.force_origin_w - pos_w  # spring pulls body toward origin
        self.force_applied_w = clamp_norm(
            self.force_kp_scaled * project_pos_diff(diff, self.force_dir_w),
            self.max_force * self.force_alpha,
        )
        n = root_quat_w.shape[0]
        q = np.broadcast_to(root_quat_w[:, None, :], (n, self.M, 4))
        self.force_applied_b = np_quat_apply_batched(np_quat_conjugate_batched(q), self.force_applied_w)

        torque_w = compute_eccentric_torque(quat_w, self.force_pos_delta, self.force_applied_w)
        return self.force_applied_w, torque_w

    def as_pre_step_wrench(self, force_body_ids: np.ndarray, torso_id: int):
        """Return a ``set_pre_step_wrench`` callback that recomputes the compliant
        wrench each physics substep and applies it via the Phase-1
        ``apply_body_wrench`` contract (force + torque) for the 6 force bodies plus
        the torso (net-wrench correction), reading the current pose from the backend.
        """
        force_body_ids = np.asarray(force_body_ids).reshape(-1)
        all_ids = np.concatenate([force_body_ids, [torso_id]])

        def fn(backend, *args) -> None:
            substep = getattr(self, "_force_substep", 0)
            pos6 = backend.get_body_pos_w(force_body_ids)  # (N,6,3)
            quat6 = backend.get_body_quat_w(force_body_ids)  # (N,6,4)
            root_quat = backend.get_base_quat()  # (N,4)
            torso_pos = backend.get_body_pos_w(np.array([torso_id]))[:, 0, :]  # (N,3)

            # GH step branch (motion_tracking.py:1045-1051): compliance -> force_apply;
            # non-compliant + max_force>0 -> force_apply_perturb; else (no_force) -> no wrench.
            if self.compliance:
                f6, tau6 = self.force_apply(substep, pos_w=pos6, quat_w=quat6, root_quat_w=root_quat)
            elif self.max_force > 0.0:
                f6, tau6 = self.force_apply_perturb(substep, quat_w=quat6, root_quat_w=root_quat)
            else:  # no_force: buffers stay zero, no external wrench
                if substep == 0:
                    self.force_applied_w[:] = 0.0
                    self.force_applied_b[:] = 0.0
                f6 = np.zeros((self.N, self.M, 3), dtype=np.float64)
                tau6 = np.zeros((self.N, self.M, 3), dtype=np.float64)

            d_f, d_m = limit_net_wrench_about_torso(
                pos6, f6, tau6, torso_pos, self.net_force_limit, self.net_torque_limit
            )
            force = np.concatenate([f6, d_f[:, None, :]], axis=1)  # (N,7,3)
            torque = np.concatenate([tau6, d_m[:, None, :]], axis=1)
            backend.apply_body_wrench(all_ids, force, torque)
            self._force_substep = substep + 1

        return fn

    def force_apply_perturb(self, substep, *, quat_w, root_quat_w):
        """Extreme non-compliant random perturbation force on the last 2 force bodies
        (GH ``force_apply_perturb`` :1016-1031). No admittance; the perturbation is a
        TemporalLerp value (constant within a control step). Returns (force_w, torque_w)
        each (N, M, 3); only the last 2 bodies carry force."""
        from unilab.utils.rotation import np_quat_apply_batched, np_quat_conjugate_batched

        self.force_applied_w[:, -2:] = self.perturb_force.current  # (N,2,3); first M-2 stay 0
        n = root_quat_w.shape[0]
        q = np.broadcast_to(root_quat_w[:, None, :], (n, self.M, 4))
        self.force_applied_b = np_quat_apply_batched(np_quat_conjugate_batched(q), self.force_applied_w)
        # eccentric torque = cross(quat_apply(quat_w, force_pos_delta), force_applied_w).
        # force_pos_delta is not resampled on the perturb path (GH), so it holds its reset value.
        torque_w = compute_eccentric_torque(quat_w, self.force_pos_delta, self.force_applied_w)
        return self.force_applied_w, torque_w

    def force_update_perturb_and_target(
        self, *, root_pos_w: np.ndarray, root_quat_w: np.ndarray, ref_keypoints_w: np.ndarray
    ) -> None:
        """Advance the non-compliant force curriculum + target (GH
        ``force_update_perturb_and_target`` :925-956). Runs once per control step in
        before_update for the non-compliant variants (no_force + extreme).

        Resample (trapezoid timing): timer<=0 -> transit U(20,50) + hold U(20,100), a new
        ``rand_points_isotropic(K, 2, max_force*force_alpha)`` perturbation with per-body
        ``rand<0.5`` enable, transitioned over ``transit`` steps. The force TARGET tracks
        the reference keypoints DIRECTLY (no admittance); ``force_keypoint_vel`` is the
        adjacent-control-step difference. ``ref_keypoints_w`` are the M force-body reference
        keypoints (world), (N, M, 3)."""
        from unilab.utils.rotation import np_quat_apply_batched, np_quat_conjugate_batched

        self.force_sample_timer -= 1
        need = np.nonzero(self.force_sample_timer <= 0)[0]
        if need.size > 0:
            transit = self._rng.integers(20, 50, size=need.size)
            hold = self._rng.integers(20, 100, size=need.size)
            self.force_sample_timer[need] = (transit + hold).astype(np.int32)
            force = rand_points_isotropic(
                need.size, 2, self.max_force * self.force_alpha, self._rng
            )  # (K, 2, 3)
            enable = (self._rng.random((need.size, 2)) < 0.5)[:, :, None]
            self.perturb_force.set(need, end=force * enable, total_steps=transit)

        # force target: track the reference keypoints directly (no admittance).
        self.force_keypoint_w[:] = np.asarray(ref_keypoints_w, dtype=np.float64)
        n = root_pos_w.shape[0]
        q = np.broadcast_to(root_quat_w[:, None, :], (n, self.M, 4))
        self.force_keypoint_b[:] = np_quat_apply_batched(
            np_quat_conjugate_batched(q), self.force_keypoint_w - root_pos_w[:, None, :]
        )
        if self.last_reset_env_ids is not None:
            self.force_keypoint_w_prev[self.last_reset_env_ids] = self.force_keypoint_w[
                self.last_reset_env_ids
            ]
        self.force_keypoint_vel_w[:] = (self.force_keypoint_w - self.force_keypoint_w_prev) / self.step_dt
        self.force_keypoint_w_prev[:] = self.force_keypoint_w




