"""A2 quadruped + P7v3 (5-DOF) arm + UMI gripper position-force control task.

Runs on the Unitree A2 quadruped carrying a P7v3 arm (joint3 upper-arm ROLL and
joint5 wrist ROLL frozen -> 5 active arm joints joint1,2,4,6,7) with a UMI
gripper. The legs plus arm joints (12 legs + 5 arm = 17 actuators; the arm count
is cfg-derived from ``asset.arm_joint_names``) are controlled by the RL policy
through a Python PD law fed to ``<motor>`` actuators, and the policy jointly
tracks end-effector position and commanded forces. Locomotion, the full
observation/command layout, the external-force push controller, the reward set,
and the CSE estimator targets are all implemented here.

Known gaps (deferred): rough-flat terrain randomization — this task runs on the
flat MJCF floor only.

Control law, recomputed every physics substep:

    target  = default_dof_pos + action * motor_strength * action_scale
    torque  = kp * (target - q) - kd * qd
    torque  = clip(torque, -torque_limit, torque_limit)

The end-effector goal is expressed in a yaw-aligned, gravity-referenced sphere
frame (terrain-relative z).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr.types import ResetPlan
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.a2arm.base import (
    A2ArmBaseCfg,
    A2ArmBaseEnv,
    A2ArmSensor,
    Asset,
    build_a2arm_position_gains,
)
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext, orientation, run_reward_dispatch
from unilab.utils.rotation import (
    np_quat_apply,
    np_quat_apply_inverse,
    np_yaw_quat,
)

# Command vector layout (15 dims).
CMD_VEL = slice(0, 3)  # base lin_vel_x, lin_vel_y, ang_vel_yaw
CMD_EE_POS = slice(3, 6)  # EE goal in sphere coords [radius, pitch, yaw]
CMD_EE_ORN = slice(6, 9)  # EE goal orientation [roll, pitch, yaw]
CMD_EE_FORCE = slice(9, 12)  # EE force command [fx, fy, fz]
CMD_BASE_FORCE = slice(12, 15)  # base force command [fx, fy, fz]
NUM_COMMANDS = 15

NUM_LEG = 12


def _default_model_file() -> str:
    return str(ASSETS_ROOT_PATH / "robots" / "a2arm" / "scene_pos_force.xml")


def _default_scene() -> SceneCfg:
    return SceneCfg(model_file=_default_model_file())


def sphere2cart(sphere: np.ndarray) -> np.ndarray:
    """Sphere [l, pitch, yaw] -> cart [x, y, z]."""
    l = sphere[..., 0]
    pitch = sphere[..., 1]
    yaw = sphere[..., 2]
    x = l * np.cos(pitch) * np.cos(yaw)
    y = l * np.cos(pitch) * np.sin(yaw)
    z = l * np.sin(pitch)
    return np.stack([x, y, z], axis=-1)


def cart2sphere(cart: np.ndarray) -> np.ndarray:
    """Cart [x, y, z] -> sphere [l, pitch, yaw]."""
    cart = np.asarray(cart)
    xy_len = np.linalg.norm(cart[..., :2], axis=-1)
    l = np.linalg.norm(cart, axis=-1)
    pitch = np.arctan2(cart[..., 2], xy_len)
    yaw = np.arctan2(cart[..., 1], cart[..., 0])
    return np.stack([l, pitch, yaw], axis=-1)


def _roll_pitch_from_quat(quat: np.ndarray) -> np.ndarray:
    """Extract (roll, pitch) from a (w, x, y, z) quaternion batch."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return np.stack([roll, pitch], axis=1)


def _soft_dof_pos_limits(hard: np.ndarray, soft: float) -> np.ndarray:
    """Shrink hard joint limits to the middle ``soft`` fraction about each
    midpoint: ``[m - 0.5*r*soft, m + 0.5*r*soft]``."""
    lower = hard[:, 0]
    upper = hard[:, 1]
    mid = (lower + upper) / 2.0
    half_range = (upper - lower) / 2.0
    return np.stack([mid - half_range * soft, mid + half_range * soft], axis=1)


def _expand_leg_torque_limit(value: float | list[float]) -> np.ndarray:
    """Expand a leg torque-limit spec to a ``(NUM_LEG,)`` array. Accepts a scalar
    (uniform), a length-3 per-leg ``(hip, thigh, calf)`` spec tiled across 4 legs,
    or a full ``length-NUM_LEG`` list used as-is (A2: hip/thigh 120, calf 180)."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full((NUM_LEG,), float(arr), dtype=np.float64)
    if arr.shape == (3,):
        return np.tile(arr, 4)
    if arr.shape == (NUM_LEG,):
        return arr
    raise ValueError(
        f"leg_torque_limit must be a scalar, length-3, or length-{NUM_LEG}; got shape {arr.shape}"
    )


class _ForceSchedule:
    """Per-env trapezoidal force episodes (ramp up -> hold -> ramp down).

    Each env idles for a random interval, then runs one force episode toward a
    random target. The episode ramps up over ``push_duration`` (sampled from
    ``duration_range``), HOLDS at the target for a FIXED ``settling`` window
    (gripper 0.5 s / base 1.0 s), then ramps down over ``push_duration`` again;
    total length is ``2 * push_duration + settling``. ``prob`` is the probability
    that any given episode actually fires; otherwise the force stays zero for
    that interval. The returned force is per-env ``(N, 3)`` in newtons.
    """

    def __init__(
        self,
        num_envs: int,
        mag_range: Sequence[float],
        interval_range: Sequence[int],
        duration_range: Sequence[int],
        prob: float,
        dtype: Any,
        z_scale: float = 1.0,
        settling: int = 0,
    ):
        self._n = num_envs
        self._mag = mag_range
        self._interval = interval_range
        self._duration = duration_range
        self._prob = float(prob)
        self._dtype = dtype
        # Fixed peak-hold window in control steps.
        self._settling = max(0, int(settling))
        # Attenuate / zero the z-component for base forces: base-ext z is scaled
        # by force_z_base_ext_scale (0.05); base-cmd z is 0.
        self._z_scale = float(z_scale)
        self.current = np.zeros((num_envs, 3), dtype=dtype)
        self._target = np.zeros((num_envs, 3), dtype=dtype)
        self._active = np.zeros((num_envs,), dtype=bool)
        self._elapsed = np.zeros((num_envs,), dtype=np.int32)
        self._ramp = np.ones((num_envs,), dtype=np.int32)
        self._hold = np.zeros((num_envs,), dtype=np.int32)
        self._timer = np.zeros((num_envs,), dtype=np.int32)
        self.reset(np.arange(num_envs, dtype=np.int32))

    def reset(self, env_ids: np.ndarray) -> None:
        n = len(env_ids)
        if n == 0:
            return
        self.current[env_ids] = 0.0
        self._target[env_ids] = 0.0
        self._active[env_ids] = False
        self._elapsed[env_ids] = 0
        self._timer[env_ids] = np.random.randint(self._interval[0], self._interval[1] + 1, size=n)

    def _start_episodes(self, env_ids: np.ndarray) -> None:
        n = len(env_ids)
        fire = np.random.uniform(size=n) < self._prob
        fire_ids = env_ids[fire]
        if len(fire_ids) > 0:
            mags = np.random.uniform(self._mag[0], self._mag[1], size=(len(fire_ids), 3))
            mags[:, 2] *= self._z_scale
            self._target[fire_ids] = mags.astype(self._dtype)
            # ``push_duration`` is the ramp length; the hold is a FIXED settling
            # window, independent of the sampled ramp.
            push_duration = np.random.randint(
                self._duration[0], self._duration[1] + 1, size=len(fire_ids)
            )
            self._ramp[fire_ids] = np.maximum(1, push_duration)
            self._hold[fire_ids] = self._settling
            self._elapsed[fire_ids] = 0
            self._active[fire_ids] = True
        # Non-firing envs simply re-arm their idle timer.
        idle_ids = env_ids[~fire]
        if len(idle_ids) > 0:
            self._timer[idle_ids] = np.random.randint(
                self._interval[0], self._interval[1] + 1, size=len(idle_ids)
            )

    def step(self, enabled: bool) -> np.ndarray:
        """Advance one control step and return the current force ``(N, 3)``."""
        if not enabled:
            self.current[:] = 0.0
            return self.current
        # Idle envs: count down; start an episode when the timer elapses.
        idle = ~self._active
        self._timer[idle] -= 1
        due = idle & (self._timer <= 0)
        if np.any(due):
            self._start_episodes(np.flatnonzero(due).astype(np.int32))
        # Active envs: evaluate the trapezoidal profile.
        active_ids = np.flatnonzero(self._active).astype(np.int32)
        if len(active_ids) > 0:
            e = self._elapsed[active_ids].astype(self._dtype)
            ramp = self._ramp[active_ids].astype(self._dtype)
            hold = self._hold[active_ids].astype(self._dtype)
            total = 2.0 * ramp + hold
            frac = np.ones_like(e)
            up = e < ramp
            frac[up] = e[up] / ramp[up]
            down = e >= (ramp + hold)
            frac[down] = np.clip((total[down] - e[down]) / ramp[down], 0.0, 1.0)
            self.current[active_ids] = self._target[active_ids] * frac[:, None]
            self._elapsed[active_ids] += 1
            done = self._elapsed[active_ids] >= total.astype(np.int32)
            if np.any(done):
                done_ids = active_ids[done]
                self._active[done_ids] = False
                self.current[done_ids] = 0.0
                self._target[done_ids] = 0.0
                self._timer[done_ids] = np.random.randint(
                    self._interval[0], self._interval[1] + 1, size=len(done_ids)
                )
        return self.current


# ── Configuration dataclasses ────────────────────────────────────────────────


@dataclass
class ObsScales:
    lin_vel: float = 2.0
    ang_vel: float = 0.25
    dof_pos: float = 1.0
    dof_vel: float = 0.05
    ee_force: float = 0.01
    base_force: float = 0.01
    ee_sphe_radius: float = 0.5
    ee_sphe_pitch: float = 1.0
    ee_sphe_yaw: float = 1.3
    ee_orn: float = 1.0


@dataclass
class PosForceNoiseConfig:
    # Noise is added to the ALREADY-SCALED observation, so each scale below is
    # noise_scales[x] * obs_scales[x] (with noise_level=1):
    #   dof_pos: 0.01 * 1.0  = 0.01   dof_vel: 1.5 * 0.05 = 0.075
    #   gravity/orn: 0.05    ang_vel: 0.2 * 0.25(applied later) -> 0.2 here
    level: float = 1.0
    scale_joint_angle: float = 0.01
    scale_joint_vel: float = 0.075
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_ang_vel: float = 0.2
    scale_orn: float = 0.05


@dataclass
class SphereCenter:
    x_offset: float = 0.2  # forward of base, in yaw frame
    y_offset: float = 0.0
    z_invariant_offset: float = 0.6  # terrain-relative height of the sphere center


@dataclass
class GoalEEConfig:
    command_mode: str = "sphere"
    arm_induced_pitch: float = 0.38
    sphere_center: SphereCenter = field(default_factory=SphereCenter)
    # Sphere sampling ranges [radius, pitch, yaw].
    pos_l: list[float] = field(default_factory=lambda: [0.25, 0.60])
    pos_p: list[float] = field(default_factory=lambda: [-2.0 * np.pi / 5.0, 2.0 * np.pi / 5.0])
    pos_y: list[float] = field(default_factory=lambda: [-3.0 * np.pi / 5.0, 3.0 * np.pi / 5.0])
    delta_orn_r: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    delta_orn_p: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    delta_orn_y: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    init_pos_start: list[float] = field(default_factory=lambda: [0.5, np.pi / 4.0, 0.0])
    init_pos_end: list[float] = field(default_factory=lambda: [0.5, 0.0, 0.0])
    traj_time: list[float] = field(default_factory=lambda: [1.0, 3.0])
    hold_time: list[float] = field(default_factory=lambda: [0.5, 2.0])
    collision_upper_limits: list[float] = field(default_factory=lambda: [0.25, 0.2, -0.15])
    collision_lower_limits: list[float] = field(default_factory=lambda: [-0.7, -0.2, -0.8])
    underground_limit: float = -0.7
    num_collision_check_samples: int = 10
    num_resample_attempts: int = 10


@dataclass
class PosForceCommandsConfig:
    lin_vel_x: list[float] = field(default_factory=lambda: [-0.6, 0.6])
    lin_vel_y: list[float] = field(default_factory=lambda: [-0.4, 0.4])
    ang_vel_yaw: list[float] = field(default_factory=lambda: [-0.6, 0.6])
    lin_vel_x_clip: float = 0.1
    lin_vel_y_clip: float = 0.1
    ang_vel_yaw_clip: float = 0.2
    zero_vel_cmd_prob: float = 0.3
    zero_vel_cmd_prob_after_force: float = 0.8
    resample_time_s: float = 5.0
    # Force command magnitudes (N). cmd = commanded force the policy must exert
    # (drives the EE/base setpoint offset); ext = external disturbance force
    # applied to the body (becomes the realized force in the observation).
    max_push_force_xyz_gripper_cmd: list[float] = field(default_factory=lambda: [-18.0, 18.0])
    max_push_force_xyz_gripper_ext: list[float] = field(default_factory=lambda: [-15.0, 15.0])
    max_push_force_xyz_base_cmd: list[float] = field(default_factory=lambda: [-25.0, 25.0])
    max_push_force_xyz_base_ext: list[float] = field(default_factory=lambda: [-20.0, 20.0])
    # Force-episode timing in seconds [min, max]. DISTINCT cmd vs ext idle
    # intervals per body; ``*_duration_s`` is the ramp (push_duration) and
    # ``settling_time_force_*_s`` is the fixed peak-hold window.
    push_gripper_interval_s_cmd: list[float] = field(default_factory=lambda: [5.0, 10.0])
    push_gripper_interval_s_ext: list[float] = field(default_factory=lambda: [6.0, 12.0])
    push_gripper_duration_s: list[float] = field(default_factory=lambda: [0.5, 1.5])
    push_base_interval_s_cmd: list[float] = field(default_factory=lambda: [5.0, 10.0])
    push_base_interval_s_ext: list[float] = field(default_factory=lambda: [8.0, 14.0])
    push_base_duration_s: list[float] = field(default_factory=lambda: [0.5, 1.5])
    # Fixed peak-hold ("settling") time per body.
    settling_time_force_gripper_s: float = 0.5
    settling_time_force_base_s: float = 1.0
    gripper_forced_prob_cmd: float = 0.8
    gripper_forced_prob_ext: float = 0.8
    base_forced_prob_cmd: float = 0.5
    base_forced_prob_ext: float = 0.5
    # Force/compliance stiffness used to offset the EE position / base velocity
    # setpoint: ee_goal += F/gripper_force_kp, base_vel_cmd += F/base_force_kd.
    gripper_force_kp: float = 300.0
    base_force_kd: float = 200.0
    # Attenuate the external base force's vertical component to 5%.
    force_z_base_ext_scale: float = 0.05
    # Gripper force vertical (z) attenuation. The arm's vertical wrench capacity at
    # extended reach is ~19 N (shoulder-pitch torque saturates: long moment arm +
    # gravity pre-load), vs 40+ N horizontal — so full vertical gripper force
    # saturates the arm and destabilizes training. Scale <1 caps the vertical
    # component; applied to BOTH cmd and ext so the reward setpoint (ext+cmd)/kp
    # stays consistent with the reduced physical force. 1.0 = no change (default).
    force_z_gripper_cmd_scale: float = 1.0
    force_z_gripper_ext_scale: float = 1.0
    # Force curriculum: external forces start after this many CONTROL steps
    # (144000 = iteration 6000 * num_steps_per_env 24).
    force_start_step: int = 144000
    # Force-MAGNITUDE curriculum (staircase). After force_start_step, all four
    # force channels (gripper/base × cmd/ext) are multiplied by a stage factor so
    # the magnitude grows in gears instead of jumping to full. ``scales`` is the
    # per-gear factor (e.g. [0.33, 0.66, 1.0]); ``stage_steps`` is how many CONTROL
    # steps each gear lasts. EMPTY scales = DISABLED = full force from force_start.
    # Applying the SAME factor to cmd and ext keeps the reward/obs setpoint offset
    # (ext+cmd)/kp consistent with the applied physical force, so early gears are a
    # weaker version of the SAME task.
    force_curriculum_scales: list[float] = field(default_factory=list)
    force_curriculum_stage_steps: int = 1


@dataclass
class PosForceControlConfig:
    action_scale: float = 0.25
    arm_action_scale: float = 0.25
    simulate_action_latency: bool = False
    # 3-control-step action delay (sim2real latency model).
    action_delay_steps: int = 3
    # PD gains; reused via build_a2arm_position_gains.
    Kp: float = 35.0
    Kd: float = 0.5
    leg_kp: float | list[float] = 60.0
    leg_kd: float | list[float] = 2.0
    arm_kp: list[float] = field(default_factory=lambda: [95.0, 115.0, 100.0, 52.0, 54.0, 55.0])
    arm_kd: list[float] = field(default_factory=lambda: [3.5, 3.8, 2.5, 1.5, 1.5, 1.5])
    # Per-joint torque limits (legs + joint1-3: 24 Nm, wrist joint4-6: 8 Nm).
    leg_torque_limit: float | list[float] = 24.0
    arm_torque_limit: list[float] = field(default_factory=lambda: [24.0, 24.0, 24.0, 8.0, 8.0, 8.0])


@dataclass
class PosForceDomainRandConfig:
    # Field names MUST match the shared DR applier contract in unilab.dr.dr_utils
    # (it reads them via getattr; unrecognized names silently no-op).
    randomize_ground_friction: bool = True
    # Multiplier on the model's default ground friction (default ~1.0 => ~[0.5, 1.8]).
    ground_friction_multiplier_range: list[float] = field(default_factory=lambda: [0.5, 1.8])
    # The foot geom has priority=1 in the MJCF, so it OVERRIDES the floor friction
    # at the contact -> randomizing the ground (above) never reaches the feet.
    # Randomize the FOOT friction directly (absolute per-env range) so contact
    # friction actually varies.
    randomize_foot_friction: bool = True
    foot_friction_range: list[float] = field(default_factory=lambda: [0.5, 1.8])
    randomize_base_mass: bool = True
    added_mass_range: list[float] = field(default_factory=lambda: [0.0, 4.0])
    random_com: bool = True
    com_offset_x: list[float] = field(default_factory=lambda: [-0.05, 0.05])
    com_offset_y: list[float] = field(default_factory=lambda: [-0.05, 0.05])
    com_offset_z: list[float] = field(default_factory=lambda: [-0.05, 0.05])
    randomize_motor_strength: bool = True
    leg_motor_strength_range: list[float] = field(default_factory=lambda: [0.85, 1.15])
    arm_motor_strength_range: list[float] = field(default_factory=lambda: [0.85, 1.15])
    # Gripper payload mass (0-0.12 kg added to the wrist/gripper link).
    randomize_gripper_mass: bool = True
    gripper_added_mass_range: list[float] = field(default_factory=lambda: [0.0, 0.12])
    # Base-velocity impulse push: every ``push_interval`` control steps, overwrite
    # the base x/y linear velocity with U(-vmax, vmax). Applied in-env by
    # ``A2ArmPosForceEnv._maybe_apply_velocity_push``. The shared force-based
    # interval push (build_interval_push_plan) is left OFF because a small xfrc
    # force on a ~15 kg base is a near-no-op; ``push_robots`` gates only that path.
    push_robots: bool = False
    velocity_push: bool = True
    push_interval: int = 400
    max_push_vel_xy: float = 0.3
    # Standing-env push amplification; 1.0 = no amplification (default).
    velocity_push_standing_scale: float = 1.0
    push_body_name: str = "base"


@dataclass
class GaitConfig:
    cycle_time: float = 0.64
    target_joint_pos_scale: float = 0.17
    target_joint_pos_thd: float = 0.5


@dataclass
class RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float = 0.25
    tracking_ee_sigma: float = 1.0
    base_height_target: float = 0.40
    soft_dof_pos_limit: float = 0.8
    soft_torque_limit: float = 0.9
    max_contact_force: float = 200.0
    # Leg gait-reference tracking: exp(-L1_err * ref_dof_scale).
    ref_dof_scale: float = 0.1
    # stand_still: exp(-L1_err * stand_still_scale) at zero command.
    stand_still_scale: float = 0.05
    feet_air_time_threshold: float = 0.5
    feet_height_target: float = 0.10
    feet_height_high_target: float = 0.20


@dataclass
class HistoryConfig:
    # Actor sees a long proprioceptive history (frame_stack=32); the CSE
    # estimator compresses it. The critic sees the current privileged step, so
    # the estimator target is critic_obs[:, :12] (estimator.target_start=0).
    num_actor_history: int = 32
    num_critic_history: int = 1


@dataclass
class ResetConfig:
    leg_pos_scale_range: list[float] = field(default_factory=lambda: [0.5, 1.5])
    arm_init_dof_pos_range: float = 0.3
    yaw_range: float = np.pi / 2.0


@registry.envcfg("A2ArmPosForce")
@dataclass
class A2ArmPosForceCfg(A2ArmBaseCfg):
    """A2 + P7v3 (joint3+joint5-frozen, 5-DOF) + UMI gripper v3 pos-force task.

    Self-contained flattened config. Active arm joints, in tree/qpos/actuator
    order, are joint1,2,4,6,7 (joint3 and joint5 dropped); every length-5 arm
    array below is indexed in that order. Reward weights + the force curriculum
    live in the task YAML.
    """

    scene: SceneCfg = field(default_factory=_default_scene)
    model_file: str = field(default_factory=_default_model_file)
    max_episode_seconds: float = 20.0
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02
    # Normalization clamps: cap the raw action and the final observation,
    # preventing the action_rate / obs feedback runaway when a policy update
    # overshoots. None disables.
    clip_actions: float | None = 100.0
    clip_observations: float | None = 100.0
    obs_scales: ObsScales = field(default_factory=ObsScales)
    noise_config: PosForceNoiseConfig = field(default_factory=PosForceNoiseConfig)  # type: ignore[assignment]
    # Active arm joints joint1,2,4,6,7 (joint3 and joint5 frozen). This drives the
    # arm dof-index lookups, so it MUST list exactly the hinges present in
    # a2arm.xml, in tree order.
    asset: Asset = field(  # type: ignore[assignment]
        default_factory=lambda: Asset(
            base_name="base_link",
            ee_body_name="end_link",
            arm_joint_names=("joint1", "joint2", "joint4", "joint6", "joint7"),
        )
    )
    control_config: PosForceControlConfig = field(  # type: ignore[assignment]
        default_factory=lambda: PosForceControlConfig(
            # A2 leg torque capacity: hip/thigh 120, calf 180 Nm.
            leg_kp=[100.0, 100.0, 150.0],
            leg_kd=[4.0, 4.0, 6.0],
            leg_torque_limit=[120.0, 120.0, 180.0],
            # P7v3 arm, order joint1,2,4,6,7 (j1/j2/j4 positioning, j6/j7 wrist).
            # forcerange contract: arm_torque_limit MUST equal the a2arm.xml motor
            # forceranges (j1/j2/j4=30, j6/j7=10) or MuJoCo re-clamps the env-PD
            # torque.
            arm_kp=[90.0, 120.0, 70.0, 30.0, 30.0],
            arm_kd=[5.5, 10.5, 5.5, 1.0, 1.0],
            arm_torque_limit=[30.0, 30.0, 30.0, 10.0, 10.0],
        )
    )
    commands: PosForceCommandsConfig = field(
        default_factory=lambda: PosForceCommandsConfig(
            # Base force OFF by default here; the task YAML re-enables it with its
            # own magnitudes.
            force_start_step=192000,  # 8000 iter x 24 steps
            max_push_force_xyz_gripper_cmd=[-30.0, 30.0],
            max_push_force_xyz_gripper_ext=[-30.0, 30.0],
            max_push_force_xyz_base_cmd=[-60.0, 60.0],
            max_push_force_xyz_base_ext=[-50.0, 50.0],
            base_forced_prob_cmd=0.0,
            base_forced_prob_ext=0.0,
        )
    )
    goal_ee: GoalEEConfig = field(
        default_factory=lambda: GoalEEConfig(
            # Terrain-relative sphere center at shoulder height. Must stay in sync
            # with the keyframe base z so the EE-to-center vector (and thus
            # init_pos_start/end and the arm keyframe) stays FK self-consistent.
            sphere_center=SphereCenter(x_offset=0.2, y_offset=0.0, z_invariant_offset=0.735),
            # P7v3 (5-DOF) reach: sampling range IK-verified safe.
            pos_l=[0.28, 0.55],
            pos_p=[np.radians(-60), np.radians(72)],
            pos_y=[np.radians(-108), np.radians(108)],
            # Home keyframe FK (joint1,2,4,6,7 = -0.0461,-0.7427,-1.069,0.1581,
            # -0.4808): end_link lands at sphere [0.4494 m, 0.8115 rad, 0] from the
            # center above (FK self-consistency verified by test_a2arm_home_pose_fk).
            init_pos_start=[0.4494, 0.8115, 0.0],
            init_pos_end=[0.4494, 0.0, 0.0],
        )
    )
    domain_rand: PosForceDomainRandConfig = field(
        default_factory=lambda: PosForceDomainRandConfig(
            push_body_name="base_link",
            added_mass_range=[0.0, 4.0],
            com_offset_x=[-0.08, 0.08],
            com_offset_y=[-0.08, 0.08],
            com_offset_z=[-0.08, 0.08],
            # UMI gripper mass ~0.3 kg already modeled; randomize payload 0-0.15 kg.
            gripper_added_mass_range=[0.0, 0.10],
        )
    )
    # Longer A2 legs -> slightly slower stride, larger swing amplitude.
    gait: GaitConfig = field(
        default_factory=lambda: GaitConfig(cycle_time=0.70, target_joint_pos_scale=0.18)
    )
    reward_config: RewardConfig | None = None
    history: HistoryConfig = field(default_factory=HistoryConfig)
    reset: ResetConfig = field(default_factory=ResetConfig)
    sensor: A2ArmSensor = field(default_factory=A2ArmSensor)  # type: ignore[assignment]


# ── Domain randomization provider ────────────────────────────────────────────


class A2ArmPosForceDRProvider(LocomotionDRProvider):
    """Reset/interval DR for the pos-force task.

    Velocity commands are sampled into the 15-dim command vector; motor-strength
    is owned by the env (sampled here and cached) so it can scale the action in
    the Python PD law and feed the privileged observation. EE goals and history
    buffers are reset through env hooks.
    """

    def __init__(
        self,
        *,
        base_body_mass: np.ndarray | None = None,
        base_geom_friction: np.ndarray | None = None,
        ground_geom_id: int | None = None,
    ):
        self._base_body_mass = base_body_mass
        self._base_geom_friction = base_geom_friction
        self._ground_geom_id = ground_geom_id

    def _get_reset_randomization_baselines(
        self, env: Any
    ) -> tuple[np.ndarray | None, np.ndarray | None, int | None, np.ndarray | None]:
        return (self._base_body_mass, self._base_geom_friction, self._ground_geom_id, None)

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        return env._sample_velocity_commands(num_reset)

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        plan = super().build_reset_plan(env, env_ids)
        env._apply_reset_dofs(plan.qpos, env_ids)
        env._sample_motor_strength(env_ids)
        env._apply_gripper_mass_dr(plan.randomization, env_ids)
        env._apply_foot_friction_dr(plan.randomization, env_ids)
        env._cache_reset_dr(env_ids, plan.randomization)
        env.reset_ee_goals(env_ids, plan.info_updates["commands"])
        env._reset_force_schedules(env_ids)
        env._history_obs_buf[env_ids] = 0.0
        env._history_critic_buf[env_ids] = 0.0
        env._gait_indices[env_ids] = 0.0
        env._feet_air_time[env_ids] = 0.0
        env._last_contacts[env_ids] = False
        env._first_contact[env_ids] = False
        env._last_dof_vel[env_ids] = 0.0
        env._action_history[env_ids] = 0.0
        return plan

    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: np.ndarray,
        info_updates: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        sliced_info: dict[str, Any] = {}
        for k, v in info_updates.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == env._num_envs:
                sliced_info[k] = v[env_ids]
            else:
                sliced_info[k] = v
        actor_raw, critic_raw = env._compute_raw_obs(sliced_info, env_ids=env_ids)
        return cast(
            "dict[str, np.ndarray]",
            env._update_history(actor_raw, critic_raw, env_ids=env_ids),
        )


# ── Environment ──────────────────────────────────────────────────────────────


@registry.env("A2ArmPosForce", sim_backend="mujoco")
class A2ArmPosForceEnv(A2ArmBaseEnv):
    _cfg: A2ArmPosForceCfg

    def __init__(self, cfg: A2ArmPosForceCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        if backend_type != "mujoco":
            raise ValueError(f"A2ArmPosForce currently supports only mujoco, got {backend_type!r}")

        # A2Arm P7v3/adapter meshes are hosted on Hugging Face. Resolve them on
        # this cold initialization path before MuJoCo parses the scene XML.
        from unilab.assets.hub import resolve_robot_asset_dir

        resolve_robot_asset_dir("robots/a2arm/meshes", marker="adapter_plate.STL")

        scene = cfg.scene if cfg.scene is not None else SceneCfg(model_file=cfg.model_file)
        backend = create_backend(
            backend_type,
            scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
            # Forward the thread-pool chunk_size config. Without this,
            # create_backend defaults adaptive_chunk_size to False (overriding the
            # cfg default of True), so chunk tuning never runs and mujoco.rollout
            # falls back to a non-tuned default -- leaving physics throughput on
            # the table. chunk_size only affects env->thread partitioning; physics
            # results are identical. bench_nsteps is left at its default (1) to
            # match this task's external-PD slow path, which calls pool.step with
            # nstep=1 per substep.
            chunk_size=cfg.chunk_size,
            adaptive_chunk_size=cfg.adaptive_chunk_size,
        )
        super().__init__(cfg, backend, num_envs)
        self._num_arm = len(cfg.asset.arm_joint_names)
        if self._num_action != NUM_LEG + self._num_arm:
            raise ValueError(
                f"A2ArmPosForce expects {NUM_LEG + self._num_arm} actuators "
                f"(NUM_LEG={NUM_LEG} + arm_joint_names={self._num_arm}), got {self._num_action}"
            )

        dtype = get_global_dtype()
        self._np_dtype = dtype
        self._gravity_sensor = "upvector"
        # ±100 action/observation clamps (None disables either).
        self._clip_actions = None if cfg.clip_actions is None else float(cfg.clip_actions)
        self._clip_obs = None if cfg.clip_observations is None else float(cfg.clip_observations)
        self._reward_cfg = cfg.reward_config
        self._enable_reward_log = True

        # PD law tables (NUM_LEG + num_arm,).
        gains = build_a2arm_position_gains(cfg.control_config, self._num_arm)  # type: ignore[arg-type]
        self._kp = gains["kp"].astype(np.float64)
        self._kd = gains["kd"].astype(np.float64)
        self._torque_limits = np.concatenate(
            [
                _expand_leg_torque_limit(cfg.control_config.leg_torque_limit),
                np.asarray(cfg.control_config.arm_torque_limit, dtype=np.float64),
            ]
        )
        self._action_scale = np.concatenate(
            [
                np.full((NUM_LEG,), cfg.control_config.action_scale, dtype=dtype),
                np.full((self._num_arm,), cfg.control_config.arm_action_scale, dtype=dtype),
            ]
        )
        self._motor_strength = np.ones((num_envs, self._num_action), dtype=dtype)
        # Action-delay ring buffer (action_delay control steps).
        self._action_delay_steps = max(0, int(cfg.control_config.action_delay_steps))
        self._action_history = np.zeros(
            (num_envs, self._action_delay_steps + 1, self._num_action), dtype=dtype
        )
        self._last_torque = np.zeros((num_envs, self._num_action), dtype=dtype)
        self._motor_targets = np.tile(self.default_angles.astype(dtype), (num_envs, 1))
        self._backend.set_pre_step_control(self._pre_step_pd_control)

        # Privileged DR readback caches.
        self._dr_friction = np.zeros((num_envs, 1), dtype=dtype)
        self._dr_base_mass = np.zeros((num_envs, 1), dtype=dtype)
        self._dr_base_com = np.zeros((num_envs, 3), dtype=dtype)
        self._dr_gripper_mass = np.zeros((num_envs, 1), dtype=dtype)
        # (DR readback caches above are populated at every reset.)
        # Gripper-mass DR: resolve the wrist/gripper body and cache the full
        # per-body mass table so we can inject a per-env mass delta at reset.
        self._gripper_body_id = int(self._backend.get_body_ids([cfg.asset.ee_body_name])[0])
        self._full_body_mass = self._backend.get_body_mass().astype(np.float64)

        # EE goal buffers.
        self._init_ee_goal_buffers(num_envs)
        self._arm_base_offset = np.asarray([0.0045, 0.0, 0.0945], dtype=dtype)
        self._ee_center_offset = np.asarray(
            [
                cfg.goal_ee.sphere_center.x_offset,
                cfg.goal_ee.sphere_center.y_offset,
                cfg.goal_ee.sphere_center.z_invariant_offset,
            ],
            dtype=dtype,
        )

        # Gait buffers.
        self._gait_indices = np.zeros((num_envs,), dtype=dtype)
        self._ref_dof_pos = np.tile(self.default_angles[:NUM_LEG].astype(dtype), (num_envs, 1))
        self._stance_mask = np.zeros((num_envs, 4), dtype=dtype)

        # Feet / dof-acceleration reward state.
        num_feet = len(cfg.sensor.feet_force)
        self._foot_contact = np.zeros((num_envs, num_feet), dtype=bool)
        self._last_contacts = np.zeros((num_envs, num_feet), dtype=bool)
        self._first_contact = np.zeros((num_envs, num_feet), dtype=bool)
        self._feet_air_time = np.zeros((num_envs, num_feet), dtype=dtype)
        self._air_time_snapshot = np.zeros((num_envs, num_feet), dtype=dtype)
        self._last_dof_vel = np.zeros((num_envs, self._num_action), dtype=dtype)
        # Hard joint position limits (18 actuated joints), for dof_pos_limits.
        joint_range = self._backend.get_joint_range()
        if joint_range is None:
            raise ValueError("backend.get_joint_range() returned None; required for dof_pos_limits")
        # Penalize dof_pos against SOFT limits (middle soft_dof_pos_limit of each
        # range), not the hard joint limits, so the -10 penalty bites earlier.
        # _reward_dof_pos_limits reads these.
        hard_dof_pos_limits = np.asarray(joint_range, dtype=dtype)[: self._num_action]
        self._dof_pos_limits = _soft_dof_pos_limits(
            hard_dof_pos_limits, self._reward_cfg.soft_dof_pos_limit
        )

        # Force application bodies (resolved once, cold path).
        self._ee_body_id = int(self._backend.get_body_ids([cfg.asset.ee_body_name])[0])
        self._base_body_id = int(self._backend.get_body_ids([cfg.asset.base_name])[0])
        self._force_body_ids = np.asarray([self._ee_body_id, self._base_body_id], dtype=np.int32)
        self._gripper_force_kp = float(cfg.commands.gripper_force_kp)
        self._base_force_kd = float(cfg.commands.base_force_kd)
        self._force_start_step = int(cfg.commands.force_start_step)
        # Force-magnitude curriculum (empty scales = disabled = full force).
        self._force_curriculum_scales = [float(s) for s in cfg.commands.force_curriculum_scales]
        self._force_curriculum_stage_steps = max(1, int(cfg.commands.force_curriculum_stage_steps))

        # Realized external force buffers (world frame) and commanded forces.
        self._force_ee_world = np.zeros((num_envs, 3), dtype=dtype)
        self._force_base_world = np.zeros((num_envs, 3), dtype=dtype)
        self._force_ee_cmd = np.zeros((num_envs, 3), dtype=dtype)  # yaw-local frame
        self._force_base_cmd = np.zeros((num_envs, 3), dtype=dtype)  # yaw-local frame

        ctrl_dt = cfg.ctrl_dt
        cmds = cfg.commands

        def _steps(seconds: list[float]) -> tuple[int, int]:
            return (max(1, int(seconds[0] / ctrl_dt)), max(1, int(seconds[1] / ctrl_dt)))

        # Distinct cmd vs ext idle intervals per body; shared ramp
        # (push_duration) and a fixed settling-time hold per body.
        g_int_cmd = _steps(cmds.push_gripper_interval_s_cmd)
        g_int_ext = _steps(cmds.push_gripper_interval_s_ext)
        b_int_cmd = _steps(cmds.push_base_interval_s_cmd)
        b_int_ext = _steps(cmds.push_base_interval_s_ext)
        g_dur = _steps(cmds.push_gripper_duration_s)
        b_dur = _steps(cmds.push_base_duration_s)
        g_settle = max(0, int(cmds.settling_time_force_gripper_s / ctrl_dt))
        b_settle = max(0, int(cmds.settling_time_force_base_s / ctrl_dt))
        self._sched_gripper_cmd = _ForceSchedule(
            num_envs,
            tuple(cmds.max_push_force_xyz_gripper_cmd),
            g_int_cmd,
            g_dur,
            cmds.gripper_forced_prob_cmd,
            dtype,
            z_scale=float(cmds.force_z_gripper_cmd_scale),
            settling=g_settle,
        )
        self._sched_gripper_ext = _ForceSchedule(
            num_envs,
            tuple(cmds.max_push_force_xyz_gripper_ext),
            g_int_ext,
            g_dur,
            cmds.gripper_forced_prob_ext,
            dtype,
            z_scale=float(cmds.force_z_gripper_ext_scale),
            settling=g_settle,
        )
        self._sched_base_cmd = _ForceSchedule(
            num_envs,
            tuple(cmds.max_push_force_xyz_base_cmd),
            b_int_cmd,
            b_dur,
            cmds.base_forced_prob_cmd,
            dtype,
            z_scale=0.0,  # commanded base force z-component is zeroed.
            settling=b_settle,
        )
        self._sched_base_ext = _ForceSchedule(
            num_envs,
            tuple(cmds.max_push_force_xyz_base_ext),
            b_int_ext,
            b_dur,
            cmds.base_forced_prob_ext,
            dtype,
            z_scale=float(cmds.force_z_base_ext_scale),
            settling=b_settle,
        )

        # Command resample timer.
        self._cmd_resample_steps = max(1, int(cfg.commands.resample_time_s / cfg.ctrl_dt))
        self._cmd_timer = np.random.randint(
            0, self._cmd_resample_steps, size=(num_envs,), dtype=np.int32
        )

        self._commands_scale = self._build_commands_scale()
        self._init_reward_functions()

        # Single-step obs dims.
        self._actor_one_step_dim = self._actor_single_obs_dim()
        self._critic_one_step_dim = self._critic_single_obs_dim()
        H_a = cfg.history.num_actor_history
        H_c = cfg.history.num_critic_history
        self._history_obs_buf = np.zeros((num_envs, H_a * self._actor_one_step_dim), dtype=dtype)
        self._history_critic_buf = np.zeros(
            (num_envs, H_c * self._critic_one_step_dim), dtype=dtype
        )

        base_body_mass = backend.get_body_mass() if cfg.domain_rand.randomize_base_mass else None
        base_geom_friction = None
        ground_geom_id = None
        if cfg.domain_rand.randomize_ground_friction:
            base_geom_friction = backend.get_geom_friction()
            ground_geom_id = backend.get_geom_id(cfg.asset.ground)
        self._ground_geom_id = ground_geom_id
        # Foot collision geoms (priority=1) govern the foot-floor contact friction,
        # so the friction DR must target THESE, not the floor. Cache their ids and
        # the full geom-friction table (used to seed the payload when ground DR is
        # off). The privileged friction readback also reads a foot geom.
        self._base_geom_friction_full = backend.get_geom_friction()
        self._foot_geom_ids = np.asarray(
            [backend.get_geom_id(name) for name in ("FL", "FR", "RL", "RR")], dtype=np.int32
        )
        self._init_domain_randomization(
            A2ArmPosForceDRProvider(
                base_body_mass=base_body_mass,
                base_geom_friction=base_geom_friction,
                ground_geom_id=ground_geom_id,
            )
        )

    # ── obs spec ──────────────────────────────────────────────────────────

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        H_a = self._cfg.history.num_actor_history
        H_c = self._cfg.history.num_critic_history
        return {
            "obs": H_a * self._actor_one_step_dim,
            "critic": H_c * self._critic_one_step_dim,
        }

    def _actor_single_obs_dim(self) -> int:
        # body_orient(2) + ang_vel(3) + dof_diff(num_action) + dof_vel(num_action)
        # + last_actions(num_action) + sin/cos(2) + commands(15)
        return int(
            2 + 3 + self._num_action + self._num_action + self._num_action + 2 + NUM_COMMANDS
        )

    def _critic_single_obs_dim(self) -> int:
        # CSE targets front block (12): base_lin_vel(3) + ee_pos_sphere(3)
        #   + force_ee(3) + force_base(3)
        # then: leg_ref_diff(12) + dr_block(6) + motor_strength(num_action) + stance(4)
        #   + contact_mask(4) + proj_gravity(3) + ee_goal_offset_sphere(3)
        #   + actor_core(...)
        cse = 12
        privileged = NUM_LEG + 6 + self._num_action + 4 + 4 + 3 + 3
        return int(cse + privileged + self._actor_single_obs_dim())

    # ── EE goal sampling (yaw + gravity frame) ────────────────────────────

    def _init_ee_goal_buffers(self, num_envs: int) -> None:
        dtype = get_global_dtype()
        self._curr_ee_goal_sphere = np.zeros((num_envs, 3), dtype=dtype)
        self._curr_ee_goal_cart = np.zeros((num_envs, 3), dtype=dtype)
        self._curr_ee_goal_world = np.zeros((num_envs, 3), dtype=dtype)
        self._ee_start_sphere = np.zeros((num_envs, 3), dtype=dtype)
        self._ee_goal_sphere = np.zeros((num_envs, 3), dtype=dtype)
        self._ee_goal_orn_euler = np.zeros((num_envs, 3), dtype=dtype)
        self._arm_goal_timer = np.zeros((num_envs,), dtype=np.int32)
        self._traj_steps = np.ones((num_envs,), dtype=np.int32)
        self._traj_total_steps = np.ones((num_envs,), dtype=np.int32)

    def get_ee_goal_spherical_center(self) -> np.ndarray:
        """Sphere center: base XY (yaw frame) + offset, terrain-relative z."""
        base_pos = self._backend.get_base_pos()
        base_yaw_quat = np_yaw_quat(self._backend.get_base_quat())
        center = np.zeros((self._num_envs, 3), dtype=self._np_dtype)
        center[:, :2] = base_pos[:, :2]
        center = center + np_quat_apply(
            base_yaw_quat, np.tile(self._ee_center_offset, (self._num_envs, 1))
        )
        return center

    def _collision_check(self, starts: np.ndarray, goals: np.ndarray) -> np.ndarray:
        cfg = self._cfg.goal_ee
        n = max(2, cfg.num_collision_check_samples)
        t = np.linspace(0.0, 1.0, n, dtype=self._np_dtype)
        path_sphere = starts[:, None, :] + (goals - starts)[:, None, :] * t[None, :, None]
        path_cart = sphere2cart(path_sphere.reshape(-1, 3)).reshape(len(starts), n, 3)
        upper = np.asarray(cfg.collision_upper_limits, dtype=self._np_dtype)
        lower = np.asarray(cfg.collision_lower_limits, dtype=self._np_dtype)
        inside = np.all(path_cart < upper, axis=2) & np.all(path_cart > lower, axis=2)
        underground = np.any(path_cart[..., 2] < float(cfg.underground_limit), axis=1)
        return np.any(inside, axis=1) | underground

    def _resample_ee_goal_sphere(self, env_ids: np.ndarray) -> None:
        cfg = self._cfg.goal_ee
        n = len(env_ids)
        start = self._ee_start_sphere[env_ids]
        candidates = np.tile(np.asarray(cfg.init_pos_end, dtype=self._np_dtype), (n, 1))
        remaining = np.arange(n, dtype=np.int32)
        for _ in range(max(1, cfg.num_resample_attempts)):
            m = len(remaining)
            l = np.random.uniform(*cfg.pos_l, size=m).astype(self._np_dtype)
            p = np.random.uniform(*cfg.pos_p, size=m).astype(self._np_dtype)
            y = np.random.uniform(*cfg.pos_y, size=m).astype(self._np_dtype)
            new_goals = np.stack([l, p, y], axis=1)
            candidates[remaining] = new_goals
            unsafe = self._collision_check(start[remaining], new_goals)
            remaining = remaining[unsafe]
            if len(remaining) == 0:
                break
        self._ee_goal_sphere[env_ids] = candidates

    def reset_ee_goals(self, env_ids: np.ndarray, commands: np.ndarray) -> None:
        """Reset EE goal trajectory: start at init_pos_start, sample an end goal."""
        env_ids = np.asarray(env_ids, dtype=np.int32).reshape(-1)
        if len(env_ids) == 0:
            return
        cfg = self._cfg.goal_ee
        self._ee_start_sphere[env_ids] = np.asarray(cfg.init_pos_start, dtype=self._np_dtype)
        self._curr_ee_goal_sphere[env_ids] = self._ee_start_sphere[env_ids]
        self._resample_ee_goal_sphere(env_ids)
        dt = self._cfg.ctrl_dt
        traj_t = np.random.uniform(*cfg.traj_time, size=len(env_ids))
        hold_t = np.random.uniform(*cfg.hold_time, size=len(env_ids))
        traj_s = np.maximum(1, np.round(traj_t / dt).astype(np.int32))
        hold_s = np.maximum(0, np.round(hold_t / dt).astype(np.int32))
        self._traj_steps[env_ids] = traj_s
        self._traj_total_steps[env_ids] = traj_s + hold_s
        self._arm_goal_timer[env_ids] = 0
        # Write the starting sphere command into the (sliced) command vector.
        commands[:, CMD_EE_POS] = self._curr_ee_goal_sphere[env_ids]

    def _update_ee_goal_trajectory(self, commands: np.ndarray) -> None:
        """Advance the EE goal along start->goal, then resample after the hold."""
        timer = self._arm_goal_timer
        traj = self._traj_steps.astype(self._np_dtype)
        t = np.clip(timer.astype(self._np_dtype) / np.maximum(traj, 1.0), 0.0, 1.0)
        self._curr_ee_goal_sphere = (
            self._ee_start_sphere + (self._ee_goal_sphere - self._ee_start_sphere) * t[:, None]
        )
        done = timer >= self._traj_total_steps
        if np.any(done):
            done_ids = np.flatnonzero(done).astype(np.int32)
            self._ee_start_sphere[done_ids] = self._ee_goal_sphere[done_ids]
            self._resample_ee_goal_sphere(done_ids)
            self._arm_goal_timer[done_ids] = 0
            dt = self._cfg.ctrl_dt
            cfg = self._cfg.goal_ee
            traj_t = np.random.uniform(*cfg.traj_time, size=len(done_ids))
            hold_t = np.random.uniform(*cfg.hold_time, size=len(done_ids))
            traj_s = np.maximum(1, np.round(traj_t / dt).astype(np.int32))
            hold_s = np.maximum(0, np.round(hold_t / dt).astype(np.int32))
            self._traj_steps[done_ids] = traj_s
            self._traj_total_steps[done_ids] = traj_s + hold_s
        self._arm_goal_timer += 1
        self._curr_ee_goal_cart = sphere2cart(self._curr_ee_goal_sphere)
        base_yaw_quat = np_yaw_quat(self._backend.get_base_quat())
        self._curr_ee_goal_world = self.get_ee_goal_spherical_center() + np_quat_apply(
            base_yaw_quat, self._curr_ee_goal_cart
        )
        commands[:, CMD_EE_POS] = self._curr_ee_goal_sphere

    def _ee_pos_sphere_measured(self) -> np.ndarray:
        """Measured EE position in the yaw+gravity sphere frame (CSE target)."""
        ee_world = self._ee_world_pos()
        base_yaw_quat = np_yaw_quat(self._backend.get_base_quat())
        ee_local = np_quat_apply_inverse(
            base_yaw_quat, ee_world - self.get_ee_goal_spherical_center()
        )
        return cart2sphere(ee_local)

    def _ee_world_pos(self) -> np.ndarray:
        """End-effector world position from the armbasepoint-relative sensors."""
        base = self._backend.get_sensor_data("armbasepoint_world_pos")
        quat = self._backend.get_sensor_data("armbasepoint_world_quat")
        rel = self._backend.get_sensor_data(self._cfg.sensor.ee_local_pos)
        return base + np_quat_apply(quat, rel)

    # ── commands ──────────────────────────────────────────────────────────

    def _build_commands_scale(self) -> np.ndarray:
        s = self._cfg.obs_scales
        return np.asarray(
            [
                s.lin_vel,
                s.lin_vel,
                s.ang_vel,
                s.ee_sphe_radius,
                s.ee_sphe_pitch,
                s.ee_sphe_yaw,
                s.ee_orn,
                s.ee_orn,
                s.ee_orn,
                s.ee_force,
                s.ee_force,
                s.ee_force,
                s.base_force,
                s.base_force,
                s.base_force,
            ],
            dtype=self._np_dtype,
        )

    def _sample_velocity_commands(self, num_reset: int) -> np.ndarray:
        cfg = self._cfg.commands
        commands = np.zeros((num_reset, NUM_COMMANDS), dtype=self._np_dtype)
        commands[:, 0] = np.random.uniform(*cfg.lin_vel_x, size=num_reset)
        commands[:, 1] = np.random.uniform(*cfg.lin_vel_y, size=num_reset)
        commands[:, 2] = np.random.uniform(*cfg.ang_vel_yaw, size=num_reset)
        # Raise the zero-velocity probability once forces are active.
        zero_prob = (
            cfg.zero_vel_cmd_prob_after_force
            if self.step_counter >= self._force_start_step
            else cfg.zero_vel_cmd_prob
        )
        zero_mask = np.random.uniform(size=num_reset) < zero_prob
        commands[zero_mask, 0:3] = 0.0
        # Deadband small commands to zero for stable standing.
        small = (
            (np.abs(commands[:, 0]) < cfg.lin_vel_x_clip)
            & (np.abs(commands[:, 1]) < cfg.lin_vel_y_clip)
            & (np.abs(commands[:, 2]) < cfg.ang_vel_yaw_clip)
        )
        commands[small, 0:3] = 0.0
        return commands

    def _command_is_moving(self, commands: np.ndarray) -> np.ndarray:
        cfg = self._cfg.commands
        return (
            (np.abs(commands[:, 0]) > cfg.lin_vel_x_clip)
            | (np.abs(commands[:, 1]) > cfg.lin_vel_y_clip)
            | (np.abs(commands[:, 2]) > cfg.ang_vel_yaw_clip)
        )

    def _resample_commands_if_due(self, info: dict) -> None:
        self._cmd_timer += 1
        due = self._cmd_timer >= self._cmd_resample_steps
        if np.any(due):
            ids = np.flatnonzero(due).astype(np.int32)
            new_cmds = self._sample_velocity_commands(len(ids))
            info["commands"][ids, 0:3] = new_cmds[:, 0:3]
            self._cmd_timer[ids] = 0

    # ── reset DOFs / motor strength ───────────────────────────────────────

    def _apply_reset_dofs(self, qpos: np.ndarray, env_ids: np.ndarray) -> None:
        """Reset: scale leg dofs, jitter arm dofs around default."""
        cfg = self._cfg.reset
        n = len(env_ids)
        base = 7  # floating base (pos3 + quat4) precedes the joint dofs
        leg_scale = np.random.uniform(*cfg.leg_pos_scale_range, size=(n, NUM_LEG)).astype(
            qpos.dtype
        )
        qpos[:, base : base + NUM_LEG] = self.default_angles[:NUM_LEG] * leg_scale
        arm_jitter = np.random.uniform(
            -cfg.arm_init_dof_pos_range, cfg.arm_init_dof_pos_range, size=(n, self._num_arm)
        ).astype(qpos.dtype)
        qpos[:, base + NUM_LEG : base + self._num_action] = (
            self.default_angles[NUM_LEG:] + arm_jitter
        )

    def _sample_motor_strength(self, env_ids: np.ndarray) -> None:
        cfg = self._cfg.domain_rand
        if not cfg.randomize_motor_strength:
            return
        n = len(env_ids)
        leg = np.random.uniform(*cfg.leg_motor_strength_range, size=(n, NUM_LEG))
        arm = np.random.uniform(*cfg.arm_motor_strength_range, size=(n, self._num_arm))
        self._motor_strength[env_ids] = np.concatenate([leg, arm], axis=1).astype(self._np_dtype)

    def _apply_gripper_mass_dr(self, randomization: Any, env_ids: np.ndarray) -> None:
        """Inject a per-env gripper-mass delta into the reset body-mass table.

        The backend composes ``body_mass`` (a full per-body table) with
        ``base_mass_delta`` (added at the base body), so setting a gripper-
        augmented table here coexists with the base-mass randomization.
        """
        cfg = self._cfg.domain_rand
        if randomization is None or not cfg.randomize_gripper_mass:
            return
        n = len(env_ids)
        delta = np.random.uniform(*cfg.gripper_added_mass_range, size=n)
        self._dr_gripper_mass[env_ids] = delta.astype(self._np_dtype).reshape(-1, 1)
        nbody = self._full_body_mass.shape[0]
        table = getattr(randomization, "body_mass", None)
        if table is None:
            table = np.broadcast_to(self._full_body_mass, (n, nbody)).copy()
        else:
            table = np.asarray(table, dtype=np.float64).copy()
        table[:, self._gripper_body_id] += delta
        randomization.body_mass = table

    def _apply_foot_friction_dr(self, randomization: Any, env_ids: np.ndarray) -> None:
        """Randomize per-env FOOT friction in the reset payload.

        The foot geom has priority=1, so it overrides the floor at the contact and
        ``randomize_ground_friction`` never reaches the feet. Set the foot geoms'
        tangential friction to one absolute ``U(foot_friction_range)`` bucket per
        env (all 4 feet share it); the priority-1 foot then imposes that friction
        on the foot-floor contact.
        """
        cfg = self._cfg.domain_rand
        if randomization is None or not cfg.randomize_foot_friction:
            return
        n = len(env_ids)
        lo, hi = cfg.foot_friction_range
        samples = np.random.uniform(lo, hi, size=n).astype(np.float64)
        table = getattr(randomization, "geom_friction", None)
        if table is None:
            ngeom = self._base_geom_friction_full.shape[0]
            table = np.broadcast_to(self._base_geom_friction_full, (n, ngeom, 3)).copy()
        else:
            table = np.asarray(table, dtype=np.float64).copy()
        for gid in self._foot_geom_ids:
            table[:, int(gid), 0] = samples
        randomization.geom_friction = table

    def _cache_reset_dr(self, env_ids: np.ndarray, randomization: Any) -> None:
        """Cache realized reset DR values for the privileged observation."""
        if randomization is None:
            return
        mass = getattr(randomization, "base_mass_delta", None)
        if mass is not None:
            self._dr_base_mass[env_ids] = np.asarray(mass, dtype=self._np_dtype).reshape(-1, 1)
        com = getattr(randomization, "base_com_offset", None)
        if com is not None:
            self._dr_base_com[env_ids] = np.asarray(com, dtype=self._np_dtype).reshape(-1, 3)
        friction = getattr(randomization, "geom_friction", None)
        if friction is not None:
            fr = np.asarray(friction, dtype=self._np_dtype)
            # Tangential friction of a FOOT geom — the priority-1 foot governs the
            # actual foot-floor contact, so this is the friction the policy feels.
            gid = int(self._foot_geom_ids[0])
            self._dr_friction[env_ids] = fr[:, gid, 0:1]

    # ── control: Python PD via per-substep pre-step hook ──────────────────

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        actions = np.asarray(actions, dtype=self._np_dtype)
        if self._clip_actions is not None:
            actions = np.clip(actions, -self._clip_actions, self._clip_actions)
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions
        # The applied (executed) action is delayed; the policy's raw current/last
        # actions above still drive observations and the action_rate rewards.
        if self._action_delay_steps > 0:
            self._action_history[:, :-1] = self._action_history[:, 1:]
            self._action_history[:, -1] = actions
            exec_actions = self._action_history[:, 0]
        elif self._cfg.control_config.simulate_action_latency:
            exec_actions = state.info["last_actions"]
        else:
            exec_actions = actions
        self._update_forces(state)
        self._maybe_apply_velocity_push(state.info.get("commands"))
        # Position targets = default + action * motor_strength * action_scale.
        self._motor_targets = (
            self.default_angles + exec_actions * self._motor_strength * self._action_scale
        ).astype(self._np_dtype)
        return self._motor_targets

    def _force_curriculum_scale(self) -> float:
        """Current force-magnitude gear factor (1.0 when the curriculum is off).

        Staircase keyed off ``step_counter`` relative to ``force_start_step``:
        gear index = (step_counter - force_start_step) // stage_steps, clamped to
        the last gear. Empty ``scales`` => disabled => 1.0 (full force).
        """
        scales = self._force_curriculum_scales
        if not scales or self.step_counter < self._force_start_step:
            return 1.0
        stage = (self.step_counter - self._force_start_step) // self._force_curriculum_stage_steps
        idx = min(max(int(stage), 0), len(scales) - 1)
        return scales[idx]

    def _update_forces(self, state: NpEnvState) -> None:
        """Advance the force episodes and stage external forces for this step.

        cmd forces are commanded (written into the command vector, yaw-local
        frame); ext forces are external disturbances applied to the EE and base
        bodies via the backend (world frame) and become the realized force in
        the observation.
        """
        enabled = self.step_counter >= self._force_start_step
        # Force-magnitude curriculum: scale all four channels by the current gear
        # (1.0 when disabled). cmd and ext share the factor so (ext+cmd)/kp stays
        # consistent with the applied physical force.
        c = self._force_curriculum_scale() if enabled else 1.0
        self._force_ee_cmd = self._sched_gripper_cmd.step(enabled) * c
        self._force_ee_world = self._sched_gripper_ext.step(enabled) * c
        self._force_base_cmd = self._sched_base_cmd.step(enabled) * c
        self._force_base_world = self._sched_base_ext.step(enabled) * c

        commands = state.info.get("commands")
        if commands is not None and commands.shape[0] == self._num_envs:
            commands[:, CMD_EE_FORCE] = self._force_ee_cmd
            commands[:, CMD_BASE_FORCE] = self._force_base_cmd

        applied = np.stack([self._force_ee_world, self._force_base_world], axis=1)
        if enabled and np.any(applied):
            self._backend.apply_body_force(self._force_body_ids, applied.astype(np.float64))

    def _maybe_apply_velocity_push(self, commands: np.ndarray | None) -> None:
        """Base-velocity impulse domain randomization.

        Every ``push_interval`` control steps, overwrite the base horizontal
        linear velocity with ``U(-max_push_vel_xy, max_push_vel_xy)`` (world
        frame), scaled by ``velocity_push_standing_scale`` for envs whose velocity
        command is ~zero (1.0 = no amplification). The write lands in the
        authoritative physics state and is consumed by the next physics substep,
        so it acts as a one-shot velocity impulse.
        """
        dr = self._cfg.domain_rand
        if not dr.velocity_push or dr.push_interval <= 0:
            return
        if self.step_counter <= 0 or self.step_counter % dr.push_interval != 0:
            return
        vmax = float(dr.max_push_vel_xy)
        vel = np.random.uniform(-vmax, vmax, size=(self._num_envs, 2)).astype(self._np_dtype)
        if commands is not None and dr.velocity_push_standing_scale != 1.0:
            standing = ~self._command_is_moving(commands)
            vel[standing] *= float(dr.velocity_push_standing_scale)
        # Overwrite the base horizontal velocity through the declared setter. We do
        # NOT write through get_base_lin_vel()'s return value: the SimBackend
        # contract does not promise it is a live view, and a backend returning a
        # copy would drop the push silently. Read (copy) -> patch x/y -> set.
        lin_vel = self._backend.get_base_lin_vel().copy()
        lin_vel[:, 0:2] = vel
        self._backend.set_base_lin_vel(lin_vel)

    def _reset_force_schedules(self, env_ids: np.ndarray) -> None:
        for sched in (
            self._sched_gripper_cmd,
            self._sched_gripper_ext,
            self._sched_base_cmd,
            self._sched_base_ext,
        ):
            sched.reset(env_ids)
        self._force_ee_world[env_ids] = 0.0
        self._force_base_world[env_ids] = 0.0
        self._force_ee_cmd[env_ids] = 0.0
        self._force_base_cmd[env_ids] = 0.0

    def _pre_step_pd_control(self, backend: Any, targets: np.ndarray) -> np.ndarray:
        q = backend.get_dof_pos()
        qd = backend.get_dof_vel()
        torque = self._kp * (targets - q) - self._kd * qd
        torque = np.clip(torque, -self._torque_limits, self._torque_limits)
        self._last_torque = torque.astype(self._np_dtype)
        return torque

    # ── gait (trot reference) ─────────────────────────────────────────────

    def _update_gait(self, info: dict) -> None:
        cfg = self._cfg.gait
        moving = self._command_is_moving(info["commands"])
        self._gait_indices = np.remainder(
            self._gait_indices + self._cfg.ctrl_dt / cfg.cycle_time, 1.0
        )
        self._gait_indices[~moving] = 0.0
        sin_pos = np.sin(2.0 * np.pi * self._gait_indices)
        thd = cfg.target_joint_pos_thd
        sin_l = sin_pos + thd
        sin_r = sin_pos - thd
        stance = np.zeros((self._num_envs, 4), dtype=self._np_dtype)
        stance[:, 0] = sin_l >= 0
        stance[:, 3] = sin_l >= 0
        stance[:, 1] = sin_r < 0
        stance[:, 2] = sin_r < 0
        self._stance_mask = stance

        ref = np.tile(self.default_angles[:NUM_LEG], (self._num_envs, 1)).astype(self._np_dtype)
        scale_1 = cfg.target_joint_pos_scale / (1.0 - thd)
        scale_2 = scale_1 * 2.0
        sl = np.where(sin_l > 0, 0.0, sin_l)
        sr = np.where(sin_r < 0, 0.0, sin_r)
        ref[:, 1] -= sl * scale_1
        ref[:, 2] += sl * scale_2
        ref[:, 10] -= sl * scale_1
        ref[:, 11] += sl * scale_2
        ref[:, 4] += sr * scale_1
        ref[:, 5] -= sr * scale_2
        ref[:, 7] += sr * scale_1
        ref[:, 8] -= sr * scale_2
        self._ref_dof_pos = ref

    # ── feet / contact sensor readers ─────────────────────────────────────

    def _get_foot_force_vec(self) -> np.ndarray:
        """Per-foot contact force vector (num_envs, 4, 3)."""
        return np.stack(
            [self._backend.get_sensor_data(n) for n in self._cfg.sensor.feet_force_vec], axis=1
        )

    def _get_foot_vel(self) -> np.ndarray:
        """Per-foot world linear velocity (num_envs, 4, 3)."""
        return np.stack(
            [self._backend.get_sensor_data(n) for n in self._cfg.sensor.feet_vel], axis=1
        )

    def _get_thigh_pos(self) -> np.ndarray:
        """Per-thigh world position (num_envs, 4, 3)."""
        return np.stack(
            [self._backend.get_sensor_data(n) for n in self._cfg.sensor.thigh_pos], axis=1
        )

    def _get_undesired_contacts(self) -> np.ndarray:
        """Binary contact flags on penalised bodies (num_envs, K)."""
        return np.stack(
            [self._backend.get_sensor_data(n)[:, 0] for n in self._cfg.sensor.undesired_contact],
            axis=1,
        )

    def _update_contact_timers(self, contact: np.ndarray) -> None:
        """feet_air_time bookkeeping (contact filtering)."""
        contact_filt = contact | self._last_contacts
        self._last_contacts = contact
        self._first_contact = (self._feet_air_time > 0.0) & contact_filt
        self._feet_air_time = self._feet_air_time + self._cfg.ctrl_dt
        # Snapshot the air time the reward consumes, then reset feet in contact.
        self._air_time_snapshot = self._feet_air_time.copy()
        self._feet_air_time = self._feet_air_time * (~contact_filt)

    # ── step ──────────────────────────────────────────────────────────────

    def update_state(self, state: NpEnvState) -> NpEnvState:
        info = state.info
        self._resample_commands_if_due(info)
        self._update_gait(info)
        self._update_ee_goal_trajectory(info["commands"])
        info["torques"] = self._last_torque

        self._foot_contact = self.get_foot_contact() > 0.5
        self._update_contact_timers(self._foot_contact)

        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._gravity_sensor)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()

        env_ids = np.arange(self._num_envs, dtype=np.int32)
        actor_raw, critic_raw = self._compute_raw_obs(info, env_ids=env_ids)
        obs = self._update_history(actor_raw, critic_raw, env_ids=env_ids)
        reward = self._compute_reward(info, linvel, gyro, gravity, dof_pos, dof_vel)
        terminated = self._compute_terminated(self._backend.get_base_quat())
        # dof acceleration uses the previous step's velocity; update after rewards.
        self._last_dof_vel = dof_vel.copy()
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_terminated(self, base_quat: np.ndarray) -> np.ndarray:
        # Termination: fall when |pitch| > 1.0 rad or |roll| > 0.8 rad.
        roll_pitch = _roll_pitch_from_quat(base_quat)
        roll = roll_pitch[:, 0]
        pitch = roll_pitch[:, 1]
        return (np.abs(pitch) > 1.0) | (np.abs(roll) > 0.8)

    # ── observations ──────────────────────────────────────────────────────

    def _compute_raw_obs(self, info: dict, env_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (actor_raw, critic_raw) single-step observations for env_ids."""
        dtype = self._np_dtype
        s = self._cfg.obs_scales

        base_quat = self._backend.get_base_quat()[env_ids]
        base_ang_vel = self._backend.get_base_ang_vel()[env_ids]
        gravity = self._backend.get_sensor_data(self._gravity_sensor)[env_ids]
        dof_pos = self.get_dof_pos()[env_ids]
        dof_vel = self.get_dof_vel()[env_ids]
        base_lin_vel = self.get_local_linvel()[env_ids]
        commands = info["commands"]
        commands = commands if commands.shape[0] == len(env_ids) else commands[env_ids]
        last_actions = info.get(
            "current_actions", np.zeros((len(env_ids), self._num_action), dtype=dtype)
        )
        last_actions = (
            last_actions if last_actions.shape[0] == len(env_ids) else last_actions[env_ids]
        )

        body_rp = _roll_pitch_from_quat(base_quat)
        diff = (dof_pos - self.default_angles) * s.dof_pos
        dvel = dof_vel * s.dof_vel
        sin_phase = np.sin(2.0 * np.pi * self._gait_indices[env_ids])[:, None]
        cos_phase = np.cos(2.0 * np.pi * self._gait_indices[env_ids])[:, None]
        cmd_scaled = commands * self._commands_scale

        # Noise (actor only).
        nz = self._cfg.noise_config
        body_rp_n = self._obs_noise(body_rp, nz.scale_orn)
        ang_vel_n = self._obs_noise(base_ang_vel, nz.scale_ang_vel)
        diff_n = self._obs_noise(diff, nz.scale_joint_angle)
        dvel_n = self._obs_noise(dvel, nz.scale_joint_vel)

        actor_raw = np.concatenate(
            [
                body_rp_n,
                ang_vel_n * s.ang_vel,
                diff_n,
                dvel_n,
                last_actions,
                sin_phase,
                cos_phase,
                cmd_scaled,
            ],
            axis=1,
            dtype=dtype,
        )

        # CSE estimator targets (front 12 of critic obs).
        ee_pos_sphere = self._ee_pos_sphere_measured()[env_ids]
        force_ee_local = np_quat_apply_inverse(
            np_yaw_quat(base_quat), self._force_ee_world[env_ids]
        )
        force_base_local = np_quat_apply_inverse(
            np_yaw_quat(base_quat), self._force_base_world[env_ids]
        )
        ee_sphere_scaled = ee_pos_sphere * np.asarray(
            [s.ee_sphe_radius, s.ee_sphe_pitch, s.ee_sphe_yaw], dtype=dtype
        )
        cse_targets = np.concatenate(
            [
                base_lin_vel * s.lin_vel,
                ee_sphere_scaled,
                force_ee_local * s.ee_force,
                force_base_local * s.base_force,
            ],
            axis=1,
            dtype=dtype,
        )

        leg_ref_diff = dof_pos[:, :NUM_LEG] - self._ref_dof_pos[env_ids]
        dr_block = np.concatenate(
            [
                self._dr_friction[env_ids],
                self._dr_base_mass[env_ids],
                self._dr_base_com[env_ids],
                self._dr_gripper_mass[env_ids],
            ],
            axis=1,
        )
        motor_strength = self._motor_strength[env_ids] - 1.0
        contact_mask = self.get_foot_contact()[env_ids].astype(dtype)

        # Force-offset EE goal expressed in the sphere frame (privileged).
        base_yaw_quat = np_yaw_quat(base_quat)
        force_cmd_world = np_quat_apply(base_yaw_quat, self._force_ee_cmd[env_ids])
        forces_offset = self._force_ee_world[env_ids] + force_cmd_world
        goal_offset_world = (
            forces_offset / self._gripper_force_kp + self._curr_ee_goal_world[env_ids]
        )
        center = self.get_ee_goal_spherical_center()[env_ids]
        goal_offset_local = np_quat_apply_inverse(base_yaw_quat, goal_offset_world - center)
        ee_goal_offset_sphere = cart2sphere(goal_offset_local) * np.asarray(
            [s.ee_sphe_radius, s.ee_sphe_pitch, s.ee_sphe_yaw], dtype=dtype
        )

        critic_core = np.concatenate(
            [
                body_rp,
                base_ang_vel * s.ang_vel,
                diff,
                dvel,
                last_actions,
                sin_phase,
                cos_phase,
                cmd_scaled,
            ],
            axis=1,
            dtype=dtype,
        )
        critic_raw = np.concatenate(
            [
                cse_targets,
                leg_ref_diff,
                dr_block,
                motor_strength,
                self._stance_mask[env_ids],
                contact_mask,
                gravity,
                ee_goal_offset_sphere,
                critic_core,
            ],
            axis=1,
            dtype=dtype,
        )
        return actor_raw, critic_raw

    def _clip_obs_dict(self, buf_a: np.ndarray, buf_c: np.ndarray) -> dict[str, np.ndarray]:
        """Return the obs/critic dict with the ±clip_observations clamp applied.

        Clamps the OUTPUT only; the stored history buffers keep raw values and are
        re-clamped on the next read.
        """
        if self._clip_obs is not None:
            lo, hi = -self._clip_obs, self._clip_obs
            return {"obs": np.clip(buf_a, lo, hi), "critic": np.clip(buf_c, lo, hi)}
        return {"obs": np.array(buf_a), "critic": np.array(buf_c)}

    def _update_history(
        self, actor_raw: np.ndarray, critic_raw: np.ndarray, env_ids: np.ndarray
    ) -> dict[str, np.ndarray]:
        H_a = self._cfg.history.num_actor_history
        H_c = self._cfg.history.num_critic_history
        a_dim = self._actor_one_step_dim
        c_dim = self._critic_one_step_dim
        # Per-step fast path: update_state passes the full identity range, so shift
        # the ring buffers IN PLACE instead of allocating via concatenate. The old
        # code touched each (N, H*dim) buffer ~4x per step (fancy-index read,
        # concatenate alloc, fancy-index write-back, return copy); in place it is 2x
        # (one shift + one return copy). At large num_envs the buffers are
        # cold-cache, so this roughly halves update_state cost (measured 21->11ms
        # at 2048 envs). Partial env_ids (reset) keeps the original scatter path.
        if len(env_ids) == self._num_envs:
            buf_a = self._history_obs_buf
            if H_a > 1:
                buf_a[:, :-a_dim] = buf_a[:, a_dim:]  # left shift (numpy buffers overlap)
                buf_a[:, -a_dim:] = actor_raw
            else:
                buf_a[:] = actor_raw
            buf_c = self._history_critic_buf
            if H_c > 1:
                # Newest-frame-FIRST: right shift. Source overlaps dest on the right,
                # so copy the moved block first (H_c is small, the temp is cheap).
                buf_c[:, c_dim:] = buf_c[:, :-c_dim].copy()
                buf_c[:, :c_dim] = critic_raw
            else:
                buf_c[:] = critic_raw
            return self._clip_obs_dict(buf_a, buf_c)
        # Reset / partial path: original scatter semantics for an env_ids subset.
        buf_a = self._history_obs_buf[env_ids]
        buf_c = self._history_critic_buf[env_ids]
        if H_a > 1:
            buf_a = np.concatenate([buf_a[:, a_dim:], actor_raw], axis=1)
        else:
            buf_a = actor_raw
        if H_c > 1:
            # Critic history is newest-frame-FIRST so the current-frame CSE target
            # block stays at offset 0 (estimator.target_start=0) regardless of H_c.
            buf_c = np.concatenate([critic_raw, buf_c[:, :-c_dim]], axis=1)
        else:
            buf_c = critic_raw
        self._history_obs_buf[env_ids] = buf_a
        self._history_critic_buf[env_ids] = buf_c
        return self._clip_obs_dict(buf_a, buf_c)

    # ── rewards ───────────────────────────────────────────────────────────

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": self._reward_tracking_lin_vel,
            "tracking_lin_vel_force_world": self._reward_tracking_lin_vel_force_world,
            "tracking_ee_force_world": self._reward_tracking_ee_force_world,
            "tracking_ang_vel": self._reward_tracking_ang_vel,
            "lin_vel_z": self._reward_lin_vel_z,
            "ang_vel_xy": self._reward_ang_vel_xy,
            "orientation": orientation,
            "alive": self._reward_alive,
            "ref_dof_leg": self._reward_ref_dof_leg,
            "action_rate": self._reward_action_rate,
            "action_rate_arm": self._reward_action_rate_arm,
            "torques": self._reward_torques,
            "dof_vel": self._reward_dof_vel,
            "dof_vel_arm": self._reward_dof_vel_arm,
            "dof_acc": self._reward_dof_acc,
            "dof_acc_arm": self._reward_dof_acc_arm,
            "base_height": self._reward_base_height,
            "hip_pos": self._reward_hip_pos,
            "torque_limits": self._reward_torque_limits,
            "dof_pos_limits": self._reward_dof_pos_limits,
            "stand_still": self._reward_stand_still,
            "collision": self._reward_collision,
            "feet_contact_number": self._reward_feet_contact_number,
            "feet_air_time": self._reward_feet_air_time,
            "feet_height": self._reward_feet_height,
            "feet_height_high": self._reward_feet_height_high,
            "feet_pos_xy": self._reward_feet_pos_xy,
            "feet_drag": self._reward_feet_drag,
            "feet_contact_forces": self._reward_feet_contact_forces,
        }

    def _compute_reward(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> np.ndarray:
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            gravity=gravity,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            num_envs=self._num_envs,
            default_angles=self.default_angles,
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height_target=self._reward_cfg.base_height_target,
            base_height=self._backend.get_base_pos()[:, 2],
        )
        self._last_reward_ctx = ctx
        # Shared reduction: also emits per-term means into info["log"]["reward/*"]
        # so TensorBoard shows each reward component's contribution.
        return run_reward_dispatch(
            scales=self._reward_cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )

    def _reward_tracking_lin_vel(self, ctx: RewardContext) -> np.ndarray:
        err = np.sum(np.square(ctx.info["commands"][:, 0:2] - ctx.linvel[:, 0:2]), axis=1)
        return np.exp(-err / ctx.tracking_sigma)

    def _reward_tracking_ang_vel(self, ctx: RewardContext) -> np.ndarray:
        err = np.square(ctx.info["commands"][:, 2] - ctx.gyro[:, 2])
        return np.exp(-err / ctx.tracking_sigma)

    def _reward_tracking_ee_force_world(self, ctx: RewardContext) -> np.ndarray:
        """EE force tracking: reward the EE at the force-offset goal.

        Under EE stiffness ``gripper_force_kp`` the end-effector settles at
        ``goal + (ext + cmd) / kp``; rewarding that offset position teaches the
        policy to comply with the external force and exert the commanded force.
        """
        base_yaw_quat = np_yaw_quat(self._backend.get_base_quat())
        force_cmd_world = np_quat_apply(base_yaw_quat, self._force_ee_cmd)
        forces_offset = self._force_ee_world + force_cmd_world
        goal_offset = forces_offset / self._gripper_force_kp + self._curr_ee_goal_world
        ee_world = self._ee_world_pos()
        ee_pos_error = np.sum(np.abs(ee_world - goal_offset), axis=1)
        return np.exp(-ee_pos_error / self._reward_cfg.tracking_ee_sigma * 2.0)

    def _reward_tracking_lin_vel_force_world(self, ctx: RewardContext) -> np.ndarray:
        """Base velocity tracking with a force-induced setpoint offset.

        Under base damping ``base_force_kd`` the base settles at
        ``cmd_vel + (ext + cmd) / kd``; tracking that offset velocity teaches
        compliance with / production of base force.
        """
        base_yaw_quat = np_yaw_quat(self._backend.get_base_quat())
        force_base_local = np_quat_apply_inverse(base_yaw_quat, self._force_base_world)
        forces_offset = force_base_local + self._force_base_cmd
        vel_cmd = ctx.info["commands"][:, 0:2]
        base_vel_offset = forces_offset[:, 0:2] / self._base_force_kd + vel_cmd
        cfg = self._cfg.commands
        moving = (
            (np.abs(base_vel_offset[:, 0]) > cfg.lin_vel_x_clip)
            | (np.abs(base_vel_offset[:, 1]) > cfg.lin_vel_y_clip)
            | (np.abs(ctx.info["commands"][:, 2]) > cfg.ang_vel_yaw_clip)
        )
        base_vel_offset = base_vel_offset * moving[:, None]
        lin_err = np.sum(np.square(base_vel_offset - ctx.linvel[:, 0:2]), axis=1)
        return np.exp(-lin_err / ctx.tracking_sigma)

    def _reward_lin_vel_z(self, ctx: RewardContext) -> np.ndarray:
        return np.square(ctx.linvel[:, 2])

    def _reward_ang_vel_xy(self, ctx: RewardContext) -> np.ndarray:
        return np.sum(np.square(ctx.gyro[:, 0:2]), axis=1)

    def _reward_alive(self, ctx: RewardContext) -> np.ndarray:
        return np.ones((ctx.num_envs,), dtype=self._np_dtype)

    def _reward_ref_dof_leg(self, ctx: RewardContext) -> np.ndarray:
        # L1 joint error, exp(-err * 0.1) (not L2-squared).
        err = np.sum(np.abs(ctx.dof_pos[:, :NUM_LEG] - self._ref_dof_pos), axis=1)
        return np.exp(-err * self._reward_cfg.ref_dof_scale)

    def _reward_action_rate(self, ctx: RewardContext) -> np.ndarray:
        # Legs only; arm has its own action_rate_arm term.
        cur = ctx.info.get("current_actions")
        last = ctx.info.get("last_actions")
        if cur is None or last is None:
            return np.zeros((ctx.num_envs,), dtype=self._np_dtype)
        return np.sum(np.square(cur[:, :NUM_LEG] - last[:, :NUM_LEG]), axis=1)

    def _reward_action_rate_arm(self, ctx: RewardContext) -> np.ndarray:
        cur = ctx.info.get("current_actions")
        last = ctx.info.get("last_actions")
        if cur is None or last is None:
            return np.zeros((ctx.num_envs,), dtype=self._np_dtype)
        return np.sum(np.square(cur[:, NUM_LEG:] - last[:, NUM_LEG:]), axis=1)

    def _reward_torques(self, ctx: RewardContext) -> np.ndarray:
        # Legs only (slices [:, :12]).
        return np.sum(np.square(self._last_torque[:, :NUM_LEG]), axis=1)

    def _reward_dof_vel(self, ctx: RewardContext) -> np.ndarray:
        # Legs only (slices [:, :12]).
        dof_vel = np.asarray(ctx.dof_vel)
        return np.sum(np.square(dof_vel[:, :NUM_LEG]), axis=1)

    def _reward_dof_vel_arm(self, ctx: RewardContext) -> np.ndarray:
        dof_vel = np.asarray(ctx.dof_vel)
        return np.sum(np.square(dof_vel[:, NUM_LEG : self._num_action]), axis=1)

    def _reward_dof_acc(self, ctx: RewardContext) -> np.ndarray:
        dof_vel = np.asarray(ctx.dof_vel)
        acc = (self._last_dof_vel[:, :NUM_LEG] - dof_vel[:, :NUM_LEG]) / self._cfg.ctrl_dt
        return np.sum(np.square(acc), axis=1)

    def _reward_dof_acc_arm(self, ctx: RewardContext) -> np.ndarray:
        dof_vel = np.asarray(ctx.dof_vel)
        acc = (
            self._last_dof_vel[:, NUM_LEG : self._num_action]
            - dof_vel[:, NUM_LEG : self._num_action]
        ) / self._cfg.ctrl_dt
        return np.sum(np.square(acc), axis=1)

    def _reward_base_height(self, ctx: RewardContext) -> np.ndarray:
        return np.square(ctx.base_height - ctx.base_height_target)

    def _reward_hip_pos(self, ctx: RewardContext) -> np.ndarray:
        hip_idx = np.asarray([0, 3, 6, 9])
        return np.sum(np.square(ctx.dof_pos[:, hip_idx] - self.default_angles[hip_idx]), axis=1)

    def _reward_torque_limits(self, ctx: RewardContext) -> np.ndarray:
        margin = self._reward_cfg.soft_torque_limit * self._torque_limits
        over = np.abs(self._last_torque) - margin
        return np.sum(np.clip(over, 0.0, None), axis=1)

    def _reward_dof_pos_limits(self, ctx: RewardContext) -> np.ndarray:
        # Penalize positions outside the hard joint range (all actuated joints).
        lower = self._dof_pos_limits[:, 0]
        upper = self._dof_pos_limits[:, 1]
        out = np.clip(lower - ctx.dof_pos[:, : self._num_action], 0.0, None)
        out += np.clip(ctx.dof_pos[:, : self._num_action] - upper, 0.0, None)
        return np.sum(out, axis=1)

    def _reward_stand_still(self, ctx: RewardContext) -> np.ndarray:
        # Reward holding the default leg pose at zero command; off when walking.
        err = np.sum(np.abs(ctx.dof_pos[:, :NUM_LEG] - self.default_angles[:NUM_LEG]), axis=1)
        rew = np.exp(-err * self._reward_cfg.stand_still_scale)
        rew[self._command_is_moving(ctx.info["commands"])] = 0.0
        return rew

    def _reward_collision(self, ctx: RewardContext) -> np.ndarray:
        contacts = self._get_undesired_contacts()
        return np.sum((contacts > 0.5).astype(self._np_dtype), axis=1)

    def _reward_feet_contact_number(self, ctx: RewardContext) -> np.ndarray:
        # Reward foot contact matching the gait stance phase (+1 / -0.3).
        contact = self._foot_contact
        stance = self._stance_mask > 0.5
        reward = np.where(contact == stance, 1.0, -0.3)
        return np.mean(reward, axis=1).astype(self._np_dtype)

    def _reward_feet_air_time(self, ctx: RewardContext) -> np.ndarray:
        thd = self._reward_cfg.feet_air_time_threshold
        rew = np.sum((self._air_time_snapshot - thd) * self._first_contact, axis=1)
        rew = rew * self._command_is_moving(ctx.info["commands"])
        return rew.astype(self._np_dtype)

    def _reward_feet_height(self, ctx: RewardContext) -> np.ndarray:
        # Front feet only: encourage clearance up to the target (clamp max=0).
        feet_z = self.get_foot_pos()[:, :2, 2]
        rew = np.clip(np.max(feet_z, axis=1) - self._reward_cfg.feet_height_target, None, 0.0)
        rew[~self._command_is_moving(ctx.info["commands"])] = 0.0
        return rew.astype(self._np_dtype)

    def _reward_feet_height_high(self, ctx: RewardContext) -> np.ndarray:
        # Penalize lifting any foot above the high threshold (clamp min=0).
        feet_z = self.get_foot_pos()[:, :, 2]
        rew = np.clip(np.max(feet_z, axis=1) - self._reward_cfg.feet_height_high_target, 0.0, None)
        rew[~self._command_is_moving(ctx.info["commands"])] = 0.0
        return rew.astype(self._np_dtype)

    def _reward_feet_pos_xy(self, ctx: RewardContext) -> np.ndarray:
        feet_xy = self.get_foot_pos()[:, :, :2]
        thigh_xy = self._get_thigh_pos()[:, :, :2]
        diff = np.linalg.norm(feet_xy - thigh_xy, axis=2)
        return np.mean(diff, axis=1).astype(self._np_dtype)

    def _reward_feet_drag(self, ctx: RewardContext) -> np.ndarray:
        feet_vel = np.sum(np.abs(self._get_foot_vel()), axis=2)
        foot_forces = np.linalg.norm(self._get_foot_force_vec(), axis=2)
        return np.sum(foot_forces * feet_vel, axis=1).astype(self._np_dtype)

    def _reward_feet_contact_forces(self, ctx: RewardContext) -> np.ndarray:
        forces = np.linalg.norm(self._get_foot_force_vec(), axis=2)
        over = np.clip(forces - self._reward_cfg.max_contact_force, 0.0, None)
        return np.sum(over, axis=1).astype(self._np_dtype)
