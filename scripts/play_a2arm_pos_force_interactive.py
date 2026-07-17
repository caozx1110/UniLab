"""Interactive MuJoCo viewer for the quadruped+Arm force-position task (CSE-PPO).

Opens a live MuJoCo window with a single quadruped+arm robot on the same flat
ground used during play, drives it with a trained CSE-PPO policy, and lets you
teleop it from the keyboard:

  * base velocity      : forward/back, strafe left/right, turn left/right
  * EE target position : spherical [radius, pitch, yaw] (the policy's reach goal)
  * External pushes     : impulse forces on the base and on the end-effector

Push magnitudes and the EE init point are read from the env config, so the
script works for the A2Arm pos-force task via task=.

Usage (mirrors scripts/train_cse_ppo.py checkpoint resolution):

    # A2 + P7v3 (5-DOF, joint3+joint5 frozen) + UMI gripper
    uv run scripts/play_a2arm_pos_force_interactive.py \
        task=a2arm_pos_force/mujoco \
        algo.load_run=<run-dir-or-checkpoint-path> \
        algo.checkpoint=<iteration-or-filename>

Camera (MuJoCo viewer): mouse-drag rotate, scroll zoom, right-drag pan.
"""

# pyright: reportAttributeAccessIssue=false, reportArgumentType=false

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mujoco
import mujoco.viewer

from unilab.algos.torch.cse_ppo.runner import CSEOnPolicyRunner
from unilab.base.backend.mujoco.xml import materialize_scene_visual_override
from unilab.envs.locomotion.a2arm.pos_force import (
    CMD_BASE_FORCE,
    CMD_EE_FORCE,
    CMD_EE_POS,
    CMD_VEL,
    sphere2cart,
)
from unilab.training import (
    BackendAdapter,
    create_env,
    ensure_registries,
    parse_checkpoint_path,
)
from unilab.training.rsl_rl import RslRlVecEnvWrapper
from unilab.utils.rotation import np_quat_apply, np_quat_apply_inverse, np_yaw_quat

# ── GLFW key codes (mujoco passive viewer key_callback) ────────────────────
_KEY_SPACE = ord(" ")
_KEY_BACKSPACE = 259
_KEY_RIGHT, _KEY_LEFT, _KEY_DOWN, _KEY_UP = 262, 263, 264, 265
_KEY_PAGE_UP, _KEY_PAGE_DOWN = 266, 267

# Initial EE target in spherical coords [radius, pitch, yaw] (closest in-range
# point to the reset pose).
_EE_INIT_SPHERE = (0.25, 0.68, 0.0)

# Teleop step sizes.
_STEP_VX = 0.1
_STEP_VY = 0.1
_STEP_VYAW = 0.1
_STEP_EE_RADIUS = 0.02
_STEP_EE_PITCH = 0.05
_STEP_EE_YAW = 0.05

# Impulse pushes (world frame). One key tap arms a single TRAPEZOIDAL force
# episode matching the training force profile: ramp up over push_duration, HOLD
# at peak for the per-body settling time (gripper 0.5 s, base 1.0 s), ramp down.
# Magnitudes are read per-robot from the env config (ext gripper/base force
# ranges) in ``_impulse_from_env``. A sudden step-impulse is out-of-distribution
# — the concurrent estimator integrates ~0.64 s of history, so a sharp spike just
# shoves the robot before the policy can react. The per-body ramp/hold step
# counts are read from the env config in ``_push_timing_from_env``.


def _backend_adapter(cfg: DictConfig) -> BackendAdapter:
    return BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="ppo_cse",
        scene_materializer=materialize_scene_visual_override,
    )


def _algo_config_dict(cfg: DictConfig) -> dict[str, Any]:
    raw = OmegaConf.to_container(cfg.algo, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("cfg.algo must resolve to a dict")
    return cast(dict[str, Any], raw)


def _select_device(cfg: DictConfig) -> str:
    configured = OmegaConf.select(cfg, "training.device")
    if configured not in (None, ""):
        return str(configured)
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ───────────────────────────────────────────────────────────────────────────
# Teleop state
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class _Ranges:
    vx: tuple[float, float]
    vy: tuple[float, float]
    vyaw: tuple[float, float]
    radius: tuple[float, float]
    pitch: tuple[float, float]
    yaw: tuple[float, float]


@dataclass
class TeleopState:
    """Keyboard-driven commands shared between the viewer callback and loop."""

    ranges: _Ranges
    # Push trapezoid timing in control steps, matched to the training force
    # profile (ramp = push_duration, hold = per-body settling time). Overwritten
    # from the env config in ``_push_timing_from_env``; defaults assume 50 Hz.
    ee_ramp: int = 25
    ee_hold: int = 25
    base_ramp: int = 25
    base_hold: int = 50
    # Push magnitudes (N), read per-robot from the env config (ext force ranges).
    impulse_ee_n: float = 15.0
    impulse_base_n: float = 20.0
    # Nominal EE init [radius, pitch, yaw], clamped to the robot's goal_ee ranges.
    ee_init: np.ndarray = field(
        default_factory=lambda: np.asarray(_EE_INIT_SPHERE, dtype=np.float64)
    )
    base_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    ee_sphere: np.ndarray = field(
        default_factory=lambda: np.asarray(_EE_INIT_SPHERE, dtype=np.float64)
    )
    # Current applied push force (world frame); driven by a trapezoidal episode.
    ee_force: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    base_force: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    # Per-body trapezoid episode state: target vector + step counter.
    _ee_target: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    _base_target: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    _ee_step: int = -1
    _base_step: int = -1
    # Sustained-force mode: when hold_mode is on, a push ramps up and HOLDS at the
    # target indefinitely (no ramp-down) until hold is toggled off (smooth
    # ramp-down) or forces are cleared. Per-body flags record whether the CURRENT
    # episode is a held one.
    hold_mode: bool = False
    _ee_held: bool = False
    _base_held: bool = False

    def __post_init__(self) -> None:
        self.ee_sphere[:] = self.ee_init

    def reset(self) -> None:
        self.base_vel[:] = 0.0
        self.ee_sphere[:] = self.ee_init
        self.clear_forces()

    # -- velocity / EE nudges (clipped to the env's command ranges) ----------
    def nudge_vel(self, axis: int, delta: float) -> None:
        self.base_vel[axis] += delta
        rng = (self.ranges.vx, self.ranges.vy, self.ranges.vyaw)[axis]
        self.base_vel[axis] = float(np.clip(self.base_vel[axis], rng[0], rng[1]))

    def zero_vel(self) -> None:
        self.base_vel[:] = 0.0

    def nudge_ee(self, axis: int, delta: float) -> None:
        self.ee_sphere[axis] += delta
        rng = (self.ranges.radius, self.ranges.pitch, self.ranges.yaw)[axis]
        self.ee_sphere[axis] = float(np.clip(self.ee_sphere[axis], rng[0], rng[1]))

    def reset_ee(self) -> None:
        self.ee_sphere[:] = self.ee_init

    # -- trapezoidal pushes (ramp-up -> hold -> ramp-down, in training range) -
    def push_ee(self, axis: int, sign: float) -> None:
        self._ee_target[:] = 0.0
        self._ee_target[axis] = sign * self.impulse_ee_n
        self._ee_step = 0
        self._ee_held = self.hold_mode

    def push_base(self, axis: int, sign: float) -> None:
        self._base_target[:] = 0.0
        self._base_target[axis] = sign * self.impulse_base_n
        self._base_step = 0
        self._base_held = self.hold_mode

    def toggle_hold(self) -> None:
        """Flip sustained-force mode. Turning it OFF releases any held force with a
        smooth ramp-down (so the robot is not shocked by an instantaneous drop)."""
        self.hold_mode = not self.hold_mode
        if not self.hold_mode:
            self.release_held()

    def release_held(self) -> None:
        """Convert any currently-held episode to its ramp-down phase (1 -> 0 over
        ``ramp``), matching the trailing edge of the training trapezoid."""
        if self._ee_held:
            self._ee_step = max(1, self.ee_ramp) + self.ee_hold  # start of ramp-down
            self._ee_held = False
        if self._base_held:
            self._base_step = max(1, self.base_ramp) + self.base_hold
            self._base_held = False

    def clear_forces(self) -> None:
        self.ee_force[:] = 0.0
        self.base_force[:] = 0.0
        self._ee_target[:] = 0.0
        self._base_target[:] = 0.0
        self._ee_step = -1
        self._base_step = -1
        self._ee_held = False
        self._base_held = False

    @staticmethod
    def _trapezoid_frac(step: int, ramp: int, hold: int) -> float:
        """Trapezoid: linear ramp up over ``ramp``, hold, linear ramp down."""
        ramp = max(1, ramp)
        total = 2 * ramp + hold
        if step < ramp:
            return step / ramp
        if step < ramp + hold:
            return 1.0
        return float(np.clip((total - step) / ramp, 0.0, 1.0))

    def advance_forces(self) -> None:
        """Update the applied push force to this step's trapezoid value.

        Call once per control step BEFORE stepping the env, so the force the
        physics applies follows the smooth ramp/hold the policy was trained on.
        """
        if self._ee_step >= 0:
            if self._ee_held:
                # Ramp up over ``ee_ramp`` then hold at the target indefinitely.
                frac = min(1.0, self._ee_step / max(1, self.ee_ramp))
                self.ee_force[:] = self._ee_target * frac
                self._ee_step += 1
            else:
                ee_total = 2 * max(1, self.ee_ramp) + self.ee_hold
                self.ee_force[:] = self._ee_target * self._trapezoid_frac(
                    self._ee_step, self.ee_ramp, self.ee_hold
                )
                self._ee_step += 1
                if self._ee_step > ee_total:
                    self.ee_force[:] = 0.0
                    self._ee_step = -1
        if self._base_step >= 0:
            if self._base_held:
                frac = min(1.0, self._base_step / max(1, self.base_ramp))
                self.base_force[:] = self._base_target * frac
                self._base_step += 1
            else:
                base_total = 2 * max(1, self.base_ramp) + self.base_hold
                self.base_force[:] = self._base_target * self._trapezoid_frac(
                    self._base_step, self.base_ramp, self.base_hold
                )
                self._base_step += 1
                if self._base_step > base_total:
                    self.base_force[:] = 0.0
                    self._base_step = -1

    def describe(self) -> str:
        return (
            f"vel(x={self.base_vel[0]:+.2f} y={self.base_vel[1]:+.2f} "
            f"yaw={self.base_vel[2]:+.2f}) "
            f"ee[l={self.ee_sphere[0]:.2f} p={self.ee_sphere[1]:+.2f} "
            f"y={self.ee_sphere[2]:+.2f}] "
            f"F_ee={np.round(self.ee_force, 1)} F_base={np.round(self.base_force, 1)} "
            f"hold={'ON' if self.hold_mode else 'off'}"
        )


def _ranges_from_env(env: Any) -> _Ranges:
    c = env.cfg.commands
    g = env.cfg.goal_ee
    return _Ranges(
        vx=(float(c.lin_vel_x[0]), float(c.lin_vel_x[1])),
        vy=(float(c.lin_vel_y[0]), float(c.lin_vel_y[1])),
        vyaw=(float(c.ang_vel_yaw[0]), float(c.ang_vel_yaw[1])),
        radius=(float(g.pos_l[0]), float(g.pos_l[1])),
        pitch=(float(g.pos_p[0]), float(g.pos_p[1])),
        yaw=(float(g.pos_y[0]), float(g.pos_y[1])),
    )


def _impulse_from_env(env: Any) -> dict[str, float]:
    """Teleop push magnitudes (N), matched per-robot to the training EXT force
    ranges (the external-disturbance ranges, not the commanded ones)."""
    c = env.cfg.commands
    return {
        "impulse_ee_n": float(max(abs(x) for x in c.max_push_force_xyz_gripper_ext)),
        "impulse_base_n": float(max(abs(x) for x in c.max_push_force_xyz_base_ext)),
    }


def _ee_init_from_env(env: Any) -> np.ndarray:
    """Nominal EE init (closest in-range point to the reset pose), clamped to the
    robot's spherical goal ranges [radius, pitch, yaw]."""
    g = env.cfg.goal_ee
    r, p, y = _EE_INIT_SPHERE
    return np.asarray(
        [
            float(np.clip(r, g.pos_l[0], g.pos_l[1])),
            float(np.clip(p, g.pos_p[0], g.pos_p[1])),
            float(np.clip(y, g.pos_y[0], g.pos_y[1])),
        ],
        dtype=np.float64,
    )


# ───────────────────────────────────────────────────────────────────────────
# Env teleop patch: freeze the auto command/EE/force overwrites
# ───────────────────────────────────────────────────────────────────────────


def _patch_env_for_teleop(env: Any, teleop: TeleopState) -> None:
    """Replace the env's automatic command/EE/force updates with teleop reads.

    The env normally (1) resamples base velocity on a timer, (2) walks the EE
    goal along an internal trajectory, and (3) drives external forces from a
    random schedule that only activates after ``force_start_step``. For
    interactive validation we suppress all three and drive them from
    ``teleop`` instead.
    """

    # (1) Never resample base velocity — keyboard owns commands[:, 0:3].
    env._resample_commands_if_due = lambda info: None  # noqa: ARG005

    # (2) Pin the EE goal to the teleop sphere; still keep curr_ee_goal_world in
    #     sync for marker rendering and any obs that read it.
    def _teleop_ee_goal(commands: np.ndarray) -> None:
        env._curr_ee_goal_sphere[:] = teleop.ee_sphere
        env._curr_ee_goal_cart = sphere2cart(env._curr_ee_goal_sphere)
        base_yaw_quat = np_yaw_quat(env._backend.get_base_quat())
        env._curr_ee_goal_world = env.get_ee_goal_spherical_center() + np_quat_apply(
            base_yaw_quat, env._curr_ee_goal_cart
        )
        commands[:, CMD_EE_POS] = env._curr_ee_goal_sphere

    env._update_ee_goal_trajectory = _teleop_ee_goal

    # (3) Apply only the teleop impulse forces (world frame), from step 0. The
    #     commanded-force channels stay zero, so these read as pure external
    #     disturbances the policy must reject.
    def _teleop_forces(state: Any) -> None:
        env._force_ee_world[:] = teleop.ee_force
        env._force_base_world[:] = teleop.base_force
        env._force_ee_cmd[:] = 0.0
        env._force_base_cmd[:] = 0.0
        commands = state.info.get("commands")
        if commands is not None and commands.shape[0] == env._num_envs:
            commands[:, CMD_EE_FORCE] = 0.0
            commands[:, CMD_BASE_FORCE] = 0.0
        applied = np.stack([env._force_ee_world, env._force_base_world], axis=1)
        if np.any(applied):
            env._backend.apply_body_force(env._force_body_ids, applied.astype(np.float64))

    env._update_forces = _teleop_forces

    # (4) Disable the random base-velocity impulse DR (UniFP _push_robots, applied
    #     in apply_action independently of _update_forces). During interactive
    #     validation every disturbance must be user-driven; otherwise the base
    #     gets uncommanded ~0.3-0.75 m/s kicks every push_interval steps.
    env._maybe_apply_velocity_push = lambda commands: None  # noqa: ARG005

    # (5) Clean deploy, matching UniFP's strict sim2sim. The policy is TRAINED
    #     with observation noise + domain randomization for robustness, but it
    #     must be EVALUATED on a clean, deterministic world — otherwise per-step
    #     obs noise makes the arm visibly jitter, and random mass/COM/motor draws
    #     skew the feel (a weak motor draw looks like a weak policy).
    env._cfg.noise_config.level = 0.0
    dr = env._cfg.domain_rand
    dr.randomize_base_mass = False
    dr.random_com = False
    dr.randomize_motor_strength = False
    dr.randomize_gripper_mass = False
    dr.randomize_ground_friction = False


def _push_timing_from_env(env: Any) -> dict[str, int]:
    """Teleop push ramp/hold (in control steps) matched to the training profile:
    ramp = the low end of push_duration, hold = the per-body settling time."""
    c = env.cfg.commands
    dt = float(env.cfg.ctrl_dt)

    def steps(seconds: float) -> int:
        return max(1, int(float(seconds) / dt))

    return {
        "ee_ramp": steps(c.push_gripper_duration_s[0]),
        "ee_hold": steps(c.settling_time_force_gripper_s),
        "base_ramp": steps(c.push_base_duration_s[0]),
        "base_hold": steps(c.settling_time_force_base_s),
    }


# ───────────────────────────────────────────────────────────────────────────
# Viewer markers
# ───────────────────────────────────────────────────────────────────────────


def _add_sphere(scene: Any, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_arrow(
    scene: Any, p0: np.ndarray, vec: np.ndarray, scale: float, width: float, rgba: np.ndarray
) -> None:
    length = float(np.linalg.norm(vec))
    if length < 1e-6 or scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    p1 = np.asarray(p0, dtype=np.float64) + np.asarray(vec, dtype=np.float64) * scale
    mujoco.mjv_connector(
        geom, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(p0, dtype=np.float64), p1
    )
    scene.ngeom += 1


def _add_line(scene: Any, p0: np.ndarray, p1: np.ndarray, width: float, rgba: np.ndarray) -> None:
    """Draw a thin capsule between two world points (wireframe edge)."""
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        width,
        np.asarray(p0, dtype=np.float64),
        np.asarray(p1, dtype=np.float64),
    )
    scene.ngeom += 1


def _sphere_pt(
    center: np.ndarray, base_yaw_quat: np.ndarray, l: float, pitch: float, yaw: float
) -> np.ndarray:
    """One sampling-sphere point (l, pitch, yaw) -> world, matching the env's
    goal transform: center + R_yaw @ sphere2cart([l, pitch, yaw])."""
    cart = sphere2cart(np.asarray([[l, pitch, yaw]], dtype=np.float64))
    world = center + np_quat_apply(base_yaw_quat, cart)
    return np.asarray(world[0], dtype=np.float64)


def _draw_sample_range(viewer: Any, env: Any) -> None:
    """Wireframe of the EE goal sampling volume (goal_ee.pos_l/pos_p/pos_y).

    The sampling region is a spherical box (radius x pitch x yaw range) centered
    on ``get_ee_goal_spherical_center()`` in the base-yaw frame — exactly the
    frame the env samples goals in. We render the two radial shells (inner l0,
    outer l1) as pitch/yaw grid curves plus the connecting radial edges, so the
    reachable goal cloud is visible during teleop.
    """
    scene = viewer.user_scn
    g = env.cfg.goal_ee
    l0, l1 = float(g.pos_l[0]), float(g.pos_l[1])
    p0, p1 = float(g.pos_p[0]), float(g.pos_p[1])
    y0, y1 = float(g.pos_y[0]), float(g.pos_y[1])

    center = np.asarray(env.get_ee_goal_spherical_center()[0], dtype=np.float64)
    base_yaw_quat = np_yaw_quat(env._backend.get_base_quat())[0:1]

    width = 0.006
    rgba_shell = np.array([0.2, 0.5, 1.0, 0.5])  # blue: pitch/yaw grid on both shells
    rgba_edge = np.array([1.0, 0.85, 0.1, 0.7])  # amber: radial edges (l0<->l1)

    n_p, n_y = 7, 9  # grid resolution along pitch / yaw
    pitches = np.linspace(p0, p1, n_p)
    yaws = np.linspace(y0, y1, n_y)

    def pt(l: float, p: float, y: float) -> np.ndarray:
        return _sphere_pt(center, base_yaw_quat, l, p, y)

    for l in (l0, l1):
        # yaw curves at fixed pitch (span the yaw range)
        for p in pitches:
            prev = pt(l, p, yaws[0])
            for y in yaws[1:]:
                cur = pt(l, p, y)
                _add_line(scene, prev, cur, width, rgba_shell)
                prev = cur
        # pitch curves at fixed yaw (span the pitch range)
        for y in yaws:
            prev = pt(l, pitches[0], y)
            for p in pitches[1:]:
                cur = pt(l, p, y)
                _add_line(scene, prev, cur, width, rgba_shell)
                prev = cur

    # Radial edges connecting inner & outer shell at the 4 corners of the p/y box.
    for p in (p0, p1):
        for y in (y0, y1):
            _add_line(scene, pt(l0, p, y), pt(l1, p, y), width, rgba_edge)

    # Sphere center marker (small cyan sphere).
    _add_sphere(scene, center, 0.02, np.array([0.1, 0.9, 0.9, 0.9]))


def _draw_markers(viewer: Any, env: Any, teleop: TeleopState, show_range: bool = True) -> None:
    scene = viewer.user_scn
    scene.ngeom = 0

    # EE goal sampling volume wireframe (draw first so markers render on top).
    if show_range:
        try:
            _draw_sample_range(viewer, env)
        except Exception as exc:  # never let viz break the play loop
            print(f"[play] sample-range viz skipped: {exc}")

    # EE target (green) and measured EE position (orange).
    goal_world = np.asarray(env._curr_ee_goal_world[0], dtype=np.float64)
    _add_sphere(scene, goal_world, 0.04, np.array([0.1, 0.9, 0.2, 0.9]))
    try:
        ee_world = np.asarray(env._ee_world_pos()[0], dtype=np.float64)
        _add_sphere(scene, ee_world, 0.03, np.array([1.0, 0.6, 0.1, 0.9]))
    except Exception:
        ee_world = None

    # Force arrows (red=EE push, magenta=base push), 0.01 m per N.
    if ee_world is not None and np.any(teleop.ee_force):
        _add_arrow(scene, ee_world, teleop.ee_force, 0.01, 0.012, np.array([1.0, 0.1, 0.1, 0.95]))
    if np.any(teleop.base_force):
        base_pos = np.asarray(env._backend.get_base_pos()[0], dtype=np.float64)
        _add_arrow(scene, base_pos, teleop.base_force, 0.01, 0.015, np.array([1.0, 0.1, 0.8, 0.95]))


def _print_legend() -> None:
    print(
        "[play] Keyboard teleop:\n"
        "  Base move : W/S vx | A/D vy | Q/E turn(yaw) | Z stop\n"
        "  EE target : U/J radius | I/K pitch | O/L yaw | P reset target\n"
        "  EE push   : arrows = fx/fy, PageUp/PageDn = fz (trapezoid push)\n"
        "  Base push : 1/2 fx | 3/4 fy | 5/6 fz (trapezoid push)\n"
        "  Hold mode : H toggle (push then STAYS on; H again = ramp down)\n"
        "  Forces off: F   |   Reset: Backspace   |   Pause: Space\n"
        "  Sample range viz: G (toggle blue reachable-goal shell wireframe)"
    )


def _print_force_estimate(runner: Any, env: Any, obs_tensor: Any) -> None:
    """Print the CSE estimator's EXT-force estimate vs the true applied force.

    Diagnostic for compliance behaviour: the policy only yields to a force it can
    perceive. In teleop the commanded-force channel is 0, so the ONLY way the
    policy knows about the push is the estimator latent. If ``est`` stays ~0 while
    ``true`` is large, the policy cannot perceive the force (OOD sustained force /
    weak observability) and will fall back to the stiff passive PD — NOT a policy
    'tracking' failure. Values are in the yaw-local frame; the estimator target is
    ``critic_obs[6:9]`` (force_ee) and ``[9:12]`` (force_base), scaled by
    obs_scales, so we divide the scale back out."""
    est = runner.actor_critic.estimator
    pred = est.predict(obs_tensor).detach().cpu().numpy()[0]  # 12-dim CSE target estimate
    s = env._cfg.obs_scales
    est_ee = pred[6:9] / max(float(s.ee_force), 1e-8)
    est_base = pred[9:12] / max(float(s.base_force), 1e-8)
    byq = np_yaw_quat(env._backend.get_base_quat())
    true_ee = np_quat_apply_inverse(byq, env._force_ee_world)[0]
    true_base = np_quat_apply_inverse(byq, env._force_base_world)[0]
    print(
        f"[force] EE est={np.round(est_ee, 1)} |{np.linalg.norm(est_ee):5.1f}N  "
        f"true={np.round(true_ee, 1)} |{np.linalg.norm(true_ee):5.1f}N   ||   "
        f"BASE est|{np.linalg.norm(est_base):5.1f}N true|{np.linalg.norm(true_base):5.1f}N"
    )


# ───────────────────────────────────────────────────────────────────────────
# Main play loop
# ───────────────────────────────────────────────────────────────────────────


def play_interactive(cfg: DictConfig, device: str) -> None:
    rl_cfg = _algo_config_dict(cfg)
    load_path, load_path_dir = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
    if load_path is None or not Path(load_path).exists():
        print(
            "[play] Could not resolve a CSE-PPO checkpoint. Pass "
            "algo.load_run=<run-dir-or-checkpoint-path> (and optionally "
            "algo.checkpoint=<iteration-or-filename>)."
        )
        return
    ckpt_keys = set(torch.load(load_path, map_location="cpu", weights_only=True).keys())
    if "actor_state_dict" not in ckpt_keys:
        print(f"[play] {load_path} is not a CSE-PPO checkpoint (keys: {ckpt_keys}). Aborting.")
        return
    print(f"[play] Loading checkpoint: {load_path}")

    env_cfg_override = cast(dict[str, Any], _backend_adapter(cfg).build_play_env_cfg_override())
    env = create_env(cfg, num_envs=1, env_cfg_override=env_cfg_override, sim_backend="mujoco")
    wrapped_env = RslRlVecEnvWrapper(env, device=device)
    runner = CSEOnPolicyRunner(wrapped_env, rl_cfg, log_dir=None, device=device)
    runner.load(str(load_path))
    policy = runner.get_inference_policy(device=device)

    # Teleop owns commands / EE goal / forces from here on.
    env.set_autoreset(False)
    teleop = TeleopState(
        ranges=_ranges_from_env(env),
        ee_init=_ee_init_from_env(env),
        **_impulse_from_env(env),
        **_push_timing_from_env(env),
    )
    _patch_env_for_teleop(env, teleop)

    # Dedicated viewer model + data (single env → use the decoder model directly).
    parent_xml, _ = _mujoco_visual_xml_paths(env)
    viz_model = mujoco.MjModel.from_xml_path(str(parent_xml))
    viz_data = mujoco.MjData(viz_model)
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    base_body_id = max(1, mujoco.mj_name2id(viz_model, mujoco.mjtObj.mjOBJ_BODY, "base"))
    ctrl_dt = float(env.cfg.ctrl_dt)

    paused = {"v": False}
    reset_requested = {"v": False}
    show_range = {"v": True}

    def _key_callback(keycode: int) -> None:
        # Global
        if keycode == _KEY_SPACE:
            paused["v"] = not paused["v"]
            print(f"[play] {'paused' if paused['v'] else 'resumed'}")
            return
        if keycode in (ord("G"), ord("g")):
            show_range["v"] = not show_range["v"]
            print(f"[play] sample-range viz {'on' if show_range['v'] else 'off'}")
            return
        if keycode == _KEY_BACKSPACE:
            # Defer to the main loop so obs stays in sync with the reset.
            reset_requested["v"] = True
            return
        if keycode in (ord("F"), ord("f")):
            teleop.clear_forces()
            print("[play] forces cleared")
            return

        # Base velocity
        if keycode in (ord("W"), ord("w")):
            teleop.nudge_vel(0, +_STEP_VX)
        elif keycode in (ord("S"), ord("s")):
            teleop.nudge_vel(0, -_STEP_VX)
        elif keycode in (ord("A"), ord("a")):
            teleop.nudge_vel(1, +_STEP_VY)
        elif keycode in (ord("D"), ord("d")):
            teleop.nudge_vel(1, -_STEP_VY)
        elif keycode in (ord("Q"), ord("q")):
            teleop.nudge_vel(2, +_STEP_VYAW)
        elif keycode in (ord("E"), ord("e")):
            teleop.nudge_vel(2, -_STEP_VYAW)
        elif keycode in (ord("Z"), ord("z")):
            teleop.zero_vel()
        # Sustained-force mode: H toggles hold (next push ramps up and STAYS on;
        # toggling off ramps the held force back down).
        elif keycode in (ord("H"), ord("h")):
            teleop.toggle_hold()
        # EE target sphere
        elif keycode in (ord("U"), ord("u")):
            teleop.nudge_ee(0, +_STEP_EE_RADIUS)
        elif keycode in (ord("J"), ord("j")):
            teleop.nudge_ee(0, -_STEP_EE_RADIUS)
        elif keycode in (ord("I"), ord("i")):
            teleop.nudge_ee(1, +_STEP_EE_PITCH)
        elif keycode in (ord("K"), ord("k")):
            teleop.nudge_ee(1, -_STEP_EE_PITCH)
        elif keycode in (ord("O"), ord("o")):
            teleop.nudge_ee(2, +_STEP_EE_YAW)
        elif keycode in (ord("L"), ord("l")):
            teleop.nudge_ee(2, -_STEP_EE_YAW)
        elif keycode in (ord("P"), ord("p")):
            teleop.reset_ee()
        # EE impulse push: arrows = fx/fy, PageUp/PageDn = fz
        elif keycode == _KEY_UP:
            teleop.push_ee(0, +1.0)
        elif keycode == _KEY_DOWN:
            teleop.push_ee(0, -1.0)
        elif keycode == _KEY_LEFT:
            teleop.push_ee(1, +1.0)
        elif keycode == _KEY_RIGHT:
            teleop.push_ee(1, -1.0)
        elif keycode == _KEY_PAGE_UP:
            teleop.push_ee(2, +1.0)
        elif keycode == _KEY_PAGE_DOWN:
            teleop.push_ee(2, -1.0)
        # Base impulse push: number keys 1/2 = fx, 3/4 = fy, 5/6 = fz
        elif keycode == ord("1"):
            teleop.push_base(0, +1.0)
        elif keycode == ord("2"):
            teleop.push_base(0, -1.0)
        elif keycode == ord("3"):
            teleop.push_base(1, +1.0)
        elif keycode == ord("4"):
            teleop.push_base(1, -1.0)
        elif keycode == ord("5"):
            teleop.push_base(2, +1.0)
        elif keycode == ord("6"):
            teleop.push_base(2, -1.0)
        else:
            return
        print(f"[play] {teleop.describe()}")

    obs = wrapped_env.reset()[0]["actor"]
    _print_legend()
    print("[play] Opening viewer — close the window or press Esc to quit.")
    diag = {"i": 0}  # throttles the force-estimate print (every 25 control steps)

    with mujoco.viewer.launch_passive(viz_model, viz_data, key_callback=_key_callback) as viewer:
        viewer.cam.distance = 2.5
        with torch.inference_mode():
            while viewer.is_running():
                t0 = time.perf_counter()

                if reset_requested["v"]:
                    reset_requested["v"] = False
                    obs = wrapped_env.reset()[0]["actor"]
                    teleop.reset()
                    print("[play] reset")

                if not paused["v"]:
                    # Advance the trapezoidal push and write teleop base velocity
                    # before stepping, so this step's physics + observation follow
                    # them (EE goal + forces are read by the patched hooks during
                    # the step).
                    teleop.advance_forces()
                    if env.state is not None:
                        env.state.info["commands"][:, CMD_VEL] = teleop.base_vel
                    obs = wrapped_env.step(policy(obs))[0]["actor"]
                    diag["i"] += 1
                    if diag["i"] % 25 == 0:
                        _print_force_estimate(runner, env, obs)

                phys = env.get_physics_state_snapshot()[0].astype(np.float64)
                mujoco.mj_setState(viz_model, viz_data, phys, state_spec)
                mujoco.mj_forward(viz_model, viz_data)

                base_pos = viz_data.xpos[base_body_id]
                viewer.cam.lookat[:] = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]

                _draw_markers(viewer, env, teleop, show_range=show_range["v"])
                viewer.sync()

                sleep = ctrl_dt - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)

    print("[play] Done.")


def _mujoco_visual_xml_paths(env: Any) -> tuple[Path, Path]:
    scene = getattr(env.cfg, "scene", None)
    static_model_file = None if scene is None else getattr(scene, "model_file", None)
    backend = getattr(env, "_backend", None)
    parent_xml = getattr(backend, "scene_visual_model_file", None) or static_model_file
    if parent_xml is None:
        raise ValueError("MuJoCo viewer requires cfg.scene or backend scene_visual_model_file.")
    robot_xml = static_model_file or parent_xml
    return Path(parent_xml), Path(robot_xml)


@hydra.main(version_base="1.3", config_path="../conf/ppo_cse", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()
    device = _select_device(cfg)
    print(f"[play] Device: {device}")
    play_interactive(cfg, device)


if __name__ == "__main__":
    main()
