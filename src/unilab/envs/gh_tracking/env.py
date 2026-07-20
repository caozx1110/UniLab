"""GH humanoid tracking environment — Phase 9 live integration.

Wires the Phase 1-7 components into the NpEnv lifecycle. Backend substep loop
(apply_action seed + registered pre-step control / pre-step wrench / post-step
telemetry hooks) reproduces GH ``_update_sim``; ``update_state`` reproduces GH
``before_update -> termination.update -> _compute_reward -> _compute_obs``
(blueprint §四). Reset (D2 order) runs through the DR provider.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.np_env import NpEnv, NpEnvState
from unilab.envs.gh_tracking import observations as obs_terms
from unilab.envs.gh_tracking import rewards as rwd
from unilab.envs.gh_tracking.config import GHTrackingCfg
from unilab.envs.gh_tracking.motion_dataset import (
    BODY_NAMES,
    FUTURE_STEPS,
    JOINT_NAMES,
    motion_to_init_qpos_qvel,
)
from unilab.envs.gh_tracking.observations import ObsState
from unilab.envs.gh_tracking.terminations import (
    apply_terminate_gate,
    compute_truncation,
    sample_reset_episode_length,
)

# --- g1_gh model structure (GH canonical) ---------------------------------- #
# keypoints (11) / height (4) / feet (2): tests/.../test_observations.py + §六.
_KEYPOINT_BODIES = (
    "head_mimic", "left_hand_mimic", "right_hand_mimic",
    "left_wrist_roll_link", "right_wrist_roll_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_knee_link", "right_knee_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
)
_HEIGHT_BODIES = ("pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link")
_FEET_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
# force bodies in USD order [L,L,R,R,L,R] (blueprint §五, source-compatible): the
# constant left/right masks over THIS order are non-anatomical (quirk 3).
_FORCE_BODIES = (
    "left_shoulder_yaw_link", "left_wrist_roll_link",
    "right_shoulder_yaw_link", "right_wrist_roll_link",
    "left_hand_mimic", "right_hand_mimic",
)
_TORSO_BODY = "torso_link"

# Reward group -> [(term, weight)] (blueprint §八 table). Signs live in the term
# formulas (penalties return negative). feet_air_time_ref (weight 10.0) wired in T10.1
# via the stateful FeetAirTimeRef + motion feet_standing (GH locomotion.py:163-213).
_REWARD_GROUPS: dict[str, list[tuple[str, float]]] = {
    "impedance": [
        ("force_reward", 2.0),
        ("force_exd_penalty", 6.0),
        ("force_target_tracking", 2.0),
        ("force_target_vel_tracking", 1.0),
        ("keypoint_tracking_imp", 2.0),
    ],
    "tracking": [
        ("lower_keypoint_tracking", 2.0),
        ("root_pos_tracking", 0.5),
        ("root_rot_tracking", 0.5),
        ("root_vel_tracking", 1.0),
        ("root_ang_vel_tracking", 1.0),
        ("joint_pos_tracking", 1.0),
        ("joint_vel_tracking", 0.5),
    ],
    "loco": [
        ("survival", 5.0),
        ("impact_force_l2", 4.0),
        ("feet_slip", 2.0),
        ("joint_vel_l2", 5e-4),
        ("action_rate_l2", 0.1),
        ("feet_air_time_ref", 10.0),
        ("joint_pos_limits", 1.0),
    ],
}
# Per-term exp-kernel sigma lists — EXACT from GH G1_gentle.yaml:55-69 (reward_sigma).
# calc_exp_sigma is sum_i exp(-err/sigma_i)/len (GH motion_tracking.py:43-49; the YAML's
# "# product" comments are misleading — the code sums). keypoint_imp reuses the keypoint
# sigma (GH keypoint_tracking_imp -> _calc_exp_sigma(err, reward_sigma["keypoint"])).
# NOTE: these are the reward sigmas, NOT the _cum_error termination scales
# (pos/0.3, rot/0.7, keypoint/0.25) which live in rewards.py CUM_*_SCALE.
_REWARD_SIGMA: dict[str, list[float]] = {
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
# tracking-joint subset patterns (GH joint_patterns override: waist3 + hip6 + knee2 +
# wrist6 = 17 joints; blueprint §八 / rewards.resolve_tracking_joints).
_TRACKING_JOINT_PATTERNS = [
    "waist_.*", ".*_hip_.*", ".*_knee_.*", ".*_wrist_.*",
]

# Force-origin resample constants (GH motion_tracking.py:577/586).
_FORCE_ORIGIN_SAMPLE_PROB = 0.5
_FORCE_ORIGIN_TRANSIT_RANGE = (25, 100)


def _quat_from_angle_axis(angle: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """(w,x,y,z) quat from angle (n,) + axis (n,3) (axis normalized internally),
    matching GH ``quat_from_angle_axis`` used for root-orientation init noise."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-8)
    half = 0.5 * np.asarray(angle, dtype=np.float64)
    return np.concatenate([np.cos(half)[:, None], np.sin(half)[:, None] * axis], axis=-1)


@registry.env("GHTracking", sim_backend="mujoco")
@registry.env("GHTracking", sim_backend="motrix")
class GHTrackingEnv(NpEnv):
    """GH humanoid tracking environment — student-teacher distillation with force-based control."""

    _cfg: GHTrackingCfg

    def __init__(self, cfg: GHTrackingCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        # Construct backend with add_body_sensors=True BEFORE super().__init__
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            add_body_sensors=True,  # Force system needs body-pose/contact queries
            base_name=cfg.asset.base_name,
            **env_backend_kwargs(cfg),
        )
        super().__init__(cfg, backend, num_envs)

        # Action space: unbounded Box over the actuated DOF (see Task 9.1).
        self._init_action_space()

        # --- resolve model structure (cold path) ---
        self._keypoint_ids = backend.get_body_ids(list(_KEYPOINT_BODIES))
        self._height_ids = backend.get_body_ids(list(_HEIGHT_BODIES))
        self._feet_ids = backend.get_body_ids(list(_FEET_BODIES))
        self._force_ids = backend.get_body_ids(list(_FORCE_BODIES))
        self._torso_id = int(backend.get_body_ids([_TORSO_BODY])[0])
        num_dof = int(backend.num_actuators)  # 29
        # default joint pose = keyframe joints (qpos[7:] after free-joint root)
        self._default_joint_pos = np.asarray(
            backend.get_keyframe_qpos(cfg.asset.keyframe_name)[7:], dtype=np.float64
        )
        body_mass = np.asarray(backend.get_body_mass(), dtype=np.float64)
        self._mass_total = float(body_mass.sum())
        num_com_bodies = int(body_mass.shape[0])

        # joint soft limits for joint_pos_limits reward: mid ∓ 0.5·range·soft_factor
        # (GH rewards/locomotion.py:274-275). Ranges from the model (cold path).
        jr = backend.get_joint_range()
        _sf = float(cfg.reward.joint_pos_limits_soft_factor)
        if jr is not None:
            jr = np.asarray(jr, dtype=np.float64)  # (num_dof, 2) [lo, hi]
            _mid = 0.5 * (jr[:, 0] + jr[:, 1])
            _half = 0.5 * (jr[:, 1] - jr[:, 0])
            self._joint_soft_lo = _mid - _half * _sf
            self._joint_soft_hi = _mid + _half * _sf
        else:  # pragma: no cover - MuJoCo always returns ranges for g1_gh
            self._joint_soft_lo = np.full(num_dof, -1e9)
            self._joint_soft_hi = np.full(num_dof, 1e9)

        # hand-grid samples for the compliant force-origin resample (real GH file
        # scripts/data_process/hand_grid_samples.pt -> bundled npz; torso frame,
        # (5414, 3, 3) per side for the 3 left/right-masked force bodies).
        _hg = np.load(str(ASSETS_ROOT_PATH / "robots" / "g1_gh" / "hand_grid_samples.npz"))
        self._hand_grid_left = _hg["left"].astype(np.float64)
        self._hand_grid_right = _hg["right"].astype(np.float64)

        # env-owned master RNG (shared by the per-control-step action/force draws)
        self._rng = np.random.default_rng(cfg.motion.seed)

        # --- Phase 4: action pipeline (residual -> per-substep joint targets) ---
        from unilab.envs.gh_tracking.action import GHActionPipeline

        self.action_pipeline = GHActionPipeline(
            joint_names=list(JOINT_NAMES),
            action_scaling=dict(cfg.control_config.action_scale),  # per-joint regex (G1_gentle.yaml:13-22)
            default_joint_pos=self._default_joint_pos,
            num_envs=num_envs,
            decimation=cfg.sim_substeps,
            max_delay=cfg.action.max_delay,
            alpha_range=cfg.action.alpha_range,
            alpha_wide=cfg.action.alpha_wide,
            alpha_jit_scale=cfg.action.alpha_jit_scale,
            boot_protect=cfg.action.boot_protect,
        )

        # --- Phase 5: compliant force system ---
        from unilab.envs.gh_tracking.force_system import ForceSystem

        self.force_system = ForceSystem(
            num_envs=num_envs,
            num_force_bodies=cfg.force.num_force_bodies,
            physics_dt=cfg.sim_dt,
            step_dt=cfg.ctrl_dt,
            max_force=cfg.force.max_force,
            net_force_limit=cfg.force.net_force_limit,
            net_torque_limit=cfg.force.net_torque_limit,
            force_alpha=cfg.force.force_alpha,
            compliance=cfg.force.compliance,
            seed=cfg.force.seed,
        )

        # --- Phase 6: observations (owns per-substep telemetry buffers) ---
        # actuator_offset is shared by reference with the action pipeline's zero
        # offset (GH joint_pos_history subtracts the action manager's offset).
        from unilab.envs.gh_tracking.observations import ObservationManager

        self.obs_manager = ObservationManager(
            num_envs=num_envs,
            mass_total=self._mass_total,
            actuator_offset=self.action_pipeline.offset,
            num_joints=num_dof,
            num_keypoints=len(_KEYPOINT_BODIES),
            num_force_bodies=cfg.force.num_force_bodies,
            num_feet=len(_FEET_BODIES),
            boot_indicator_max=cfg.obs.boot_indicator_max,
            seed=cfg.obs.seed,
        )

        # --- Phase 7: termination / domain randomization / student root cache ---
        from unilab.envs.gh_tracking.domain_rand import GHDomainRand
        from unilab.envs.gh_tracking.rewards import StudentRootCache
        from unilab.envs.gh_tracking.terminations import CumErrorTermination

        self.termination = CumErrorTermination(
            num_envs, thres=cfg.termination.thres, min_steps=cfg.termination.min_steps
        )
        self.dr = GHDomainRand(
            num_envs, num_joints=num_dof, num_com_bodies=num_com_bodies, seed=cfg.domain_rand.seed
        )
        self.student_cache = StudentRootCache(num_envs, steps=50)

        # motion-frame body indices (into the 27 motion BODY_NAMES) for the 11
        # keypoints and 6 force bodies, plus each force body's slot among the 11
        # keypoints (keypoint_tracking_imp compliant-target substitution).
        self._motion_kp_idx = np.array([BODY_NAMES.index(n) for n in _KEYPOINT_BODIES], dtype=np.int64)
        self._motion_force_idx = np.array([BODY_NAMES.index(n) for n in _FORCE_BODIES], dtype=np.int64)
        self._force_in_kp_idx = np.array(
            [_KEYPOINT_BODIES.index(n) for n in _FORCE_BODIES], dtype=np.int64
        )
        # lower-body keypoints (knees + ankles) for lower_keypoint_tracking
        self._lower_kp_slots = np.array(
            [_KEYPOINT_BODIES.index(n) for n in
             ("left_knee_link", "right_knee_link", "left_ankle_roll_link", "right_ankle_roll_link")],
            dtype=np.int64,
        )
        # 17-joint tracking subset (waist/hip/knee/wrist), GH joint_patterns override
        self._track_joint_ids = rwd.resolve_tracking_joints(list(JOINT_NAMES), _TRACKING_JOINT_PATTERNS)

        # per-env runtime counters
        self._motion_t = np.zeros(num_envs, dtype=np.int64)
        self._episode_length = np.zeros(num_envs, dtype=np.int64)
        # previous-control-step tracking-joint pos, for joint_vel_tracking diff
        self._prev_track_joint_pos = np.zeros((num_envs, self._track_joint_ids.shape[0]), dtype=np.float64)

        # joint-velocity 2-slot sub-sampler for joint_vel_l2 (fed per substep in Task 9.4)
        self._joint_vel_2slot = rwd.JointVel2Slot(num_envs, num_dof)

        # feet air-time reward state (T10.1) + motion feet-standing reference buffer
        self._feet_air = rwd.FeetAirTimeRef(
            num_envs, len(_FEET_BODIES), thres=cfg.reward.feet_air_time_thres, step_dt=cfg.ctrl_dt
        )
        self._motion_feet_idx = np.array(
            [BODY_NAMES.index(n) for n in _FEET_BODIES], dtype=np.int64
        )
        self._feet_standing = np.zeros((num_envs, len(_FEET_BODIES)), dtype=bool)

        # reward manager (3-group vector · step_dt); reward context populated each step
        self._rc: dict[str, Any] = {}
        self._reward_manager = self._build_reward_manager()

        # last absolute joint target written to sim (priv.applied_action obs term)
        self._last_joint_target = np.broadcast_to(
            self._default_joint_pos, (num_envs, num_dof)
        ).copy()

        # --- Phase 3: weighted motion dataset (guarded on data availability) ---
        # Empty dirs (default / no real data yet, DP2) -> deferred; reset/step
        # tests and the owner YAML provide (synthetic) dataset dirs + weights.
        from unilab.envs.gh_tracking.motion_dataset import WeightedMotionDataset

        self.motion: WeightedMotionDataset | None = None
        self._motion_len = np.zeros(num_envs, dtype=np.int64)
        self._motion_slice = None
        self._student_reset_ids: np.ndarray | None = None  # just-reset envs pending cache fill
        if cfg.motion.dirs:
            self.motion = WeightedMotionDataset(
                dirs=list(cfg.motion.dirs),
                weights=list(cfg.motion.weights),
                env_size=num_envs,
                max_step=cfg.motion.max_step,
                seed=cfg.motion.seed,
                sample_once=cfg.motion.sample_once,
            )
            # per-env fixed-clip lengths (sample_once); used for clip-finished truncation
            self._motion_len = self.motion.reset(np.arange(num_envs))

        # State buffers (Phase 7 D2 producer/consumer: reward writes, obs reads).
        self._cum_error = np.zeros((num_envs, 3))

        # Register the per-substep actuator-control hook (Phase 4). The backend calls
        # it every physics substep to recompute delayed/lerped joint targets; wrap it
        # to capture the final absolute joint target (priv.applied_action obs term).
        _pipe_ctrl = self.action_pipeline.as_pre_step_control()

        def _control_hook(bk: Any, ctrl: np.ndarray) -> np.ndarray:
            tgt = _pipe_ctrl(bk, ctrl)
            self._last_joint_target = np.asarray(tgt, dtype=np.float64)
            return tgt

        backend.set_pre_step_control(_control_hook)

        # Per-substep compliant-wrench recompute (Phase 5): apply_body_wrench on the
        # 6 force bodies + torso from the current body pose, zeroed each substep.
        backend.set_pre_step_wrench(
            self.force_system.as_pre_step_wrench(self._force_ids, self._torso_id)
        )
        # Per-substep telemetry (Phase 6): joint 2-slot (substep%2 -> avg 2/3), root
        # linvel EMA (4x), contact-force history (3) — fires every substep incl. last.
        self._post_substep = 0
        backend.set_post_step_callback(self._post_step_telemetry)

        # Domain-randomization provider drives the reset lifecycle (D2 order) and
        # materializes the backend pool (NpEnv._init_domain_randomization).
        from unilab.envs.gh_tracking.dr_provider import GHTrackingDRProvider

        self._init_domain_randomization(GHTrackingDRProvider(self))

    def _init_action_space(self) -> None:
        nu = self._backend.num_actuators
        self._action_space = gym.spaces.Box(-np.inf, np.inf, (nu,), dtype=float)

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space  # type: ignore[no-any-return]

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        """Three observation groups per GH algorithm (DENYLIST field)."""
        return {
            "policy": 450,
            "priv": 717,
            "priv_critic": 3,
        }

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        """Ingest a control-step action; return a seed ctrl.

        The Phase 4 pipeline recomputes per-substep joint targets inside the
        registered ``set_pre_step_control`` hook (communication delay, alpha lerp,
        boot protection). This method only ingests the raw action for the step
        (``start_control_step``: clamp to [-10, 10], roll the action buffer, jitter
        alpha) and returns the default-pose seed of the correct shape; the backend
        overrides it every substep via the hook.
        """
        actions = np.asarray(actions, dtype=np.float64)
        # start-of-control-step substep counters for the per-substep hooks
        self._post_substep = 0
        self.force_system.reset_force_substep()
        self.action_pipeline.start_control_step(actions, self._rng)
        return np.broadcast_to(
            self._default_joint_pos, (self._num_envs, self.action_pipeline.action_dim)
        ).copy()

    def _post_step_telemetry(self, backend: Any) -> None:
        """Per-substep telemetry hook (Phase 1 post-step contract: fires after every
        physics substep incl. the last). Samples the joint 2-slot (substep%2 -> the
        substep 2/3 mean feeds the joint-pos history and joint_vel_l2), the root
        linear-velocity EMA (4x/control step), and the 3-frame contact-force history.
        """
        substep = self._post_substep
        state = self._build_obs_state()
        self.obs_manager.post_step(substep, state)
        self._joint_vel_2slot.post_step(substep, backend.get_dof_vel())
        self._post_substep = substep + 1

    # ------------------------------------------------------------------ #
    # Reward wiring                                                        #
    # ------------------------------------------------------------------ #
    def _build_reward_manager(self) -> rwd.RewardManager:
        """Build the 3-group RewardManager. term_fns read the per-step reward
        context ``self._rc`` (passed as the compute ``state``)."""
        sig = _REWARD_SIGMA
        fns = {
            # loco
            "survival": lambda s: rwd.survival(s["n"]),
            "action_rate_l2": lambda s: rwd.action_rate_l2(s["action_buf"]),
            "impact_force_l2": lambda s: rwd.impact_force_l2(
                s["net_force_hist"], s["first_contact"], s["mass_total"]),
            "feet_slip": lambda s: rwd.feet_slip(s["in_contact"], s["feet_vel_xy"]),
            "joint_vel_l2": lambda s: rwd.joint_vel_l2(s["joint_vel_mean"]),
            "joint_pos_limits": lambda s: rwd.joint_pos_limits(
                s["joint_pos"], s["soft_lo"], s["soft_hi"], s["soft_factor"]),
            "feet_air_time_ref": lambda s: s["feet_air_time_reward"],
            # tracking
            "lower_keypoint_tracking": lambda s: rwd.lower_keypoint_tracking(
                s["lower_actual_kp_w"], s["lower_target_kp_w"], sig["lower_keypoint"]),
            "root_pos_tracking": lambda s: rwd.root_pos_tracking(
                s["root_pos_w"], s["reward_root_pos_w"], sig["root_pos"])[0],
            "root_rot_tracking": lambda s: rwd.root_rot_tracking(
                s["root_quat_w"], s["reward_root_quat_w"], sig["root_rot"])[0],
            "root_vel_tracking": lambda s: rwd.root_vel_tracking(s["root_vel_err"], sig["root_vel"]),
            "root_ang_vel_tracking": lambda s: rwd.root_vel_tracking(
                s["root_ang_vel_err"], sig["root_ang_vel"]),
            "joint_pos_tracking": lambda s: rwd.joint_pos_tracking(
                s["track_joint_actual"], s["track_joint_target"], sig["joint_pos"]),
            "joint_vel_tracking": lambda s: rwd.joint_vel_tracking(
                s["track_joint_vel_diff"], s["track_joint_vel_target"], sig["joint_vel"]),
            # impedance
            "force_reward": lambda s: rwd.force_reward(
                s["force_applied_w"], s["force_expected_w"], s["force_safe_limit"], sig["force"]),
            "force_exd_penalty": lambda s: rwd.force_exd_penalty(
                s["force_applied_w"], s["force_expected_w"], s["force_safe_limit"]),
            "force_target_tracking": lambda s: rwd.force_target_tracking(
                s["force_body_w"], s["force_keypoint_w"], sig["force_target"]),
            "force_target_vel_tracking": lambda s: rwd.force_target_vel_tracking(
                s["force_body_vel_w"], s["force_keypoint_vel_w"], sig["force_vel"]),
            "keypoint_tracking_imp": lambda s: rwd.keypoint_tracking_imp(
                s["actual_kp_w"], s["target_kp_w"], s["force_keypoint_w"],
                s["force_in_kp_idx"], sig["keypoint_imp"])[0],
        }
        return rwd.RewardManager(
            groups=_REWARD_GROUPS, term_fns=fns, step_dt=self._cfg.ctrl_dt,
            student_train=self._cfg.student_train,  # T10.3: student forces reward progress=1.0
        )

    # ------------------------------------------------------------------ #
    # Lifecycle: update_state (GH _step steps 2-7, blueprint §四)          #
    # ------------------------------------------------------------------ #
    def update_state(self, state: NpEnvState) -> NpEnvState:
        """Reproduce GH ``_step`` after the substep loop: before_update ->
        termination.update (BEFORE reward, 1-step _cum_error lag) -> reward
        (·step_dt per group) -> observation (priv_critic consumes _cum_error) ->
        termination/truncation decision."""
        self._before_update()
        # termination counter uses the PREVIOUS step's _cum_error (1-step lag)
        self.termination.update(self._cum_error)
        reward_vec = self._compute_reward()  # writes THIS step's _cum_error
        reward = reward_vec.sum(axis=-1)  # scalar for NpEnv; 3-vector stashed for GAE (Phase 10)
        obs = self._compute_obs()  # priv_critic == _cum_error (consumer)

        terminated = apply_terminate_gate(self.termination.terminated(), self._episode_length)[:, 0]
        truncated = compute_truncation(
            self._episode_length, self._cfg.termination.max_episode_length, self._motion_finished()
        )[:, 0]
        info = dict(state.info)
        info["reward_vec"] = reward_vec  # (N,3) for modewise GAE in Phase 10
        return state.replace(
            obs=obs, reward=reward, terminated=terminated, truncated=truncated, info=info
        )

    def _before_update(self) -> None:
        """GH command_manager.before_update: advance motion frame, gather future
        frames, run the admittance integration + force curriculum (blueprint §四.2/§五)."""
        self._motion_t = self._motion_t + 1
        self._episode_length = self._episode_length + 1
        if self.motion is not None:
            self._motion_slice = self.motion.get_slice(
                np.arange(self._num_envs), self._motion_t, FUTURE_STEPS
            )
            # student root cache: fill 50 future frames for just-reset envs at the
            # POST-increment t (GH update_reward_target last_reset_env_ids branch,
            # motion_tracking.py:458-461) — aligns the 1-frame boundary with GH.
            if self._student_reset_ids is not None:
                self._fill_student_cache(self._student_reset_ids)
                self._student_reset_ids = None
            # motion feet-standing: reference foot XY speed < 0.1 (GH motion_tracking.py:492).
            # feet_vel = (future[1] - future[0]) foot positions / their time gap.
            dt01 = float(FUTURE_STEPS[1] - FUTURE_STEPS[0]) * self._cfg.ctrl_dt
            feet_bp = self._motion_slice.body_pos_w[:, :, self._motion_feet_idx, :].astype(np.float64)
            feet_vel = (feet_bp[:, 1] - feet_bp[:, 0]) / dt01  # (N, 2, 3)
            self._feet_standing = np.linalg.norm(feet_vel[:, :, :2], axis=-1) < 0.1
            # compliant admittance force machinery (GH before_update:1032-1044, gentle branch).
            # Non-compliant variants (no_force / extreme) skip it; the extreme perturb-target
            # update lands in Task B.
            if self.force_system.compliance:
                root_pos_w = self._backend.get_base_pos()
                root_quat_w = self._backend.get_base_quat()
                ref_kp_force_w = self._motion_slice.body_pos_w[:, 0][:, self._motion_force_idx, :].astype(
                    np.float64
                )
                self.force_system.force_update_origin_and_target(
                    root_pos_w=root_pos_w, root_quat_w=root_quat_w, ref_keypoints_w=ref_kp_force_w
                )
                self.force_system.last_reset_env_ids = None  # consumed by the update above
                self.force_system.force_schedule(self._rng)
                # set spring origins for envs that resampled a new force this step
                self._resample_force_origin(self.force_system._need_origin_resample)
            else:
                self.force_system.last_reset_env_ids = None  # clear pending reset (no compliant update)

    def _motion_finished(self) -> np.ndarray:
        if self.motion is None:
            return np.zeros((self._num_envs, 1), dtype=bool)
        return (self._motion_t >= (self._motion_len - 1)).reshape(-1, 1)

    @staticmethod
    def _body_frame_vel_err(cur_quat, cur_vel_w, ref_quat, ref_vel_w) -> np.ndarray:
        """GH root_vel / root_ang_vel_tracking error: ‖ref_vel_b - cur_vel_b‖, each
        velocity rotated into its OWN root frame (motion_tracking.py:369-406)."""
        from unilab.utils.rotation import np_quat_apply_inverse

        cur_b = np_quat_apply_inverse(
            np.asarray(cur_quat, dtype=np.float64), np.asarray(cur_vel_w, dtype=np.float64))
        ref_b = np_quat_apply_inverse(
            np.asarray(ref_quat, dtype=np.float64), np.asarray(ref_vel_w, dtype=np.float64))
        return np.linalg.norm(ref_b - cur_b, axis=-1, keepdims=True)

    def _select_reward_root_target(
        self, root_pos_w, root_quat_w, m_pos0, m_quat0, env_origins
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reward root target (GH update_reward_target, motion_tracking.py:442-484):
        teacher (train) = current motion frame + env_origins; student (adapt/finetune) =
        StudentRootCache head, rolling in a robot-anchored ref t->t+50 displacement tail.

        Cache-fill boundary (fill at reset-t vs GH's post-increment t) is a 1-frame
        element-wise-parity detail (§十二:721, needs GH runtime dump)."""
        if not self._cfg.student_train:
            return rwd.teacher_reward_target(m_pos0, m_quat0, env_origins)
        if self.motion is not None:
            plus = self.motion.get_slice(
                np.arange(self._num_envs), self._motion_t, np.array([50], dtype=np.int64))
            ref_pos_t_plus = plus.root_pos_w[:, 0].astype(np.float64)
            ref_quat_t_plus = plus.root_quat_w[:, 0].astype(np.float64)
        else:
            ref_pos_t_plus, ref_quat_t_plus = m_pos0, m_quat0
        return self.student_cache.step(
            root_pos_w, root_quat_w, m_pos0, m_quat0, ref_pos_t_plus, ref_quat_t_plus)

    def _compute_reward(self) -> np.ndarray:
        """Compute the 3-group reward vector and (as producer) the instantaneous
        _cum_error (root-pos/rot/keypoint normalized errors, teacher/student target)."""
        bk = self._backend
        n = self._num_envs
        env_origins = np.zeros((n, 3), dtype=np.float64)  # MuJoCo: each env at world origin
        root_pos_w = bk.get_base_pos()
        root_quat_w = bk.get_base_quat()
        actual_kp_w = bk.get_body_pos_w(self._keypoint_ids)  # (N,11,3)

        # motion references (current frame) for tracking + reward target
        if self.motion is not None:
            m_pos0 = self._motion_slice.root_pos_w[:, 0].astype(np.float64)
            m_quat0 = self._motion_slice.root_quat_w[:, 0].astype(np.float64)
            m_linvel0 = self._motion_slice.root_lin_vel_w[:, 0].astype(np.float64)
            m_angvel0 = self._motion_slice.root_ang_vel_w[:, 0].astype(np.float64)
            target_kp_w = self._motion_slice.body_pos_w[:, 0][:, self._motion_kp_idx, :].astype(
                np.float64
            )
            track_target = self._motion_slice.joint_pos[:, 0][:, self._track_joint_ids].astype(
                np.float64
            )
            m_joint_vel = self._motion_slice.joint_vel[:, 0][:, self._track_joint_ids].astype(
                np.float64
            )
        else:
            m_pos0, m_quat0 = root_pos_w, root_quat_w
            m_linvel0, m_angvel0 = bk.get_base_lin_vel(), bk.get_base_ang_vel_world()
            target_kp_w = actual_kp_w
            track_target = bk.get_dof_pos()[:, self._track_joint_ids]
            m_joint_vel = np.zeros((n, self._track_joint_ids.shape[0]), dtype=np.float64)
        # reward root target: teacher = current motion frame; student = 50-cache head (T10.3)
        reward_root_pos_w, reward_root_quat_w = self._select_reward_root_target(
            root_pos_w, root_quat_w, m_pos0, m_quat0, env_origins
        )

        # --- _cum_error producer (instantaneous normalized errors) ---
        _, cum_pos = rwd.root_pos_tracking(root_pos_w, reward_root_pos_w, _REWARD_SIGMA["root_pos"])
        _, cum_rot = rwd.root_rot_tracking(root_quat_w, reward_root_quat_w, _REWARD_SIGMA["root_rot"])
        _, cum_kp = rwd.keypoint_tracking(actual_kp_w, target_kp_w, _REWARD_SIGMA["keypoint"])
        self._cum_error[:, 0] = cum_pos[:, 0]
        self._cum_error[:, 1] = cum_rot[:, 0]
        self._cum_error[:, 2] = cum_kp[:, 0]

        # --- feet air-time (T10.1): stateful step -> reward + shared first_contact ---
        feet_air_reward, first_contact = self._feet_air.step(
            bk.get_body_contact_state(self._feet_ids),
            bk.get_body_pos_w(self._feet_ids)[:, :, 2],
            self._feet_standing,
        )

        # --- assemble the reward context (self._rc) for the term_fns ---
        fs = self.force_system
        track_actual = bk.get_dof_pos()[:, self._track_joint_ids]
        rc = self._rc
        rc.clear()
        rc.update(
            n=n,
            mass_total=self._mass_total,
            action_buf=self.action_pipeline.action_buf,
            net_force_hist=self.obs_manager.contact.history,
            first_contact=first_contact.astype(np.float64),  # T10.1: shared with impact_force_l2
            feet_air_time_reward=feet_air_reward,  # T10.1: precomputed stateful term value
            in_contact=bk.get_body_contact_state(self._feet_ids).astype(np.float64),
            feet_vel_xy=bk.get_body_lin_vel_w(self._feet_ids)[:, :, :2],
            joint_vel_mean=self._joint_vel_2slot.mean(),
            joint_pos=bk.get_dof_pos(),
            soft_lo=np.broadcast_to(self._joint_soft_lo, (n, self._joint_soft_lo.shape[0])),
            soft_hi=np.broadcast_to(self._joint_soft_hi, (n, self._joint_soft_hi.shape[0])),
            soft_factor=float(self._cfg.reward.joint_pos_limits_soft_factor),
            lower_actual_kp_w=actual_kp_w[:, self._lower_kp_slots, :],
            lower_target_kp_w=target_kp_w[:, self._lower_kp_slots, :],
            root_pos_w=root_pos_w,
            reward_root_pos_w=reward_root_pos_w,
            root_quat_w=root_quat_w,
            reward_root_quat_w=reward_root_quat_w,
            root_vel_err=self._body_frame_vel_err(
                root_quat_w, bk.get_base_lin_vel(), m_quat0, m_linvel0),
            root_ang_vel_err=self._body_frame_vel_err(
                root_quat_w, bk.get_base_ang_vel_world(), m_quat0, m_angvel0),
            track_joint_actual=track_actual,
            track_joint_target=track_target,
            track_joint_vel_diff=(track_actual - self._prev_track_joint_pos) / self._cfg.ctrl_dt,
            track_joint_vel_target=m_joint_vel,
            force_applied_w=fs.force_applied_w,
            force_expected_w=fs.force_expected_w,
            force_safe_limit=fs.force_safe_limit_tl.current,
            force_body_w=bk.get_body_pos_w(self._force_ids),
            force_keypoint_w=fs.force_keypoint_w,
            force_body_vel_w=bk.get_body_lin_vel_w(self._force_ids),
            force_keypoint_vel_w=fs.force_keypoint_vel_w,
            actual_kp_w=actual_kp_w,
            target_kp_w=target_kp_w,
            force_in_kp_idx=self._force_in_kp_idx,
        )
        self._prev_track_joint_pos = track_actual.copy()
        return self._reward_manager.compute(rc)  # (N, 3)

    def _build_obs_state(self) -> ObsState:
        """Assemble the 26-field ObsState from backend + motion + force + action."""
        from unilab.utils.rotation import np_quat_apply_inverse

        bk = self._backend
        n = self._num_envs
        root_quat_w = bk.get_base_quat()
        grav = np_quat_apply_inverse(root_quat_w, np.tile([0.0, 0.0, -1.0], (n, 1)))
        if self.motion is not None and self._motion_slice is not None:
            ms = self._motion_slice
            m_root_pos = ms.root_pos_w.astype(np.float64)
            m_root_quat = ms.root_quat_w.astype(np.float64)
            m_root_linvel = ms.root_lin_vel_w.astype(np.float64)
            m_joint = ms.joint_pos.astype(np.float64)
            m_body_kp = ms.body_pos_w[:, :, self._motion_kp_idx, :].astype(np.float64)
        else:
            s = FUTURE_STEPS.shape[0]
            m_root_pos = np.zeros((n, s, 3))
            m_root_quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, s, 1)).astype(np.float64)
            m_root_linvel = np.zeros((n, s, 3))
            m_joint = np.zeros((n, s, len(JOINT_NAMES)))
            m_body_kp = np.zeros((n, s, len(_KEYPOINT_BODIES), 3))
        fs = self.force_system
        return ObsState(
            root_pos_w=bk.get_base_pos(),
            root_quat_w=root_quat_w,
            root_ang_vel_b=bk.get_base_ang_vel(),
            projected_gravity_b=grav,
            env_origins=np.zeros((n, 3), dtype=np.float64),
            joint_pos_target=self._last_joint_target,
            applied_torque=bk.get_actuator_effort(),
            action_buf=self.action_pipeline.prev_actions(),
            body_pos_w_height=bk.get_body_pos_w(self._height_ids),
            body_pos_w_kp=bk.get_body_pos_w(self._keypoint_ids),
            body_lin_vel_w_kp=bk.get_body_lin_vel_w(self._keypoint_ids),
            motion_root_pos_w=m_root_pos,
            motion_root_quat_w=m_root_quat,
            motion_root_lin_vel_w=m_root_linvel,
            motion_joint_pos=m_joint,
            motion_body_pos_w_kp=m_body_kp,
            force_keypoint_b=fs.force_keypoint_b,
            force_applied_b=fs.force_applied_b,
            force_expected_b=fs.force_expected_b,
            force_sample_timer=fs.force_sample_timer.reshape(n, 1).astype(np.float64),
            force_safe_limit=fs.force_safe_limit_tl.current,
            boot_indicator=np.zeros((n, 1)),  # TODO(parity): GH boot_indicator source
            cum_error=self._cum_error,
            joint_pos=bk.get_dof_pos(),
            root_lin_vel_w=bk.get_base_lin_vel(),
            net_contact_force=bk.get_body_net_contact_force_w(self._feet_ids),
        )

    def _compute_obs(self) -> dict[str, np.ndarray]:
        """Roll the control-step obs histories then assemble the 3 groups."""
        state = self._build_obs_state()
        self.obs_manager.update(state)  # control-step history roll (after all substeps)
        return self.obs_manager.compute(state)

    # ------------------------------------------------------------------ #
    # Reset backbone (D2 order). The DR provider (dr_provider.py) wraps    #
    # _build_reset_plan_state (pre set_state) + _build_reset_observation   #
    # (post set_state); _reset_idx is the standalone helper for tests.     #
    # ------------------------------------------------------------------ #
    def _build_reset_plan_state(
        self, env_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """GH D2 reset (pre set_state): sample motion start -> noised init qpos/qvel;
        reset action/force/termination + fill student cache + sample per-reset DR.

        钉死不变量 (Phase 3/4 记账): boot-protect caches the NOISED init joint pose
        (read motion -> add init noise -> set_boot_protect_pose -> write sim)."""
        env_ids = np.asarray(env_ids)
        if self.motion is None:
            raise RuntimeError("motion dataset required for reset (set cfg.motion.dirs)")

        from unilab.utils.rotation import np_quat_mul

        starts = self.motion.sample_start_offsets(env_ids, self._cfg.motion.zero_init_prob, self._rng)
        self._motion_t[env_ids] = starts
        slice0 = self.motion.get_slice(env_ids, starts, FUTURE_STEPS)
        qpos, qvel = motion_to_init_qpos_qvel(slice0)

        # D2 init-state noise, per field randn.clamp(-1,1)*value (GH motion_tracking.py:208-235).
        nz = self._cfg.init_noise
        rng = self._rng
        m = qpos.shape[0]
        # root position (z-noise clamped >= 0: only lift, never sink)
        rp = np.clip(rng.standard_normal((m, 3)), -1.0, 1.0) * nz.root_pos
        rp[:, 2] = np.maximum(rp[:, 2], 0.0)
        qpos[:, 0:3] += rp
        # root orientation: quat_mul(random_quat(randn*root_ori, rand_axis), motion_quat)
        angle = np.clip(rng.standard_normal(m), -1.0, 1.0) * nz.root_ori
        qpos[:, 3:7] = np_quat_mul(_quat_from_angle_axis(angle, rng.random((m, 3))), qpos[:, 3:7])
        # root lin/ang vel: qvel[0:3] world lin, qvel[3:6] local ang. Isotropic Gaussian
        # noise is rotation-invariant in distribution, so a local-frame add on qvel[3:6]
        # matches GH's world-frame ang-vel noise (same per-component variance).
        qvel[:, 0:3] += np.clip(rng.standard_normal((m, 3)), -1.0, 1.0) * nz.root_lin_vel
        qvel[:, 3:6] += np.clip(rng.standard_normal((m, 3)), -1.0, 1.0) * nz.root_ang_vel
        # joint pos/vel noise (boot-protect caches the NOISED joint pose)
        qpos[:, 7:] += np.clip(rng.standard_normal(qpos[:, 7:].shape), -1.0, 1.0) * nz.joint_pos
        qvel[:, 6:] += np.clip(rng.standard_normal(qvel[:, 6:].shape), -1.0, 1.0) * nz.joint_vel
        noised_init_joint_pos = qpos[:, 7:].copy()

        self.action_pipeline.reset(env_ids, self._rng)
        self.action_pipeline.set_boot_protect_pose(env_ids, noised_init_joint_pos)
        # per-reset batch-shared joint zero offset -> action offset (obs shares the ref)
        self.action_pipeline.offset[env_ids] = self.dr.sample_joint_offset(env_ids)
        # per-reset motor params + startup-once material/COM (cached) sampled
        self.dr.sample_motor(env_ids)
        self.dr.startup_material()
        self.dr.startup_com()

        self.force_system.force_reset(env_ids)
        self.force_system.last_reset_env_ids = env_ids
        self.termination.reset(env_ids)
        self._feet_air.reset(env_ids)
        self._episode_length[env_ids] = sample_reset_episode_length(
            len(env_ids), self._num_envs, self._cfg.termination.max_episode_length, self._rng
        )
        # student cache fill is deferred to the next _before_update (GH last_reset_env_ids
        # branch, post-increment t) — aligns the 1-frame boundary with GH.
        self._student_reset_ids = np.asarray(env_ids)

        info_updates = {"boot_protect": noised_init_joint_pos}
        return qpos, qvel, info_updates

    def _fill_student_cache(self, env_ids: np.ndarray) -> None:
        """Fill the student root cache with 50 future motion frames from the current
        motion frame ``_motion_t`` (GH update_reward_target, post-increment t)."""
        env_ids = np.asarray(env_ids)
        fill = self.motion.get_slice(env_ids, self._motion_t[env_ids], np.arange(50, dtype=np.int64))
        env_origins = np.zeros((len(env_ids), 3), dtype=np.float64)
        self.student_cache.reset(
            env_ids,
            fill.root_pos_w.astype(np.float64),
            fill.root_quat_w.astype(np.float64),
            env_origins,
        )

    def _resample_force_origin(self, env_ids: np.ndarray) -> None:
        """GH force_reset_origin (motion_tracking.py:794-828): set the spring origins
        for envs that resampled a new force. start = current force-body root-frame
        position; with prob 0.5 per side (the ``force_type != 4`` gate is always true
        under quirk 2) the endpoint is a real hand-grid sample transformed
        torso->world->root; the origin lerps start->end over U(25,100) control steps."""
        env_ids = np.asarray(env_ids)
        if env_ids.size == 0:
            return
        from unilab.utils.rotation import np_quat_apply_batched, np_quat_apply_inverse

        bk = self._backend
        fs = self.force_system
        k = env_ids.shape[0]

        pos_w = bk.get_body_pos_w(self._force_ids)[env_ids]  # (k,6,3)
        root_pos = bk.get_base_pos()[env_ids][:, None, :]  # (k,1,3)
        root_quat = bk.get_base_quat()[env_ids]  # (k,4)
        q6 = np.broadcast_to(root_quat[:, None, :], (k, fs.M, 4)).reshape(-1, 4)
        pos_b_start = np_quat_apply_inverse(q6, (pos_w - root_pos).reshape(-1, 3)).reshape(pos_w.shape)
        pos_b_end = pos_b_start.copy()

        torso_pos = bk.get_body_pos_w(np.array([self._torso_id]))[env_ids][:, 0, :]  # (k,3)
        torso_quat = bk.get_body_quat_w(np.array([self._torso_id]))[env_ids][:, 0, :]  # (k,4)
        left_slots = np.flatnonzero(fs.left_mask.reshape(-1))  # {0,2,4}
        right_slots = np.flatnonzero(fs.right_mask.reshape(-1))  # {1,3,5}

        for slots, grid in ((left_slots, self._hand_grid_left), (right_slots, self._hand_grid_right)):
            take = np.flatnonzero(self._rng.random(k) < _FORCE_ORIGIN_SAMPLE_PROB)
            if take.size == 0:
                continue
            idx = np.floor(self._rng.random(take.size) * grid.shape[0]).astype(np.int64)
            link_torso = grid[idx]  # (m,3,3) torso frame
            tq = np.broadcast_to(torso_quat[take][:, None, :], link_torso.shape[:2] + (4,))
            link_w = np_quat_apply_batched(tq, link_torso) + torso_pos[take][:, None, :]  # (m,3,3)
            rq = np.broadcast_to(root_quat[take][:, None, :], link_torso.shape[:2] + (4,)).reshape(-1, 4)
            link_b = np_quat_apply_inverse(
                rq, (link_w - root_pos[take]).reshape(-1, 3)
            ).reshape(link_torso.shape)
            pos_b_end[np.ix_(take, slots)] = link_b

        steps = self._rng.integers(*_FORCE_ORIGIN_TRANSIT_RANGE, size=k)
        fs.force_origin_tl.set(env_ids, start=pos_b_start, end=pos_b_end, total_steps=steps)

    def _build_reset_observation(self, env_ids: np.ndarray) -> dict[str, np.ndarray]:
        """GH D2 reset (post set_state): refresh the motion slice, reset the obs
        history from the freshly-set sim state, and return the reset-env obs subset."""
        env_ids = np.asarray(env_ids)
        self._motion_slice = self.motion.get_slice(
            np.arange(self._num_envs), self._motion_t, FUTURE_STEPS
        )
        obs_state = self._build_obs_state()
        self.obs_manager.reset(env_ids, obs_state)
        self._prev_track_joint_pos[env_ids] = (
            self._backend.get_dof_pos()[env_ids][:, self._track_joint_ids]
        )
        obs = self.obs_manager.compute(obs_state)
        return {k: v[env_ids] for k, v in obs.items()}

    def _reset_idx(self, env_ids: np.ndarray) -> None:
        """Standalone reset (bypasses the DR manager) for unit tests. The DR provider
        runs the same _build_reset_plan_state / _build_reset_observation around the
        manager's set_state call."""
        env_ids = np.asarray(env_ids)
        qpos, qvel, _info = self._build_reset_plan_state(env_ids)
        self._backend.set_state(env_ids, qpos, qvel)
        self._build_reset_observation(env_ids)
