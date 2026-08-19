"""T800 adapter for the shared biped walk-flat task."""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.g1.base import Asset
from unilab.envs.locomotion.g1.joystick import (
    G1WalkControlConfig,
    G1WalkEnv,
    G1WalkFlatCfg,
    G1WalkRewardConfig,
)
from unilab.utils.rotation import np_quat_apply_inverse, np_yaw_quat

T800_ACTIVE_JOINT_NAMES = (
    "J00_HIP_PITCH_L",
    "J01_HIP_ROLL_L",
    "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L",
    "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R",
    "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R",
    "J10_ANKLE_PITCH_R",
    "J11_ANKLE_ROLL_R",
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
    "J18_SHOULDER_PITCH_R",
    "J19_SHOULDER_ROLL_R",
    "J20_SHOULDER_YAW_R",
    "J21_ELBOW_PITCH_R",
    "J22_ELBOW_YAW_R",
)

T800_ACTION_SCALE = (
    0.5,
    0.2,
    0.2,
    0.5,
    0.5,
    0.2,
    0.5,
    0.2,
    0.2,
    0.5,
    0.5,
    0.2,
    0.2,
    0.2,
    0.05,
    0.2,
    0.05,
    0.2,
    0.2,
    0.05,
    0.2,
    0.05,
)


def compute_lateral_feet_penalty(
    left_foot: np.ndarray,
    right_foot: np.ndarray,
    base_quat: np.ndarray,
    *,
    min_width: float,
    sigma: float,
) -> np.ndarray:
    """Return a soft lower-bound penalty for signed foot width in the heading frame."""
    foot_delta_world = np.asarray(left_foot) - np.asarray(right_foot)
    foot_delta_heading = np_quat_apply_inverse(np_yaw_quat(base_quat), foot_delta_world)
    width_deficit = np.maximum(float(min_width) - foot_delta_heading[:, 1], 0.0)
    return np.asarray(1.0 - np.exp(-np.square(width_deficit / float(sigma))))


def resolve_required_indices(available: tuple[str, ...], required: tuple[str, ...]) -> np.ndarray:
    """Resolve a stable required-name order and fail closed on model drift."""
    if len(set(available)) != len(available):
        raise ValueError("duplicate actuator names in T800 model")
    if len(set(required)) != len(required):
        raise ValueError("duplicate required T800 actuator names")

    lookup = {name: index for index, name in enumerate(available)}
    missing = [name for name in required if name not in lookup]
    if missing:
        raise ValueError(f"missing required T800 actuator names: {missing}")
    return np.asarray([lookup[name] for name in required], dtype=np.int32)


def expand_active_targets(
    active_targets: np.ndarray,
    full_default_targets: np.ndarray,
    active_actuator_indices: np.ndarray,
) -> np.ndarray:
    """Expand policy targets while preserving inactive position targets."""
    active_targets = np.asarray(active_targets)
    full_default_targets = np.asarray(full_default_targets)
    active_actuator_indices = np.asarray(active_actuator_indices, dtype=np.int32)
    if active_targets.ndim != 2:
        raise ValueError(f"T800 actions must be rank 2, got shape {active_targets.shape}")
    if active_targets.shape[1] != active_actuator_indices.size:
        raise ValueError(
            "T800 active target width does not match actuator mapping: "
            f"{active_targets.shape[1]} != {active_actuator_indices.size}"
        )

    ctrl = np.broadcast_to(
        full_default_targets,
        (active_targets.shape[0], full_default_targets.size),
    ).copy()
    ctrl[:, active_actuator_indices] = active_targets
    return ctrl


class T800Asset(Asset):
    base_name = "LINK_BASE"
    foot_name = "LINK_FOOT"
    ground = "floor"


class T800InitState:
    pos = [0.0, 0.0, 1.0165]


@dataclass
class T800WalkControlConfig(G1WalkControlConfig):
    action_scale: list[float] = field(default_factory=lambda: list(T800_ACTION_SCALE))  # type: ignore[assignment]


@dataclass
class T800WalkRewardConfig(G1WalkRewardConfig):
    close_feet_lateral_threshold: float = 0.20
    close_feet_lateral_sigma: float = 0.04


class T800WalkEnv(G1WalkEnv):
    """Expose 22 policy joints over the complete 25-actuator T800 model."""

    _active_actuator_indices: np.ndarray
    _active_dof_pos_indices: np.ndarray
    _active_dof_vel_indices: np.ndarray
    _full_default_ctrl: np.ndarray

    def __init__(
        self,
        cfg: T800WalkFlatCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> None:
        # Resolve visual assets before the backend parses t800.xml. Each
        # directory is an independent Hugging Face snapshot contract.
        from unilab.assets.hub import resolve_robot_asset_dir

        resolve_robot_asset_dir("robots/t800/assets", marker="LINK_BASE.obj")
        resolve_robot_asset_dir("robots/t800/textures", marker="LINK_BASE.png")
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)

    def _init_reward_functions(self) -> None:
        super()._init_reward_functions()
        self._reward_fns["penalty_close_feet_lateral"] = self._reward_close_feet_lateral

    def _reward_close_feet_lateral(self, ctx: object) -> np.ndarray:
        del ctx
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        return compute_lateral_feet_penalty(
            left_foot,
            right_foot,
            self._backend.get_base_quat(),
            min_width=self._reward_cfg.close_feet_lateral_threshold,
            sigma=self._reward_cfg.close_feet_lateral_sigma,
        )

    def _init_action_space(self) -> None:
        self._active_actuator_indices = resolve_required_indices(
            self._backend.get_actuator_names(), T800_ACTIVE_JOINT_NAMES
        )
        ctrl_range = self._backend.get_actuator_ctrl_range()[self._active_actuator_indices]
        self._action_space = gym.spaces.Box(
            ctrl_range[:, 0],
            ctrl_range[:, 1],
            (len(T800_ACTIVE_JOINT_NAMES),),
            dtype=float,
        )

    def _init_buffers(self) -> None:
        self._active_dof_pos_indices = self._backend.get_joint_dof_pos_indices(
            T800_ACTIVE_JOINT_NAMES
        )
        self._active_dof_vel_indices = self._backend.get_joint_dof_vel_indices(
            T800_ACTIVE_JOINT_NAMES
        )

        actuator_names = self._backend.get_actuator_names()
        actuator_dof_pos_indices = self._backend.get_joint_dof_pos_indices(actuator_names)
        raw_qpos = np.asarray(self._backend.get_keyframe_qpos(self._keyframe_name))
        root_qpos_dim = raw_qpos.size - self._backend.num_dof_vel
        if root_qpos_dim != 7:
            raise ValueError(
                f"T800 requires a free-joint root with 7 qpos values, got {root_qpos_dim}"
            )

        self._init_qpos = raw_qpos
        self._init_qvel = np.asarray(self._backend.get_init_qvel())
        dof_qpos = raw_qpos[root_qpos_dim:]
        self.default_angles = np.asarray(dof_qpos[self._active_dof_pos_indices])
        self._full_default_ctrl = np.asarray(dof_qpos[actuator_dof_pos_indices])

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # gyro(3) + gravity(3) + 3 * active joints(66) + command(3) + phase(2)
        return {"obs": 77, "critic": 80}

    def get_dof_pos(self) -> np.ndarray:
        return self._backend.get_dof_pos()[:, self._active_dof_pos_indices]

    def get_dof_vel(self) -> np.ndarray:
        return self._backend.get_dof_vel()[:, self._active_dof_vel_indices]

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[1] != self._num_action:
            raise ValueError(
                f"T800 actions must have shape (num_envs, {self._num_action}), got {actions.shape}"
            )
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions

        gait_phase = state.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        gait_phase[:, 0] = (gait_phase[:, 0] + self._gait_phase_delta) % (2 * np.pi)
        gait_phase[:, 1] = (gait_phase[:, 1] + self._gait_phase_delta) % (2 * np.pi)
        state.info["gait_phase"] = gait_phase

        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else actions
        )
        action_scale = np.asarray(self._cfg.control_config.action_scale, dtype=actions.dtype)
        if action_scale.shape != (self._num_action,):
            raise ValueError(
                f"T800 action_scale must match the 22 policy joints, got shape {action_scale.shape}"
            )
        active_targets = exec_actions * action_scale + self.default_angles
        return expand_active_targets(
            active_targets,
            self._full_default_ctrl,
            self._active_actuator_indices,
        )

    def build_symmetry_augmentation(self, *, device: str):
        del device
        return None


@registry.envcfg("T800WalkFlat")
@dataclass
class T800WalkFlatCfg(G1WalkFlatCfg):
    reward_config: T800WalkRewardConfig | None = None
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "t800" / "scene_flat.xml")
        )
    )
    init_state: T800InitState = field(default_factory=T800InitState)  # type: ignore[assignment]
    asset: T800Asset = field(default_factory=T800Asset)  # type: ignore[assignment]
    control_config: T800WalkControlConfig = field(default_factory=T800WalkControlConfig)  # type: ignore[assignment]
    sim_dt: float = 0.002
    ctrl_dt: float = 0.01

    def validate(self) -> None:
        super().validate()
        if self.reward_config is None:
            return
        if self.reward_config.close_feet_lateral_threshold < 0.0:
            raise ValueError("close_feet_lateral_threshold must be non-negative")
        if self.reward_config.close_feet_lateral_sigma <= 0.0:
            raise ValueError("close_feet_lateral_sigma must be positive")


registry.register_env("T800WalkFlat", T800WalkEnv, sim_backend="mujoco")
