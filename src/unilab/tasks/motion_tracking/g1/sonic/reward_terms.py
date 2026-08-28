"""SONIC v1 release-only Manager-Based reward terms."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np

from unilab.managers import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.tasks.motion_tracking.common.manager_terms import MotionCommand
from unilab.utils.rotation import np_quat_apply_batched, np_quat_apply_inverse_batched

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


SONIC_VR_POINT_BODY_NAMES: tuple[str, ...] = (
    "torso_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
SONIC_VR_POINT_OFFSETS = np.asarray(
    ((0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    dtype=np.float32,
)
SONIC_VR_POINT_OFFSETS.setflags(write=False)
SONIC_ANKLE_JOINT_NAMES: tuple[str, ...] = (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _command(env: ManagerBasedRlEnv, command_name: str) -> MotionCommand:
    try:
        command = env.command_manager.get_term(command_name)
    except KeyError as exc:
        raise KeyError(f"Motion command term '{command_name}' not found") from exc
    if not isinstance(command, MotionCommand):
        raise TypeError(
            f"Command term '{command_name}' is {type(command).__name__}, expected MotionCommand"
        )
    return command


def _body_ids(command: MotionCommand, body_names: tuple[str, ...]) -> np.ndarray:
    configured = tuple(command.cfg.body_names)
    missing = [name for name in body_names if name not in configured]
    if missing:
        raise ValueError(f"Bodies {missing} are not tracked by the motion command")
    return np.asarray([configured.index(name) for name in body_names], dtype=np.intp)


def anti_shake_ang_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    body_names: tuple[str, ...] | list[str],
    threshold: float = 1.5,
) -> np.ndarray:
    """Mean squared excess angular speed for selected tracked bodies."""

    command = _command(env, command_name)
    names = tuple(body_names)
    if not names:
        raise ValueError("anti_shake_ang_vel body_names must not be empty")
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(
        threshold, (int, float, np.number)
    ):
        raise TypeError("anti_shake_ang_vel threshold must be a real number")
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or threshold_value < 0.0:
        raise ValueError("anti_shake_ang_vel threshold must be finite and non-negative")
    indices = _body_ids(command, names)
    speeds = np.linalg.norm(command.robot_body_ang_vel_w[:, indices], axis=-1)
    excess = np.maximum(speeds - threshold_value, 0.0)
    return np.mean(np.square(excess), axis=-1)


def tracking_vr_5point_local(
    env: ManagerBasedRlEnv,
    command_name: str,
    body_names: tuple[str, ...] | list[str] = SONIC_VR_POINT_BODY_NAMES,
    std: float = 0.1,
) -> np.ndarray:
    """Track SONIC's torso and wrist points in each pelvis-local frame."""

    command = _command(env, command_name)
    names = tuple(body_names)
    if names != SONIC_VR_POINT_BODY_NAMES:
        raise ValueError(
            "tracking_vr_5point_local body_names must be the SONIC release points "
            f"{SONIC_VR_POINT_BODY_NAMES}"
        )
    if isinstance(std, (bool, np.bool_)) or not isinstance(std, (int, float, np.number)):
        raise TypeError("tracking_vr_5point_local std must be a real number")
    std_value = float(std)
    if not math.isfinite(std_value) or std_value <= 0.0:
        raise ValueError("tracking_vr_5point_local std must be finite and positive")

    point_ids = _body_ids(command, names)
    reference_pos = command.body_pos_w[:, point_ids]
    reference_quat = command.body_quat_w[:, point_ids]
    robot_pos = command.robot_body_pos_w[:, point_ids]
    robot_quat = command.robot_body_quat_w[:, point_ids]
    reference_points = reference_pos + np_quat_apply_batched(
        reference_quat, SONIC_VR_POINT_OFFSETS[None, :, :]
    )
    robot_points = robot_pos + np_quat_apply_batched(robot_quat, SONIC_VR_POINT_OFFSETS[None, :, :])
    reference_local = np_quat_apply_inverse_batched(
        command.anchor_quat_w[:, None, :],
        reference_points - command.anchor_pos_w[:, None, :],
    )
    robot_local = np_quat_apply_inverse_batched(
        command.robot_anchor_quat_w[:, None, :],
        robot_points - command.robot_anchor_pos_w[:, None, :],
    )
    squared_error = np.sum(np.square(reference_local - robot_local), axis=-1)
    return np.exp(-np.mean(squared_error, axis=-1) / (std_value**2))


class sonic_feet_acc_l2(ManagerTermBase):
    """Reset-aware finite-difference acceleration for SONIC's four ankles."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError("sonic_feet_acc_l2 asset_cfg must be SceneEntityCfg")
        if tuple(asset_cfg.joint_names or ()) != SONIC_ANKLE_JOINT_NAMES:
            raise ValueError(
                "sonic_feet_acc_l2 asset_cfg joint_names must exactly match the SONIC ankle order"
            )
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        joint_ids, joint_names = self._entity.find_joints(
            SONIC_ANKLE_JOINT_NAMES, preserve_order=True
        )
        if tuple(joint_names) != SONIC_ANKLE_JOINT_NAMES:
            raise ValueError("sonic_feet_acc_l2 entity joint order does not match SONIC ankles")
        self._joint_ids = np.asarray(joint_ids, dtype=np.intp)
        self._previous = self._entity.data.joint_vel[:, self._joint_ids].copy()

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        ids = np.arange(self.num_envs, dtype=np.intp)
        if env_ids is not None:
            ids = ids[env_ids]
        self._previous[ids] = self._entity.data.joint_vel[np.ix_(ids, self._joint_ids)]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> np.ndarray:
        del asset_cfg
        step_dt = float(env.step_dt)
        if not math.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError("sonic_feet_acc_l2 env.step_dt must be finite and positive")
        velocity = self._entity.data.joint_vel[:, self._joint_ids]
        acceleration = (velocity - self._previous) / step_dt
        self._previous[:] = velocity
        return np.sum(np.square(acceleration), axis=-1)


__all__ = [
    "SONIC_ANKLE_JOINT_NAMES",
    "SONIC_VR_POINT_BODY_NAMES",
    "SONIC_VR_POINT_OFFSETS",
    "anti_shake_ang_vel",
    "sonic_feet_acc_l2",
    "tracking_vr_5point_local",
]
