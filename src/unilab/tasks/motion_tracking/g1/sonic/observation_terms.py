"""Release-ABI observation terms owned by the Manager-Based SONIC task.

The generic manager provides the temporal lifecycle.  In particular, actor
and critic history are configured with :class:`ObservationTermCfg` history
buffers, which backfill reset rows and expose chronological (oldest-to-newest)
frames.  This module deliberately contains only current numeric term formulas
and the non-public tokenizer handoff.  It never parses a model, XML, manifest,
or motion file in ``reset``/``step``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from unilab.utils.rotation import (
    np_matrix_first_two_cols_from_quat,
    np_quat_apply_batched,
    np_quat_conjugate_batched,
    np_quat_mul_batched,
)

from .actions import SONIC_POLICY_JOINT_ORDER
from .manager_terms import SONIC_JOINT_ORDER
from .observations import SONIC_TOKENIZER_OBSERVATION_DIM

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


SONIC_NUM_FUTURE_FRAMES = 10
SONIC_NUM_JOINTS = 29
SONIC_NUM_BODIES = 14
SONIC_FUTURE_COMMAND_DIM = SONIC_NUM_FUTURE_FRAMES * SONIC_NUM_JOINTS * 2
SONIC_VR_BODY_NAMES: tuple[str, ...] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "torso_link",
)
SONIC_VR_BODY_OFFSETS = np.asarray(
    ((0.18, -0.025, 0.0), (0.18, 0.025, 0.0), (0.0, 0.0, 0.35)),
    dtype=np.float32,
)
SONIC_WRIST_POLICY_INDICES = np.asarray((23, 24, 25, 26, 27, 28), dtype=np.intp)
SONIC_LOWER_BODY_POLICY_INDICES = np.asarray(
    [SONIC_POLICY_JOINT_ORDER.index(name) for name in SONIC_JOINT_ORDER[:12]],
    dtype=np.intp,
)
for _value in (SONIC_VR_BODY_OFFSETS, SONIC_WRIST_POLICY_INDICES, SONIC_LOWER_BODY_POLICY_INDICES):
    _value.setflags(write=False)


@dataclass(frozen=True)
class SonicFutureReference:
    """One full-batch numeric future reference cached by ``SonicMotionCommand``."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    smpl_joint_pos: np.ndarray
    smpl_joints: np.ndarray
    smpl_root_quat_w: np.ndarray


def _command(env: ManagerBasedRlEnv, command_name: str):
    from .manager_terms import SonicMotionCommand

    try:
        command = env.command_manager.get_term(command_name)
    except KeyError as exc:
        raise KeyError(f"SONIC motion command term '{command_name}' not found") from exc
    if not isinstance(command, SonicMotionCommand):
        raise TypeError(
            f"SONIC observation requires command '{command_name}' to be SonicMotionCommand, "
            f"got {type(command).__name__}"
        )
    return command


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    from unilab.tasks.motion_tracking.common.manager_terms import motion_anchor_pos_b as _term

    return _term(env, command_name)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    from unilab.tasks.motion_tracking.common.manager_terms import motion_anchor_ori_b as _term

    return _term(env, command_name)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    from unilab.tasks.motion_tracking.common.manager_terms import robot_body_pos_b as _term

    return _term(env, command_name)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    from unilab.tasks.motion_tracking.common.manager_terms import robot_body_ori_b as _term

    return _term(env, command_name)


def _validate_policy_rows(value: np.ndarray, *, name: str, rows: int) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != (rows, SONIC_NUM_JOINTS):
        shape = getattr(value, "shape", None)
        raise ValueError(f"SONIC {name} must have shape ({rows}, 29), got {shape}")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"SONIC {name} must have a floating dtype, got {value.dtype}")
    return value


def _policy_joint_values(command, values: np.ndarray, *, name: str) -> np.ndarray:
    values = _validate_policy_rows(values, name=name, rows=command.num_envs)
    return np.take(values, command.policy_to_joint, axis=1)


def sonic_joint_pos_rel(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
) -> np.ndarray:
    """Release policy-order joint position relative to episode default."""

    command = _command(env, command_name)
    values = (
        command.robot_joint_pos - command.robot.data.default_joint_pos - command.joint_default_bias
    )
    return _policy_joint_values(command, values, name="joint position")


def sonic_joint_vel_rel(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
) -> np.ndarray:
    """Release policy-order joint velocity relative to the entity default."""

    command = _command(env, command_name)
    values = command.robot_joint_vel - command.robot.data.default_joint_vel
    return _policy_joint_values(command, values, name="joint velocity")


def sonic_last_action(env: ManagerBasedRlEnv) -> np.ndarray:
    """Return the 29-D release policy action kept by ``ActionManager``."""

    values = env.action_manager.action
    if not isinstance(values, np.ndarray) or values.shape != (env.num_envs, SONIC_NUM_JOINTS):
        shape = getattr(values, "shape", None)
        raise ValueError(
            "SONIC observations require exactly one 29-D policy-order action term; "
            f"ActionManager.action has shape {shape}"
        )
    return values


def _anchor_local_vector(command, values_w: np.ndarray) -> np.ndarray:
    anchor_quat = command.robot_anchor_quat_w
    return np_quat_apply_batched(np_quat_conjugate_batched(anchor_quat), values_w)


def sonic_base_lin_vel(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
) -> np.ndarray:
    """Pelvis-anchor linear velocity expressed in the current anchor frame."""

    command = _command(env, command_name)
    return _anchor_local_vector(command, command.robot_anchor_lin_vel_w)


def sonic_base_ang_vel(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
) -> np.ndarray:
    """Pelvis-anchor angular velocity expressed in the current anchor frame."""

    command = _command(env, command_name)
    return _anchor_local_vector(command, command.robot_anchor_ang_vel_w)


def sonic_projected_gravity(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
) -> np.ndarray:
    """World down vector expressed in the current SONIC anchor frame."""

    command = _command(env, command_name)
    gravity_w = np.empty((env.num_envs, 3), dtype=command.robot_anchor_quat_w.dtype)
    gravity_w.fill(0.0)
    gravity_w[:, 2] = -1.0
    return _anchor_local_vector(command, gravity_w)


def _zero_smpl_reference(command) -> tuple[np.ndarray, np.ndarray]:
    dtype = command.motion.joint_pos.dtype
    joints = np.zeros((command.num_envs, SONIC_NUM_FUTURE_FRAMES, 24, 3), dtype=dtype)
    root = np.zeros((command.num_envs, SONIC_NUM_FUTURE_FRAMES, 4), dtype=dtype)
    root[..., 0] = 1.0
    return joints, root


def _make_future_reference(command) -> SonicFutureReference:
    cached = command.get_future_reference_cache()
    if cached is not None:
        if not isinstance(cached, SonicFutureReference):
            raise TypeError("SONIC command future-reference cache has an unexpected type")
        return cached

    frame_indices = command.future_frame_indices()
    fields = command.motion.gather_fields(
        ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w"), frame_indices.reshape(-1)
    )
    num_envs = command.num_envs
    joint_pos = np.take(
        fields["joint_pos"].reshape(num_envs, SONIC_NUM_FUTURE_FRAMES, SONIC_NUM_JOINTS),
        command.policy_to_joint,
        axis=-1,
    )
    joint_vel = np.take(
        fields["joint_vel"].reshape(num_envs, SONIC_NUM_FUTURE_FRAMES, SONIC_NUM_JOINTS),
        command.policy_to_joint,
        axis=-1,
    )
    body_pos = fields["body_pos_w"].reshape(num_envs, SONIC_NUM_FUTURE_FRAMES, SONIC_NUM_BODIES, 3)
    body_quat = fields["body_quat_w"].reshape(
        num_envs, SONIC_NUM_FUTURE_FRAMES, SONIC_NUM_BODIES, 4
    )

    smpl_indices = command.future_frame_indices(smpl=True)
    smpl_joint_fields = command.motion.gather_fields(("joint_pos",), smpl_indices.reshape(-1))
    smpl_joint_pos = np.take(
        smpl_joint_fields["joint_pos"].reshape(num_envs, SONIC_NUM_FUTURE_FRAMES, SONIC_NUM_JOINTS),
        command.policy_to_joint,
        axis=-1,
    )
    if command.has_smpl_reference:
        smpl_fields = command.motion.gather_fields(
            ("smpl_joints", "smpl_root_quat_w"), smpl_indices.reshape(-1)
        )
        smpl_joints = smpl_fields["smpl_joints"].reshape(num_envs, SONIC_NUM_FUTURE_FRAMES, 24, 3)
        smpl_root = smpl_fields["smpl_root_quat_w"].reshape(num_envs, SONIC_NUM_FUTURE_FRAMES, 4)
    else:
        smpl_joints, smpl_root = _zero_smpl_reference(command)

    reference = SonicFutureReference(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        smpl_joint_pos=smpl_joint_pos,
        smpl_joints=smpl_joints,
        smpl_root_quat_w=smpl_root,
    )
    command.set_future_reference_cache(reference)
    return reference


def sonic_future_command(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
) -> np.ndarray:
    """Return the release 10-frame policy-order future ``q``/``qdot`` command."""

    command = _command(env, command_name)
    reference = _make_future_reference(command)
    result = np.concatenate(
        (
            reference.joint_pos.reshape(env.num_envs, -1),
            reference.joint_vel.reshape(env.num_envs, -1),
        ),
        axis=1,
    )
    if result.shape != (env.num_envs, SONIC_FUTURE_COMMAND_DIM):
        raise RuntimeError(f"SONIC future command has invalid shape {result.shape}")
    return result


def _future_anchor_rot6d(command, reference: SonicFutureReference) -> np.ndarray:
    anchor = reference.body_quat_w[:, :, command.anchor_body_idx]
    relative = np_quat_mul_batched(
        np_quat_conjugate_batched(command.robot_anchor_quat_w)[:, None, :], anchor
    )
    return np_matrix_first_two_cols_from_quat(relative)


def _tokenizer_corruption(command, value: np.ndarray) -> np.ndarray:
    if not command.cfg.params.tokenizer_enable_corruption:
        return value
    noise = command._env.rng.uniform(-0.05, 0.05, size=value.shape).astype(value.dtype)
    return value + noise


def _tokenizer_layout(
    command,
    reference: SonicFutureReference,
) -> np.ndarray:
    num_envs = command.num_envs
    # ``reference`` is already materialized by the critic's future-command
    # term in the normal manager group order.  The tokenizer layout only used
    # the old ``future_command`` argument to recover its dtype; rebuilding and
    # concatenating that 580-D array here duplicated a sizeable hot-path
    # allocation for every environment step.
    dtype = reference.joint_pos.dtype
    anchor_ori_b = _future_anchor_rot6d(command, reference)[:, 0]
    future_ori_b = _future_anchor_rot6d(command, reference)
    command_z = reference.body_pos_w[:, :, command.anchor_body_idx, 2:3]
    lower = np.concatenate(
        (
            reference.joint_pos[:, :, SONIC_LOWER_BODY_POLICY_INDICES].reshape(num_envs, -1),
            reference.joint_vel[:, :, SONIC_LOWER_BODY_POLICY_INDICES].reshape(num_envs, -1),
        ),
        axis=1,
    )
    vr_rows = np.asarray(
        [command.cfg.body_names.index(name) for name in SONIC_VR_BODY_NAMES], dtype=np.intp
    )
    future_body_pos = reference.body_pos_w[:, 0, vr_rows]
    future_body_quat = reference.body_quat_w[:, 0, vr_rows]
    anchor_pos = reference.body_pos_w[:, 0, command.anchor_body_idx]
    anchor_quat = reference.body_quat_w[:, 0, command.anchor_body_idx]
    vr_pos_w = future_body_pos + np_quat_apply_batched(
        future_body_quat, SONIC_VR_BODY_OFFSETS[None, :, :].astype(dtype, copy=False)
    )
    vr_pos = np_quat_apply_batched(
        np_quat_conjugate_batched(anchor_quat)[:, None, :], vr_pos_w - anchor_pos[:, None, :]
    ).reshape(num_envs, -1)
    vr_quat = np_quat_mul_batched(
        np_quat_conjugate_batched(anchor_quat)[:, None, :], future_body_quat
    ).reshape(num_envs, -1)
    smpl_local = np_quat_apply_batched(
        np_quat_conjugate_batched(reference.smpl_root_quat_w)[:, :, None, :],
        reference.smpl_joints,
    )
    smpl_ori_b = np_matrix_first_two_cols_from_quat(
        np_quat_mul_batched(
            np_quat_conjugate_batched(command.robot_anchor_quat_w)[:, None, :],
            reference.smpl_root_quat_w,
        )
    )
    # Training tokenizer corruption is independent of actor observation noise.
    # It uses the task RNG on the numeric hot path and never accesses assets.
    anchor_ori_token = _tokenizer_corruption(command, anchor_ori_b)
    future_ori_token = _tokenizer_corruption(command, future_ori_b)
    smpl_local_token = _tokenizer_corruption(command, smpl_local)
    smpl_ori_token = _tokenizer_corruption(command, smpl_ori_b)
    result = np.concatenate(
        (
            command.encoder_index,
            np.concatenate((reference.joint_pos, reference.joint_vel), axis=-1).reshape(
                num_envs, -1
            ),
            command_z.reshape(num_envs, -1),
            command_z[:, 0],
            anchor_ori_token,
            future_ori_token.reshape(num_envs, -1),
            lower,
            vr_pos,
            vr_quat,
            smpl_local_token.reshape(num_envs, -1),
            smpl_ori_token.reshape(num_envs, -1),
            reference.smpl_joint_pos[:, :, SONIC_WRIST_POLICY_INDICES].reshape(num_envs, -1),
        ),
        axis=1,
    )
    if result.shape != (num_envs, SONIC_TOKENIZER_OBSERVATION_DIM):
        raise RuntimeError(f"SONIC tokenizer observation has invalid shape {result.shape}")
    return result


def sonic_tokenizer_observation(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
) -> np.ndarray:
    """Compute and task-publish the hidden 1761-D release tokenizer input.

    Configure this in an un-mapped ``tokenizer`` observation group.  The
    function returns the ordinary manager term matrix, while the command cache
    gives the task's PPO runner a supported public provider without extending
    ``NpEnvState.obs``.
    """

    command = _command(env, command_name)
    reference = _make_future_reference(command)
    result = _tokenizer_layout(command, reference)
    command.write_tokenizer_observations(result)
    return result


__all__ = [
    "SONIC_FUTURE_COMMAND_DIM",
    "SonicFutureReference",
    "motion_anchor_ori_b",
    "motion_anchor_pos_b",
    "robot_body_ori_b",
    "robot_body_pos_b",
    "sonic_base_ang_vel",
    "sonic_base_lin_vel",
    "sonic_future_command",
    "sonic_joint_pos_rel",
    "sonic_joint_vel_rel",
    "sonic_last_action",
    "sonic_projected_gravity",
    "sonic_tokenizer_observation",
]
