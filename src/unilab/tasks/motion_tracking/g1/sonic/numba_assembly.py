"""Numba fixed-layout assembly for the SONIC tokenizer observation.

The Manager observation contract still requires a term to return a full
``(num_envs, 1761)`` matrix, including on a row-scoped reset.  This kernel
therefore writes into an owner-owned output buffer and accepts an explicit row
index array: reset calls update only the rows that the ObservationManager will
consume, while untouched rows retain their previous values.  No simulator or
asset access occurs here.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

SONIC_TOKENIZER_DIM = 1761
_OFF_ENCODER = 0
_OFF_JOINT = 3
_OFF_COMMAND_Z = 583
_OFF_COMMAND_Z_SCALAR = 593
_OFF_ANCHOR_ORI = 594
_OFF_FUTURE_ORI = 600
_OFF_LOWER = 660
_OFF_VR_POS = 900
_OFF_VR_QUAT = 909
_OFF_SMPL_LOCAL = 921
_OFF_SMPL_ORI = 1641
_OFF_SMPL_WRIST = 1701


@njit(inline="always")
def _quat_mul(
    w1: float,
    x1: float,
    y1: float,
    z1: float,
    w2: float,
    x2: float,
    y2: float,
    z2: float,
) -> tuple[float, float, float, float]:
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


@njit(inline="always")
def _quat_apply(
    w: float,
    x: float,
    y: float,
    z: float,
    vx: float,
    vy: float,
    vz: float,
) -> tuple[float, float, float]:
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


@njit(inline="always")
def _quat_apply_inverse(
    w: float,
    x: float,
    y: float,
    z: float,
    vx: float,
    vy: float,
    vz: float,
) -> tuple[float, float, float]:
    return _quat_apply(w, -x, -y, -z, vx, vy, vz)


@njit(inline="always")
def _rot6d(w: float, x: float, y: float, z: float) -> tuple[float, float, float, float, float, float]:
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return (
        1.0 - 2.0 * (yy + zz),
        2.0 * (xy - wz),
        2.0 * (xy + wz),
        1.0 - 2.0 * (xx + zz),
        2.0 * (xz - wy),
        2.0 * (yz + wx),
    )


@njit(cache=True, nogil=True, parallel=True)
def assemble_sonic_tokenizer_observations_kernel(
    rows: np.ndarray,
    encoder_index: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    smpl_joint_pos: np.ndarray,
    smpl_joints: np.ndarray,
    smpl_root_quat_w: np.ndarray,
    robot_anchor_quat_w: np.ndarray,
    anchor_body_idx: int,
    vr_body_indices: np.ndarray,
    lower_policy_indices: np.ndarray,
    wrist_policy_indices: np.ndarray,
    vr_body_offsets: np.ndarray,
    output: np.ndarray,
) -> None:
    """Write the release 1761-D tokenizer layout for selected environment rows."""

    for row_idx in prange(rows.shape[0]):
        env_idx = rows[row_idx]

        # Encoder selection (3), then ten frames of [q, qdot] in policy order.
        for component in range(3):
            output[env_idx, _OFF_ENCODER + component] = encoder_index[env_idx, component]
        for frame in range(10):
            frame_joint_offset = _OFF_JOINT + frame * 58
            for joint in range(29):
                output[env_idx, frame_joint_offset + joint] = joint_pos[env_idx, frame, joint]
                output[env_idx, frame_joint_offset + 29 + joint] = joint_vel[env_idx, frame, joint]

            output[env_idx, _OFF_COMMAND_Z + frame] = body_pos_w[env_idx, frame, anchor_body_idx, 2]
        output[env_idx, _OFF_COMMAND_Z_SCALAR] = body_pos_w[env_idx, 0, anchor_body_idx, 2]

        # Relative reference anchor orientation and all ten future orientations.
        robot_w = robot_anchor_quat_w[env_idx, 0]
        robot_x = robot_anchor_quat_w[env_idx, 1]
        robot_y = robot_anchor_quat_w[env_idx, 2]
        robot_z = robot_anchor_quat_w[env_idx, 3]
        for frame in range(10):
            motion_w = body_quat_w[env_idx, frame, anchor_body_idx, 0]
            motion_x = body_quat_w[env_idx, frame, anchor_body_idx, 1]
            motion_y = body_quat_w[env_idx, frame, anchor_body_idx, 2]
            motion_z = body_quat_w[env_idx, frame, anchor_body_idx, 3]
            rel_w, rel_x, rel_y, rel_z = _quat_mul(
                robot_w, -robot_x, -robot_y, -robot_z,
                motion_w, motion_x, motion_y, motion_z,
            )
            rot6d = _rot6d(rel_w, rel_x, rel_y, rel_z)
            target = _OFF_FUTURE_ORI + frame * 6
            for component in range(6):
                output[env_idx, target + component] = rot6d[component]
            if frame == 0:
                for component in range(6):
                    output[env_idx, _OFF_ANCHOR_ORI + component] = rot6d[component]

        # Lower-body q/qdot command.  The release layout stores all ten q
        # frames followed by all ten qdot frames (rather than interleaving
        # q/qdot per frame).
        for frame in range(10):
            target = _OFF_LOWER + frame * 12
            for lower_joint in range(12):
                source_joint = lower_policy_indices[lower_joint]
                output[env_idx, target + lower_joint] = joint_pos[env_idx, frame, source_joint]
                vel_target = _OFF_LOWER + 120 + frame * 12
                output[env_idx, vel_target + lower_joint] = joint_vel[env_idx, frame, source_joint]

        # Three-point VR targets from the first future frame, local to the
        # first-frame reference anchor.
        anchor_pos_x = body_pos_w[env_idx, 0, anchor_body_idx, 0]
        anchor_pos_y = body_pos_w[env_idx, 0, anchor_body_idx, 1]
        anchor_pos_z = body_pos_w[env_idx, 0, anchor_body_idx, 2]
        anchor_qw = body_quat_w[env_idx, 0, anchor_body_idx, 0]
        anchor_qx = body_quat_w[env_idx, 0, anchor_body_idx, 1]
        anchor_qy = body_quat_w[env_idx, 0, anchor_body_idx, 2]
        anchor_qz = body_quat_w[env_idx, 0, anchor_body_idx, 3]
        for point in range(3):
            body = vr_body_indices[point]
            body_qw = body_quat_w[env_idx, 0, body, 0]
            body_qx = body_quat_w[env_idx, 0, body, 1]
            body_qy = body_quat_w[env_idx, 0, body, 2]
            body_qz = body_quat_w[env_idx, 0, body, 3]
            off_x, off_y, off_z = _quat_apply(
                body_qw, body_qx, body_qy, body_qz,
                vr_body_offsets[point, 0], vr_body_offsets[point, 1], vr_body_offsets[point, 2],
            )
            world_x = body_pos_w[env_idx, 0, body, 0] + off_x
            world_y = body_pos_w[env_idx, 0, body, 1] + off_y
            world_z = body_pos_w[env_idx, 0, body, 2] + off_z
            local_x, local_y, local_z = _quat_apply_inverse(
                anchor_qw, anchor_qx, anchor_qy, anchor_qz,
                world_x - anchor_pos_x, world_y - anchor_pos_y, world_z - anchor_pos_z,
            )
            pos_target = _OFF_VR_POS + point * 3
            output[env_idx, pos_target] = local_x
            output[env_idx, pos_target + 1] = local_y
            output[env_idx, pos_target + 2] = local_z
            rel_w, rel_x, rel_y, rel_z = _quat_mul(
                anchor_qw, -anchor_qx, -anchor_qy, -anchor_qz,
                body_qw, body_qx, body_qy, body_qz,
            )
            quat_target = _OFF_VR_QUAT + point * 4
            output[env_idx, quat_target] = rel_w
            output[env_idx, quat_target + 1] = rel_x
            output[env_idx, quat_target + 2] = rel_y
            output[env_idx, quat_target + 3] = rel_z

        # SMPL local joints, root orientation, and policy-order wrists.
        for frame in range(10):
            root_w = smpl_root_quat_w[env_idx, frame, 0]
            root_x = smpl_root_quat_w[env_idx, frame, 1]
            root_y = smpl_root_quat_w[env_idx, frame, 2]
            root_z = smpl_root_quat_w[env_idx, frame, 3]
            local_target = _OFF_SMPL_LOCAL + frame * 72
            for joint in range(24):
                local_x, local_y, local_z = _quat_apply_inverse(
                    root_w, root_x, root_y, root_z,
                    smpl_joints[env_idx, frame, joint, 0],
                    smpl_joints[env_idx, frame, joint, 1],
                    smpl_joints[env_idx, frame, joint, 2],
                )
                target = local_target + joint * 3
                output[env_idx, target] = local_x
                output[env_idx, target + 1] = local_y
                output[env_idx, target + 2] = local_z
            rel_w, rel_x, rel_y, rel_z = _quat_mul(
                robot_w, -robot_x, -robot_y, -robot_z,
                root_w, root_x, root_y, root_z,
            )
            ori6d = _rot6d(rel_w, rel_x, rel_y, rel_z)
            ori_target = _OFF_SMPL_ORI + frame * 6
            for component in range(6):
                output[env_idx, ori_target + component] = ori6d[component]
            wrist_target = _OFF_SMPL_WRIST + frame * 6
            for wrist in range(6):
                output[env_idx, wrist_target + wrist] = smpl_joint_pos[
                    env_idx, frame, wrist_policy_indices[wrist]
                ]


__all__ = ["SONIC_TOKENIZER_DIM", "assemble_sonic_tokenizer_observations_kernel"]
