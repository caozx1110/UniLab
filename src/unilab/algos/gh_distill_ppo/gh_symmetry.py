"""Concrete GH symmetry transforms for the three obs groups + action (Phase 10.5).

Builds the left-right mirror ``SymmetryTransform`` for policy(450) / priv(717) /
priv_critic(3) / action(29) by concatenating per-term transforms in the SAME order as
the Phase-6 observation assembly (envs/gh_tracking/observations.py ``compute``).

**Signs come from GH's explicit tables, never guessed** (assets/humanoid.py:173-211,
ported verbatim below and expanded with GH's ``mirrored``):
- joint-space terms use ``joint_symmetry_mapping`` — per-joint ``(sign, other_joint)``
  with NON-trivial signs (hip_roll/hip_yaw/ankle_roll/waist_yaw/waist_roll/shoulder_roll/
  shoulder_yaw/wrist_yaw/wrist_roll = -1; the rest +1).
- cartesian terms use ``spatial_symmetry_mapping`` (name->mirror) with the fixed Y-flip
  ``sign=(1,-1,1)`` (GH cartesian_space_symmetry).
- fixed per-term signs come straight from GH observations.py / motion_tracking.py
  (root_ang_vel [-1,1,-1], gravity [1,-1,1], command/force_priv compositions).

Element-wise ordering vs a GH runtime dump is ⏳ (§十二:724, needs GH runtime); this module
locks the table signs, the builders, the Phase-6 concat order, dims, involution, and
known (perm,sign) assertions against the GH tables.
"""
from __future__ import annotations

import torch

from unilab.algos.gh_distill_ppo.symmetry import SymmetryTransform
from unilab.envs.gh_tracking.motion_dataset import JOINT_NAMES

# --- GH tables, verbatim from assets/humanoid.py:173-211 (pre-mirror base dicts) --- #
_JOINT_SYMMETRY_BASE: dict[str, tuple[int, str]] = {
    "left_hip_pitch_joint": (1, "right_hip_pitch_joint"),
    "left_hip_roll_joint": (-1, "right_hip_roll_joint"),
    "left_hip_yaw_joint": (-1, "right_hip_yaw_joint"),
    "left_knee_joint": (1, "right_knee_joint"),
    "left_ankle_pitch_joint": (1, "right_ankle_pitch_joint"),
    "left_ankle_roll_joint": (-1, "right_ankle_roll_joint"),
    "waist_yaw_joint": (-1, "waist_yaw_joint"),
    "waist_roll_joint": (-1, "waist_roll_joint"),
    "waist_pitch_joint": (1, "waist_pitch_joint"),
    "left_shoulder_pitch_joint": (1, "right_shoulder_pitch_joint"),
    "left_shoulder_roll_joint": (-1, "right_shoulder_roll_joint"),
    "left_shoulder_yaw_joint": (-1, "right_shoulder_yaw_joint"),
    "left_elbow_joint": (1, "right_elbow_joint"),
    "left_wrist_yaw_joint": (-1, "right_wrist_yaw_joint"),
    "left_wrist_roll_joint": (-1, "right_wrist_roll_joint"),
    "left_wrist_pitch_joint": (1, "right_wrist_pitch_joint"),
}
_SPATIAL_SYMMETRY_BASE: dict[str, str] = {
    "left_hip_pitch_link": "right_hip_pitch_link",
    "left_hip_roll_link": "right_hip_roll_link",
    "left_hip_yaw_link": "right_hip_yaw_link",
    "left_knee_link": "right_knee_link",
    "left_ankle_pitch_link": "right_ankle_pitch_link",
    "left_ankle_roll_link": "right_ankle_roll_link",
    "pelvis": "pelvis",
    "torso_link": "torso_link",
    "waist_yaw_link": "waist_yaw_link",
    "waist_roll_link": "waist_roll_link",
    "left_shoulder_pitch_link": "right_shoulder_pitch_link",
    "left_shoulder_roll_link": "right_shoulder_roll_link",
    "left_shoulder_yaw_link": "right_shoulder_yaw_link",
    "left_elbow_link": "right_elbow_link",
    "left_wrist_yaw_link": "right_wrist_yaw_link",
    "left_wrist_roll_link": "right_wrist_roll_link",
    "left_wrist_pitch_link": "right_wrist_pitch_link",
    "right_hand_mimic": "left_hand_mimic",
    "head_mimic": "head_mimic",
}


def _mirrored_joint(base: dict[str, tuple[int, str]]) -> dict[str, tuple[int, str]]:
    """GH symmetry_utils.mirrored (joint branch): add ``other -> (sign, this)``."""
    out = dict(base)
    for k, (sign, other) in base.items():
        out[other] = (sign, k)
    return out


def _mirrored_spatial(base: dict[str, str]) -> dict[str, str]:
    """GH symmetry_utils.mirrored (cartesian branch): add ``mirror -> this``."""
    out = dict(base)
    for k, v in base.items():
        out[v] = k
    return out


GH_JOINT_SYMMETRY_MAPPING = _mirrored_joint(_JOINT_SYMMETRY_BASE)      # all 29 joints
GH_SPATIAL_SYMMETRY_MAPPING = _mirrored_spatial(_SPATIAL_SYMMETRY_BASE)

# g1_gh body-name lists (match envs/gh_tracking/env.py + GH asset order).
KEYPOINT_BODIES = (
    "head_mimic", "left_hand_mimic", "right_hand_mimic",
    "left_wrist_roll_link", "right_wrist_roll_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_knee_link", "right_knee_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
)
HEIGHT_BODIES = ("pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link")
FEET_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
FORCE_BODIES = (
    "left_shoulder_yaw_link", "left_wrist_roll_link",
    "right_shoulder_yaw_link", "right_wrist_roll_link",
    "left_hand_mimic", "right_hand_mimic",
)


# --- GH builders (utils/symmetry.py:59-106) ------------------------------- #
def joint_space_symmetry(joint_names) -> SymmetryTransform:
    """Per-joint mirror perm+sign from GH_JOINT_SYMMETRY_MAPPING (GH:59-83)."""
    names = list(joint_names)
    missing = set(names) - set(GH_JOINT_SYMMETRY_MAPPING)
    if missing:
        raise ValueError(f"joint symmetry mapping missing joints: {sorted(missing)}")
    ids, signs = [], []
    for name in names:
        sign, other = GH_JOINT_SYMMETRY_MAPPING[name]
        ids.append(names.index(other))
        signs.append(float(sign))
    return SymmetryTransform(torch.tensor(ids, dtype=torch.long), torch.tensor(signs))


def cartesian_space_symmetry(body_names, sign=(1, -1, 1)) -> SymmetryTransform:
    """Per-body mirror perm with fixed axis sign (GH:85-106). ``sign`` has one entry
    per spatial coord (``(1,-1,1)`` = Y-flip; ``(1,)`` for a lone Z/height)."""
    names = list(body_names)
    k = len(sign)
    missing = set(names) - set(GH_SPATIAL_SYMMETRY_MAPPING)
    if missing:
        raise ValueError(f"spatial symmetry mapping missing bodies: {sorted(missing)}")
    ids, signs = [], []
    for name in names:
        other = GH_SPATIAL_SYMMETRY_MAPPING[name]
        base = names.index(other) * k
        ids.extend(base + j for j in range(k))
        signs.extend(sign)
    return SymmetryTransform(torch.tensor(ids, dtype=torch.long), torch.tensor(signs, dtype=torch.float32))


def _fixed(signs) -> SymmetryTransform:
    """Identity-perm transform with a fixed sign vector (GH per-term signs)."""
    return SymmetryTransform(torch.arange(len(signs)), torch.tensor(signs, dtype=torch.float32))


def _command_sym(n_future: int = 5) -> SymmetryTransform:
    """GH command_sym (motion_tracking.py:1087-1091): heights[1]*S ⊕ pos_diff[1,-1]*(S-1)
    ⊕ heading[1,-1]*(S-1) ⊕ safe_limit[1] = 5+8+8+1 = 22."""
    return SymmetryTransform.cat([
        _fixed([1]).repeat(n_future),
        _fixed([1, -1]).repeat(n_future - 1),
        _fixed([1, -1]).repeat(n_future - 1),
        _fixed([1]),
    ])


def _force_priv_sym() -> SymmetryTransform:
    """GH force_priv_sym (motion_tracking.py:1104-1106): cartesian(force bodies,[1,-1,1])
    over the 3 force blocks (keypoint_b/applied_b/expected_b) ⊕ timer[1] = 6*3*3+1 = 55."""
    return SymmetryTransform.cat([
        cartesian_space_symmetry(FORCE_BODIES, sign=[1, -1, 1]).repeat(3),
        _fixed([1]),
    ])


def build_gh_symmetry(n_future: int = 5) -> dict[str, SymmetryTransform]:
    """Assemble the four GH symmetry transforms, concatenated in Phase-6 obs order.

    Returns ``{"policy":(450), "priv":(717), "priv_critic":(3), "action":(29)}``.
    """
    J = joint_space_symmetry(JOINT_NAMES)                       # 29
    ANGVEL = _fixed([-1, 1, -1])                                # root ang-vel / axis-angle
    GRAV = _fixed([1, -1, 1])                                   # projected gravity / linvel
    KP = cartesian_space_symmetry(KEYPOINT_BODIES, sign=[1, -1, 1])  # 33

    policy = SymmetryTransform.cat([
        _fixed([1]),                     # boot_indicator_state        [0:1]
        _command_sym(n_future),          # command                     [1:23]
        J.repeat(n_future),              # target_joint_pos_obs         [23:168]
        GRAV.repeat(n_future),           # target_projected_gravity_b   [168:183]
        ANGVEL,                          # root_ang_vel_history[0]      [183:186]
        GRAV,                            # projected_gravity_history[0] [186:189]
        J.repeat(6),                     # joint_pos_history[0,1,2,3,4,8] [189:363]
        J.repeat(3),                     # prev_actions(steps=3)        [363:450]
    ])

    priv = SymmetryTransform.cat([
        GRAV.repeat(n_future),           # target_pos_b_obs             [0:15]
        GRAV.repeat(n_future),           # target_linvel_b_obs          [15:30]
        ANGVEL.repeat(n_future),         # relative_quat_obs            [30:45]
        _force_priv_sym(),               # force_priv                   [45:100]
        cartesian_space_symmetry(HEIGHT_BODIES, sign=(1,)),  # body_height   [100:104]
        cartesian_space_symmetry(FEET_BODIES, sign=[1, -1, 1]),  # contact_forces [104:110]
        GRAV,                            # root_linvel_b (EMA)          [110:113]
        ANGVEL.repeat(9),                # root_ang_vel_history[0..8]   [113:140]
        GRAV.repeat(9),                  # projected_gravity_history[0..8] [140:167]
        J.repeat(9),                     # joint_pos_history[0..8]      [167:428]
        KP,                              # current_keypoint_b           [428:461]
        KP,                              # current_keypoint_vel_b       [461:494]
        KP.repeat(n_future),             # target_keypoints_diff_b_obs  [494:659]
        J,                               # applied_action               [659:688]
        J,                               # applied_torque               [688:717]
    ])

    priv_critic = _fixed([1, 1, 1])      # cum_error (identity, GH motion_tracking.py:363)
    action = J                           # joint-space mirror (GH action.py:99)
    return {"policy": policy, "priv": priv, "priv_critic": priv_critic, "action": action}
