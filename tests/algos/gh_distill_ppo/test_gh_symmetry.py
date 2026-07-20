"""T10.5: concrete GH symmetry transforms — dims + involution + KNOWN (perm,sign)
assertions against the GH tables (assets/humanoid.py:173-211).

involution alone is satisfied by any self-consistent permutation, so it does NOT prove
the signs are right. These tests additionally pin specific (perm, sign) entries to the
verbatim GH joint_symmetry_mapping values and the cartesian (1,-1,1) Y-flip.
"""
import torch

from unilab.algos.gh_distill_ppo.gh_symmetry import (
    GH_JOINT_SYMMETRY_MAPPING,
    KEYPOINT_BODIES,
    build_gh_symmetry,
    cartesian_space_symmetry,
    joint_space_symmetry,
)
from unilab.envs.gh_tracking.motion_dataset import JOINT_NAMES


def test_group_dims():
    sym = build_gh_symmetry()
    assert sym["policy"].perm.shape == (450,)
    assert sym["priv"].perm.shape == (717,)
    assert sym["priv_critic"].perm.shape == (3,)
    assert sym["action"].perm.shape == (29,)


def test_all_groups_involution():
    sym = build_gh_symmetry()
    for k, t in sym.items():
        x = torch.randn(4, t.perm.shape[0])
        torch.testing.assert_close(t(t(x)), x, msg=f"{k} not involution")


def test_joint_mapping_covers_all_29_and_is_paired():
    # every joint present, and mapping is a sign-consistent involution (GH table)
    assert set(GH_JOINT_SYMMETRY_MAPPING) == set(JOINT_NAMES)
    for name, (sign, other) in GH_JOINT_SYMMETRY_MAPPING.items():
        s2, back = GH_JOINT_SYMMETRY_MAPPING[other]
        assert back == name and s2 == sign        # paired with equal sign


def test_action_signs_match_gh_table_known_joints():
    """The parity anti-trap: specific (perm, sign) pinned to GH humanoid.py:173-190."""
    act = joint_space_symmetry(JOINT_NAMES)
    idx = {n: i for i, n in enumerate(JOINT_NAMES)}

    def check(this, other, sign):
        i = idx[this]
        assert act.perm[i].item() == idx[other], f"{this} should map to {other}"
        assert act.signs[i].item() == sign, f"{this} sign should be {sign}"

    check("left_hip_roll_joint", "right_hip_roll_joint", -1.0)    # non-trivial -1
    check("right_hip_roll_joint", "left_hip_roll_joint", -1.0)
    check("left_hip_yaw_joint", "right_hip_yaw_joint", -1.0)      # -1
    check("left_knee_joint", "right_knee_joint", 1.0)             # +1
    check("left_hip_pitch_joint", "right_hip_pitch_joint", 1.0)   # +1
    check("left_ankle_roll_joint", "right_ankle_roll_joint", -1.0)
    check("left_shoulder_roll_joint", "right_shoulder_roll_joint", -1.0)
    check("left_wrist_pitch_joint", "right_wrist_pitch_joint", 1.0)
    check("waist_yaw_joint", "waist_yaw_joint", -1.0)             # self-map, sign -1
    check("waist_pitch_joint", "waist_pitch_joint", 1.0)          # self-map, sign +1


def test_cartesian_keypoint_is_y_flip_and_swaps_left_right():
    """cartesian_space_symmetry uses sign=(1,-1,1) (Y-flip) and swaps L/R bodies."""
    kp = cartesian_space_symmetry(KEYPOINT_BODIES, sign=[1, -1, 1])
    ki = {n: i for i, n in enumerate(KEYPOINT_BODIES)}
    lk, rk = ki["left_knee_link"], ki["right_knee_link"]
    # left_knee's 3-vector block maps to right_knee's block with signs [1,-1,1]
    assert kp.perm[3 * lk: 3 * lk + 3].tolist() == [3 * rk, 3 * rk + 1, 3 * rk + 2]
    assert kp.signs[3 * lk: 3 * lk + 3].tolist() == [1.0, -1.0, 1.0]
    # head_mimic self-maps, still Y-flipped
    h = ki["head_mimic"]
    assert kp.perm[3 * h: 3 * h + 3].tolist() == [3 * h, 3 * h + 1, 3 * h + 2]
    assert kp.signs[3 * h: 3 * h + 3].tolist() == [1.0, -1.0, 1.0]


def test_priv_critic_is_identity():
    sym = build_gh_symmetry()
    pc = sym["priv_critic"]
    assert pc.perm.tolist() == [0, 1, 2]
    assert pc.signs.tolist() == [1.0, 1.0, 1.0]
