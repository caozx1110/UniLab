"""Structural acceptance tests for the GH-specific G1 asset (g1_gh).

The GH task rewrites G1 in MuJoCo with: 3 mimic welded bodies (USD-extracted
offsets), per-joint armature + zero joint friction, fully disabled
self-collision, full-precision humanoid.py PD gains, and the Phase-1 sensor
contract (actfrc / netcontact / contactfound). These tests encode the
authoritative humanoid.py / USD values directly as the oracle.
"""

from __future__ import annotations

import numpy as np
import mujoco

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.base.scene import SceneCfg

_G1GH = ASSETS_ROOT_PATH / "robots" / "g1_gh"
ROBOT = str(_G1GH / "robot.xml")
SCENE = str(_G1GH / "scene_flat.xml")


def _bk(path: str, base: str | None = "pelvis", n: int = 1):
    bk = MuJoCoBackend(SceneCfg(model_file=path), n, 0.005, base_name=base)
    bk.materialize()
    return bk


# --- authoritative oracles (humanoid.py + USD §5.1) ------------------------- #

_ARM_LEG = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
_WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
_ARM = [
    f"{s}_{j}_joint"
    for s in ("left", "right")
    for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw")
]
ALL_JOINTS = _ARM_LEG + _WAIST + _ARM  # 29


def _group(j: str) -> tuple[float, float, float, float]:
    """Return (kp, kv, armature, effort) for a joint, transcribed from humanoid.py."""
    if "hip_roll" in j or "knee" in j:
        return (99.09842777666113, 6.3088018534966395, 0.025101925, 139.0)
    if "hip_yaw" in j or "hip_pitch" in j or "waist_yaw" in j:
        return (40.17923847137318, 2.5578897650279457, 0.010177520, 88.0)
    if "waist_roll" in j or "waist_pitch" in j or "ankle" in j:
        return (28.50124619574858, 1.814445686584846, 0.00721945, 50.0)
    if "wrist_pitch" in j or "wrist_yaw" in j:
        return (16.77832748089279, 1.06814150219, 0.00425, 5.0)
    # shoulder_pitch/roll/yaw, elbow, wrist_roll
    return (14.25062309787429, 0.907222843292423, 0.003609725, 25.0)


MIMIC = {
    "head_mimic": ("torso_link", (0.0100035, 0.0, 0.41)),
    "left_hand_mimic": ("left_wrist_yaw_link", (0.115996, 0.0, 0.0)),
    "right_hand_mimic": ("right_wrist_yaw_link", (0.115996, 0.0, 0.0)),
}


# --- 2.1 robot.xml structure ------------------------------------------------ #


def test_mimic_bodies_parent_offset_massless_nocollision() -> None:
    m = _bk(ROBOT).model
    for name, (parent, off) in MIMIC.items():
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        assert bid >= 0, f"missing mimic body {name}"
        pid = int(m.body_parentid[bid])
        assert mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, pid) == parent
        np.testing.assert_allclose(m.body_pos[bid], off, atol=1e-6)
        np.testing.assert_allclose(m.body_mass[bid], 0.005, atol=1e-9)
        assert int(m.body_geomnum[bid]) == 0  # no collision/visual geom on marker


def test_per_joint_armature_and_zero_frictionloss() -> None:
    m = _bk(ROBOT).model
    for j in ALL_JOINTS:
        _, _, arm, _ = _group(j)
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        assert jid >= 0, f"missing joint {j}"
        dof = int(m.jnt_dofadr[jid])
        # atol reflects MuJoCo XML serialization precision (~6 sig figs); still
        # discriminates the 5 distinct motor armatures (min gap ~6e-4).
        np.testing.assert_allclose(m.dof_armature[dof], arm, atol=1e-6, err_msg=j)
        np.testing.assert_allclose(m.dof_frictionloss[dof], 0.0, atol=1e-9, err_msg=j)


def test_actuator_gains_full_precision_humanoid_py() -> None:
    m = _bk(ROBOT).model
    for j in ALL_JOINTS:
        kp, kv, _, fr = _group(j)
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
        assert aid >= 0, f"missing actuator {j}"
        # rtol reflects XML serialization precision (~6 sig figs). Still rejects
        # USD drive:stiffness values (e.g. knee 230.09 vs humanoid.py 99.098).
        np.testing.assert_allclose(m.actuator_gainprm[aid, 0], kp, rtol=1e-4, err_msg=j)
        np.testing.assert_allclose(m.actuator_biasprm[aid, 1], -kp, rtol=1e-4, err_msg=j)
        np.testing.assert_allclose(m.actuator_biasprm[aid, 2], -kv, rtol=1e-4, err_msg=j)
        np.testing.assert_allclose(m.actuator_forcerange[aid], (-fr, fr), atol=1e-6, err_msg=j)


def test_self_collision_fully_disabled_via_conaffinity() -> None:
    m = _bk(ROBOT).model
    collidable = [g for g in range(m.ngeom) if int(m.geom_contype[g]) == 1]
    assert collidable, "expected collision geoms present"
    # Fully off (not just a few <exclude>): every collidable geom has conaffinity 0,
    # so no robot-robot pair can collide while ground contact (ground conaffinity=1)
    # still works.
    assert all(int(m.geom_conaffinity[g]) == 0 for g in collidable)


def test_actfrc_sensors_present_effort_readable() -> None:
    bk = _bk(ROBOT)
    eff = bk.get_actuator_effort()  # Phase-1 contract; requires actfrc_<actuator>
    assert eff.shape == (1, 29)


# --- 2.2 scene + GH keyframe + foot net-contact sensors --------------------- #

# GH default stand pose joint targets, in actuator/joint order (humanoid.py).
STAND_CTRL = [
    -0.28, 0.0, 0.0, 0.5, -0.23, 0.0,   # left leg
    -0.28, 0.0, 0.0, 0.5, -0.23, 0.0,   # right leg
    0.0, 0.0, 0.0,                       # waist
    0.35, 0.16, 0.0, 0.87, 0.0, 0.0, 0.0,   # left arm
    0.35, -0.16, 0.0, 0.87, 0.0, 0.0, 0.0,  # right arm
]


def test_scene_keyframe_is_gh_default_pose() -> None:
    q = _bk(SCENE).get_keyframe_qpos("stand")
    assert q.shape == (36,)
    np.testing.assert_allclose(q[:3], (0.0, 0.0, 0.74), atol=1e-6)  # root pos, z=0.74
    np.testing.assert_allclose(q[3:7], (1.0, 0.0, 0.0, 0.0), atol=1e-6)  # root quat
    np.testing.assert_allclose(q[7:36], STAND_CTRL, atol=1e-6)  # 29 joint defaults


def test_keyframe_not_in_robot_xml() -> None:
    # AGENTS.md: keyframe belongs in the task-level scene, never robot.xml.
    robot_txt = open(ROBOT).read()
    assert "<keyframe" not in robot_txt
    assert "<key " not in robot_txt


def test_foot_net_contact_sensors_wired_to_feet() -> None:
    # Contract wiring only (no physics): the scene declares netcontact_<body> and
    # contactfound_<body> for both feet, resolvable via the Phase-1 read methods.
    bk = _bk(SCENE)
    bid = bk.get_body_ids(["left_ankle_roll_link", "right_ankle_roll_link"])
    assert bk.get_body_net_contact_force_w(bid).shape == (1, 2, 3)  # KeyError if missing
    assert bk.get_body_contact_state(bid).shape == (1, 2)


def test_foot_net_contact_bears_weight_when_freshly_standing() -> None:
    # Sanity that the foot net-contact sensor reads REAL foot-ground contact: after
    # a short settle from the stand keyframe the feet bear ~the robot's weight.
    # NOTE: this is NOT user9d1fc795 PD-stability claim — MuJoCo's explicit position actuator
    # sags (tau/kp) and the static pose eventually collapses vs Isaac's implicit
    # PD (B5 approximation, out of scope for Phase 2). Hence the short settle.
    bk = _bk(SCENE, base="pelvis")
    q = bk.get_keyframe_qpos("stand").astype(np.float64)
    nv = int(bk.model.nv)
    bk.set_state(np.array([0]), q[None], np.zeros((1, nv)))
    ctrl = np.array(STAND_CTRL, dtype=np.float32)[None]  # (1, 29) position targets
    for _ in range(60):
        bk.step(ctrl, nsteps=1)

    bid = bk.get_body_ids(["left_ankle_roll_link", "right_ankle_roll_link"])
    fz = bk.get_body_net_contact_force_w(bid)[0, :, 2]  # net up-force per foot
    total_mg = float(bk.model.body_mass.sum()) * 9.81
    assert (fz > 0).all()  # both feet push up
    assert fz.sum() > 0.5 * total_mg  # feet bear most of the weight
    assert bk.get_body_contact_state(bid).all()


# --- 2.3 force-body order + keypoint bodies (contract guards for Phase 5/6) --- #

# GH force-apply bodies in USD order [L,L,R,R,L,R] (the de-facto mask quirk).
FORCE_ORDER = [
    "left_shoulder_yaw_link", "left_wrist_roll_link",
    "right_shoulder_yaw_link", "right_wrist_roll_link",
    "left_hand_mimic", "right_hand_mimic",
]
# GH keypoint-tracking bodies (11).
KEYPOINTS = [
    "head_mimic", "left_hand_mimic", "right_hand_mimic",
    "left_wrist_roll_link", "right_wrist_roll_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_knee_link", "right_knee_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
]


def test_force_body_order_resolvable_distinct() -> None:
    ids = _bk(SCENE).get_body_ids(FORCE_ORDER)
    assert ids.shape == (6,)
    assert (ids >= 0).all()
    assert len(set(ids.tolist())) == 6  # 6 distinct force bodies


def test_keypoint_bodies_resolvable() -> None:
    ids = _bk(SCENE).get_body_ids(KEYPOINTS)
    assert ids.shape == (11,)
    assert (ids >= 0).all()
    assert len(set(ids.tolist())) == 11
