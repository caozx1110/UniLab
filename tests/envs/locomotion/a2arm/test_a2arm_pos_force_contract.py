"""Contract tests for the A2ArmPosForce environment (A2 + P7v3 + UMI gripper).

The a2arm MJCF is a 5-DOF arm: joint3 (upper-arm ROLL, the redundant 7th DOF)
and joint5 (wrist ROLL) are frozen, leaving active arm joints joint1,2,4,6,7
(a 2-DOF wrist). The env is ``A2ArmPosForceEnv``. These tests prove the model +
config + env chain constructs and steps in MuJoCo with the joint3+joint5-frozen
joint set, the P7v3 arm torque limits (j1/j2/j4=30, j6/j7=10), the sphere centre,
and the FK-verified home pose.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

_A2ARM_TRAINING_CONTRACT_SHA256 = "5327b8883e690ce7e39b96f7b9315d5125fb84528c81c86778ae2ae99a474e08"

_A2_SHARED_MESH_NAMES = (
    "base_link",
    "left_front_Link1",
    "left_front_Link2",
    "left_front_Link3",
    "left_front_Link4",
    "left_hind_Link1",
    "left_hind_Link2",
    "left_hind_Link3",
    "left_hind_Link4",
    "right_front_Link1",
    "right_front_Link2",
    "right_front_Link3",
    "right_front_Link4",
    "right_hind_Link1",
    "right_hind_Link2",
    "right_hind_Link3",
    "right_hind_Link4",
)

_UMI_MESH_DERIVED_BODY_NAMES = (
    "umi_l11",
    "umi_l12",
    "umi_l61",
    "umi_l62",
    "umi_l21",
    "umi_l22",
    "umi_l23",
    "umi_l31",
    "umi_l32",
    "umi_l41",
    "umi_l42",
    "umi_l51",
    "umi_l52",
    "umi_l53",
)

_P7_VISUAL_MESH_FACE_BUDGETS = {
    "adapter_plate.STL": 10_000,
    "p7_v3/base_link.STL": 10_000,
    "p7_v3/link1.STL": 10_000,
    "p7_v3/link2.STL": 30_000,
    "p7_v3/link3.STL": 10_000,
    "p7_v3/link4.STL": 10_000,
    "p7_v3/link5.STL": 10_000,
    "p7_v3/link6.STL": 10_000,
    "p7_v3/link7.STL": 10_000,
}


def _rounded(values):
    return np.round(np.asarray(values, dtype=np.float64), 12).tolist()


def _object_names(mujoco, model, object_type, count):
    return [mujoco.mj_id2name(model, object_type, i) for i in range(count)]


def _a2arm_xml_root():
    from unilab.assets import ASSETS_ROOT_PATH

    robot_dir = ASSETS_ROOT_PATH / "robots" / "a2arm"
    return ET.parse(robot_dir / "a2arm.xml").getroot(), robot_dir


def _local_stl_bytes(robot_dir: Path) -> int:
    return sum(path.stat().st_size for path in (robot_dir / "meshes").rglob("*.STL"))


def _binary_stl_face_count(path: Path) -> int:
    with path.open("rb") as stl:
        stl.seek(80)
        count = struct.unpack("<I", stl.read(4))[0]
    assert path.stat().st_size == 84 + 50 * count, f"expected binary STL: {path}"
    return count


def _a2arm_training_contract_sha256() -> str:
    import mujoco

    from unilab.assets import ASSETS_ROOT_PATH

    model = mujoco.MjModel.from_xml_path(
        str(ASSETS_ROOT_PATH / "robots" / "a2arm" / "scene_pos_force.xml")
    )
    physical_geom_ids = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    snapshot = {
        "dimensions": [
            model.nq,
            model.nv,
            model.nu,
            model.nbody,
            model.njnt,
            model.nsite,
            model.nsensor,
        ],
        "names": {
            "body": _object_names(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, model.nbody),
            "joint": _object_names(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
            "actuator": _object_names(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
            "site": _object_names(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, model.nsite),
            "sensor": _object_names(mujoco, model, mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor),
        },
        "option": {
            "timestep": model.opt.timestep,
            "gravity": _rounded(model.opt.gravity),
            "integrator": int(model.opt.integrator),
            "solver": int(model.opt.solver),
            "iterations": model.opt.iterations,
            "ls_iterations": model.opt.ls_iterations,
        },
        "body": {
            "parentid": model.body_parentid.tolist(),
            "mass": _rounded(model.body_mass),
            "ipos": _rounded(model.body_ipos),
            "iquat": _rounded(model.body_iquat),
            "inertia": _rounded(model.body_inertia),
        },
        "joint": {
            "type": model.jnt_type.tolist(),
            "bodyid": model.jnt_bodyid.tolist(),
            "qposadr": model.jnt_qposadr.tolist(),
            "dofadr": model.jnt_dofadr.tolist(),
            "pos": _rounded(model.jnt_pos),
            "axis": _rounded(model.jnt_axis),
            "range": _rounded(model.jnt_range),
        },
        "dof": {
            "damping": _rounded(model.dof_damping),
            "frictionloss": _rounded(model.dof_frictionloss),
            "armature": _rounded(model.dof_armature),
        },
        "actuator": {
            "trntype": model.actuator_trntype.tolist(),
            "trnid": model.actuator_trnid.tolist(),
            "gear": _rounded(model.actuator_gear),
            "ctrlrange": _rounded(model.actuator_ctrlrange),
            "forcerange": _rounded(model.actuator_forcerange),
            "gainprm": _rounded(model.actuator_gainprm),
            "biasprm": _rounded(model.actuator_biasprm),
        },
        "site": {
            "bodyid": model.site_bodyid.tolist(),
            "pos": _rounded(model.site_pos),
            "quat": _rounded(model.site_quat),
        },
        "sensor": {
            "type": model.sensor_type.tolist(),
            "objtype": model.sensor_objtype.tolist(),
            "objid": model.sensor_objid.tolist(),
            "reftype": model.sensor_reftype.tolist(),
            "refid": model.sensor_refid.tolist(),
            "adr": model.sensor_adr.tolist(),
            "dim": model.sensor_dim.tolist(),
        },
        "physical_geom": {
            "names": [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(i))
                for i in physical_geom_ids
            ],
            "type": model.geom_type[physical_geom_ids].tolist(),
            "bodyid": model.geom_bodyid[physical_geom_ids].tolist(),
            "pos": _rounded(model.geom_pos[physical_geom_ids]),
            "quat": _rounded(model.geom_quat[physical_geom_ids]),
            "size": _rounded(model.geom_size[physical_geom_ids]),
            "contype": model.geom_contype[physical_geom_ids].tolist(),
            "conaffinity": model.geom_conaffinity[physical_geom_ids].tolist(),
            "condim": model.geom_condim[physical_geom_ids].tolist(),
            "priority": model.geom_priority[physical_geom_ids].tolist(),
            "friction": _rounded(model.geom_friction[physical_geom_ids]),
            "solref": _rounded(model.geom_solref[physical_geom_ids]),
            "solimp": _rounded(model.geom_solimp[physical_geom_ids]),
            "margin": _rounded(model.geom_margin[physical_geom_ids]),
            "gap": _rounded(model.geom_gap[physical_geom_ids]),
        },
        "state": {
            "qpos0": _rounded(model.qpos0),
            "key_qpos": _rounded(model.key_qpos[home]),
            "key_ctrl": _rounded(model.key_ctrl[home]),
        },
    }
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def test_a2arm_training_contract_signature():
    pytest.importorskip("mujoco", reason="mujoco not installed")
    assert _a2arm_training_contract_sha256() == _A2ARM_TRAINING_CONTRACT_SHA256


def test_a2arm_visual_assets_are_reduced_without_runtime_download():
    root, robot_dir = _a2arm_xml_root()
    meshes = {mesh.get("name"): mesh.get("file") for mesh in root.findall("./asset/mesh")}

    assert all(meshes[name] == f"../../a2/assets/a2/{name}.STL" for name in _A2_SHARED_MESH_NAMES)
    assert not any(name.startswith("umi_") for name in meshes)
    assert not (robot_dir / "meshes" / "umi_gripper_v3").exists()
    assert _local_stl_bytes(robot_dir) <= 5_600_000


def test_a2arm_umi_mesh_derived_inertials_are_explicit():
    root, _ = _a2arm_xml_root()
    bodies = {body.get("name"): body for body in root.findall(".//body")}

    assert all(bodies[name].find("inertial") is not None for name in _UMI_MESH_DERIVED_BODY_NAMES)


def test_a2arm_visual_mesh_face_budgets():
    _, robot_dir = _a2arm_xml_root()
    meshes_dir = robot_dir / "meshes"

    for relative_path, budget in _P7_VISUAL_MESH_FACE_BUDGETS.items():
        face_count = _binary_stl_face_count(meshes_dir / relative_path)
        assert face_count <= budget


def _skip_if_no_mujoco():
    pytest.importorskip("mujoco", reason="mujoco not installed")
    try:
        from mujoco_uni.batch_env import BatchEnvPool  # noqa: F401
    except Exception:
        pytest.skip("mujoco_uni.batch_env not available")


def _ensure_registered() -> None:
    from unilab.base import registry

    registry.ensure_registries()
    if not registry.contains("A2ArmPosForce"):
        importlib.import_module("unilab.envs.locomotion.a2arm.pos_force")


def _default_reward_cfg(**overrides):
    from unilab.envs.locomotion.a2arm.pos_force import RewardConfig

    cfg = dict(
        scales={
            "tracking_lin_vel_force_world": 2.0,
            "tracking_ee_force_world": 2.0,
            "tracking_ang_vel": 1.0,
            "ref_dof_leg": 1.0,
            "alive": 1.5,
            "base_height": -2.0,
            "torques": -5.0e-6,
            "feet_contact_number": 2.0,
            "feet_air_time": 1.0,
            "feet_height": 1.0,
            "feet_height_high": -15.0,
            "feet_pos_xy": -0.5,
            "feet_drag": -8.0e-4,
            "feet_contact_forces": -1.0e-3,
            "collision": -5.0,
            "dof_pos_limits": -10.0,
            "stand_still": 0.5,
        },
        tracking_sigma=0.25,
        base_height_target=0.45,
        max_contact_force=400.0,
        feet_height_target=0.12,
        feet_height_high_target=0.24,
    )
    cfg.update(overrides)
    return RewardConfig(**cfg)


def _make_a2arm_env(num_envs: int = 2):
    from unilab.base import registry

    _ensure_registered()
    return registry.make(
        "A2ArmPosForce",
        sim_backend="mujoco",
        num_envs=num_envs,
        env_cfg_override={"reward_config": _default_reward_cfg()},
    )


def test_a2arm_pos_force_registered():
    """Registers without MuJoCo (decorators run on module import)."""
    from unilab.base import registry

    _ensure_registered()
    assert registry.contains("A2ArmPosForce")


def test_a2arm_exported_from_package():
    """Cfg/Env are exported from the a2arm package (public API, __all__)."""
    import unilab.envs.locomotion.a2arm as pkg

    assert "A2ArmPosForceCfg" in pkg.__all__
    assert "A2ArmPosForceEnv" in pkg.__all__
    assert hasattr(pkg, "A2ArmPosForceCfg")
    assert hasattr(pkg, "A2ArmPosForceEnv")


def test_a2arm_config_geometry():
    """Sphere centre (0.2, 0, 0.75); re-optimized home target [0.4494 m, 0.8115 rad, 0]."""
    from unilab.envs.locomotion.a2arm.pos_force import A2ArmPosForceCfg

    cfg = A2ArmPosForceCfg()
    assert cfg.goal_ee.sphere_center.x_offset == pytest.approx(0.2)
    assert cfg.goal_ee.sphere_center.z_invariant_offset == pytest.approx(0.735, abs=0.01)
    # Home target [radius, pitch, yaw] (recomputed from the joint3+joint5-frozen default FK).
    assert cfg.goal_ee.init_pos_start[0] == pytest.approx(0.4494, abs=1e-3)
    assert cfg.goal_ee.init_pos_start[1] == pytest.approx(0.8115, abs=1e-3)
    assert cfg.goal_ee.init_pos_start[2] == pytest.approx(0.0)
    assert cfg.goal_ee.init_pos_end == pytest.approx([0.4494, 0.0, 0.0], abs=1e-3)
    # Arm arrays are length-5, indexed in tree order joint1,2,4,6,7.
    assert list(cfg.control_config.arm_torque_limit) == [30.0, 30.0, 30.0, 10.0, 10.0]
    assert list(cfg.control_config.arm_kp) == [90.0, 120.0, 70.0, 30.0, 30.0]
    assert list(cfg.control_config.arm_kd) == [5.5, 10.5, 5.5, 1.0, 1.0]


def test_a2arm_mjcf_joint_set():
    """The frozen-joint change is structural: 5 arm hinges = {j1,j2,j4,j6,j7},
    joint3 AND joint5 absent, 17 actuators, arm forceranges 30/30/30/10/10.
    Compiles the scene MJCF directly (no batch backend needed)."""
    pytest.importorskip("mujoco", reason="mujoco not installed")
    import mujoco

    from unilab.assets import ASSETS_ROOT_PATH

    model_path = str(ASSETS_ROOT_PATH / "robots" / "a2arm" / "scene_pos_force.xml")
    m = mujoco.MjModel.from_xml_path(model_path)

    arm_joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    arm_joints = [j for j in arm_joints if j and j.startswith("joint")]
    assert arm_joints == ["joint1", "joint2", "joint4", "joint6", "joint7"]
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint3") == -1
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint5") == -1

    assert m.nu == 17
    arm_forceranges = m.actuator_forcerange[12:]
    expected = np.array([[-30, 30], [-30, 30], [-30, 30], [-10, 10], [-10, 10]], dtype=float)
    assert np.allclose(arm_forceranges, expected)
    j7 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint7")
    assert np.allclose(m.jnt_range[j7], [-1.5708, 1.2217], atol=1e-3)


def test_a2arm_home_pose_fk():
    """Home keyframe FK: end_link sits at [0.4494 m, 0.8115 rad, 0] from the sphere
    centre (0.2, 0, 0.75), matching init_pos_start (reset self-consistency)."""
    pytest.importorskip("mujoco", reason="mujoco not installed")
    import mujoco

    from unilab.assets import ASSETS_ROOT_PATH
    from unilab.envs.locomotion.a2arm.pos_force import A2ArmPosForceCfg

    model_path = str(ASSETS_ROOT_PATH / "robots" / "a2arm" / "scene_pos_force.xml")
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    kf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    d.qpos[:] = m.key_qpos[kf]
    mujoco.mj_forward(m, d)

    cfg = A2ArmPosForceCfg().goal_ee
    # Sphere centre at the keyframe (base at origin, yaw 0): base_xy + offset, z.
    center = np.array(
        [
            cfg.sphere_center.x_offset,
            cfg.sphere_center.y_offset,
            cfg.sphere_center.z_invariant_offset,
        ]
    )
    ee_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "endpoint")
    rel = d.site_xpos[ee_id] - center
    radius = float(np.linalg.norm(rel))
    xy = float(np.linalg.norm(rel[:2]))
    pitch = float(np.arctan2(rel[2], xy))
    yaw = float(np.arctan2(rel[1], rel[0]))

    assert radius == pytest.approx(cfg.init_pos_start[0], abs=0.005)
    assert pitch == pytest.approx(cfg.init_pos_start[1], abs=0.02)
    assert yaw == pytest.approx(0.0, abs=0.02)


def test_a2arm_obs_layout_matches_go2_contract():
    """A2Arm is a 17-DOF variant (5-DOF arm), so obs/critic layout is scaled down."""
    _skip_if_no_mujoco()
    env = _make_a2arm_env(num_envs=2)
    spec = env.obs_groups_spec
    assert set(spec) == {"obs", "critic"}
    assert spec["obs"] == env._cfg.history.num_actor_history * env._actor_single_obs_dim()
    assert spec["critic"] == env._cfg.history.num_critic_history * env._critic_single_obs_dim()
    assert spec["obs"] == 32 * 73


@pytest.mark.slow
def test_a2arm_constructs_with_17_dof_and_per_joint_torque():
    _skip_if_no_mujoco()
    env = _make_a2arm_env()
    assert env._num_action == 17
    # A2 leg limits hip/thigh 120, calf 180 (tiled x4); P7v3 arm 30/30/30/10/10.
    assert np.allclose(env._torque_limits[:12], [120, 120, 180] * 4)
    assert np.allclose(env._torque_limits[12:], [30, 30, 30, 10, 10])


@pytest.mark.slow
def test_a2arm_init_step_runs_and_torque_within_limits():
    """End-to-end: init + steps must run (all sensors/geoms resolve) with finite
    obs/reward, and the Python PD torque stays within the per-joint limits."""
    _skip_if_no_mujoco()
    env = _make_a2arm_env(num_envs=2)
    critic_dim = env._cfg.history.num_critic_history * env._critic_single_obs_dim()
    state = env.init_state()
    assert state.obs["obs"].shape == (2, 32 * 73)
    assert state.obs["critic"].shape == (2, critic_dim)
    for _ in range(10):
        state = env.step(np.zeros((2, 17), dtype=np.float64))
    assert np.isfinite(state.reward).all()
    assert np.isfinite(state.obs["obs"]).all()
    assert np.isfinite(state.obs["critic"]).all()
    assert np.all(np.abs(env._last_torque) <= env._torque_limits + 1e-6)


@pytest.mark.slow
def test_a2arm_set_base_lin_vel_write_through():
    """The velocity-push DR routes through ``backend.set_base_lin_vel`` (not a
    write through ``get_base_lin_vel()``'s return value). Prove the setter's write
    actually reaches the state read back by the getter, and that a shape mismatch
    is rejected rather than silently mis-applied."""
    _skip_if_no_mujoco()
    env = _make_a2arm_env(num_envs=2)
    env.init_state()

    target = np.zeros((2, 3), dtype=np.float64)
    target[:, 0] = 0.7
    target[:, 1] = -0.4
    env._backend.set_base_lin_vel(target)
    assert np.allclose(env._backend.get_base_lin_vel(), target)

    with pytest.raises(ValueError):
        env._backend.set_base_lin_vel(np.zeros((2, 2), dtype=np.float64))
