"""Contract tests for the A2ArmPosForce environment (A2 + P7v3 + UMI gripper).

The a2arm MJCF is a 5-DOF arm: joint3 (upper-arm ROLL, the redundant 7th DOF)
and joint5 (wrist ROLL) are frozen, leaving active arm joints joint1,2,4,6,7
(a 2-DOF wrist). The env is ``A2ArmPosForceEnv``. These tests prove the model +
config + env chain constructs and steps in MuJoCo with the joint3+joint5-frozen
joint set, the P7v3 arm torque limits (j1/j2/j4=30, j6/j7=10), the sphere centre,
and the FK-verified home pose.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest


def _skip_if_no_mujoco():
    pytest.importorskip("mujoco", reason="mujoco not installed")
    try:
        from mujoco.batch_env import BatchEnvPool  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env not available")


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
    assert list(cfg.control_config.arm_kd) == [3.0, 4.0, 2.0, 1.0, 1.0]


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
