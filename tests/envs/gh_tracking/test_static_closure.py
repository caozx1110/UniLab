"""Static numeric closure — reward σ / init_noise / action_scaling / soft_factor /
feet_air_time thres / hand_grid asserted against the EXACT GH G1_gentle.yaml values.

These are static config constants (copyable now from GH config, no GH runtime needed),
distinct from DP2 runtime element-wise parity. Locked here so they can't silently drift.
"""
import numpy as np

from unilab.envs.gh_tracking.action import resolve_action_scaling
from unilab.envs.gh_tracking.config import GHTrackingCfg
from unilab.envs.gh_tracking.env import _REWARD_SIGMA
from unilab.envs.gh_tracking.motion_dataset import JOINT_NAMES


def test_reward_sigma_matches_gh_config():
    """reward_sigma, exact from G1_gentle.yaml:55-69 (keypoint_imp reuses keypoint σ)."""
    assert _REWARD_SIGMA == {
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


def test_init_noise_matches_gh_config():
    """init_noise, exact from G1_gentle.yaml:48-54."""
    nz = GHTrackingCfg().init_noise
    assert (nz.root_pos, nz.root_ori, nz.root_lin_vel) == (0.03, 0.1, 0.1)
    assert (nz.root_ang_vel, nz.joint_pos, nz.joint_vel) == (0.1, 0.1, 0.1)


def test_action_scaling_regex_matches_gh_and_resolves_per_joint():
    """action_scaling, exact from G1_gentle.yaml:13-22 (per-joint, NOT uniform 0.25)."""
    scaling = GHTrackingCfg().control_config.action_scale
    assert scaling == {
        ".*elbow_joint": 1.0, ".*shoulder.*": 1.0, ".*wrist.*": 1.0,
        ".*hip_roll.*": 0.25, ".*hip_yaw.*": 0.25, ".*hip_pitch.*": 0.5,
        ".*knee.*": 0.5, ".*waist.*": 0.25, ".*ankle.*": 0.5,
    }
    ids, scale = resolve_action_scaling(list(JOINT_NAMES), scaling)  # every joint matches exactly one
    assert ids.shape == (29,) and scale.shape == (29,)
    by_name = dict(zip(JOINT_NAMES, scale))
    assert by_name["left_elbow_joint"] == 1.0
    assert by_name["left_shoulder_pitch_joint"] == 1.0
    assert by_name["left_wrist_roll_joint"] == 1.0
    assert by_name["left_hip_roll_joint"] == 0.25
    assert by_name["left_hip_yaw_joint"] == 0.25
    assert by_name["waist_yaw_joint"] == 0.25
    assert by_name["left_hip_pitch_joint"] == 0.5
    assert by_name["left_knee_joint"] == 0.5
    assert by_name["left_ankle_pitch_joint"] == 0.5
    assert not np.allclose(scale, scale[0]), "must be per-joint, not uniform"


def test_reward_static_constants_match_gh_config():
    """joint_pos_limits soft_factor + feet_air_time thres, exact from G1_gentle.yaml:133-134."""
    r = GHTrackingCfg().reward
    assert r.joint_pos_limits_soft_factor == 0.9
    assert r.feet_air_time_thres == 0.8


def test_hand_grid_samples_bundled_from_real_file():
    """Real GH hand_grid_samples bundled (torso frame, (5414,3,3) per side)."""
    from unilab.assets import ASSETS_ROOT_PATH

    hg = np.load(str(ASSETS_ROOT_PATH / "robots" / "g1_gh" / "hand_grid_samples.npz"))
    assert hg["left"].shape == (5414, 3, 3)
    assert hg["right"].shape == (5414, 3, 3)


def test_env_wires_static_values_into_components():
    """Env construction wires the per-joint scaling + real joint soft limits."""
    from unilab.envs.gh_tracking.env import GHTrackingEnv

    env = GHTrackingEnv(GHTrackingCfg(), num_envs=1, backend_type="mujoco")
    try:
        scale = np.asarray(env.action_pipeline.action_scaling)
        assert scale.shape == (29,)
        assert set(np.round(scale, 6).tolist()) == {0.25, 0.5, 1.0}  # per-joint, not uniform
        # joint soft limits from real model ranges (not the ±1e9 fallback)
        assert env._joint_soft_lo.shape == (29,)
        assert np.all(env._joint_soft_hi > env._joint_soft_lo)
        assert np.all(np.abs(env._joint_soft_lo) < 1e8)
    finally:
        env.close()
