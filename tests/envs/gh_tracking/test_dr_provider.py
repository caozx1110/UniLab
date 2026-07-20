"""Task 9.5: DR provider — D2 reset order + student-cache 50-fill + force origin.

The provider wires the NpEnv autoreset lifecycle, so init_state() drives a full GH
reset (D2 order) here. Synthetic motion backs the env (DP2: no real data yet).
"""
import numpy as np

from unilab.envs.gh_tracking.motion_dataset import write_synthetic_dataset


def _make_env(tmp_path, n=2):
    from unilab.envs.gh_tracking.config import GHTrackingCfg
    from unilab.envs.gh_tracking.env import GHTrackingEnv

    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[200, 300], seed=0)
    cfg = GHTrackingCfg()
    cfg.motion.dirs = [str(tmp_path / "interx")]
    cfg.motion.weights = [1.0]
    return GHTrackingEnv(cfg, num_envs=n, backend_type="mujoco"), cfg


def test_d2_reset_boot_protect_is_noised_pose(tmp_path):
    """D2: reset stores the NOISED init joint pose as boot-protect (info + pipeline)."""
    env, _cfg = _make_env(tmp_path, 2)
    state = env.init_state()

    assert "boot_protect" in state.info
    assert state.info["boot_protect"].shape == (2, 29)
    # the action pipeline caches the same noised pose it will hold during boot delay
    np.testing.assert_allclose(env.action_pipeline.boot_protect_pose, state.info["boot_protect"])
    env.close()


def test_student_cache_50_frame_fill_at_reset(tmp_path):
    """Student root cache filled with 50 future motion frames at reset (Phase 7 ⑤)."""
    env, _cfg = _make_env(tmp_path, 2)
    env.init_state()

    assert env.student_cache.steps == 50
    assert env.student_cache.ts_root_pos_w.shape == (2, 50, 3)
    assert env.student_cache.ts_root_quat_w.shape == (2, 50, 4)
    assert np.any(env.student_cache.ts_root_pos_w != 0.0), "cache must be filled from motion"
    env.close()


def test_dr_joint_offset_applied_and_shared_with_obs(tmp_path):
    """Per-reset batch-shared joint zero offset -> action offset, shared with obs."""
    env, _cfg = _make_env(tmp_path, 2)
    # the obs joint-history offset is the SAME array as the action-manager offset
    assert env.obs_manager.priv_joint.offset is env.action_pipeline.offset
    env.init_state()
    assert env.action_pipeline.offset.shape == (2, 29)
    # a batch-shared per-joint draw: both envs share the same offset row
    np.testing.assert_allclose(env.action_pipeline.offset[0], env.action_pipeline.offset[1])
    env.close()


def test_init_state_produces_three_obs_groups(tmp_path):
    """Full init_state via the DR provider returns the 3 obs groups (right shapes)."""
    env, _cfg = _make_env(tmp_path, 4)
    state = env.init_state()

    assert set(state.obs.keys()) == {"policy", "priv", "priv_critic"}
    assert state.obs["policy"].shape == (4, 450)
    assert state.obs["priv"].shape == (4, 717)
    assert state.obs["priv_critic"].shape == (4, 3)
    env.close()
