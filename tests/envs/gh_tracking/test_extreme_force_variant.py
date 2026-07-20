"""Task B: G1_extreme_force variant (compliance=False, max_force>0).

GH motion_tracking.py: non-compliant random perturbation force on the last 2 force bodies
(no admittance). force_apply_perturb (:1016-1031): force_applied_w[:,-2:]=perturb_force.current,
eccentric torque cross(quat_apply(quat,force_pos_delta), force_applied_w).
force_update_perturb_and_target (:925-956): timer -> resample (transit U(20,50)+hold U(20,100),
rand_points_isotropic(K,2,max_force*alpha), per-body rand<0.5 enable, TemporalLerp transition);
force target tracks the reference keypoints directly (no admittance); vel = adjacent-step diff.
"""
import numpy as np

from unilab.envs.gh_tracking.force_system import ForceSystem
from unilab.envs.gh_tracking.motion_dataset import write_synthetic_dataset


def _extreme_fs(n=2, max_force=30.0, seed=0):
    return ForceSystem(num_envs=n, physics_dt=0.005, step_dt=0.02,
                       max_force=max_force, compliance=False, seed=seed)


# --- force_apply_perturb: 2-body force + eccentric torque + net wrench --- #

def test_force_apply_perturb_last_two_bodies_and_eccentric_torque():
    from unilab.utils.rotation import np_quat_apply_batched

    fs = _extreme_fs(2, max_force=30.0)
    # known perturbation force on the 2 bodies + known eccentric offset
    fs.perturb_force.value[:] = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])  # (N,2,3)
    fs.force_pos_delta[:] = 0.05  # nonzero -> exercise the eccentric-torque formula
    n = 2
    quat6 = np.zeros((n, fs.M, 4)); quat6[..., 0] = 1.0   # identity
    root_quat = np.zeros((n, 4)); root_quat[:, 0] = 1.0

    f6, tau6 = fs.force_apply_perturb(0, quat_w=quat6, root_quat_w=root_quat)
    assert f6.shape == (n, fs.M, 3) and tau6.shape == (n, fs.M, 3)
    np.testing.assert_array_equal(f6[:, :-2], 0.0)                     # only last 2 get force
    np.testing.assert_allclose(f6[:, -2:], fs.perturb_force.current)
    # eccentric torque = cross(quat_apply(quat, force_pos_delta), force_applied_w)
    delta_w = np_quat_apply_batched(quat6, fs.force_pos_delta)
    np.testing.assert_allclose(tau6, np.cross(delta_w, f6, axis=-1))


def test_perturb_force_norm_bounded_and_per_body_enable():
    fs = _extreme_fs(64, max_force=30.0, seed=1)
    fs.force_sample_timer[:] = 0                     # force a resample this call
    ref = np.zeros((64, fs.M, 3))
    rootp = np.zeros((64, 3)); rootq = np.zeros((64, 4)); rootq[:, 0] = 1.0
    fs.force_update_perturb_and_target(root_pos_w=rootp, root_quat_w=rootq, ref_keypoints_w=ref)
    # timer resampled into transit(20,50)+hold(20,100) -> [40,150)
    assert np.all((fs.force_sample_timer >= 40) & (fs.force_sample_timer < 150))
    # perturb end target norm <= max_force (rand_points_isotropic ball radius)
    end = fs.perturb_force._end
    assert np.all(np.linalg.norm(end, axis=-1) <= 30.0 + 1e-6)


def test_force_target_tracks_reference_keypoint_no_admittance():
    fs = _extreme_fs(2, max_force=30.0)
    fs.force_sample_timer[:] = 100                   # no resample
    ref = np.random.default_rng(0).standard_normal((2, fs.M, 3))
    rootp = np.zeros((2, 3)); rootq = np.zeros((2, 4)); rootq[:, 0] = 1.0
    fs.last_reset_env_ids = np.arange(2)             # first call -> prev seeded, vel 0
    fs.force_update_perturb_and_target(root_pos_w=rootp, root_quat_w=rootq, ref_keypoints_w=ref)
    # target == reference keypoint directly (no admittance integration)
    np.testing.assert_allclose(fs.force_keypoint_w, ref)
    np.testing.assert_allclose(fs.force_keypoint_vel_w, 0.0)  # prev seeded on reset
    # second step: vel = (kp - prev)/step_dt
    fs.last_reset_env_ids = None
    ref2 = ref + 0.1
    fs.force_update_perturb_and_target(root_pos_w=rootp, root_quat_w=rootq, ref_keypoints_w=ref2)
    np.testing.assert_allclose(fs.force_keypoint_vel_w, (ref2 - ref) / 0.02, rtol=1e-6)


# --- env extreme smoke + gentle-unchanged ------------------------------- #

def _make_env(tmp_path, compliance, max_force, n=2):
    from unilab.envs.gh_tracking.config import GHTrackingCfg
    from unilab.envs.gh_tracking.env import GHTrackingEnv
    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[120, 200], seed=0)
    cfg = GHTrackingCfg()
    cfg.motion.dirs = [str(tmp_path / "interx")]; cfg.motion.weights = [1.0]
    cfg.force.compliance = compliance; cfg.force.max_force = max_force
    return GHTrackingEnv(cfg, num_envs=n, backend_type="mujoco"), cfg


def test_extreme_env_smoke_finite(tmp_path):
    env, _cfg = _make_env(tmp_path, compliance=False, max_force=30.0, n=2)
    assert env.force_system.compliance is False
    state = env.init_state()
    assert state.obs["policy"].shape == (2, 450)
    for _ in range(5):
        state = env.step(np.zeros((2, 29)))
        assert np.isfinite(state.reward).all()
        assert np.isfinite(state.info["reward_vec"]).all()
        # perturb force only on the last 2 force bodies
        np.testing.assert_array_equal(env.force_system.force_applied_w[:, :-2], 0.0)
    env.close()


def test_gentle_path_unchanged(tmp_path):
    """compliance=True must still run the admittance path (no perturb)."""
    env, _cfg = _make_env(tmp_path, compliance=True, max_force=30.0, n=2)
    assert env.force_system.compliance is True
    env.init_state()
    env.step(np.zeros((2, 29)))
    # gentle populates the admittance compliant keypoint (perturb path never runs)
    assert env.force_system.force_keypoint_w.shape == (2, 6, 3)
    env.close()


def test_extreme_task_config_composes():
    """conf/.../mujoco_extreme_force.yaml aligns with GH G1_extreme_force.yaml."""
    from hydra import compose, initialize_config_dir
    from pathlib import Path

    conf = str(Path(__file__).resolve().parents[3] / "conf" / "gh_distill")
    with initialize_config_dir(config_dir=conf, version_base="1.3"):
        cfg = compose(config_name="config",
                      overrides=["phase=train", "task=gh_tracking/mujoco_extreme_force"])
    assert cfg.training.task_name == "GHTracking"
    assert cfg.env.force.max_force == 30.0
    assert cfg.env.force.compliance is False
    assert cfg.reward.impedance.force_reward.enabled is False
    assert cfg.reward.impedance.force_exd_penalty.enabled is False
