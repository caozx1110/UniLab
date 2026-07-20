"""Task A: G1_no_force variant (compliance=False, max_force=0).

GH motion_tracking.py:1045-1051 — with compliance=False and max_force<=0, step() enters
NEITHER force branch, so no external wrench is applied (nur _limit_net_wrench_about_torso on
zero buffers). obs/reward/DR/3-phase are identical to gentle; the impedance force terms are
still computed on zero force and must stay finite (GH G1_no_force.yaml keeps them running,
only disabling force_reward/force_exd_penalty in its reward curriculum).
"""
import numpy as np

from unilab.envs.gh_tracking.force_system import ForceSystem
from unilab.envs.gh_tracking.motion_dataset import write_synthetic_dataset


def _make_no_force_env(tmp_path, n=2):
    from unilab.envs.gh_tracking.config import GHTrackingCfg
    from unilab.envs.gh_tracking.env import GHTrackingEnv

    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[120, 200], seed=0)
    cfg = GHTrackingCfg()
    cfg.motion.dirs = [str(tmp_path / "interx")]
    cfg.motion.weights = [1.0]
    cfg.force.compliance = False   # G1_no_force
    cfg.force.max_force = 0.0
    return GHTrackingEnv(cfg, num_envs=n, backend_type="mujoco"), cfg


# --- ForceSystem-level: no_force applies zero wrench --------------------- #

def test_force_system_no_force_pre_step_wrench_is_zero():
    fs = ForceSystem(num_envs=2, physics_dt=0.005, step_dt=0.02,
                     max_force=0.0, compliance=False)
    assert fs.compliance is False
    applied = {}

    class _Backend:
        def get_body_pos_w(self, ids):
            return np.random.randn(2, len(np.atleast_1d(ids)), 3)
        def get_body_quat_w(self, ids):
            q = np.zeros((2, len(np.atleast_1d(ids)), 4)); q[..., 0] = 1.0; return q
        def get_base_quat(self):
            q = np.zeros((2, 4)); q[:, 0] = 1.0; return q
        def apply_body_wrench(self, ids, force, torque):
            applied["force"] = force.copy(); applied["torque"] = torque.copy()

    fn = fs.as_pre_step_wrench(np.arange(6), torso_id=6)
    fs.reset_force_substep()
    fn(_Backend())
    # every applied wrench component is exactly zero (no external force)
    np.testing.assert_array_equal(applied["force"], 0.0)
    np.testing.assert_array_equal(applied["torque"], 0.0)
    np.testing.assert_array_equal(fs.force_applied_w, 0.0)
    np.testing.assert_array_equal(fs.force_applied_b, 0.0)


# --- env-level: no_force lifecycle + finite reward ---------------------- #

def test_no_force_env_zero_buffers_and_finite_reward(tmp_path):
    env, _cfg = _make_no_force_env(tmp_path, 2)
    assert env.force_system.compliance is False

    state = env.init_state()
    assert state.obs["policy"].shape == (2, 450)      # obs dims unchanged
    assert state.obs["priv"].shape == (2, 717)
    assert state.obs["priv_critic"].shape == (2, 3)

    for _ in range(5):
        state = env.step(np.zeros((2, 29)))
        # no external force applied throughout
        np.testing.assert_array_equal(env.force_system.force_applied_w, 0.0)
        np.testing.assert_array_equal(env.force_system.force_applied_b, 0.0)
        # impedance reward terms (force_reward/force_target/...) stay finite at zero force
        assert np.isfinite(state.reward).all()
        assert np.isfinite(state.info["reward_vec"]).all()
    env.close()


def test_no_force_task_config_composes():
    """conf/gh_distill/task/gh_tracking/mujoco_no_force.yaml aligns with GH G1_no_force.yaml."""
    from hydra import compose, initialize_config_dir
    from pathlib import Path

    conf = str(Path(__file__).resolve().parents[3] / "conf" / "gh_distill")
    with initialize_config_dir(config_dir=conf, version_base="1.3"):
        cfg = compose(config_name="config",
                      overrides=["phase=train", "task=gh_tracking/mujoco_no_force"])
    assert cfg.training.task_name == "GHTracking"
    assert cfg.env.force.max_force == 0.0
    assert cfg.env.force.compliance is False
    assert cfg.reward.impedance.force_reward.enabled is False        # GH disables these
    assert cfg.reward.impedance.force_exd_penalty.enabled is False
