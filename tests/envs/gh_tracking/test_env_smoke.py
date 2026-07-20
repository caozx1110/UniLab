"""Task 9.7: GHTrackingEnv full-lifecycle integration smoke test.

init_state -> step x10 -> close with all Phase 1-7 components wired: DR-provider reset
(D2 order), apply_action + backend.step (per-substep control/force/telemetry hooks),
update_state (1-step lag, 3-group reward, _cum_error producer/consumer), autoreset.
Synthetic weighted motion backs the env (DP2: no real data yet).
"""
import numpy as np

from unilab.envs.gh_tracking.motion_dataset import write_synthetic_dataset


def test_env_smoke_init_step_close(tmp_path):
    from unilab.envs.gh_tracking.config import GHTrackingCfg
    from unilab.envs.gh_tracking.env import GHTrackingEnv

    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[120, 200], seed=1)
    write_synthetic_dataset(str(tmp_path / "lafan"), clip_lengths=[80, 300], seed=2)
    cfg = GHTrackingCfg()
    cfg.motion.dirs = [str(tmp_path / "interx"), str(tmp_path / "lafan")]
    cfg.motion.weights = [0.5, 0.5]

    n = 4
    env = GHTrackingEnv(cfg, num_envs=n, backend_type="mujoco")

    # init_state triggers a full reset (all envs) through the DR provider
    state = env.init_state()
    assert set(state.obs.keys()) == {"policy", "priv", "priv_critic"}
    assert state.obs["policy"].shape == (n, 450)
    assert state.obs["priv"].shape == (n, 717)
    assert state.obs["priv_critic"].shape == (n, 3)

    rng = np.random.default_rng(0)
    for _ in range(10):
        actions = rng.standard_normal((n, 29)).clip(-1.0, 1.0)
        state = env.step(actions)
        assert state.obs["policy"].shape == (n, 450)
        assert state.obs["priv"].shape == (n, 717)
        assert state.obs["priv_critic"].shape == (n, 3)
        assert state.reward.shape == (n,)
        assert state.terminated.shape == (n,)
        assert state.truncated.shape == (n,)
        assert np.isfinite(state.reward).all(), "reward must be finite"
        assert np.isfinite(state.obs["policy"]).all(), "policy obs must be finite"

    env.close()
