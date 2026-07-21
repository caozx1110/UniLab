import numpy as np
import pytest
from unilab.envs.gh_tracking.config import GHTrackingCfg
from unilab.envs.gh_tracking import gh_tracking_numba as NB
from unilab.envs.gh_tracking.motion_dataset import write_synthetic_dataset


def _make_env(tmp_path, numba: bool):
    from unilab.envs.gh_tracking.env import GHTrackingEnv
    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[200, 300], seed=0)
    cfg = GHTrackingCfg()
    cfg.motion.dirs = [str(tmp_path / "interx")]
    cfg.motion.weights = [1.0]
    cfg.numba_acceleration = numba
    return GHTrackingEnv(cfg, num_envs=8, backend_type="mujoco")


def test_config_defaults_off():
    cfg = GHTrackingCfg()
    assert cfg.numba_acceleration is False
    assert cfg.numba_num_threads is None


def test_accelerator_delegation_matches_numpy_update_state(tmp_path):
    # numba path (delegating) must equal the numpy path bit-for-bit in Task 1,
    # because the accelerator just calls the same numpy functions.
    np.random.seed(0)
    env_np = _make_env(tmp_path, numba=False)
    s_np = env_np.init_state()
    s_np = env_np.step(np.zeros((8, env_np._backend.num_actuators), dtype=np.float32))

    np.random.seed(0)
    env_nb = _make_env(tmp_path, numba=True)
    assert env_nb._numba_accelerator is not None
    s_nb = env_nb.init_state()
    s_nb = env_nb.step(np.zeros((8, env_nb._backend.num_actuators), dtype=np.float32))

    for g in ("policy", "priv", "priv_critic"):
        np.testing.assert_allclose(s_nb.obs[g], s_np.obs[g], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(s_nb.reward, s_np.reward, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(s_nb.terminated, s_np.terminated)
