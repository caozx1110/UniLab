"""Task 2: fused prange reward kernel parity (numba fp32 vs numpy fallback).

The kernel replaces only the reward aggregation inside the numba accelerator;
obs/termination still delegate to numpy. Parity is a tolerance (fp32 + fastmath),
NOT bit-identical: rtol=1e-4, atol=1e-5. Column order [impedance, tracking, loco].

Uses the manual-drive fixture (write_synthetic_dataset + cfg.motion.dirs) — a
GHTrackingCfg with no motion dirs raises on reset. Both envs are seeded identically
(cfg.motion.seed) and driven with the same zero actions, so their trajectories —
and thus their per-step reward contexts — match; only the reward math backend differs.
"""
import numpy as np
import pytest

from unilab.base.np_env import NpEnvState
from unilab.envs.gh_tracking.gh_tracking_numba import is_available
from unilab.envs.gh_tracking.motion_dataset import write_synthetic_dataset


def _minimal_state(n: int) -> NpEnvState:
    return NpEnvState(
        obs={},
        reward=np.zeros(n),
        terminated=np.zeros(n, dtype=bool),
        truncated=np.zeros(n, dtype=bool),
        info={"steps": np.zeros(n, dtype=np.uint32)},
    )


def _make_env(tmp_path, numba: bool, n: int):
    from unilab.envs.gh_tracking.config import GHTrackingCfg
    from unilab.envs.gh_tracking.env import GHTrackingEnv

    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[200, 300], seed=0)
    cfg = GHTrackingCfg()
    cfg.numba_acceleration = numba
    cfg.motion.dirs = [str(tmp_path / "interx")]
    cfg.motion.weights = [1.0]
    env = GHTrackingEnv(cfg, num_envs=n, backend_type="mujoco")
    env._reset_idx(np.arange(n))
    return env, cfg


def _rollout_reward(tmp_path, numba: bool, n: int = 8, steps: int = 3):
    env, cfg = _make_env(tmp_path, numba, n)
    state = _minimal_state(n)
    rewards = []
    for _ in range(steps):
        ctrl = env.apply_action(np.zeros((n, 29)), state)
        env._backend.step(ctrl, cfg.sim_substeps)
        state = env.update_state(state)
        rewards.append(state.info["reward_vec"].copy())
    env.close()
    return np.stack(rewards)


@pytest.mark.skipif(not is_available(), reason="numba not installed")
def test_reward_uses_kernel(tmp_path):
    env, _ = _make_env(tmp_path, numba=True, n=4)
    assert env._numba_accelerator._reward_from_kernel is True
    env.close()


@pytest.mark.skipif(not is_available(), reason="numba not installed")
def test_reward_vec_parity_numba_vs_numpy(tmp_path):
    r_np = _rollout_reward(tmp_path, numba=False)
    r_nb = _rollout_reward(tmp_path, numba=True)
    np.testing.assert_allclose(r_nb, r_np, rtol=1e-4, atol=1e-5)
