"""Task 3: fused prange obs kernel parity (numba fp32 vs numpy fallback).

The kernel replaces only the obs assembly inside the numba accelerator; the
stateful telemetry roll + rng noise (``obs_manager.update``) stays in numpy and
runs exactly once per step (same call-site as the numpy ``_compute_obs`` path),
so the injected noise streams match. Parity is a tolerance (fp32 + fastmath),
NOT bit-identical: rtol=1e-4, atol=1e-5.

Uses the manual-drive fixture (write_synthetic_dataset + cfg.motion.dirs) — a
GHTrackingCfg with no motion dirs raises on reset. Both envs are seeded
identically (cfg.motion.seed + cfg.obs.seed) and driven with the same zero
actions, so their trajectories — and thus their per-step obs — match; only the
obs math backend differs.
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


def _rollout_obs(tmp_path, numba: bool, n: int = 8, steps: int = 3):
    env, cfg = _make_env(tmp_path, numba, n)
    state = _minimal_state(n)
    out = []
    for _ in range(steps):
        ctrl = env.apply_action(np.zeros((n, 29)), state)
        env._backend.step(ctrl, cfg.sim_substeps)
        state = env.update_state(state)
        out.append({g: state.obs[g].copy() for g in ("policy", "priv", "priv_critic")})
    env.close()
    return out


@pytest.mark.skipif(not is_available(), reason="numba not installed")
def test_obs_uses_kernel(tmp_path):
    env, _ = _make_env(tmp_path, numba=True, n=4)
    # _obs_from_kernel is only flipped True inside _compute_obs_dict, so a step
    # must actually run before we can assert the kernel executed.
    n = 4
    env.step(np.zeros((n, env._backend.num_actuators), dtype=np.float32))
    assert env._numba_accelerator._obs_from_kernel is True
    env.close()


@pytest.mark.skipif(not is_available(), reason="numba not installed")
def test_obs_parity_numba_vs_numpy(tmp_path):
    a = _rollout_obs(tmp_path, numba=False)
    b = _rollout_obs(tmp_path, numba=True)
    for da, db in zip(a, b):
        for g in ("policy", "priv", "priv_critic"):
            np.testing.assert_allclose(db[g], da[g], rtol=1e-4, atol=1e-5)
