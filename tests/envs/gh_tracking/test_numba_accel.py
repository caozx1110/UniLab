import numpy as np
import pytest
from unilab.envs.gh_tracking.config import GHTrackingCfg
from unilab.envs.gh_tracking import gh_tracking_numba as NB
from unilab.envs.gh_tracking.motion_dataset import write_synthetic_dataset


def _make_env(tmp_path, numba: bool, num_envs: int = 8):
    from unilab.envs.gh_tracking.env import GHTrackingEnv
    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[200, 300], seed=0)
    cfg = GHTrackingCfg()
    cfg.motion.dirs = [str(tmp_path / "interx")]
    cfg.motion.weights = [1.0]
    cfg.numba_acceleration = numba
    return GHTrackingEnv(cfg, num_envs=num_envs, backend_type="mujoco")


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


@pytest.mark.skipif(not NB.is_available(), reason="numba not installed")
def test_update_state_full_parity_multistep(tmp_path):
    # Task 4 (re-scoped): no kernel fusion. This is the real safety gate for the
    # whole numba-acceleration feature — it locks the FULL update_state output
    # (all three obs groups + reward + terminated) across a multi-step rollout,
    # comparing the numba-accelerated path (Task 2 reward kernel + Task 3 obs
    # kernel, termination on numpy) against the pure-numpy path bit-for-tolerance.
    #
    # Both rollouts use the identical synthetic dataset and identical
    # np.random.seed(0) before construction, so the two envs evolve through the
    # same trajectory (same resets, same domain-rand draws, same obs noise
    # streams) — the only intended difference is the reward/obs compute backend.
    n = 32

    def rollout(numba):
        np.random.seed(0)
        env = _make_env(tmp_path, numba, num_envs=n)
        env.init_state()
        frames = []
        for _ in range(10):
            s = env.step(np.zeros((n, env._backend.num_actuators), dtype=np.float32))
            frames.append((
                s.obs["policy"].copy(),
                s.obs["priv"].copy(),
                s.obs["priv_critic"].copy(),
                s.reward.copy(),
                s.terminated.copy(),
            ))
        env.close()
        return frames

    frames_np = rollout(False)
    frames_nb = rollout(True)

    for i, ((pa, va, ca, ra, ta), (pb, vb, cb, rb, tb)) in enumerate(
        zip(frames_np, frames_nb)
    ):
        np.testing.assert_allclose(
            pb, pa, rtol=1e-4, atol=1e-5, err_msg=f"policy obs mismatch at step {i}"
        )
        np.testing.assert_allclose(
            vb, va, rtol=1e-4, atol=1e-5, err_msg=f"priv obs mismatch at step {i}"
        )
        np.testing.assert_allclose(
            cb, ca, rtol=1e-4, atol=1e-5, err_msg=f"priv_critic obs mismatch at step {i}"
        )
        np.testing.assert_allclose(
            rb, ra, rtol=1e-4, atol=1e-5, err_msg=f"reward mismatch at step {i}"
        )
        np.testing.assert_array_equal(tb, ta, err_msg=f"terminated mismatch at step {i}")
