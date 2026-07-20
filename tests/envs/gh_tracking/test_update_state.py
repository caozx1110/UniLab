"""Task 9.3: update_state ordering (1-step _cum_error lag) + producer/consumer.

Driven via the _reset_idx backbone + a manual apply_action/backend.step/update_state
(the autoreset lifecycle needs the DR provider from Task 9.5). A synthetic motion
dataset backs the env (DP2: no real data yet).
"""
import numpy as np

from unilab.base.np_env import NpEnvState
from unilab.envs.gh_tracking.motion_dataset import write_synthetic_dataset


def _minimal_state(n: int) -> NpEnvState:
    return NpEnvState(
        obs={},
        reward=np.zeros(n),
        terminated=np.zeros(n, dtype=bool),
        truncated=np.zeros(n, dtype=bool),
        info={"steps": np.zeros(n, dtype=np.uint32)},
    )


def _make_env(tmp_path, n=2):
    from unilab.envs.gh_tracking.config import GHTrackingCfg
    from unilab.envs.gh_tracking.env import GHTrackingEnv

    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[200, 300], seed=0)
    cfg = GHTrackingCfg()
    cfg.motion.dirs = [str(tmp_path / "interx")]
    cfg.motion.weights = [1.0]
    env = GHTrackingEnv(cfg, num_envs=n, backend_type="mujoco")  # __init__ materializes (DR init)
    env._reset_idx(np.arange(n))
    return env, cfg


def _drive_one_step(env, cfg, n):
    state = _minimal_state(n)
    ctrl = env.apply_action(np.zeros((n, 29)), state)
    env._backend.step(ctrl, cfg.sim_substeps)
    return env.update_state(state)


def test_termination_update_before_reward_1_step_lag(tmp_path):
    """termination.update() must run BEFORE _compute_reward() (1-step _cum_error lag)."""
    env, cfg = _make_env(tmp_path, 2)
    order: list[str] = []
    orig_update = env.termination.update
    orig_reward = env._compute_reward
    env.termination.update = lambda ce: (order.append("update"), orig_update(ce))[1]
    env._compute_reward = lambda: (order.append("reward"), orig_reward())[1]

    _drive_one_step(env, cfg, 2)

    assert order == ["update", "reward"], "termination.update must precede _compute_reward"
    env.close()


def test_cum_error_producer_consumer(tmp_path):
    """Tracking reward writes _cum_error (3 comps); priv_critic obs consumes it."""
    env, cfg = _make_env(tmp_path, 2)
    new_state = _drive_one_step(env, cfg, 2)

    assert env._cum_error.shape == (2, 3)
    assert set(new_state.obs.keys()) == {"policy", "priv", "priv_critic"}
    assert new_state.obs["policy"].shape == (2, 450)
    assert new_state.obs["priv"].shape == (2, 717)
    assert new_state.obs["priv_critic"].shape == (2, 3)
    np.testing.assert_allclose(new_state.obs["priv_critic"], env._cum_error)
    env.close()


def test_reward_is_three_group_vector_summed_scalar(tmp_path):
    """reward_vec is (N,3) per-group·step_dt; scalar reward = sum, stashed for GAE."""
    env, cfg = _make_env(tmp_path, 2)
    new_state = _drive_one_step(env, cfg, 2)

    assert new_state.reward.shape == (2,)
    assert new_state.info["reward_vec"].shape == (2, 3)
    np.testing.assert_allclose(new_state.reward, new_state.info["reward_vec"].sum(axis=-1))
    env.close()
