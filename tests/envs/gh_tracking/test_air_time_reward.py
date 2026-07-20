"""T10.1: feet_air_time_ref + first_contact + motion feet_standing.

Unit-tests the stateful FeetAirTimeRef (GH locomotion.py:163-213 formula) and the
env wiring (term in the loco group, first_contact shared with impact_force_l2,
feet_standing sourced from the motion slice).
"""
import numpy as np

from unilab.base.np_env import NpEnvState
from unilab.envs.gh_tracking.motion_dataset import write_synthetic_dataset
from unilab.envs.gh_tracking.rewards import FeetAirTimeRef


def test_feet_air_time_ref_penalizes_short_air_at_landing():
    air = FeetAirTimeRef(num_envs=1, num_feet=2, thres=0.8, step_dt=0.02)
    fs = np.zeros((1, 2), dtype=bool)      # motion not standing
    height = np.full((1, 2), 0.1)

    for _ in range(3):                     # airborne: reward_time grows, no landing
        r, fc = air.step(np.zeros((1, 2), dtype=bool), height, fs)
        np.testing.assert_allclose(r, 0.0)
        assert not fc.any()
    assert np.all(air.reward_time > 0.0)

    r, fc = air.step(np.ones((1, 2), dtype=bool), height, fs)   # land
    assert fc.all(), "first_contact fires on the landing step"
    assert np.all(r < 0.0), "short air time (< thres) is penalized at first contact"
    np.testing.assert_allclose(air.reward_time, 0.0)            # zeroed on contact

    r2, fc2 = air.step(np.ones((1, 2), dtype=bool), height, fs)  # still in contact
    assert not fc2.any()
    np.testing.assert_allclose(r2, 0.0)


def test_feet_standing_xor_penalizes_disagreement():
    """contact_diff = feet_standing ^ current_contact decays reward_time by step_dt."""
    air = FeetAirTimeRef(num_envs=1, num_feet=2, thres=0.8, step_dt=0.02)
    standing = np.ones((1, 2), dtype=bool)   # reference expects foot planted
    # robot airborne while reference standing -> disagreement -> reward_time decreases
    r, fc = air.step(np.zeros((1, 2), dtype=bool), np.full((1, 2), 0.1), standing)
    assert np.all(air.reward_time < 0.0)


# --- env wiring ---------------------------------------------------------- #

def _minimal_state(n):
    return NpEnvState(obs={}, reward=np.zeros(n), terminated=np.zeros(n, bool),
                      truncated=np.zeros(n, bool), info={"steps": np.zeros(n, np.uint32)})


def _make_env(tmp_path, n=2):
    from unilab.envs.gh_tracking.config import GHTrackingCfg
    from unilab.envs.gh_tracking.env import GHTrackingEnv
    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[200, 300], seed=0)
    cfg = GHTrackingCfg()
    cfg.motion.dirs = [str(tmp_path / "interx")]
    cfg.motion.weights = [1.0]
    env = GHTrackingEnv(cfg, num_envs=n, backend_type="mujoco")
    env._reset_idx(np.arange(n))
    return env, cfg


def _drive_one_step(env, cfg, n):
    state = _minimal_state(n)
    ctrl = env.apply_action(np.zeros((n, 29)), state)
    env._backend.step(ctrl, cfg.sim_substeps)
    return env.update_state(state)


def test_feet_air_time_ref_wired_into_loco_group(tmp_path):
    from unilab.envs.gh_tracking.env import _REWARD_GROUPS

    assert ("feet_air_time_ref", 10.0) in _REWARD_GROUPS["loco"]
    env, cfg = _make_env(tmp_path, 2)
    new_state = _drive_one_step(env, cfg, 2)

    # first_contact now real (shared with impact_force_l2), feet_standing from motion
    assert env._rc["first_contact"].shape == (2, 2)
    assert env._feet_standing.shape == (2, 2) and env._feet_standing.dtype == bool
    # reward stays finite + 3-group vector shape intact
    assert np.isfinite(new_state.reward).all()
    assert new_state.info["reward_vec"].shape == (2, 3)
    env.close()
