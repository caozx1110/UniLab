"""Task 9.4: per-substep force wrench + post-substep telemetry callbacks.

Verifies both hooks are registered on the backend (real attr names
_pre_step_wrench_fn / _post_step_callback_fn) and fire once per physics substep,
covering the last substep (a pre-step hook alone cannot).
"""
import numpy as np

from unilab.base.np_env import NpEnvState
from unilab.envs.gh_tracking.motion_dataset import write_synthetic_dataset


def _minimal_state(n: int) -> NpEnvState:
    return NpEnvState(
        obs={}, reward=np.zeros(n), terminated=np.zeros(n, dtype=bool),
        truncated=np.zeros(n, dtype=bool), info={"steps": np.zeros(n, dtype=np.uint32)},
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


def test_pre_step_wrench_and_post_step_callbacks_registered(tmp_path):
    """Both per-substep hooks must be registered in __init__ (Phase 1 real attrs)."""
    env, _cfg = _make_env(tmp_path, 2)
    assert env._backend._pre_step_wrench_fn is not None
    assert env._backend._post_step_callback_fn is not None
    env.close()


def test_callbacks_fire_every_substep_including_last(tmp_path):
    """Force-wrench and telemetry hooks fire once per physics substep, incl. the last."""
    env, cfg = _make_env(tmp_path, 2)
    state = _minimal_state(2)
    ctrl = env.apply_action(np.zeros((2, 29)), state)
    assert env._post_substep == 0  # substep counters reset at start of control step
    assert env.force_system._force_substep == 0

    env._backend.step(ctrl, cfg.sim_substeps)

    assert env._post_substep == cfg.sim_substeps, "post-step telemetry must fire every substep"
    assert env.force_system._force_substep == cfg.sim_substeps, "force wrench must fire every substep"
    env.close()
