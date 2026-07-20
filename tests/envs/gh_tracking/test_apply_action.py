"""Task 9.2: component init + apply_action (Phase 4 boot-protected residual pipeline).

apply_action is exercised in isolation from the full reset lifecycle (the DR
provider lands in Task 9.5): the action pipeline is reset directly and a minimal
NpEnvState is passed, since apply_action drives the pipeline's internal buffers.
"""
import numpy as np

from unilab.base.np_env import NpEnvState


def _minimal_state(n: int) -> NpEnvState:
    return NpEnvState(
        obs={},
        reward=np.zeros(n),
        terminated=np.zeros(n, dtype=bool),
        truncated=np.zeros(n, dtype=bool),
        info={},
    )


def test_apply_action_returns_ctrl_and_ingests():
    """apply_action returns (N, num_dof) ctrl and ingests the raw action (Phase 4)."""
    from unilab.envs.gh_tracking.env import GHTrackingEnv
    from unilab.envs.gh_tracking.config import GHTrackingCfg

    cfg = GHTrackingCfg()
    env = GHTrackingEnv(cfg, num_envs=2, backend_type="mujoco")
    env.action_pipeline.reset(np.arange(2), np.random.default_rng(0))

    actions = np.full((2, 29), 0.5)
    ctrl = env.apply_action(actions, _minimal_state(2))

    assert ctrl.shape == (2, 29), "ctrl should be (num_envs, num_dof)"
    # start_control_step must ingest the (clamped) raw action at action_buf slot 0
    np.testing.assert_allclose(env.action_pipeline.action_buf[:, 0, :], np.clip(actions, -10.0, 10.0))
    env.close()


def test_apply_action_clamps_and_pre_step_control_hook_registered():
    """Raw action clamped to [-10, 10]; per-substep actuator-control hook is wired."""
    from unilab.envs.gh_tracking.env import GHTrackingEnv
    from unilab.envs.gh_tracking.config import GHTrackingCfg

    cfg = GHTrackingCfg()
    env = GHTrackingEnv(cfg, num_envs=2, backend_type="mujoco")
    env.action_pipeline.reset(np.arange(2), np.random.default_rng(0))

    # Phase 4 as_pre_step_control hook must be registered in __init__
    assert env._backend._pre_step_control_fn is not None

    actions = np.full((2, 29), 50.0)  # exceeds the +-10 raw-action clamp
    env.apply_action(actions, _minimal_state(2))
    np.testing.assert_allclose(env.action_pipeline.action_buf[:, 0, :], 10.0)
    env.close()
