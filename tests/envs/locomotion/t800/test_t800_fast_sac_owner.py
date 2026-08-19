"""FastSAC owner and runtime contracts for T800 walk-flat."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.envs.locomotion.t800.joystick import T800WalkEnv

ROOT_DIR = Path(__file__).parents[4]
CONF_DIR = ROOT_DIR / "conf" / "offpolicy"


def _compose_t800_sac_owner():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose(
            "config",
            overrides=["algo=sac", "task=sac/t800_walk_flat/mujoco"],
        )


def _build_observation_probe(cfg: Any) -> T800WalkEnv:
    env: Any = object.__new__(T800WalkEnv)
    env._num_envs = 1
    env._cfg = SimpleNamespace(
        noise_config=cfg.env.noise_config,
        reward_config=None,
        curriculum=cfg.env.curriculum,
    )
    env._reward_cfg = cfg.reward
    env.default_angles = np.zeros((1, 22), dtype=np.float32)
    env._obs_noise = lambda data, scale: np.asarray(data, dtype=np.float32)
    return env


def test_t800_fast_sac_owner_matches_g1_walk_contract_without_time_scaling():
    cfg = _compose_t800_sac_owner()

    assert cfg.training.task_name == "T800WalkFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert str(cfg.env.scene.model_file).replace("\\", "/").endswith("robots/t800/scene_flat.xml")
    assert cfg.env.sim_dt == pytest.approx(0.002)
    assert cfg.env.ctrl_dt == pytest.approx(0.01)
    assert len(cfg.env.control_config.action_scale) == 22
    assert len(cfg.reward.pose_weights) == 22
    assert cfg.algo.use_symmetry is False
    assert cfg.algo.gamma == pytest.approx(0.98488578)
    assert cfg.algo.learning_starts == 10
    assert cfg.algo.max_iterations == 10000
    assert cfg.algo.save_interval == 1000
    assert cfg.algo.updates_per_step == 8
    assert cfg.algo.policy_frequency == 4
    assert {
        "penalty_orientation",
        "penalty_ang_vel_xy",
        "penalty_action_rate",
        "alive",
    } <= set(cfg.reward.scales)


def test_t800_fast_sac_owner_uses_t800_termination_domain():
    cfg = _compose_t800_sac_owner()

    assert cfg.reward.max_tilt_deg == pytest.approx(25.0)
    assert cfg.reward.min_base_height == pytest.approx(0.7165)


def test_t800_fast_sac_observation_uses_g1_walk_layout_and_scaling():
    cfg = _compose_t800_sac_owner()
    env = _build_observation_probe(cfg)
    obs = env._compute_obs(
        {
            "commands": np.array([[0.7, 0.0, 0.2]], dtype=np.float32),
            "current_actions": np.zeros((1, 22), dtype=np.float32),
            "gait_phase": np.array([[0.3, 3.4]], dtype=np.float32),
        },
        linvel=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        gyro=np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
        gravity=np.array([[0.1, 0.2, 0.9]], dtype=np.float32),
        dof_pos=np.zeros((1, 22), dtype=np.float32),
        dof_vel=np.arange(7.0, 29.0, dtype=np.float32)[None, :],
    )

    assert env._uses_walk_observation_profile() is True
    assert env.obs_groups_spec == {"obs": 77, "critic": 80}
    assert obs["obs"].shape == (1, 77)
    assert obs["critic"].shape == (1, 80)
    np.testing.assert_allclose(obs["obs"][:, :3], [[1.0, 1.25, 1.5]])
    np.testing.assert_allclose(
        obs["obs"][:, 28:50],
        np.arange(7.0, 29.0, dtype=np.float32)[None, :] * 0.05,
    )
    np.testing.assert_allclose(obs["critic"][:, -3:], [[2.0, 4.0, 6.0]])


def test_t800_fast_sac_builder_consumes_77_80_22_without_symmetry(
    monkeypatch: pytest.MonkeyPatch,
):
    from unilab.algos.torch.fast_sac import double_buffer as owner_module

    cfg = _compose_t800_sac_owner()

    class ProbeEnv:
        obs_groups_spec = {"obs": 77, "critic": 80}
        action_space = SimpleNamespace(shape=(22,))

        def build_symmetry_augmentation(self, *, device: str):
            raise AssertionError(f"symmetry must stay disabled, got device={device}")

        def close(self):
            return None

    class RecordingLearner:
        def __init__(self, *, device: str, **kwargs):
            self.device = device
            self.kwargs = kwargs

    class RecordingRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(owner_module, "ensure_registries", lambda: None)
    monkeypatch.setattr(owner_module, "create_env", lambda *args, **kwargs: ProbeEnv())
    monkeypatch.setattr(owner_module, "FastSACLearner", RecordingLearner)
    monkeypatch.setattr(owner_module, "DoubleBufferOffPolicyRunner", RecordingRunner)

    runner = owner_module.build_sac_double_buffer_runner(
        cfg,
        env_cfg_override={},
        replay_prefetch_mode="one_tick",
        device="cuda:0",
    )

    learner = runner.kwargs["learner"]
    assert learner.kwargs["obs_dim"] == 77
    assert learner.kwargs["critic_obs_dim"] == 80
    assert learner.kwargs["action_dim"] == 22
    assert learner.kwargs["use_symmetry"] is False
    assert learner.kwargs["symmetry_augmentation"] is None
    assert runner.kwargs["num_envs"] == 2048
    assert runner.kwargs["learning_starts"] == 10
    assert runner.kwargs["updates_per_step"] == 8


@pytest.mark.slow
def test_t800_fast_sac_mujoco_reset_and_step_are_finite():
    pytest.importorskip("mujoco")
    pytest.importorskip("mujoco_uni.batch_env")
    from unilab.base import registry
    from unilab.training.backend_adapter import BackendAdapter

    cfg = _compose_t800_sac_owner()
    registry.ensure_registries()
    env_override = BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="sac",
    ).build_task_env_cfg_override()
    env = registry.make(
        "T800WalkFlat",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override=env_override,
    )
    try:
        state = env.init_state()
        assert env.action_space.shape == (22,)
        assert state.obs["obs"].shape == (1, 77)
        assert state.obs["critic"].shape == (1, 80)
        for _ in range(3):
            state = env.step(np.zeros((1, 22), dtype=np.float64))
        assert np.isfinite(state.obs["obs"]).all()
        assert np.isfinite(state.obs["critic"]).all()
        assert np.isfinite(state.reward).all()
    finally:
        env.close()
