"""Owner and I/O contracts for T800 PPO walk-flat."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.envs.locomotion.t800 import joystick as t800_joystick
from unilab.envs.locomotion.t800.joystick import (
    T800_ACTIVE_JOINT_NAMES,
    T800WalkEnv,
    expand_active_targets,
    resolve_required_indices,
)

CONF_DIR = Path(__file__).parents[4] / "conf" / "ppo"


def _compose_t800_owner():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose("config", overrides=["task=t800_walk_flat/mujoco"])


def test_t800_active_joint_contract_is_official_22_joint_walking_order():
    assert len(T800_ACTIVE_JOINT_NAMES) == 22
    assert T800_ACTIVE_JOINT_NAMES[:12] == tuple(
        (
            "J00_HIP_PITCH_L",
            "J01_HIP_ROLL_L",
            "J02_HIP_YAW_L",
            "J03_KNEE_PITCH_L",
            "J04_ANKLE_PITCH_L",
            "J05_ANKLE_ROLL_L",
            "J06_HIP_PITCH_R",
            "J07_HIP_ROLL_R",
            "J08_HIP_YAW_R",
            "J09_KNEE_PITCH_R",
            "J10_ANKLE_PITCH_R",
            "J11_ANKLE_ROLL_R",
        )
    )
    assert T800_ACTIVE_JOINT_NAMES[12:] == tuple(
        (
            "J13_SHOULDER_PITCH_L",
            "J14_SHOULDER_ROLL_L",
            "J15_SHOULDER_YAW_L",
            "J16_ELBOW_PITCH_L",
            "J17_ELBOW_YAW_L",
            "J18_SHOULDER_PITCH_R",
            "J19_SHOULDER_ROLL_R",
            "J20_SHOULDER_YAW_R",
            "J21_ELBOW_PITCH_R",
            "J22_ELBOW_YAW_R",
        )
    )
    assert "J12_TORSO_YAW" not in T800_ACTIVE_JOINT_NAMES
    assert "J23_HEAD_PITCH" not in T800_ACTIVE_JOINT_NAMES
    assert "J24_HEAD_YAW" not in T800_ACTIVE_JOINT_NAMES


def test_resolve_required_indices_fails_closed_on_missing_joint():
    with pytest.raises(ValueError, match="missing required T800 actuator"):
        resolve_required_indices(("J00_HIP_PITCH_L",), T800_ACTIVE_JOINT_NAMES)


def test_resolve_required_indices_fails_closed_on_duplicate_available_name():
    with pytest.raises(ValueError, match="duplicate actuator names"):
        resolve_required_indices(("joint", "joint"), ("joint",))


def test_expand_active_targets_holds_inactive_position_targets():
    defaults = np.arange(25, dtype=np.float64)
    active_indices = np.asarray([*range(12), *range(13, 23)], dtype=np.int32)
    active = np.full((2, 22), 99.0)

    ctrl = expand_active_targets(active, defaults, active_indices)

    assert ctrl.shape == (2, 25)
    np.testing.assert_allclose(ctrl[:, active_indices], 99.0)
    np.testing.assert_allclose(
        ctrl[:, [12, 23, 24]],
        np.broadcast_to(defaults[[12, 23, 24]], (2, 3)),
    )


def test_t800_walk_obs_groups_are_77_actor_80_critic():
    env = object.__new__(T800WalkEnv)
    assert env.obs_groups_spec == {"obs": 77, "critic": 80}


def test_t800_env_resolves_assets_and_textures_before_backend_init(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[tuple[object, ...]] = []

    def fake_resolve_robot_asset_dir(directory: str, *, marker: str):
        events.append(("resolve", directory, marker))

    def fake_g1_init(
        self: object,
        cfg: object,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> None:
        del self, cfg
        events.append(("backend", num_envs, backend_type))

    monkeypatch.setattr(
        "unilab.assets.hub.resolve_robot_asset_dir",
        fake_resolve_robot_asset_dir,
    )
    monkeypatch.setattr(t800_joystick.G1WalkEnv, "__init__", fake_g1_init)

    T800WalkEnv(SimpleNamespace(), num_envs=4, backend_type="mujoco")

    assert events == [
        ("resolve", "robots/t800/assets", "LINK_BASE.obj"),
        ("resolve", "robots/t800/textures", "LINK_BASE.png"),
        ("backend", 4, "mujoco"),
    ]


def test_t800_apply_action_advances_and_wraps_both_gait_phases():
    env: Any = object.__new__(T800WalkEnv)
    env._num_action = 22
    env._num_envs = 2
    env._gait_phase_delta = 0.75
    env._cfg = SimpleNamespace(
        control_config=SimpleNamespace(
            simulate_action_latency=False,
            action_scale=np.ones(22),
        )
    )
    env.default_angles = np.zeros(22)
    env._full_default_ctrl = np.zeros(25)
    env._active_actuator_indices = np.asarray([*range(12), *range(13, 23)])
    initial_phase = np.asarray(
        [[2 * np.pi - 0.5, 0.25], [1.0, 2 * np.pi - 0.25]],
        dtype=np.float64,
    )
    state: Any = SimpleNamespace(info={"gait_phase": initial_phase.copy()})

    env.apply_action(np.zeros((2, 22)), state)

    np.testing.assert_allclose(
        state.info["gait_phase"],
        (initial_phase + env._gait_phase_delta) % (2 * np.pi),
    )


def test_t800_ppo_mujoco_owner_composes_with_22_joint_contract():
    cfg = _compose_t800_owner()

    assert cfg.training.task_name == "T800WalkFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert str(cfg.env.scene.model_file).replace("\\", "/").endswith("robots/t800/scene_flat.xml")
    assert len(cfg.env.control_config.action_scale) == 22
    assert len(cfg.reward.pose_weights) == 22
    assert cfg.algo.obs_groups.critic == ["critic"]


def test_t800_ppo_owner_keeps_default_ppo_timebase():
    cfg = _compose_t800_owner()

    assert cfg.env.ctrl_dt == pytest.approx(0.01)
    assert cfg.algo.num_steps_per_env == 24
    assert cfg.algo.algorithm.gamma == pytest.approx(0.99)
    assert cfg.algo.algorithm.lam == pytest.approx(0.95)


def test_t800_ppo_owner_keeps_default_entropy_coef():
    cfg = _compose_t800_owner()

    assert cfg.algo.algorithm.entropy_coef == pytest.approx(0.01)


def test_t800_ppo_owner_uses_morphology_adjusted_feet_phase_profile():
    cfg = _compose_t800_owner()

    assert cfg.reward.scales.feet_phase == pytest.approx(1.5)
    assert cfg.reward.gait_frequency == pytest.approx(1.5)
    assert cfg.reward.feet_phase_swing_height == pytest.approx(0.13)
    assert cfg.reward.feet_phase_tracking_sigma == pytest.approx(0.014)


@pytest.mark.slow
def test_t800_walk_runtime_reset_step_and_inactive_joint_hold_are_finite():
    pytest.importorskip("mujoco")
    pytest.importorskip("mujoco_uni.batch_env")
    from unilab.base import registry
    from unilab.training.backend_adapter import BackendAdapter

    cfg = _compose_t800_owner()
    registry.ensure_registries()
    env_override = BackendAdapter(
        cfg,
        root_dir=Path(__file__).parents[4],
        algo_name="ppo",
    ).build_task_env_cfg_override()
    env = cast(
        T800WalkEnv,
        registry.make(
            "T800WalkFlat",
            sim_backend="mujoco",
            num_envs=1,
            env_cfg_override=env_override,
        ),
    )
    try:
        state = env.init_state()
        assert env.action_space.shape == (22,)
        assert state.obs["obs"].shape == (1, 77)
        assert state.obs["critic"].shape == (1, 80)

        ctrl = env.apply_action(np.zeros((1, 22), dtype=np.float64), state)
        assert ctrl.shape == (1, 25)
        np.testing.assert_allclose(
            ctrl[:, [12, 23, 24]],
            np.broadcast_to(env._full_default_ctrl[[12, 23, 24]], (1, 3)),
        )

        for _ in range(3):
            state = env.step(np.zeros((1, 22), dtype=np.float64))
        assert np.isfinite(state.obs["obs"]).all()
        assert np.isfinite(state.obs["critic"]).all()
        assert np.isfinite(state.reward).all()
    finally:
        env.close()
