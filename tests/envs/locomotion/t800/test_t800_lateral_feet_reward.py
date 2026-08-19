"""T800 lateral foot-separation reward contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.base.registry import apply_cfg_overrides
from unilab.envs.locomotion.t800 import joystick as t800_joystick
from unilab.utils.rotation import np_quat_apply, np_yaw_to_quat

ROOT_DIR = Path(__file__).parents[4]
CONF_DIR = ROOT_DIR / "conf" / "offpolicy"


def _compose_t800_sac_owner():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose(
            "config",
            overrides=["algo=sac", "task=sac/t800_walk_flat/mujoco"],
        )


def _penalty(
    left_foot: np.ndarray,
    right_foot: np.ndarray,
    *,
    yaw: float = 0.0,
) -> np.ndarray:
    base_quat = np_yaw_to_quat(np.asarray([yaw], dtype=np.float64))
    return t800_joystick.compute_lateral_feet_penalty(
        left_foot,
        right_foot,
        base_quat,
        min_width=0.20,
        sigma=0.04,
    )


def test_lateral_penalty_is_not_hidden_by_fore_aft_step_length():
    left = np.asarray([[0.60, 0.08, 0.0]], dtype=np.float64)
    right = np.asarray([[-0.60, -0.08, 0.0]], dtype=np.float64)

    penalty = _penalty(left, right)

    assert penalty == pytest.approx([1.0 - np.exp(-1.0)])


def test_lateral_penalty_is_invariant_to_robot_yaw():
    yaw = np.pi / 2.0
    yaw_quat = np_yaw_to_quat(np.asarray([yaw], dtype=np.float64))
    left_local = np.asarray([[0.60, 0.08, 0.0]], dtype=np.float64)
    right_local = np.asarray([[-0.60, -0.08, 0.0]], dtype=np.float64)
    left_world = np_quat_apply(yaw_quat, left_local)
    right_world = np_quat_apply(yaw_quat, right_local)

    penalty = _penalty(left_world, right_world, yaw=yaw)

    assert penalty == pytest.approx([1.0 - np.exp(-1.0)])


def test_lateral_penalty_uses_signed_width_to_penalize_crossed_feet():
    left = np.asarray([[0.0, -0.05, 0.0]], dtype=np.float64)
    right = np.asarray([[0.0, 0.05, 0.0]], dtype=np.float64)

    penalty = _penalty(left, right)

    assert penalty[0] > 0.999


def test_lateral_penalty_is_zero_at_or_above_minimum_width():
    left = np.asarray([[0.0, 0.10, 0.0], [0.0, 0.12, 0.0]], dtype=np.float64)
    right = np.asarray([[0.0, -0.10, 0.0], [0.0, -0.12, 0.0]], dtype=np.float64)
    base_quat = np_yaw_to_quat(np.zeros(2, dtype=np.float64))

    penalty = t800_joystick.compute_lateral_feet_penalty(
        left,
        right,
        base_quat,
        min_width=0.20,
        sigma=0.04,
    )

    np.testing.assert_allclose(penalty, 0.0)


def test_t800_reward_dispatch_registers_lateral_penalty():
    env: Any = object.__new__(t800_joystick.T800WalkEnv)

    env._init_reward_functions()

    assert env._reward_fns["penalty_close_feet_lateral"] == env._reward_close_feet_lateral


def test_t800_lateral_reward_is_evaluated_without_a_contact_gate():
    class BackendProbe:
        def get_sensor_data(self, name: str) -> np.ndarray:
            return {
                "left_foot_pos": np.asarray([[0.60, 0.08, 0.0]], dtype=np.float64),
                "right_foot_pos": np.asarray([[-0.60, -0.08, 0.0]], dtype=np.float64),
            }[name]

        def get_base_quat(self) -> np.ndarray:
            return np_yaw_to_quat(np.zeros(1, dtype=np.float64))

    env: Any = object.__new__(t800_joystick.T800WalkEnv)
    env._backend = BackendProbe()
    env._reward_cfg = type(
        "RewardCfg",
        (),
        {
            "close_feet_lateral_threshold": 0.20,
            "close_feet_lateral_sigma": 0.04,
        },
    )()

    penalty = env._reward_close_feet_lateral(None)

    assert penalty == pytest.approx([1.0 - np.exp(-1.0)])


def test_t800_fast_sac_owner_enables_only_the_lateral_close_feet_penalty():
    cfg = _compose_t800_sac_owner()

    assert cfg.reward.scales.penalty_close_feet_lateral == pytest.approx(-5.0)
    assert "penalty_close_feet_xy" not in cfg.reward.scales
    assert cfg.reward.close_feet_lateral_threshold == pytest.approx(0.18)
    assert cfg.reward.close_feet_lateral_sigma == pytest.approx(0.04)

    reward_override = OmegaConf.to_container(cfg.reward, resolve=True)
    assert isinstance(reward_override, dict)
    env_cfg = t800_joystick.T800WalkFlatCfg()
    apply_cfg_overrides(env_cfg, {"reward_config": reward_override})
    assert isinstance(env_cfg.reward_config, t800_joystick.T800WalkRewardConfig)
    env_cfg.validate()


def test_t800_walk_config_rejects_non_positive_lateral_penalty_sigma():
    cfg = _compose_t800_sac_owner()
    reward_override = OmegaConf.to_container(cfg.reward, resolve=True)
    assert isinstance(reward_override, dict)
    reward_override["close_feet_lateral_sigma"] = 0.0
    env_cfg = t800_joystick.T800WalkFlatCfg()
    apply_cfg_overrides(env_cfg, {"reward_config": reward_override})

    with pytest.raises(ValueError, match="close_feet_lateral_sigma must be positive"):
        env_cfg.validate()
