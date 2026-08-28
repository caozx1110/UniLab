from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unilab.managers import ManagerTermBaseCfg
from unilab.tasks.motion_tracking.g1.sonic import termination_terms as terms
from unilab.tasks.motion_tracking.g1.sonic.termination_terms import (
    sonic_anchor_height_adaptive,
    sonic_anchor_ori_full,
    sonic_body_height_adaptive,
    sonic_foot_pos_xyz,
)


def _env() -> tuple[SimpleNamespace, SimpleNamespace]:
    num_envs = 3
    names = ("pelvis", "left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link")
    command = SimpleNamespace()
    command.num_envs = num_envs
    command.cfg = SimpleNamespace(body_names=names)
    command.anchor_pos_w = np.asarray([[0, 0, 1.0], [0, 0, 0.4], [0, 0, 1.0]], dtype=np.float32)
    command.reference_anchor_height = np.asarray([1.0, 0.4, 1.0], dtype=np.float32)
    command.robot_anchor_pos_w = command.anchor_pos_w.copy()
    command.robot_anchor_pos_w[:, 2] -= np.asarray([0.1, -0.6, 0.0], dtype=np.float32)
    command.anchor_quat_w = np.tile(np.asarray([1, 0, 0, 0], dtype=np.float32), (num_envs, 1))
    command.robot_anchor_quat_w = command.anchor_quat_w.copy()
    command.body_pos_relative_w = np.zeros((num_envs, len(names), 3), dtype=np.float32)
    command.robot_body_pos_w = command.body_pos_relative_w.copy()
    command.body_pos_relative_w[0, 1, 0] = 0.29
    command.robot_body_pos_w[0, 1, 0] = 0.1
    command.body_pos_relative_w[1, 2, 2] = 0.8
    command.robot_body_pos_w[1, 2, 2] = 0.0
    env = SimpleNamespace(
        num_envs=num_envs,
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )
    return env, command


def test_release_anchor_height_adaptive_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    env, command = _env()
    monkeypatch.setattr(terms, "_command", lambda env, name: command)
    result = sonic_anchor_height_adaptive(
        env, "motion", threshold=0.15, threshold_adaptive=True, down_threshold=0.75, root_height_threshold=0.5
    )
    np.testing.assert_array_equal(result, [False, False, False])
    command.robot_anchor_pos_w[0, 2] = 0.7
    np.testing.assert_array_equal(
        sonic_anchor_height_adaptive(
            env, "motion", threshold=0.15, threshold_adaptive=True, down_threshold=0.75, root_height_threshold=0.5
        ),
        [True, False, False],
    )


def test_release_orientation_uses_squared_full_quaternion_error(monkeypatch: pytest.MonkeyPatch) -> None:
    env, command = _env()
    monkeypatch.setattr(terms, "_command", lambda env, name: command)
    angle = 0.6
    command.robot_anchor_quat_w[2] = [np.cos(angle / 2), np.sin(angle / 2), 0, 0]
    result = sonic_anchor_ori_full(env, "motion", threshold=0.2)
    np.testing.assert_array_equal(result, [False, False, True])


def test_release_body_height_and_foot_xyz_terms(monkeypatch: pytest.MonkeyPatch) -> None:
    env, command = _env()
    monkeypatch.setattr(terms, "_command", lambda env, name: command)
    body_cfg = ManagerTermBaseCfg(
        func=sonic_body_height_adaptive,
        params={
            "command_name": "motion",
            "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
        },
    )
    body_term = sonic_body_height_adaptive(body_cfg, env)
    np.testing.assert_array_equal(
        body_term(env, "motion", threshold=0.15, threshold_adaptive=True, down_threshold=0.75, root_height_threshold=0.5),
        [False, True, False],
    )
    foot_cfg = ManagerTermBaseCfg(
        func=sonic_foot_pos_xyz,
        params={"command_name": "motion", "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"]},
    )
    foot_term = sonic_foot_pos_xyz(foot_cfg, env)
    np.testing.assert_array_equal(foot_term(env, "motion", threshold=0.2), [False, True, False])


def test_terms_fail_closed_for_unknown_command() -> None:
    with pytest.raises(TypeError, match="SonicMotionCommand"):
        terms.sonic_anchor_height_adaptive(
            SimpleNamespace(
                command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace()),
            ),
            "motion",
            threshold=0.15,
        )
