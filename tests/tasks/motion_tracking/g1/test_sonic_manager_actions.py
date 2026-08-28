"""Near-risk contract checks for Manager-Based SONIC actions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unilab.tasks.motion_tracking.g1.sonic.actions import (
    SONIC_ACTION_DIM,
    SONIC_ACTION_SCALE,
    SONIC_JOINT_TO_POLICY,
    SONIC_POLICY_JOINT_ORDER,
    SONIC_POLICY_TO_JOINT,
    SonicMotionJointPositionAction,
    SonicMotionJointPositionActionCfg,
    sonic_action_scale_by_joint,
)
from unilab.tasks.motion_tracking.g1.sonic.manager_terms import SONIC_JOINT_ORDER


def test_release_action_layout_and_scale_are_fixed() -> None:
    assert len(SONIC_POLICY_JOINT_ORDER) == SONIC_ACTION_DIM
    assert (
        tuple(SONIC_POLICY_JOINT_ORDER[index] for index in SONIC_JOINT_TO_POLICY)
        == SONIC_JOINT_ORDER
    )
    assert (
        tuple(SONIC_JOINT_ORDER[index] for index in SONIC_POLICY_TO_JOINT)
        == SONIC_POLICY_JOINT_ORDER
    )
    assert SONIC_ACTION_SCALE.shape == (SONIC_ACTION_DIM,)
    assert not SONIC_ACTION_SCALE.flags.writeable

    scale_by_joint = sonic_action_scale_by_joint()
    assert scale_by_joint["left_hip_pitch_joint"] == pytest.approx(0.35066146)
    assert scale_by_joint["left_hip_yaw_joint"] == pytest.approx(0.54754645)
    assert scale_by_joint["left_ankle_pitch_joint"] == pytest.approx(0.43857732)
    assert scale_by_joint["left_wrist_pitch_joint"] == pytest.approx(0.07450087)


def test_action_config_rejects_non_release_affine_contract() -> None:
    with pytest.raises(ValueError, match="actuator_names"):
        SonicMotionJointPositionActionCfg(
            entity_name="robot",
            actuator_names=tuple(reversed(SONIC_JOINT_ORDER)),
        )
    with pytest.raises(ValueError, match="use_default_offset"):
        SonicMotionJointPositionActionCfg(entity_name="robot", use_default_offset=False)
    with pytest.raises(ValueError, match="differs from the release"):
        SonicMotionJointPositionActionCfg(
            entity_name="robot",
            scale={name: 1.0 for name in SONIC_JOINT_ORDER},
        )


def test_process_actions_clips_then_reorders_before_manager_affine() -> None:
    # The public method is exercised with minimal inherited buffers.  A full
    # environment is intentionally unnecessary: Entity application belongs to
    # the generic MotionJointPositionAction contract and is separately covered
    # by its owner tests.
    term = object.__new__(SonicMotionJointPositionAction)
    term.cfg = SimpleNamespace(action_clip_value=20.0, simulate_action_latency=False)
    term._raw_actions = np.zeros((1, SONIC_ACTION_DIM), dtype=np.float32)
    term._previous_raw_actions = np.zeros_like(term._raw_actions)
    term._processed_actions = np.zeros_like(term._raw_actions)
    term._scale = np.ones_like(term._raw_actions)
    term._offset = np.zeros_like(term._raw_actions)
    term._clip = None
    term._policy_to_target = np.asarray(SONIC_JOINT_TO_POLICY, dtype=np.intp)
    term._clipped_policy_actions = np.empty_like(term._raw_actions)
    term._target_order_actions = np.empty_like(term._raw_actions)

    policy_actions = np.arange(SONIC_ACTION_DIM, dtype=np.float32)[None, :] - 14.0
    term.process_actions(policy_actions)

    expected = np.clip(policy_actions[:, SONIC_JOINT_TO_POLICY], -20.0, 20.0)
    np.testing.assert_array_equal(term.raw_action, expected)
    np.testing.assert_array_equal(term.processed_action, expected)
    np.testing.assert_array_equal(policy_actions, np.arange(SONIC_ACTION_DIM)[None, :] - 14.0)
