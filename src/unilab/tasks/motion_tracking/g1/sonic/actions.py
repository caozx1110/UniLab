"""Task-owned Manager action contract for the 29-DoF SONIC G1 policy.

The released policy emits actions in IsaacLab's interleaved policy order,
whereas a :class:`~unilab.base.entity.Entity` accepts targets in its declared
G1 joint order.  This module resolves that layout once at construction and
keeps the runtime conversion entirely numeric.  It therefore works through
the ordinary ``ActionManager`` / ``Entity`` contract on both MuJoCo and
mjwarp, without inspecting backend-private state or assets on a hot path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np

from unilab.tasks.motion_tracking.common.manager_terms import (
    MotionJointPositionAction,
    MotionJointPositionActionCfg,
)

from .manager_terms import SONIC_JOINT_ORDER

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


SONIC_ACTION_DIM = 29

# This is the action layout published by the release checkpoint.  It is not
# the entity/backend joint order, even though both refer to the same 29 G1
# actuators.
SONIC_POLICY_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

if len(SONIC_JOINT_ORDER) != SONIC_ACTION_DIM or len(SONIC_POLICY_JOINT_ORDER) != SONIC_ACTION_DIM:
    raise RuntimeError("SONIC action contract must contain exactly 29 joints")
if set(SONIC_JOINT_ORDER) != set(SONIC_POLICY_JOINT_ORDER):
    raise RuntimeError("SONIC policy and entity joint layouts must name the same joints")

# For every column in the canonical Entity/G1 joint order, identify the
# corresponding column in the release policy action.  This is independent of
# the simulation backend because it is a task-level name contract.
SONIC_JOINT_TO_POLICY: tuple[int, ...] = tuple(
    SONIC_POLICY_JOINT_ORDER.index(name) for name in SONIC_JOINT_ORDER
)
SONIC_POLICY_TO_JOINT: tuple[int, ...] = tuple(
    SONIC_JOINT_ORDER.index(name) for name in SONIC_POLICY_JOINT_ORDER
)


def sonic_action_scale() -> np.ndarray:
    """Return fixed release action scales in ``SONIC_POLICY_JOINT_ORDER``.

    IsaacLab's implicit actuators use ``0.25 * effort_limit / stiffness``.
    The release's model-12 motor choices are explicit here rather than inferred
    from the XML, so construction is backend-independent and control steps do
    not inspect assets or model metadata.
    """

    natural_frequency = 10.0 * 2.0 * math.pi
    stiffness_5020 = 0.003609725 * natural_frequency**2
    stiffness_7520_14 = 0.010177520 * natural_frequency**2
    stiffness_7520_22 = 0.025101925 * natural_frequency**2
    stiffness_4010 = 0.00425 * natural_frequency**2
    values: dict[str, float] = {}
    for side in ("left", "right"):
        values[f"{side}_hip_pitch_joint"] = 0.25 * 139.0 / stiffness_7520_22
        values[f"{side}_hip_roll_joint"] = 0.25 * 139.0 / stiffness_7520_22
        values[f"{side}_hip_yaw_joint"] = 0.25 * 88.0 / stiffness_7520_14
        values[f"{side}_knee_joint"] = 0.25 * 139.0 / stiffness_7520_22
        values[f"{side}_ankle_pitch_joint"] = 0.25 * 50.0 / (2.0 * stiffness_5020)
        values[f"{side}_ankle_roll_joint"] = 0.25 * 50.0 / (2.0 * stiffness_5020)
        values[f"{side}_shoulder_pitch_joint"] = 0.25 * 25.0 / stiffness_5020
        values[f"{side}_shoulder_roll_joint"] = 0.25 * 25.0 / stiffness_5020
        values[f"{side}_shoulder_yaw_joint"] = 0.25 * 25.0 / stiffness_5020
        values[f"{side}_elbow_joint"] = 0.25 * 25.0 / stiffness_5020
        values[f"{side}_wrist_roll_joint"] = 0.25 * 25.0 / stiffness_5020
        values[f"{side}_wrist_pitch_joint"] = 0.25 * 5.0 / stiffness_4010
        values[f"{side}_wrist_yaw_joint"] = 0.25 * 5.0 / stiffness_4010
    values["waist_yaw_joint"] = 0.25 * 88.0 / stiffness_7520_14
    values["waist_roll_joint"] = 0.25 * 50.0 / (2.0 * stiffness_5020)
    values["waist_pitch_joint"] = 0.25 * 50.0 / (2.0 * stiffness_5020)
    return np.asarray([values[name] for name in SONIC_POLICY_JOINT_ORDER], dtype=np.float32)


SONIC_ACTION_SCALE = sonic_action_scale()
SONIC_ACTION_SCALE.setflags(write=False)


def sonic_action_scale_by_joint() -> dict[str, float]:
    """Return release scales keyed by canonical Entity/G1 joint name."""

    policy_values = dict(zip(SONIC_POLICY_JOINT_ORDER, SONIC_ACTION_SCALE, strict=True))
    return {name: float(policy_values[name]) for name in SONIC_JOINT_ORDER}


def _validate_real(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {number}")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be positive, got {number}")
    return number


def _validate_scale(value: float | dict[str, float]) -> None:
    if not isinstance(value, dict):
        raise TypeError("SonicMotionJointPositionActionCfg scale must be a per-joint dict")
    expected = sonic_action_scale_by_joint()
    if set(value) != set(expected):
        missing = sorted(set(expected).difference(value))
        extra = sorted(set(value).difference(expected))
        raise ValueError(
            "SonicMotionJointPositionActionCfg scale joint names must exactly match "
            f"SONIC_JOINT_ORDER; missing={missing}, extra={extra}"
        )
    for name, expected_value in expected.items():
        actual = _validate_real(value[name], name=f"SONIC action scale for {name}", positive=True)
        if not math.isclose(actual, expected_value, rel_tol=1.0e-6, abs_tol=1.0e-8):
            raise ValueError(
                f"SONIC action scale for {name}={actual} differs from the release value "
                f"{expected_value}"
            )


@dataclass(kw_only=True)
class SonicMotionJointPositionActionCfg(MotionJointPositionActionCfg):
    """Exact release-policy action owner for a Manager-Based SONIC task."""

    actuator_names: tuple[str, ...] | list[str] = SONIC_JOINT_ORDER
    scale: float | dict[str, float] = field(default_factory=sonic_action_scale_by_joint)
    action_clip_value: float = 20.0

    def __post_init__(self) -> None:
        if tuple(self.actuator_names) != SONIC_JOINT_ORDER:
            raise ValueError(
                "SonicMotionJointPositionActionCfg actuator_names must exactly match "
                "the canonical SONIC_JOINT_ORDER"
            )
        if self.preserve_order:
            raise ValueError(
                "SonicMotionJointPositionActionCfg does not use generic preserve_order; "
                "the task term owns the explicit policy-to-entity permutation"
            )
        if not self.use_default_offset:
            raise ValueError(
                "SonicMotionJointPositionActionCfg requires use_default_offset=true "
                "for release default joint targets"
            )
        if (
            isinstance(self.offset, bool)
            or not isinstance(self.offset, Real)
            or float(self.offset) != 0.0
        ):
            raise ValueError(
                "SonicMotionJointPositionActionCfg offset must remain the scalar 0.0; "
                "default joint targets are supplied by use_default_offset"
            )
        if self.clip is not None:
            raise ValueError(
                "SonicMotionJointPositionActionCfg uses action_clip_value before scaling; "
                "generic processed-action clip must be unset"
            )
        if not isinstance(self.simulate_action_latency, bool):
            raise TypeError(
                "SonicMotionJointPositionActionCfg simulate_action_latency must be bool"
            )
        self.action_clip_value = _validate_real(
            self.action_clip_value,
            name="SonicMotionJointPositionActionCfg action_clip_value",
            positive=True,
        )
        _validate_scale(self.scale)

    def build(self, env: ManagerBasedRlEnv) -> SonicMotionJointPositionAction:
        return SonicMotionJointPositionAction(self, env)


class SonicMotionJointPositionAction(MotionJointPositionAction):
    """Clip, reorder, and scale release actions through the Manager contract."""

    cfg: SonicMotionJointPositionActionCfg  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, cfg: SonicMotionJointPositionActionCfg, env: ManagerBasedRlEnv):
        # Dataclass construction validates normal Hydra use; keep the boundary
        # fail-closed for direct/programmatic configs too.
        cfg.__post_init__()
        super().__init__(cfg, env)
        if self.action_dim != SONIC_ACTION_DIM:
            raise ValueError(
                f"SONIC requires {SONIC_ACTION_DIM} action targets, got {self.action_dim}"
            )
        if tuple(self.target_names) != SONIC_JOINT_ORDER:
            raise ValueError(
                "SONIC entity action targets must exactly match SONIC_JOINT_ORDER; "
                f"got {tuple(self.target_names)}"
            )
        self._policy_to_target = np.asarray(SONIC_JOINT_TO_POLICY, dtype=np.intp)
        self._policy_to_target.setflags(write=False)
        self._clipped_policy_actions = np.empty_like(self._raw_actions)
        self._target_order_actions = np.empty_like(self._raw_actions)

    @property
    def policy_to_target(self) -> np.ndarray:
        """Immutable policy-column indices for the canonical entity target order."""

        return self._policy_to_target

    def process_actions(self, actions: np.ndarray) -> None:
        expected_shape = self._clipped_policy_actions.shape
        if not isinstance(actions, np.ndarray):
            raise TypeError(
                f"{type(self).__name__} expected np.ndarray, got {type(actions).__name__}"
            )
        if actions.shape != expected_shape:
            raise ValueError(
                f"{type(self).__name__} expected action shape {expected_shape}, got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError(f"{type(self).__name__} received NaN or Inf actions")

        np.clip(
            actions,
            -self.cfg.action_clip_value,
            self.cfg.action_clip_value,
            out=self._clipped_policy_actions,
        )
        np.take(
            self._clipped_policy_actions,
            self._policy_to_target,
            axis=1,
            out=self._target_order_actions,
        )
        super().process_actions(self._target_order_actions)


__all__ = [
    "SONIC_ACTION_DIM",
    "SONIC_ACTION_SCALE",
    "SONIC_JOINT_TO_POLICY",
    "SONIC_POLICY_JOINT_ORDER",
    "SONIC_POLICY_TO_JOINT",
    "SonicMotionJointPositionAction",
    "SonicMotionJointPositionActionCfg",
    "sonic_action_scale",
    "sonic_action_scale_by_joint",
]
