"""Shared base for A2 quadruped + arm locomotion tasks.

A2Arm-specific config (joint set, EE/foot sensors, PD gains) plus a thin
``LocomotionBaseEnv`` subclass that adds foot-sensor readers. Concrete tasks
(e.g. the pos-force task in ``pos_force.py``) subclass ``A2ArmBaseEnv`` and
implement their own control / observation / reward logic.

This base is intentionally lean: the pos-force task drives the legs+arm through
its own Python PD law and does not use inverse kinematics, so no IK solver lives
here. (The IK-driven ``go2_arm`` manip-loco base remains the reference for
arm tasks that need Cartesian IK.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.envs.locomotion.common.base import (
    ControlConfigBase,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
    Sensor,
)

# Quadruped leg actuator count (FL, FR, RL, RR x hip, thigh, calf).
NUM_LEG = 12


@dataclass
class ControlConfig(ControlConfigBase):
    """Base PD-gain defaults for the A2 quadruped + arm.

    ``leg_*`` / ``arm_*`` accept a scalar, a full per-actuator array, or a short
    per-joint-group spec that is tiled across the limb (see
    ``build_a2arm_position_gains`` / ``_expand_gain``). Concrete tasks override
    these with their own control config; this is only the generic base default.
    """

    Kp: float = 35.0
    Kd: float = 0.5
    leg_kp: float | list[float] | None = 60.0
    leg_kd: float | list[float] | None = 2.0
    arm_kp: float | list[float] | None = field(
        default_factory=lambda: [95.0, 115.0, 100.0, 52.0, 54.0, 55.0]
    )
    arm_kd: float | list[float] | None = field(
        default_factory=lambda: [3.5, 3.8, 2.5, 1.5, 1.5, 1.5]
    )
    # Arm residual action scale, independent from the leg action_scale.
    arm_action_scale: float = 0.0


@dataclass
class Asset:
    base_name: str = "base"
    foot_name: str = "foot"
    ground: str = "floor"
    ee_site_name: str = "endpoint"
    ee_body_name: str = "link6"
    arm_joint_names: tuple[str, ...] = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    )


@dataclass
class A2ArmSensor(Sensor):
    feet_force: list[str] = field(
        default_factory=lambda: [
            "FL_foot_contact",
            "FR_foot_contact",
            "RL_foot_contact",
            "RR_foot_contact",
        ]
    )
    feet_pos: list[str] = field(default_factory=lambda: ["FL_pos", "FR_pos", "RL_pos", "RR_pos"])
    # Foot contact-force vectors (3-dim each), foot world velocities, and thigh
    # world positions — used by the pos-force feet rewards.
    feet_force_vec: list[str] = field(
        default_factory=lambda: ["FL_foot_force", "FR_foot_force", "RL_foot_force", "RR_foot_force"]
    )
    feet_vel: list[str] = field(
        default_factory=lambda: [
            "FL_global_linvel",
            "FR_global_linvel",
            "RL_global_linvel",
            "RR_global_linvel",
        ]
    )
    thigh_pos: list[str] = field(
        default_factory=lambda: ["FL_thigh_pos", "FR_thigh_pos", "RL_thigh_pos", "RR_thigh_pos"]
    )
    # Penalised-contact sensors for the collision reward (thigh + calf + base).
    undesired_contact: list[str] = field(
        default_factory=lambda: [
            "FL_thigh_contact",
            "FR_thigh_contact",
            "RL_thigh_contact",
            "RR_thigh_contact",
            "FL_calf_contact1",
            "FR_calf_contact1",
            "RL_calf_contact1",
            "RR_calf_contact1",
            "FL_calf_contact2",
            "FR_calf_contact2",
            "RL_calf_contact2",
            "RR_calf_contact2",
            "base1_contact",
            "base2_contact",
            "base3_contact",
        ]
    )
    ee_local_pos: str = "endpoint_pos"
    ee_local_quat: str = "endpoint_quat"
    ee_local_vel: str = "endpoint_vel"
    ee_relative_pos: str = "endpoint_relative_pos"
    ee_relative_quat: str = "endpoint_relative_quat"
    arm_ref_world_quat: str = "armbasepoint_world_quat"


@dataclass
class A2ArmBaseCfg(LocomotionBaseCfg):
    control_config: ControlConfig = field(default_factory=ControlConfig)  # type: ignore[assignment]
    asset: Asset = field(default_factory=Asset)
    sensor: A2ArmSensor = field(default_factory=A2ArmSensor)  # type: ignore[assignment]
    iterations: int | None = None
    sim_dt: float = 0.01
    ctrl_dt: float = 0.02


def _expand_gain(
    name: str, value: float | list[float] | None, fallback: float, size: int
) -> np.ndarray:
    raw_value = fallback if value is None else value
    gain = np.asarray(raw_value, dtype=np.float64)
    if gain.ndim == 0:
        return np.full((size,), float(gain), dtype=np.float64)
    if gain.shape == (size,):
        return gain
    # Per-joint group spec (e.g. length-3 hip/thigh/calf) tiled across the limb,
    # so a quadruped leg can take [hip, thigh, calf] and repeat it per leg (A2).
    if gain.ndim == 1 and gain.shape[0] != 0 and size % gain.shape[0] == 0:
        return np.tile(gain, size // gain.shape[0])
    raise ValueError(
        f"{name} must be a scalar, shape ({size},), or a length dividing {size}; got {gain.shape}"
    )


def build_a2arm_position_gains(cfg: ControlConfig, num_arm: int) -> dict[str, np.ndarray]:
    leg_kp = _expand_gain("control_config.leg_kp", cfg.leg_kp, cfg.Kp, NUM_LEG)
    leg_kd = _expand_gain("control_config.leg_kd", cfg.leg_kd, cfg.Kd, NUM_LEG)
    arm_kp = _expand_gain("control_config.arm_kp", cfg.arm_kp, cfg.Kp, num_arm)
    arm_kd = _expand_gain("control_config.arm_kd", cfg.arm_kd, cfg.Kd, num_arm)
    return {
        "kp": np.concatenate([leg_kp, arm_kp]),
        "kd": np.concatenate([leg_kd, arm_kd]),
    }


class A2ArmBaseEnv(LocomotionBaseEnv):
    """Common base env for A2 quadruped + arm tasks.

    Adds foot-sensor readers on top of ``LocomotionBaseEnv``. All arm-specific
    state (joint set, EE sensors, PD gains) is carried by the config dataclasses
    above; concrete tasks implement their own control/observation logic.
    """

    _cfg: A2ArmBaseCfg

    def get_foot_pos(self) -> np.ndarray:
        """Get foot positions. Returns shape (num_envs, 4, 3)."""
        foot_pos = [self._backend.get_sensor_data(name) for name in self._cfg.sensor.feet_pos]
        return np.stack(foot_pos, axis=1)

    def get_foot_contact(self) -> np.ndarray:
        """Get foot contact values. Returns shape (num_envs, 4)."""
        contacts = [
            self._backend.get_sensor_data(name)[:, 0] for name in self._cfg.sensor.feet_force
        ]
        return np.stack(contacts, axis=1)
