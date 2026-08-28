"""SONIC v1 training termination terms.

The release uses four tracking-failure checks in addition to clip timeout:
adaptive anchor/body height checks, full anchor orientation error, and a
three-dimensional ankle position check.  These terms operate only on the
numeric state already published by :class:`SonicMotionCommand`; body indices
are resolved once when a class term is constructed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from unilab.managers import ManagerTermBase, ManagerTermBaseCfg
from unilab.utils.rotation import np_quat_error_magnitude_squared_batched

from .manager_terms import SonicMotionCommand

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, command_name: str) -> SonicMotionCommand:
    try:
        command = env.command_manager.get_term(command_name)
    except KeyError as exc:
        raise KeyError(f"SONIC motion command term '{command_name}' not found") from exc
    if not isinstance(command, SonicMotionCommand):
        raise TypeError(
            f"SONIC termination requires SonicMotionCommand, got {type(command).__name__}"
        )
    return command


def _threshold(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def sonic_anchor_height_adaptive(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float,
    threshold_adaptive: bool = False,
    down_threshold: float = 0.5,
    root_height_threshold: float = 1.0,
) -> np.ndarray:
    """Match release ``exceeded_anchor_height`` semantics."""

    command = _command(env, command_name)
    threshold_value = _threshold(threshold, name="anchor height threshold")
    down_value = _threshold(down_threshold, name="anchor height down_threshold")
    root_value = _threshold(root_height_threshold, name="anchor height root_height_threshold")
    error = np.abs(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2])
    if threshold_adaptive:
        limit = np.where(
            command.running_ref_root_height < root_value,
            down_value,
            threshold_value,
        )
    else:
        limit = threshold_value
    return error > limit


def sonic_anchor_ori_full(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float,
    asset_cfg: object | None = None,
) -> np.ndarray:
    """Terminate on the squared full quaternion error used by SONIC v1."""

    del asset_cfg
    command = _command(env, command_name)
    threshold_value = _threshold(threshold, name="anchor orientation threshold")
    error_squared = np_quat_error_magnitude_squared_batched(
        command.anchor_quat_w, command.robot_anchor_quat_w
    )
    return error_squared > threshold_value


class sonic_body_height_adaptive(ManagerTermBase):
    """Release ``exceeded_body_height`` with fixed construction-time indices."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        command_name = cfg.params.get("command_name")
        if not isinstance(command_name, str) or not command_name:
            raise ValueError("sonic_body_height_adaptive requires command_name")
        self._command_name = command_name
        command = _command(env, command_name)
        names = tuple(cfg.params.get("body_names") or ())
        if not names:
            raise ValueError("sonic_body_height_adaptive requires body_names")
        missing = [name for name in names if name not in command.cfg.body_names]
        if missing:
            raise ValueError(f"SONIC body height bodies are not tracked: {missing}")
        self._body_ids = np.asarray(
            [command.cfg.body_names.index(name) for name in names], dtype=np.intp
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        threshold: float,
        threshold_adaptive: bool = False,
        down_threshold: float = 0.5,
        root_height_threshold: float = 0.5,
        body_names: object | None = None,
    ) -> np.ndarray:
        del env, body_names
        if command_name != self._command_name:
            raise ValueError(f"term bound to '{self._command_name}', got '{command_name}'")
        command = _command(self._env, command_name)
        threshold_value = _threshold(threshold, name="body height threshold")
        down_value = _threshold(down_threshold, name="body height down_threshold")
        root_value = _threshold(root_height_threshold, name="body height root_height_threshold")
        error = np.abs(
            command.body_pos_relative_w[:, self._body_ids, 2]
            - command.robot_body_pos_w[:, self._body_ids, 2]
        )
        if threshold_adaptive:
            limit = np.where(
                command.running_ref_root_height[:, None] < root_value,
                down_value,
                threshold_value,
            )
        else:
            limit = threshold_value
        return np.any(error > limit, axis=-1)


class sonic_foot_pos_xyz(ManagerTermBase):
    """Terminate when either tracked ankle's full 3-D error exceeds a limit."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        command_name = cfg.params.get("command_name")
        if not isinstance(command_name, str) or not command_name:
            raise ValueError("sonic_foot_pos_xyz requires command_name")
        self._command_name = command_name
        command = _command(env, command_name)
        names = tuple(cfg.params.get("body_names") or ())
        if not names:
            raise ValueError("sonic_foot_pos_xyz requires body_names")
        missing = [name for name in names if name not in command.cfg.body_names]
        if missing:
            raise ValueError(f"SONIC foot bodies are not tracked: {missing}")
        self._body_ids = np.asarray(
            [command.cfg.body_names.index(name) for name in names], dtype=np.intp
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        threshold: float,
        body_names: object | None = None,
    ) -> np.ndarray:
        del env, body_names
        if command_name != self._command_name:
            raise ValueError(f"term bound to '{self._command_name}', got '{command_name}'")
        command = _command(self._env, command_name)
        threshold_value = _threshold(threshold, name="foot position threshold")
        error = command.body_pos_relative_w[:, self._body_ids] - command.robot_body_pos_w[:, self._body_ids]
        return np.any(np.linalg.norm(error, axis=-1) > threshold_value, axis=-1)


__all__ = [
    "sonic_anchor_height_adaptive",
    "sonic_anchor_ori_full",
    "sonic_body_height_adaptive",
    "sonic_foot_pos_xyz",
]
