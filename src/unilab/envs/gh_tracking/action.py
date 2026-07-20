"""GH action pipeline for the MuJoCo migration (Phase 4).

Pure-numpy port of GH ``JointPosition`` (action.py): 29-dim residual action ->
joint position targets, with a per-substep communication delay, alpha lerp
smoothing, and boot protection. Runs inside ``set_pre_step_control`` so targets
are recomputed every physics substep.

The env drives it once per control step (``start_control_step``) then the backend
calls the ``as_pre_step_control`` fn ``decimation`` times (``substep_target``).
"""

from __future__ import annotations

import re

import numpy as np


def resolve_action_scaling(
    joint_names: list[str], patterns: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve regex ``pattern -> scale`` to a per-joint scale vector.

    Mirrors GH ``resolve_matching_names_values``: each joint (in order) must match
    exactly one pattern; returns (joint_ids, scale) where ``scale`` is ordered by
    joint index.
    """
    ids: list[int] = []
    scale: list[float] = []
    for j, name in enumerate(joint_names):
        matched = [v for pat, v in patterns.items() if re.fullmatch(pat, name)]
        if len(matched) != 1:
            raise ValueError(
                f"joint '{name}' matched {len(matched)} action_scaling patterns "
                f"(expected exactly 1)"
            )
        ids.append(j)
        scale.append(float(matched[0]))
    return np.asarray(ids, dtype=np.int64), np.asarray(scale, dtype=np.float64)


class GHActionPipeline:
    """29-dim residual -> position target pipeline with delay/lerp/boot protect."""

    def __init__(
        self,
        joint_names: list[str],
        action_scaling: dict[str, float],
        default_joint_pos: np.ndarray,
        num_envs: int,
        decimation: int = 4,
        max_delay: int = 4,
        alpha_range: tuple[float, float] = (0.9, 0.9),
        alpha_wide: tuple[float, float] = (0.8, 1.0),
        alpha_jit_scale: float | None = 0.025,
        boot_protect: bool = True,
    ) -> None:
        self.joint_ids, self.action_scaling = resolve_action_scaling(joint_names, action_scaling)
        self.action_dim = len(self.joint_ids)  # 29
        self.num_envs = int(num_envs)
        self.decimation = int(decimation)
        self.max_delay = int(max_delay)
        self.alpha_range = alpha_range
        self.alpha_wide = alpha_wide
        self.alpha_jit_scale = alpha_jit_scale
        self.boot_protect = boot_protect

        # hist = max((max_delay - 1) // decimation + 1, 3)  (=3 for 4,4)
        self.hist = max((self.max_delay - 1) // self.decimation + 1, 3)

        self.default_joint_pos = np.broadcast_to(
            np.asarray(default_joint_pos, dtype=np.float64), (self.num_envs, self.action_dim)
        ).copy()
        self.offset = np.zeros((self.num_envs, self.action_dim), dtype=np.float64)

        self.action_buf = np.zeros((self.num_envs, self.hist, self.action_dim), dtype=np.float64)
        self.applied_action = np.zeros((self.num_envs, self.action_dim), dtype=np.float64)
        self.alpha = np.ones((self.num_envs, 1), dtype=np.float64)
        self.delay = np.zeros((self.num_envs, 1), dtype=np.int64)
        self.boot_delay = np.zeros((self.num_envs, 1), dtype=np.int64)
        self.boot_protect_pose = np.zeros((self.num_envs, self.action_dim), dtype=np.float64)
        self._substep = 0

    def reset(self, env_ids: np.ndarray, rng: np.random.Generator) -> None:
        env_ids = np.asarray(env_ids)
        n = len(env_ids)
        self.action_buf[env_ids] = 0.0
        self.applied_action[env_ids] = 0.0
        delay = rng.integers(0, self.max_delay + 1, size=(n, 1))
        self.delay[env_ids] = delay
        if self.boot_protect:
            self.boot_delay[env_ids] = delay
        self.alpha[env_ids] = rng.uniform(self.alpha_range[0], self.alpha_range[1], size=(n, 1))

    def set_boot_protect_pose(self, env_ids: np.ndarray, pose: np.ndarray) -> None:
        """Cache the boot-protect joint pose (the NOISED init pose, D2) per env."""
        self.boot_protect_pose[np.asarray(env_ids)] = np.asarray(pose, dtype=np.float64)

    def start_control_step(self, raw_action: np.ndarray, rng: np.random.Generator) -> None:
        """Ingest a new control-step action (GH ``__call__`` substep==0 branch)."""
        raw = np.clip(np.asarray(raw_action, dtype=np.float64), -10.0, 10.0)
        if self.alpha_jit_scale is not None:
            jit = rng.uniform(-self.alpha_jit_scale, self.alpha_jit_scale, size=(self.num_envs, 1))
            self.alpha = np.clip(self.alpha + jit, self.alpha_wide[0], self.alpha_wide[1])
        self.action_buf = np.roll(self.action_buf, shift=1, axis=1)
        self.action_buf[:, 0, :] = raw
        self._substep = 0

    def prev_actions(self) -> np.ndarray:
        """Raw action history for observations: ``action_buf[:, :3]`` (newest first)."""
        return self.action_buf[:, :3].copy()

    def substep_target(self, substep: int) -> np.ndarray:
        """Compute the joint position target for one physics substep."""
        # communication delay: which history slot this substep reads
        idx = (self.delay - substep + self.decimation - 1) // self.decimation
        idx = np.clip(idx, 0, self.hist - 1)  # (num_envs, 1)
        delayed = np.take_along_axis(self.action_buf, idx[:, :, None], axis=1)[:, 0, :]
        # alpha lerp smoothing (in place, persists across substeps)
        self.applied_action += self.alpha * (delayed - self.applied_action)
        # residual -> joint target
        pos_tgt = self.default_joint_pos + self.offset
        pos_tgt[:, self.joint_ids] += self.applied_action * self.action_scaling
        # boot protection: hold the cached noised init pose verbatim while counting down
        if self.boot_protect:
            hold = self.boot_delay > 0  # (num_envs, 1)
            pos_tgt = np.where(hold, self.boot_protect_pose, pos_tgt)
            self.boot_delay = np.maximum(self.boot_delay - 1, 0)
        self._substep = substep + 1
        return pos_tgt

    def as_pre_step_control(self):
        """Return a ``set_pre_step_control`` fn feeding per-substep joint targets."""

        def fn(backend, ctrl):
            tgt = self.substep_target(self._substep)
            return tgt.astype(ctrl.dtype)

        return fn
