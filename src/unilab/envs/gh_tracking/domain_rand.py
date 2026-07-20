"""GH domain randomization for the MuJoCo migration (Phase 7).

Ports the GH DR sampling lifecycle (``randomizations.py``): material/COM are
sampled ONCE at startup and fixed across resets (reuse UniLab's
``apply_init_randomization`` startup-once primitive); motor params (stiffness/
damping/armature) resample every reset; the joint zero offset is a per-joint draw
of shape ``(num_joints,)`` broadcast to the whole reset batch (batch-shared,
source-compatible).

Only the sliding friction coefficient maps to MuJoCo ``geom_friction[..., 0]``;
PhysX dynamic-fraction / restitution have no direct MuJoCo slot and are a
calibrated approximation (handled by the live DR wiring, not here).
"""

from __future__ import annotations

import numpy as np

COM_RANGE = (-0.02, 0.02)
STIFFNESS_RANGE = (0.9, 1.1)
DAMPING_RANGE = (0.9, 1.1)
ARMATURE_RANGE = (0.75, 1.25)
JOINT_OFFSET_RANGE = (-0.01, 0.01)
STATIC_FRICTION_RANGE = (0.3, 1.6)
DYNAMIC_FRICTION_FRAC_RANGE = (0.75, 1.0)
RESTITUTION_RANGE = (0.0, 0.2)


class GHDomainRand:
    """GH DR sampler with a startup-once cache + per-reset resampling."""

    def __init__(self, num_envs: int, num_joints: int, num_com_bodies: int, seed: int = 0) -> None:
        self.n = int(num_envs)
        self.num_joints = int(num_joints)
        self.num_com_bodies = int(num_com_bodies)
        self._rng = np.random.default_rng(seed)
        self._com: np.ndarray | None = None
        self._material: dict[str, np.ndarray] | None = None

    def startup_com(self) -> np.ndarray:
        """Per-env per-body COM offset, sampled once and cached (fixed across resets)."""
        if self._com is None:
            self._com = self._rng.uniform(*COM_RANGE, size=(self.n, self.num_com_bodies, 3))
        return self._com

    def startup_material(self) -> dict[str, np.ndarray]:
        """Per-env friction (static, dynamic fraction, restitution), sampled once and cached."""
        if self._material is None:
            static = self._rng.uniform(*STATIC_FRICTION_RANGE, size=(self.n, 1))
            frac = self._rng.uniform(*DYNAMIC_FRICTION_FRAC_RANGE, size=(self.n, 1))
            restitution = self._rng.uniform(*RESTITUTION_RANGE, size=(self.n, 1))
            self._material = {"static": static, "dynamic_frac": frac, "restitution": restitution}
        return self._material

    def sample_motor(self, env_ids: np.ndarray) -> dict[str, np.ndarray]:
        """Per-env per-joint motor-param scales, resampled every reset."""
        k = len(np.asarray(env_ids))
        return {
            "stiffness_scale": self._rng.uniform(*STIFFNESS_RANGE, size=(k, self.num_joints)),
            "damping_scale": self._rng.uniform(*DAMPING_RANGE, size=(k, self.num_joints)),
            "armature_scale": self._rng.uniform(*ARMATURE_RANGE, size=(k, self.num_joints)),
        }

    def sample_joint_offset(self, env_ids: np.ndarray) -> np.ndarray:
        """Batch-shared per-joint zero offset (GH random_joint_offset): a single
        ``(num_joints,)`` draw broadcast to every env in the reset batch."""
        k = len(np.asarray(env_ids))
        per_joint = self._rng.uniform(*JOINT_OFFSET_RANGE, size=(self.num_joints,))
        return np.broadcast_to(per_joint, (k, self.num_joints)).copy()


def material_to_geom_friction_sliding(static_friction: np.ndarray) -> np.ndarray:
    """Map the sampled static friction to MuJoCo ``geom_friction[..., 0]`` (sliding).

    Only the sliding coefficient is representable; PhysX dynamic-fraction and
    restitution are a calibrated approximation with no direct MuJoCo slot and are
    NOT written to geom_friction.
    """
    return np.asarray(static_friction, dtype=np.float64)
