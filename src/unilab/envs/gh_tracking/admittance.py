"""Admittance mass-chain for the GH force system (Phase 5).

Pure-numpy port of GH ``admittance.py``. A per-force-body virtual mass-spring
integrated with semi-implicit Euler at ``physics_dt`` (0.005), stepped 4x per
control step, which pushes the compliant contact target forward by one control
step. State ``x, v`` have shape ``(H, N, M, 3)`` with ``H=mixed_loop_steps``.
"""

from __future__ import annotations

import numpy as np


def clamp_norm(x: np.ndarray, max_norm, eps: float = 1e-6) -> np.ndarray:
    """Scale vectors along the last axis so their norm does not exceed ``max_norm``.

    ``max_norm`` may be a scalar or an array broadcastable to ``x[..., :1]``.
    """
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    scale = np.minimum(np.asarray(max_norm, dtype=x.dtype) / n, 1.0)
    return x * scale


class AdmittanceMassChain:
    """Virtual mass-spring-damper per force body (GH ``AdmittanceMassChain``)."""

    def __init__(
        self,
        num_envs: int,
        num_points: int,
        dt: float,
        mixed_loop_steps: int = 1,
        mass: float = 1.0,
        damping: float = 40.0,
        vel_clip: float = 15.0,
        acc_clip: float = 500.0,
    ) -> None:
        self.N = int(num_envs)
        self.M = int(num_points)
        self.H = int(mixed_loop_steps)
        self.dt = float(dt)
        self.mass = float(mass)
        self.damping = float(damping)
        self.vel_clip = float(vel_clip)
        self.acc_clip = float(acc_clip)
        self.x = np.zeros((self.H, self.N, self.M, 3), dtype=np.float64)
        self.v = np.zeros((self.H, self.N, self.M, 3), dtype=np.float64)

    def reset(self, env_ids: np.ndarray, x0_b: np.ndarray, v0_b: np.ndarray | None = None) -> None:
        env_ids = np.asarray(env_ids)
        self.x[:, env_ids] = np.asarray(x0_b, dtype=np.float64)[None]
        self.v[:, env_ids] = 0.0 if v0_b is None else np.asarray(v0_b, dtype=np.float64)[None]

    def step(self, F_drive_b: np.ndarray, F_ext_b: np.ndarray) -> None:
        """Semi-implicit Euler: F_damp=-c*v; a=clamp(F_total/m); v+=a*dt (clamped); x+=v*dt."""
        dt = self.dt
        f_damp = -self.damping * self.v
        f_total = F_drive_b + F_ext_b + f_damp
        a = clamp_norm(f_total / self.mass, self.acc_clip)
        self.v = clamp_norm(self.v + a * dt, self.vel_clip)
        self.x = self.x + self.v * dt
