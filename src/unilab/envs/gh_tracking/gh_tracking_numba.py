"""Optional numba (prange over env axis) acceleration for GHTrackingEnv.update_state.

Mirrors motion_tracking/g1/motion_tracking_numba.py. Path A: float32 + fastmath;
parity is rtol=1e-4/atol=1e-5, not bit-identical. Task 1 delegates to the numpy
path to prove the wiring; later tasks move reward/obs/termination into a kernel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit, prange, set_num_threads  # noqa: F401
    _NUMBA = True
except ImportError:  # pragma: no cover
    njit = prange = set_num_threads = None  # type: ignore[assignment]
    _NUMBA = False


def is_available() -> bool:
    return _NUMBA


def unsupported_terms(groups: dict) -> frozenset[str]:
    # Task 1 delegates to numpy, so every term is "supported". Later tasks
    # narrow this to terms without a kernel translation.
    return frozenset()


@dataclass(frozen=True)
class GHNumbaResult:
    reward_vec: np.ndarray          # (N, 3) [impedance, tracking, loco]
    obs: dict[str, np.ndarray]      # policy (N,450) / priv (N,717) / priv_critic (N,3)
    terminated: np.ndarray          # (N,) bool


class GHTrackingNumbaAccelerator:
    def __init__(self, num_threads: int | None) -> None:
        self.num_threads = num_threads

    @classmethod
    def from_env(cls, env, num_threads: int | None) -> "GHTrackingNumbaAccelerator":
        if not _NUMBA:
            raise RuntimeError(
                "numba_acceleration=True but numba is not importable; "
                "install numba or set numba_acceleration=False"
            )
        return cls(num_threads=num_threads)

    def compute_update_state(self, env) -> GHNumbaResult:
        # Task 1: delegate to the numpy path (proves the seam). Tasks 2-4 replace
        # the body with the fused kernel.
        if self.num_threads is not None and _NUMBA:
            set_num_threads(self.num_threads)
        reward_vec = env._compute_reward()            # (N,3), writes _cum_error
        obs = env._compute_obs()                      # dict of 3 groups
        from unilab.envs.gh_tracking.terminations import apply_terminate_gate
        terminated = apply_terminate_gate(
            env.termination.terminated(), env._episode_length)[:, 0]
        return GHNumbaResult(reward_vec=reward_vec, obs=obs, terminated=terminated)
