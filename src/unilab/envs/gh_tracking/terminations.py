"""GH termination for the MuJoCo migration (Phase 7).

Pure-numpy port of GH ``terminations.py::cum_error`` plus the ``base.py`` step
gating (first-5-step mask, clip-finished / timeout truncation, random reset
episode length). The counter increments while any ``_cum_error`` component exceeds
the threshold and resets to 0 otherwise; termination needs a STRICT ``count >
min_steps`` (i.e. 51 consecutive for min_steps=50).
"""

from __future__ import annotations

import numpy as np


class CumErrorTermination:
    def __init__(self, num_envs: int, thres: float = 1.0, min_steps: int = 50) -> None:
        self.thres = float(thres)
        self.min_steps = int(min_steps)
        self.error_exceeded_count = np.zeros((int(num_envs), 1), dtype=np.int32)

    def update(self, cum_error: np.ndarray) -> None:
        exceeded = (np.asarray(cum_error, dtype=np.float64) > self.thres).any(axis=-1, keepdims=True)
        self.error_exceeded_count[exceeded] += 1
        self.error_exceeded_count[~exceeded] = 0

    def terminated(self) -> np.ndarray:
        return self.error_exceeded_count > self.min_steps  # strict -> 51 consecutive

    def reset(self, env_ids: np.ndarray) -> None:
        self.error_exceeded_count[np.asarray(env_ids)] = 0


def apply_terminate_gate(terminated: np.ndarray, episode_length: np.ndarray) -> np.ndarray:
    """Mask terminations during the first 5 steps (GH ``terminated & (episode_length_buf > 5)``)."""
    gate = (np.asarray(episode_length).reshape(-1, 1) > 5)
    return np.asarray(terminated) & gate


def compute_truncation(
    episode_length: np.ndarray, max_episode_length: int, finished: np.ndarray
) -> np.ndarray:
    """Truncation = timeout (episode_length >= max) OR clip finished (GH base.py:488-490)."""
    timeout = np.asarray(episode_length).reshape(-1, 1) >= max_episode_length
    return timeout | np.asarray(finished).reshape(-1, 1)


def sample_reset_episode_length(
    n_reset: int, n_total: int, max_episode_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Random reset episode-length init (GH base.py:352-355): if fewer than 20% of
    envs reset, draw from [0, max//5); otherwise from [0, max)."""
    if n_reset < 0.2 * n_total:
        return rng.integers(0, max_episode_length // 5, size=n_reset)
    return rng.integers(0, max_episode_length, size=n_reset)
