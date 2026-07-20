"""TemporalLerp: time-varying tensors with start/end/duration (Phase 5).

Pure-numpy port of GH ``commands/utils.py::TemporalLerp``. Manages discrete-time
ramps over arbitrary leading-shape tensors (e.g. per-env force stiffness / spring
origin), used by the force schedule.
"""

from __future__ import annotations

import numpy as np


class TemporalLerp:
    def __init__(
        self,
        shape,
        default: float = 0.0,
        easing: str = "linear",  # "linear" | "smoothstep"
        clamp: tuple[float, float] | None = None,
    ) -> None:
        self.value = np.full(shape, default, dtype=np.float64)
        self._start = self.value.copy()
        self._end = self.value.copy()
        tshape = self.value.shape[:1] + (1,) * (self.value.ndim - 1)
        self._t = np.zeros(tshape, dtype=np.int64)
        self._T = np.ones(tshape, dtype=np.int64)  # avoid /0
        self._active = np.zeros(tshape, dtype=bool)
        self.easing = easing
        self.clamp = clamp

    def set(self, env_ids, end=None, delta=None, total_steps=0, start=None) -> None:
        if (end is None) == (delta is None):
            raise ValueError("provide exactly one of 'end' or 'delta'")
        index = slice(None) if env_ids is None else np.asarray(env_ids)
        cur_start = self.value[index] if start is None else np.asarray(start, dtype=np.float64)
        self._start[index] = cur_start
        self._end[index] = cur_start + np.asarray(delta, dtype=np.float64) if end is None \
            else np.asarray(end, dtype=np.float64)
        self._t[index] = 0
        if np.isscalar(total_steps):
            self._T[index] = int(total_steps)
        else:
            ts = np.asarray(total_steps).reshape((-1,) + (1,) * (self.value.ndim - 1))
            self._T[index] = ts
        self._active[index] = True
        self.value[index] = cur_start

    def update_time(self, steps: int = 1) -> None:
        if not self._active.any():
            return
        self._t[self._active] += int(steps)
        self._update_value()
        done = self._t >= self._T
        self._active = self._active & (~done)

    def reset(self, env_ids=None, value=None) -> None:
        index = slice(None) if env_ids is None else np.asarray(env_ids)
        if value is not None:
            self.value[index] = value
        self._start[index] = self.value[index]
        self._end[index] = self.value[index]
        self._t[index] = 0
        self._T[index] = 1
        self._active[index] = False

    @property
    def current(self) -> np.ndarray:
        return self.value

    @property
    def mask_active(self) -> np.ndarray:
        return self._active.reshape(-1)

    @property
    def mask_done(self) -> np.ndarray:
        return (~self._active).reshape(-1)

    def _ease(self, a: np.ndarray) -> np.ndarray:
        if self.easing == "linear":
            return a
        if self.easing == "smoothstep":
            return a * a * (3 - 2 * a)
        raise ValueError(f"unknown easing {self.easing!r}")

    def _update_value(self) -> None:
        frac = np.clip(self._t / self._T, 0.0, 1.0)
        self.value = self._ease(frac) * (self._end - self._start) + self._start
        if self.clamp is not None:
            self.value = np.clip(self.value, self.clamp[0], self.clamp[1])
