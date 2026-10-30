"""Actor parameter and buffer exponential moving averages."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


class ActorEMA:
    """Maintain evaluation-only actor state at one or more half-lives.

    The actor remains the source used for optimization and collection.  EMA
    snapshots include every actor ``state_dict`` entry, so BatchNorm buffers
    follow the same update schedule as parameters.
    """

    def __init__(self, actor: nn.Module, half_lives: Sequence[float] = ()) -> None:
        values = tuple(float(value) for value in half_lives)
        if len(set(values)) != len(values) or any(
            not math.isfinite(value) or value <= 0 for value in values
        ):
            raise ValueError("Actor EMA half-lives must be unique positive finite values")
        self.half_lives = values
        self.updates = 0
        self._states = [
            {key: value.detach().clone() for key, value in actor.state_dict().items()}
            for _ in self.half_lives
        ]

    @torch.no_grad()
    def update(self, actor: nn.Module) -> None:
        """Apply one actor-update sample to every configured EMA snapshot."""
        current = actor.state_dict()
        if any(target.keys() != current.keys() for target in self._states):
            raise ValueError("Actor EMA state keys changed")
        for half_life, state in zip(self.half_lives, self._states):
            alpha = -math.expm1(-math.log(2.0) / half_life)
            for key, value in current.items():
                target = state[key]
                if target.shape != value.shape or target.dtype != value.dtype:
                    raise ValueError(f"Actor EMA tensor ABI changed: {key}")
                if value.is_floating_point():
                    target.lerp_(value.detach(), alpha)
                else:
                    target.copy_(value)
        self.updates += 1

    def actor_state_dict(self, half_life: float) -> dict[str, torch.Tensor]:
        """Return a detached copy of one EMA actor snapshot."""
        try:
            index = self.half_lives.index(float(half_life))
        except ValueError as error:
            raise ValueError(f"Unknown actor EMA half-life: {half_life}") from error
        return {key: value.detach().clone() for key, value in self._states[index].items()}

    def live_state_dicts(self) -> tuple[dict[str, torch.Tensor], ...]:
        """Return live EMA storage for in-place distributed synchronization."""
        return tuple(self._states)

    def state_dict(self) -> dict[str, Any]:
        return {
            "half_lives": self.half_lives,
            "updates": self.updates,
            "states": [
                {key: value.detach().clone() for key, value in state.items()}
                for state in self._states
            ],
        }

    def load_state_dict(self, saved: dict[str, Any]) -> None:
        saved_half_lives = tuple(float(value) for value in saved.get("half_lives", ()))
        if saved_half_lives != self.half_lives or len(saved.get("states", ())) != len(self._states):
            raise ValueError("Actor EMA half-life mismatch")
        for target, source in zip(self._states, saved["states"]):
            if target.keys() != source.keys():
                raise ValueError("Actor EMA state keys mismatch")
            for key, value in target.items():
                other = source[key]
                if other.shape != value.shape or other.dtype != value.dtype:
                    raise ValueError(f"Actor EMA tensor ABI mismatch: {key}")
                if other.is_floating_point() and not bool(torch.isfinite(other).all()):
                    raise ValueError(f"Nonfinite actor EMA state: {key}")
                value.copy_(other)
        updates = int(saved.get("updates", 0))
        if updates < 0:
            raise ValueError("Invalid actor EMA update count")
        self.updates = updates
