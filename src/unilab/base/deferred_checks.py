"""Opt-in device verdict accumulation for bounded diagnostic synchronization.

The caller owns iteration boundaries. Checks accumulate even on non-publish
iterations; a transient failure cannot disappear before publication.
"""
from __future__ import annotations

import torch


class DeferredChecks:
    def __init__(self, device: torch.device | str, interval: int = 1):
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("check interval must be a positive integer")
        self.interval = interval
        self.valid = torch.ones((), dtype=torch.bool, device=device)
        self.steps = 0
        self.pending = False

    def record(self, valid: torch.Tensor) -> None:
        if valid.shape != () or valid.dtype != torch.bool or valid.device != self.valid.device:
            raise ValueError("verdict must be a scalar bool on the checks device")
        self.valid.logical_and_(valid.detach())
        self.pending = True

    def end_step(self) -> None:
        self.steps += 1
        if self.steps % self.interval == 0:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        if not bool(self.valid):
            # Keep failure latched if a caller catches the exception.
            raise ValueError("Deferred device check failed (NaN, Inf or invalid action)")
        self.valid.fill_(True)
        self.pending = False
