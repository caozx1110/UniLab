"""IndependentNormal action distribution (GHDistillPPO, Phase 10.4).

Pure port of GH ``IndependentNormal`` (learning/modules/distributions.py:139-153):
a diagonal Normal over the 29 action dims, reinterpreted as one event dim so
``log_prob``/``entropy`` sum over actions, with ``scale`` clamped to 1e-6.
"""
from __future__ import annotations

import torch
import torch.distributions as D


class IndependentNormal(D.Independent):
    """Diagonal Normal with 1 reinterpreted batch dim and ``scale.clamp_min(1e-6)``."""

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, validate_args=None) -> None:
        scale = torch.clamp_min(scale, 1e-6)
        super().__init__(D.Normal(loc, scale), 1, validate_args=validate_args)

    @property
    def scale(self) -> torch.Tensor:
        return self.base_dist.scale

    @property
    def deterministic_sample(self) -> torch.Tensor:
        """Mean action (GH: used for evaluation rollout)."""
        return self.base_dist.mean
