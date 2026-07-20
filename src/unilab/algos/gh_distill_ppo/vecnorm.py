"""GH VecNorm — decay-EMA observation normalizer (GHDistillPPO, Phase 10.7).

Reimplements ``torchrl.envs.transforms.VecNorm(obs_keys, decay=0.9999)`` — the exact
normalizer GH uses (scripts/utils/helpers.py:14,148) — over the three GH obs groups,
decoupled from torchrl's TensorDict/shared-memory machinery.

**Decay-EMA (NOT Welford running-count).** Per group, running buffers ``sum``, ``ssq``,
``count`` (init 0) are updated on each batch of shape ``(N, dim)``::

    count <- decay*count + N
    sum   <- decay*sum   + x.sum(0)
    ssq   <- decay*ssq   + (x**2).sum(0)
    mean = sum/count ;  var = ssq/count - mean**2 ;  std = sqrt(max(var,0))
    normalized = (x - mean) / clamp_min(std, eps)

Under a constant input the mean converges toward it geometrically at rate ``decay``
(0.9999) — distinct from Welford's ``1/count`` rate. Buffers live in ``state_dict`` for
the GH checkpoint ``vecnorm`` slot. ``eval()`` freezes updates (GH: train appends the
live VecNorm, adapt/finetune append ``to_observation_norm()`` = frozen).

eps=1e-4 std floor (torchrl default). Exact torchrl eps-placement / first-step transient
is an element-wise detail (⏳, needs torchrl for a byte match); the decay-EMA statistics
this reproduces are the parity-relevant part.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _buf_name(group: str, kind: str) -> str:
    return f"_{group}__{kind}"


class VecNorm(nn.Module):
    def __init__(self, shapes: dict[str, int], decay: float = 0.9999, eps: float = 1e-4) -> None:
        super().__init__()
        self.groups = list(shapes.keys())
        self.decay = float(decay)
        self.eps = float(eps)
        for g, d in shapes.items():
            self.register_buffer(_buf_name(g, "sum"), torch.zeros(int(d)))
            self.register_buffer(_buf_name(g, "ssq"), torch.zeros(int(d)))
            self.register_buffer(_buf_name(g, "count"), torch.zeros(()))

    def _stats(self, group: str):
        s = getattr(self, _buf_name(group, "sum"))
        sq = getattr(self, _buf_name(group, "ssq"))
        c = getattr(self, _buf_name(group, "count")).clamp_min(1e-8)
        mean = s / c
        var = (sq / c - mean * mean).clamp_min(0.0)
        std = var.sqrt().clamp_min(self.eps)
        return mean, std

    def update(self, obs: dict[str, torch.Tensor]) -> None:
        """Decay-EMA update over the batch dim (train only — no-op in eval mode)."""
        if not self.training:
            return
        for g in self.groups:
            if g not in obs:
                continue
            x = obs[g].detach()
            x = x.reshape(-1, x.shape[-1])
            n = float(x.shape[0])
            s = getattr(self, _buf_name(g, "sum"))
            sq = getattr(self, _buf_name(g, "ssq"))
            c = getattr(self, _buf_name(g, "count"))
            s.mul_(self.decay).add_(x.sum(dim=0))
            sq.mul_(self.decay).add_((x * x).sum(dim=0))
            c.mul_(self.decay).add_(n)

    def normalize(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {}
        for g, x in obs.items():
            if g in self.groups and getattr(self, _buf_name(g, "count")) > 0:
                mean, std = self._stats(g)
                out[g] = (x - mean) / std
            else:
                out[g] = x
        return out

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Update-if-training then normalize (GH transform semantics)."""
        self.update(obs)
        return self.normalize(obs)
