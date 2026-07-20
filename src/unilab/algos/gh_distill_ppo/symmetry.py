"""Symmetry involution transform for GHDistillPPO.

GH symmetry.py:9-41 — involution property x[perm*signs][perm*signs]=x.
"""
import torch
import torch.nn as nn


class SymmetryTransform(nn.Module):
    """GH symmetry.py:9-24 — involution transform x[perm]*signs."""
    def __init__(self, perm, signs):
        super().__init__()
        if not len(perm) == len(signs) > 0:
            raise ValueError("perm and signs must have same length and be non-empty")
        self.register_buffer("perm", torch.as_tensor(perm, dtype=torch.long))
        self.register_buffer("signs", torch.as_tensor(signs, dtype=torch.float32))

    def forward(self, x: torch.Tensor, sign=True):
        if sign:
            return x[..., self.perm] * self.signs
        else:
            return x[..., self.perm]

    @staticmethod
    def cat(transforms):
        """GH symmetry.py:29-40."""
        if not all(isinstance(t, SymmetryTransform) for t in transforms):
            raise ValueError("All transforms must be SymmetryTransform instances")
        perm = []
        signs = []
        offset = 0
        for t in transforms:
            perm.append(t.perm + offset)
            signs.append(t.signs)
            offset += t.perm.shape[0]
        return SymmetryTransform(torch.cat(perm), torch.cat(signs))
