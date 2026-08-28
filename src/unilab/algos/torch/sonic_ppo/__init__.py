"""Simulator-agnostic SONIC v1 PPO/model primitives.

The manager-based adapter owns environment collection and checkpoint lifecycle.
This package intentionally exports only the learner-side model, optimizer,
rollout storage, and cold-path release-checkpoint conversion.
"""

from .algorithm import SonicPPO
from .checkpoint import convert_official_sonic_release_checkpoint
from .model import FSQ, SonicActorCritic, UniversalToken
from .storage import SonicRolloutStorage

__all__ = [
    "FSQ",
    "SonicActorCritic",
    "SonicPPO",
    "SonicRolloutStorage",
    "UniversalToken",
    "convert_official_sonic_release_checkpoint",
]
