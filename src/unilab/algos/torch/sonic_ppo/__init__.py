"""Native, simulator-agnostic SONIC PPO building blocks.

The implementation intentionally keeps the environment boundary small: an env
must implement ``reset()`` and ``step(action)`` and may expose observations as
either tensors or ``{"actor": ..., "critic": ..., "tokenizer": ...}``.
This makes the learner useful for CPU contract tests while the production
SonicG1Tracking owner supplies the high-throughput MuJoCo environment.
"""

from .algorithm import SonicPPO
from .model import FSQ, SonicActorCritic, UniversalToken
from .runner import SonicPPORunner, train_sonic
from .storage import SonicRolloutStorage

__all__ = [
    "FSQ",
    "SonicActorCritic",
    "SonicPPO",
    "SonicPPORunner",
    "SonicRolloutStorage",
    "UniversalToken",
    "train_sonic",
]
