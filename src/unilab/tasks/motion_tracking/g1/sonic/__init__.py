"""Task-owned SONIC manager-based integration seams.

This package deliberately does not register an environment or change the
generic manager observation contract.  Later SONIC owner slices compose these
typed task-local components.
"""

from .observations import (
    SonicManagerObservationAdapter,
    SonicObservationBatch,
    SonicTokenizerObservationCache,
    SonicTokenizerObservationProvider,
)
from .runner import ManagerBasedSonicEnv, SonicManagerPPORunner

__all__ = [
    "SonicManagerObservationAdapter",
    "ManagerBasedSonicEnv",
    "SonicManagerPPORunner",
    "SonicObservationBatch",
    "SonicTokenizerObservationCache",
    "SonicTokenizerObservationProvider",
]
