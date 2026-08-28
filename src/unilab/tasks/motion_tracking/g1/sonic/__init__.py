"""Task-owned SONIC manager-based integration seams.

This package deliberately does not register an environment or change the
generic manager observation contract.  Later SONIC owner slices compose these
typed task-local components.
"""

from .manager_terms import (
    SONIC_JOINT_ORDER,
    CompactSonicMotionLoader,
    SonicMotionCommand,
    SonicMotionCommandCfg,
    SonicMotionCommandParamsCfg,
    SonicMotionManifestError,
)
from .observations import (
    SonicManagerObservationAdapter,
    SonicObservationBatch,
    SonicTokenizerObservationCache,
    SonicTokenizerObservationProvider,
)
from .runner import ManagerBasedSonicEnv, SonicManagerPPORunner

__all__ = [
    "CompactSonicMotionLoader",
    "ManagerBasedSonicEnv",
    "SONIC_JOINT_ORDER",
    "SonicManagerObservationAdapter",
    "SonicManagerPPORunner",
    "SonicMotionCommand",
    "SonicMotionCommandCfg",
    "SonicMotionCommandParamsCfg",
    "SonicMotionManifestError",
    "SonicObservationBatch",
    "SonicTokenizerObservationCache",
    "SonicTokenizerObservationProvider",
]
