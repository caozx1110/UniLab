"""Task-owned SONIC manager-based integration seams.

This package deliberately does not register an environment or change the
generic manager observation contract.  Later SONIC owner slices compose these
typed task-local components.
"""

from .actions import (
    SONIC_ACTION_DIM,
    SONIC_ACTION_SCALE,
    SONIC_JOINT_TO_POLICY,
    SONIC_POLICY_JOINT_ORDER,
    SONIC_POLICY_TO_JOINT,
    SonicMotionJointPositionAction,
    SonicMotionJointPositionActionCfg,
    sonic_action_scale,
    sonic_action_scale_by_joint,
)
from .lazy_motion_loader import BoundedLazySonicMotionLoader, LazySonicMotionData
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
    "BoundedLazySonicMotionLoader",
    "CompactSonicMotionLoader",
    "LazySonicMotionData",
    "ManagerBasedSonicEnv",
    "SONIC_ACTION_DIM",
    "SONIC_ACTION_SCALE",
    "SONIC_JOINT_ORDER",
    "SONIC_JOINT_TO_POLICY",
    "SONIC_POLICY_JOINT_ORDER",
    "SONIC_POLICY_TO_JOINT",
    "SonicManagerObservationAdapter",
    "SonicManagerPPORunner",
    "SonicMotionCommand",
    "SonicMotionCommandCfg",
    "SonicMotionJointPositionAction",
    "SonicMotionJointPositionActionCfg",
    "SonicMotionCommandParamsCfg",
    "SonicMotionManifestError",
    "SonicObservationBatch",
    "SonicTokenizerObservationCache",
    "SonicTokenizerObservationProvider",
    "sonic_action_scale",
    "sonic_action_scale_by_joint",
]
