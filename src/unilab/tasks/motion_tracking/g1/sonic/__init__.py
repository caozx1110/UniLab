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
from .observation_terms import (
    sonic_base_ang_vel,
    sonic_base_lin_vel,
    sonic_future_command,
    sonic_joint_pos_rel,
    sonic_joint_vel_rel,
    sonic_last_action,
    sonic_projected_gravity,
    sonic_tokenizer_observation,
)
from .observations import (
    SonicManagerObservationAdapter,
    SonicObservationBatch,
    SonicTokenizerObservationCache,
    SonicTokenizerObservationProvider,
)
from .runner import ManagerBasedSonicEnv, SonicManagerPPORunner
from .termination_terms import (
    sonic_anchor_height_adaptive,
    sonic_anchor_ori_full,
    sonic_body_height_adaptive,
    sonic_foot_pos_xyz,
)

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
    "sonic_base_ang_vel",
    "sonic_base_lin_vel",
    "sonic_future_command",
    "sonic_joint_pos_rel",
    "sonic_joint_vel_rel",
    "sonic_last_action",
    "sonic_projected_gravity",
    "sonic_tokenizer_observation",
    "sonic_anchor_height_adaptive",
    "sonic_anchor_ori_full",
    "sonic_body_height_adaptive",
    "sonic_foot_pos_xyz",
    "sonic_action_scale",
    "sonic_action_scale_by_joint",
]
