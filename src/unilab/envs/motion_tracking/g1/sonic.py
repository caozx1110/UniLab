"""SONIC release owner for the UniLab G1 MuJoCo environment.

The regular :class:`G1MotionTrackingEnv` intentionally keeps its historical
flat actor/critic contract.  SONIC needs a separate owner because its policy
history, tokenizer modalities and future-reference windows are part of the
network contract.  This module reuses the shared tracking/reward engine and
only owns the SONIC observation and action-order boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import gymnasium as gym
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.g1.base import NoiseConfig
from unilab.envs.motion_tracking.common.config import MotionTrackingCfg
from unilab.envs.motion_tracking.common.tracking import MotionTrackingEnv
from unilab.utils.geometry import np_write_relative_anchor_transform_pos_rot6d
from unilab.utils.rotation import (
    np_matrix_first_two_cols_from_quat,
    np_quat_apply_batched,
    np_quat_apply_inverse,
    np_quat_conjugate_batched,
    np_quat_mul_batched,
)

from ..common.motion_loader import MotionSampler

if TYPE_CHECKING:
    from unilab.training.sonic_store import SonicMotionStore


SONIC_ACTOR_OBS_DIM = 930
SONIC_CRITIC_OBS_DIM = 1645
SONIC_TOKENIZER_OBS_DIM = 1761
SONIC_RELEASE_REVISION = "c374bae5b9039cd0ee71377e654d11ce1bc69e1d"


@dataclass(frozen=True)
class SonicObservationTerm:
    """One immutable term in a flattened SONIC observation group.

    The order below is the resolved release order, not an alphabetical view of
    the term names.  ``shape`` excludes the environment batch dimension and
    ``flat_slice`` makes the policy ABI directly auditable.
    """

    name: str
    shape: tuple[int, ...]
    start: int
    stop: int

    @property
    def width(self) -> int:
        return self.stop - self.start

    @property
    def flat_slice(self) -> slice:
        return slice(self.start, self.stop)


def _sonic_layout(
    entries: Sequence[tuple[str, tuple[int, ...]]],
) -> tuple[SonicObservationTerm, ...]:
    offset = 0
    result: list[SonicObservationTerm] = []
    for name, shape in entries:
        width = int(np.prod(shape, dtype=np.int64))
        result.append(SonicObservationTerm(name, shape, offset, offset + width))
        offset += width
    return tuple(result)


# Provenance: GR00T-WholeBodyControl@SONIC_RELEASE_REVISION and IsaacLab
# v2.3.2 ObservationManager._prepare_terms.  The manager iterates the
# configclass ``__dict__``, so class declaration order wins over Hydra's
# visually listed defaults order.
#   gear_sonic/config/manager_env/observations/{policy/local_dir_hist,
#   critic/privileged_mf_hist,tokenizer/unitoken_all_noz}.yaml and
#   gear_sonic/envs/manager_env/mdp/observations.py
# The 1761-wide tokenizer is the *training* observation group.  The release
# deploy encoder's 1751 input is a different export ABI: one scalar mode plus
# the active 1750-wide union (the two unused command-z terms are omitted).
SONIC_ACTOR_OBSERVATION_TERMS = _sonic_layout(
    (
        ("base_ang_vel", (10, 3)),
        ("joint_pos", (10, 29)),
        ("joint_vel", (10, 29)),
        ("actions", (10, 29)),
        ("gravity_dir", (10, 3)),
    )
)
SONIC_CRITIC_OBSERVATION_TERMS = _sonic_layout(
    (
        ("command_multi_future", (580,)),
        ("motion_anchor_pos_b", (3,)),
        ("motion_anchor_ori_b", (6,)),
        ("body_pos", (14, 3)),
        ("body_ori", (14, 6)),
        ("base_lin_vel", (10, 3)),
        ("base_ang_vel", (10, 3)),
        ("joint_pos", (10, 29)),
        ("joint_vel", (10, 29)),
        ("actions", (10, 29)),
    )
)
SONIC_TOKENIZER_OBSERVATION_TERMS = _sonic_layout(
    (
        ("encoder_index", (3,)),
        ("command_multi_future_nonflat", (10, 58)),
        ("command_z_multi_future_nonflat", (10, 1)),
        ("command_z", (1,)),
        ("motion_anchor_ori_b", (6,)),
        ("motion_anchor_ori_b_mf_nonflat", (10, 6)),
        ("command_multi_future_lower_body", (240,)),
        ("vr_3point_local_target", (9,)),
        ("vr_3point_local_orn_target", (12,)),
        ("smpl_joints_multi_future_local_nonflat", (10, 72)),
        ("smpl_root_ori_b_multi_future", (10, 6)),
        ("joint_pos_multi_future_wrist_for_smpl", (10, 6)),
    )
)


def pack_sonic_observation_terms(
    terms: Mapping[str, np.ndarray], layout: Sequence[SonicObservationTerm]
) -> np.ndarray:
    """Validate and flatten named terms into their immutable release slices."""

    if not layout:
        raise ValueError("SONIC observation layout must not be empty")
    expected_names = tuple(term.name for term in layout)
    if set(terms) != set(expected_names):
        raise ValueError(
            "SONIC observation terms disagree with layout: "
            f"missing={sorted(set(expected_names) - set(terms))}, "
            f"extra={sorted(set(terms) - set(expected_names))}"
        )
    batch_size: int | None = None
    result: np.ndarray | None = None
    for term in layout:
        value = np.asarray(terms[term.name])
        if value.ndim < 1 or tuple(value.shape[1:]) != term.shape:
            raise ValueError(
                f"SONIC term {term.name!r} must have shape (N, {term.shape}), got {value.shape}"
            )
        if batch_size is None:
            batch_size = int(value.shape[0])
            result = np.empty((batch_size, layout[-1].stop), dtype=np.float32)
        elif value.shape[0] != batch_size:
            raise ValueError(f"SONIC term {term.name!r} has a different batch size")
        assert result is not None
        result[:, term.flat_slice] = value.reshape(value.shape[0], -1)
    assert result is not None
    return result


assert SONIC_ACTOR_OBSERVATION_TERMS[-1].stop == SONIC_ACTOR_OBS_DIM
assert SONIC_CRITIC_OBSERVATION_TERMS[-1].stop == SONIC_CRITIC_OBS_DIM
assert SONIC_TOKENIZER_OBSERVATION_TERMS[-1].stop == SONIC_TOKENIZER_OBS_DIM

# Materialized motion and MuJoCo actuator order.  The release policy itself
# uses the interleaved IsaacLab order declared separately below.
SONIC_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

SONIC_POLICY_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)
SONIC_MUJOCO_TO_POLICY: tuple[int, ...] = tuple(
    SONIC_JOINT_ORDER.index(name) for name in SONIC_POLICY_JOINT_ORDER
)
SONIC_POLICY_TO_MUJOCO: tuple[int, ...] = tuple(
    SONIC_POLICY_JOINT_ORDER.index(name) for name in SONIC_JOINT_ORDER
)
SONIC_LOWER_BODY_POLICY_INDICES: tuple[int, ...] = tuple(
    SONIC_POLICY_JOINT_ORDER.index(name) for name in SONIC_JOINT_ORDER[:12]
)

SONIC_BODY_ORDER: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

# Upstream selects these directly from its IsaacLab-policy-order future q.
SONIC_WRIST_JOINT_INDICES: tuple[int, ...] = (23, 24, 25, 26, 27, 28)

SONIC_VR_BODY_ORDER: tuple[str, ...] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "torso_link",
)
SONIC_VR_BODY_OFFSETS: tuple[tuple[float, float, float], ...] = (
    (0.18, -0.025, 0.0),
    (0.18, 0.025, 0.0),
    (0.0, 0.0, 0.35),
)


def sonic_action_scale() -> np.ndarray:
    """Return the release model-12 scale in IsaacLab policy order.

    Isaac's implicit actuator uses ``0.25 * effort_limit / stiffness``.  The
    UniLab G1 XML exposes the same position gains, so keeping this calculation
    explicit avoids silently falling back to the historical scalar ``0.25``.
    """

    natural_frequency = 10.0 * 2.0 * np.pi
    stiffness_5020 = 0.003609725 * natural_frequency**2
    stiffness_7520_14 = 0.010177520 * natural_frequency**2
    stiffness_7520_22 = 0.025101925 * natural_frequency**2
    stiffness_4010 = 0.00425 * natural_frequency**2
    values: dict[str, float] = {}
    for name in ("left", "right"):
        # G1 model-12 uses the 7520-22 actuator (139 Nm) for hip pitch.
        # Hip yaw is the 7520-14 / 88 Nm actuator below.  Mixing these two
        # entries changes the policy-to-torque contract by ~56% on hip pitch.
        values[f"{name}_hip_pitch_joint"] = 0.25 * 139.0 / stiffness_7520_22
        values[f"{name}_hip_roll_joint"] = 0.25 * 139.0 / stiffness_7520_22
        values[f"{name}_hip_yaw_joint"] = 0.25 * 88.0 / stiffness_7520_14
        values[f"{name}_knee_joint"] = 0.25 * 139.0 / stiffness_7520_22
        values[f"{name}_ankle_pitch_joint"] = 0.25 * 50.0 / (2.0 * stiffness_5020)
        values[f"{name}_ankle_roll_joint"] = 0.25 * 50.0 / (2.0 * stiffness_5020)
        values[f"{name}_shoulder_pitch_joint"] = 0.25 * 25.0 / stiffness_5020
        values[f"{name}_shoulder_roll_joint"] = 0.25 * 25.0 / stiffness_5020
        values[f"{name}_shoulder_yaw_joint"] = 0.25 * 25.0 / stiffness_5020
        values[f"{name}_elbow_joint"] = 0.25 * 25.0 / stiffness_5020
        values[f"{name}_wrist_roll_joint"] = 0.25 * 25.0 / stiffness_5020
        values[f"{name}_wrist_pitch_joint"] = 0.25 * 5.0 / stiffness_4010
        values[f"{name}_wrist_yaw_joint"] = 0.25 * 5.0 / stiffness_4010
    values["waist_yaw_joint"] = 0.25 * 88.0 / stiffness_7520_14
    values["waist_roll_joint"] = 0.25 * 50.0 / (2.0 * stiffness_5020)
    values["waist_pitch_joint"] = 0.25 * 50.0 / (2.0 * stiffness_5020)
    return np.asarray([values[name] for name in SONIC_POLICY_JOINT_ORDER], dtype=np.float32)


SONIC_ACTION_SCALE = sonic_action_scale()


@dataclass
class SonicG1TrackingCfg(MotionTrackingCfg):
    """Configuration that owns the SONIC 29-DoF observation contract."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_sonic.xml")
        )
    )
    motion_manifest: str | None = None
    motion_rank: int = 0
    motion_world_size: int = 1
    motion_shard_clips: bool = True
    motion_cache_size: int = 2
    # SONIC's local-dir history uses active uniform sensor corruption with
    # release scales; the generic G1 profile defaults to level=0.
    noise_config: NoiseConfig = field(
        default_factory=lambda: NoiseConfig(
            level=1.0,
            scale_gravity=0.05,
            scale_gyro=0.2,
            scale_joint_angle=0.01,
            scale_joint_vel=0.5,
        )
    )
    # Upstream ``sonic_release`` uses the pelvis as the motion/robot anchor.
    # The generic UniLab tracking profile historically anchors at ``torso_link``;
    # leaving that inherited default here silently changes every local-frame
    # observation, reward, and termination while preserving all tensor shapes.
    anchor_body_name: str = "pelvis"
    body_names: tuple[str, ...] = SONIC_BODY_ORDER
    ee_body_names: tuple[str, ...] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    history_length: int = 10
    num_future_frames: int = 10
    dt_future_ref_frames: float = 0.1
    smpl_num_future_frames: int = 10
    smpl_dt_future_ref_frames: float = 0.02
    smpl_y_up: bool = True
    encoder_names: tuple[str, ...] = ("g1", "teleop", "smpl")
    encoder_sample_probs: tuple[float, ...] = (1.0, 1.0, 1.0)
    teleop_sample_prob_when_smpl: float = 0.5
    tokenizer_enable_corruption: bool = True
    vr_body_names: tuple[str, ...] = SONIC_VR_BODY_ORDER
    vr_body_offsets: tuple[tuple[float, float, float], ...] = SONIC_VR_BODY_OFFSETS
    use_release_action_scale: bool = True
    # SONIC clips are materialized in the configured 14-body order. The
    # shared tracking engine otherwise assumes MuJoCo body-id indexing.
    motion_data_body_indices: tuple[int, ...] = tuple(range(len(SONIC_BODY_ORDER)))

    def __post_init__(self) -> None:
        self.sim_dt = 0.005
        self.ctrl_dt = 0.02
        self.sampling_mode = "adaptive"
        self.sensor.gyro = "pelvis_gyro"
        if self.use_release_action_scale:
            self.control_config.action_scale = SONIC_ACTION_SCALE.copy()


@registry.envcfg("SonicG1Tracking")
@dataclass
class SonicG1TrackingEnvCfg(SonicG1TrackingCfg):
    """Registered SONIC G1 configuration."""


class SonicG1TrackingEnv(MotionTrackingEnv):
    """G1 tracking env exposing actor, critic and tokenizer groups."""

    _cfg: SonicG1TrackingCfg

    def __init__(self, cfg: SonicG1TrackingCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        if (
            cfg.history_length != 10
            or cfg.num_future_frames != 10
            or cfg.smpl_num_future_frames != 10
        ):
            raise ValueError(
                "SONIC release observations require 10 history, future, and SMPL-future frames"
            )
        if tuple(cfg.encoder_names) != ("g1", "teleop", "smpl"):
            raise ValueError("SONIC release encoder_names must be ('g1', 'teleop', 'smpl')")
        self._sonic_reset_ids: np.ndarray | None = None
        self._sonic_store = self._resolve_store(cfg)
        # Snapshot immutable motion metadata on the cold path.  Observation
        # construction must only touch frame arrays; it must not repeatedly
        # traverse manifest/asset metadata in ``step``.
        self._sonic_fps = (
            max(1, int(round(self._sonic_store.manifest.clips[0].fps)))
            if self._sonic_store is not None
            else 50
        )
        self._sonic_num_bodies = (
            self._sonic_store.num_bodies if self._sonic_store is not None else len(cfg.body_names)
        )
        self._sonic_has_smpl = bool(
            self._sonic_store is not None
            and {"smpl_joints", "smpl_root_quat_w"}.issubset(self._sonic_store.arrays)
        )
        self._future_offsets = self._future_frame_offsets(
            cfg.num_future_frames, cfg.dt_future_ref_frames, self._sonic_fps
        )
        self._smpl_future_offsets = self._future_frame_offsets(
            cfg.smpl_num_future_frames, cfg.smpl_dt_future_ref_frames, self._sonic_fps
        )
        if self._sonic_store is not None:
            from unilab.training.sonic_store import SonicMotionLoader

            # Inject before the shared tracking owner constructs its loader so
            # both paths retain the bounded, rank-sharded lazy store contract.
            cfg.motion_loader = SonicMotionLoader(self._sonic_store)
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        if self._num_action != len(SONIC_JOINT_ORDER):
            raise ValueError(f"SONIC requires 29 actuators, backend exposes {self._num_action}")
        self._backend_to_policy = self._resolve_actuator_permutation()
        self._policy_to_backend = np.argsort(self._backend_to_policy)
        self._policy_default_angles = self.default_angles[self._backend_to_policy]
        self._policy_joint_range = (
            self._joint_range[self._backend_to_policy] if self._joint_range is not None else None
        )
        if len(cfg.body_names) != 14:
            raise ValueError(
                f"SONIC release observations require 14 bodies, got {len(cfg.body_names)}"
            )
        if len(cfg.vr_body_names) != 3:
            raise ValueError("SONIC release observations require three VR bodies")
        try:
            self._vr_body_rows = np.asarray(
                [tuple(cfg.body_names).index(name) for name in cfg.vr_body_names], dtype=np.int32
            )
        except ValueError as exc:
            raise ValueError("SONIC VR body names must be present in body_names") from exc
        self._vr_body_offsets = np.asarray(cfg.vr_body_offsets, dtype=np.float32)
        if self._vr_body_offsets.shape != (3, 3):
            raise ValueError("SONIC vr_body_offsets must have shape (3, 3)")
        self._history = np.zeros((num_envs, self._cfg.history_length, 93), dtype=np.float32)
        self._critic_history = np.zeros_like(self._history)
        self._encoder_index = np.zeros((num_envs, len(self._cfg.encoder_names)), dtype=np.float32)
        self._sample_encoder_indices(np.arange(num_envs, dtype=np.int32))
        self._actor_obs_width = SONIC_ACTOR_OBS_DIM
        self._critic_obs_width = SONIC_CRITIC_OBS_DIM

    @staticmethod
    def _resolve_store(cfg: SonicG1TrackingCfg) -> SonicMotionStore | None:
        if not cfg.motion_manifest:
            return None
        from unilab.training.sonic_store import load_sonic_motion_store

        return load_sonic_motion_store(
            cfg.motion_manifest,
            verify_checksums=True,
            verify_shapes=True,
            expected_joint_order=SONIC_JOINT_ORDER,
            expected_body_order=cfg.body_names,
            rank=cfg.motion_rank,
            world_size=cfg.motion_world_size,
            shard_clips=cfg.motion_shard_clips,
            cache_size=cfg.motion_cache_size,
        )

    @staticmethod
    def _future_frame_offsets(count: int, spacing: float, fps: int) -> np.ndarray:
        steps = float(spacing) * int(fps)
        rounded_steps = int(round(steps))
        if rounded_steps < 1 or not np.isclose(steps, rounded_steps, atol=1.0e-9):
            raise ValueError(
                f"SONIC future spacing={spacing} at fps={fps} must be a positive integer step"
            )
        return np.arange(count, dtype=np.int64) * rounded_steps

    def _resolve_actuator_permutation(self) -> np.ndarray:
        names = tuple(self._backend.get_actuator_names())
        normalized = tuple(name.removesuffix("_dof") for name in names)
        if set(normalized) != set(SONIC_JOINT_ORDER) or len(normalized) != len(SONIC_JOINT_ORDER):
            raise ValueError(
                "SONIC actuator names do not match the 29-DoF release order: "
                f"expected={SONIC_JOINT_ORDER}, actual={normalized}"
            )
        return np.asarray(
            [normalized.index(name) for name in SONIC_POLICY_JOINT_ORDER], dtype=np.int32
        )

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {
            "actor_obs": SONIC_ACTOR_OBS_DIM,
            "critic_obs": SONIC_CRITIC_OBS_DIM,
            "tokenizer": SONIC_TOKENIZER_OBS_DIM,
        }

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {
                name: gym.spaces.Box(-np.inf, np.inf, shape=(width,), dtype=np.float32)
                for name, width in self.obs_groups_spec.items()
            }
        )

    def _actor_obs_dim(self, n: int) -> int:
        del n
        return SONIC_ACTOR_OBS_DIM

    def _critic_base_obs_dim(self, n: int) -> int:
        del n
        return SONIC_CRITIC_OBS_DIM - len(self._cfg.body_names) * 9

    def _sample_encoder_indices(self, env_ids: np.ndarray) -> None:
        if not len(env_ids):
            return
        probabilities = np.asarray(self._cfg.encoder_sample_probs, dtype=np.float64)
        if probabilities.shape != (len(self._cfg.encoder_names),) or np.any(probabilities < 0):
            raise ValueError("encoder_sample_probs must match encoder_names and be non-negative")
        if probabilities.sum() <= 0:
            raise ValueError("encoder_sample_probs must contain a positive mass")
        teleop_probability = float(self._cfg.teleop_sample_prob_when_smpl)
        if not 0.0 <= teleop_probability <= 1.0:
            raise ValueError("teleop_sample_prob_when_smpl must be in [0, 1]")
        if not self._sonic_has_smpl:
            probabilities[2] = 0.0
            if probabilities.sum() <= 0:
                raise ValueError("encoder_sample_probs need g1 or teleop mass without SMPL data")
        probabilities /= probabilities.sum()
        choices = np.random.choice(len(probabilities), size=len(env_ids), p=probabilities)
        self._encoder_index[env_ids] = 0.0
        self._encoder_index[env_ids, choices] = 1.0
        # Release training uses a multi-hot mask for SMPL-native samples: the
        # paired G1 encoder is always active and teleop is additionally active
        # with the configured probability for latent-alignment losses.
        smpl_ids = env_ids[choices == 2]
        if len(smpl_ids):
            self._encoder_index[smpl_ids, 0] = 1.0
            use_teleop = np.random.random(len(smpl_ids)) < teleop_probability
            self._encoder_index[smpl_ids[use_teleop], 1] = 1.0

    def reset(self, env_indices: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], dict]:
        if env_indices is None:
            if self._state is None:
                state = self.init_state()
                return state.obs, state.info
            env_indices = np.arange(self._num_envs, dtype=np.int32)
        env_indices = np.asarray(env_indices, dtype=np.int32)
        self._sonic_reset_ids = env_indices
        self._sample_encoder_indices(env_indices)
        try:
            return super().reset(env_indices)
        finally:
            self._sonic_reset_ids = None

    def apply_action(self, actions: np.ndarray, state: Any) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != len(SONIC_JOINT_ORDER):
            raise ValueError(f"SONIC actions must have shape (N, 29), got {actions.shape}")
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions
        delayed = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else actions
        )
        # Scale/defaults share the IsaacLab policy ABI.  Only the completed
        # target is mapped back to the backend/MuJoCo actuator order.
        target_policy = (
            delayed * np.asarray(self._cfg.control_config.action_scale)
            + self._policy_default_angles
        )
        target_backend = target_policy[:, self._policy_to_backend]
        bias = state.info.get("default_dof_pos_bias")
        if isinstance(bias, np.ndarray):
            target_backend = target_backend + bias
        return target_backend

    def _policy_joint_values(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)[:, self._backend_to_policy]

    def _policy_defaults_for_obs(self, info: Mapping[str, Any], env_ids: np.ndarray) -> np.ndarray:
        defaults = np.broadcast_to(
            np.asarray(self._policy_default_angles, dtype=np.float32),
            (len(env_ids), len(SONIC_JOINT_ORDER)),
        ).copy()
        bias = info.get("default_dof_pos_bias")
        if not isinstance(bias, np.ndarray):
            state = getattr(self, "_state", None)
            state_info = getattr(state, "info", None)
            if isinstance(state_info, Mapping):
                bias = state_info.get("default_dof_pos_bias")
        if bias is None:
            return defaults
        bias = np.asarray(bias, dtype=np.float32)
        if bias.ndim != 2 or bias.shape[1] != len(SONIC_JOINT_ORDER):
            raise ValueError(f"default_dof_pos_bias must have shape (N, 29), got {bias.shape}")
        if bias.shape[0] == self._num_envs:
            bias = bias[env_ids]
        elif bias.shape[0] != len(env_ids):
            raise ValueError(
                "default_dof_pos_bias batch does not match observation rows: "
                f"{bias.shape[0]} versus {len(env_ids)}"
            )
        defaults += bias[:, self._backend_to_policy]
        return defaults

    def _actions_for_obs(self, info: Mapping[str, Any], env_ids: np.ndarray) -> np.ndarray:
        actions = info.get("current_actions")
        if not isinstance(actions, np.ndarray):
            state = getattr(self, "_state", None)
            state_info = getattr(state, "info", None)
            if isinstance(state_info, Mapping):
                actions = state_info.get("current_actions")
        if not isinstance(actions, np.ndarray):
            return np.zeros((len(env_ids), len(SONIC_JOINT_ORDER)), dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != len(SONIC_JOINT_ORDER):
            raise ValueError(f"current_actions must have shape (N, 29), got {actions.shape}")
        if actions.shape[0] == self._num_envs:
            actions = actions[env_ids]
        elif actions.shape[0] != len(env_ids):
            raise ValueError(
                "current_actions batch does not match observation rows: "
                f"{actions.shape[0]} versus {len(env_ids)}"
            )
        return actions

    def _tokenizer_corruption(self, data: np.ndarray, scale: float) -> np.ndarray:
        if not self._cfg.tokenizer_enable_corruption:
            return data
        seed = self._configured_obs_noise_seed()
        rng = np.random if seed is None else getattr(self, "_obs_noise_rng", None)
        if rng is None:
            rng = np.random.default_rng(seed)
            self._obs_noise_rng = rng
        return data + rng.uniform(-scale, scale, data.shape).astype(data.dtype)

    def _future_reference(self, frame_indices: np.ndarray) -> dict[str, np.ndarray]:
        if self._sonic_store is None:
            return self._zero_future_reference(len(frame_indices))
        indices = self._sonic_store.future_indices(frame_indices, self._future_offsets)
        flat = indices.reshape(-1)
        future_fields = self._sonic_store.gather_fields(
            ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w"), flat
        )
        result = {
            "joint_pos": np.take(
                future_fields["joint_pos"].reshape(len(frame_indices), -1, 29),
                SONIC_MUJOCO_TO_POLICY,
                axis=-1,
            ),
            "joint_vel": np.take(
                future_fields["joint_vel"].reshape(len(frame_indices), -1, 29),
                SONIC_MUJOCO_TO_POLICY,
                axis=-1,
            ),
            "body_pos": future_fields["body_pos_w"].reshape(
                len(frame_indices), -1, self._sonic_num_bodies, 3
            )[:, :, : len(self._cfg.body_names)],
            "body_quat": future_fields["body_quat_w"].reshape(
                len(frame_indices), -1, self._sonic_num_bodies, 4
            )[:, :, : len(self._cfg.body_names)],
        }
        smpl_indices = self._sonic_store.future_indices(frame_indices, self._smpl_future_offsets)
        smpl_joint_fields = self._sonic_store.gather_fields(
            ("joint_pos",), smpl_indices.reshape(-1)
        )
        result["smpl_joint_pos"] = np.take(
            smpl_joint_fields["joint_pos"].reshape(len(frame_indices), -1, 29),
            SONIC_MUJOCO_TO_POLICY,
            axis=-1,
        )
        if not self._sonic_has_smpl:
            result.update(self._zero_smpl_reference(len(frame_indices)))
            return result
        smpl_fields = self._sonic_store.gather_fields(
            ("smpl_joints", "smpl_root_quat_w"), smpl_indices.reshape(-1)
        )
        result.update(
            {
                "smpl_joints": smpl_fields["smpl_joints"].reshape(len(frame_indices), -1, 24, 3),
                "smpl_root_quat": smpl_fields["smpl_root_quat_w"].reshape(
                    len(frame_indices), -1, 4
                ),
            }
        )
        return result

    def _zero_smpl_reference(self, num_envs: int) -> dict[str, np.ndarray]:
        smpl_frames = self._cfg.smpl_num_future_frames
        return {
            "smpl_joints": np.zeros((num_envs, smpl_frames, 24, 3), dtype=np.float32),
            "smpl_root_quat": np.broadcast_to(
                np.asarray([1, 0, 0, 0], dtype=np.float32),
                (num_envs, smpl_frames, 4),
            ).copy(),
        }

    def _zero_future_reference(self, num_envs: int) -> dict[str, np.ndarray]:
        frames = self._cfg.num_future_frames
        result = {
            "joint_pos": np.zeros((num_envs, frames, 29), dtype=np.float32),
            "joint_vel": np.zeros((num_envs, frames, 29), dtype=np.float32),
            "body_pos": np.zeros(
                (num_envs, frames, len(self._cfg.body_names), 3), dtype=np.float32
            ),
            "body_quat": np.broadcast_to(
                np.asarray([1, 0, 0, 0], dtype=np.float32),
                (num_envs, frames, len(self._cfg.body_names), 4),
            ).copy(),
            "smpl_joint_pos": np.zeros(
                (num_envs, self._cfg.smpl_num_future_frames, 29), dtype=np.float32
            ),
        }
        result.update(self._zero_smpl_reference(num_envs))
        return result

    def _build_history(
        self,
        env_ids: np.ndarray,
        linvel: np.ndarray,
        gyro: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        robot_body_quat_w: np.ndarray,
        last_actions: np.ndarray,
        *,
        policy_default_angles: np.ndarray,
        advance_history: bool,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        dof_pos = self._policy_joint_values(dof_pos)
        dof_vel = self._policy_joint_values(dof_vel)
        anchor_quat = robot_body_quat_w[:, self.anchor_body_idx]
        gravity = np.broadcast_to(np.asarray([0.0, 0.0, -1.0], dtype=np.float32), (len(env_ids), 3))
        gravity = np.asarray(np_quat_apply_inverse(anchor_quat, gravity), dtype=np.float32)
        joint_pos = dof_pos - policy_default_angles
        noise_cfg = self._cfg.noise_config
        actor_gyro = gyro
        actor_joint_pos = joint_pos
        actor_joint_vel = dof_vel
        actor_gravity = gravity
        if noise_cfg.level > 0:
            actor_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
            actor_joint_pos = self._obs_noise(joint_pos, noise_cfg.scale_joint_angle)
            actor_joint_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
            actor_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        actor_current = np.concatenate(
            [actor_gyro, actor_joint_pos, actor_joint_vel, last_actions, actor_gravity], axis=1
        )
        critic_current = np.concatenate([linvel, gyro, joint_pos, dof_vel, last_actions], axis=1)
        reset_ids = self._sonic_reset_ids
        if reset_ids is not None or not advance_history:
            self._history[env_ids] = actor_current[:, None, :]
            self._critic_history[env_ids] = critic_current[:, None, :]
        else:
            self._history[env_ids, :-1] = self._history[env_ids, 1:]
            self._history[env_ids, -1] = actor_current
            self._critic_history[env_ids, :-1] = self._critic_history[env_ids, 1:]
            self._critic_history[env_ids, -1] = critic_current
        actor_history = self._history[env_ids]
        critic_history = self._critic_history[env_ids]
        return (
            {
                "base_ang_vel": actor_history[:, :, 0:3],
                "joint_pos": actor_history[:, :, 3:32],
                "joint_vel": actor_history[:, :, 32:61],
                "actions": actor_history[:, :, 61:90],
                "gravity_dir": actor_history[:, :, 90:93],
            },
            {
                "base_lin_vel": critic_history[:, :, 0:3],
                "base_ang_vel": critic_history[:, :, 3:6],
                "joint_pos": critic_history[:, :, 6:35],
                "joint_vel": critic_history[:, :, 35:64],
                "actions": critic_history[:, :, 64:93],
            },
        )

    def _compute_obs(
        self,
        info: dict,
        motion_data: Any,
        linvel: np.ndarray,
        gyro: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
    ) -> dict[str, np.ndarray]:
        del motion_data
        env_ids = np.asarray(info.get("env_ids", np.arange(linvel.shape[0])), dtype=np.int32)
        if env_ids.shape[0] != linvel.shape[0]:
            env_ids = np.arange(linvel.shape[0], dtype=np.int32)
        policy_default_angles = self._policy_defaults_for_obs(info, env_ids)
        last_actions = self._actions_for_obs(info, env_ids)
        is_clip_refresh = "env_ids" in info and self._sonic_reset_ids is None
        if is_clip_refresh:
            self._sample_encoder_indices(env_ids)
        actor_terms, critic_history_terms = self._build_history(
            env_ids,
            linvel,
            gyro,
            dof_pos,
            dof_vel,
            robot_body_quat_w,
            last_actions,
            policy_default_angles=policy_default_angles,
            advance_history=not is_clip_refresh,
        )

        frame_indices = self.motion_sampler.current_frames[env_ids]
        future = self._future_reference(frame_indices)
        robot_anchor_pos = robot_body_pos_w[:, self.anchor_body_idx]
        robot_anchor_quat = robot_body_quat_w[:, self.anchor_body_idx]
        ref_body_pos = future["body_pos"][:, 0]
        ref_body_quat = future["body_quat"][:, 0]
        anchor_pos = ref_body_pos[:, self.anchor_body_idx]
        anchor_quat = ref_body_quat[:, self.anchor_body_idx]
        anchor_pos_b = np.empty((len(env_ids), 3), dtype=np.float32)
        anchor_ori_b = np.empty((len(env_ids), 6), dtype=np.float32)
        np_write_relative_anchor_transform_pos_rot6d(
            robot_anchor_pos,
            robot_anchor_quat,
            anchor_pos,
            anchor_quat,
            anchor_pos_b,
            anchor_ori_b,
        )

        body_pos_b = np_quat_apply_batched(
            np_quat_conjugate_batched(robot_anchor_quat)[:, None, :],
            robot_body_pos_w - robot_anchor_pos[:, None, :],
        ).astype(np.float32)
        body_ori_b = np_matrix_first_two_cols_from_quat(
            np_quat_mul_batched(
                np.broadcast_to(
                    np_quat_conjugate_batched(robot_anchor_quat)[:, None, :],
                    robot_body_quat_w.shape,
                ),
                robot_body_quat_w,
            )
        ).astype(np.float32)

        ref_anchor_quat = future["body_quat"][:, :, self.anchor_body_idx]
        robot_anchor_quat_future = robot_anchor_quat[:, None, :]
        relative_future_quat = np_quat_mul_batched(
            np.broadcast_to(
                np_quat_conjugate_batched(robot_anchor_quat_future), ref_anchor_quat.shape
            ),
            ref_anchor_quat,
        )
        future_ori = np_matrix_first_two_cols_from_quat(relative_future_quat).reshape(
            len(env_ids), -1
        )
        command = np.concatenate(
            [
                future["joint_pos"].reshape(len(env_ids), -1),
                future["joint_vel"].reshape(len(env_ids), -1),
            ],
            axis=1,
        )
        command_z_multi = future["body_pos"][:, :, self.anchor_body_idx, 2:3]
        lower = np.concatenate(
            [
                future["joint_pos"][:, :, SONIC_LOWER_BODY_POLICY_INDICES].reshape(
                    len(env_ids), -1
                ),
                future["joint_vel"][:, :, SONIC_LOWER_BODY_POLICY_INDICES].reshape(
                    len(env_ids), -1
                ),
            ],
            axis=1,
        )
        vr_rows = self._vr_body_rows
        vr_pos_w = future["body_pos"][:, 0, vr_rows] + np_quat_apply_batched(
            future["body_quat"][:, 0, vr_rows], self._vr_body_offsets[None, :, :]
        )
        vr_pos = vr_pos_w - anchor_pos[:, None, :]
        vr_pos = np_quat_apply_batched(
            np_quat_conjugate_batched(anchor_quat)[:, None, :], vr_pos
        ).reshape(len(env_ids), -1)
        vr_quat = future["body_quat"][:, 0, vr_rows]
        vr_quat = np_quat_mul_batched(
            np.broadcast_to(np_quat_conjugate_batched(anchor_quat)[:, None, :], vr_quat.shape),
            vr_quat,
        ).reshape(len(env_ids), -1)
        smpl_joints = future["smpl_joints"]
        smpl_root = future["smpl_root_quat"]
        smpl_local = np_quat_apply_batched(
            np_quat_conjugate_batched(smpl_root)[..., None, :], smpl_joints
        )
        smpl_ori = np_matrix_first_two_cols_from_quat(
            np_quat_mul_batched(
                np.broadcast_to(
                    np_quat_conjugate_batched(robot_anchor_quat)[:, None, :], smpl_root.shape
                ),
                smpl_root,
            )
        )
        # TokenizerCfg.enable_corruption is independent of actor noise level.
        anchor_ori_token = self._tokenizer_corruption(anchor_ori_b, 0.05)
        future_ori_token = self._tokenizer_corruption(future_ori, 0.05)
        smpl_local = self._tokenizer_corruption(smpl_local, 0.05)
        smpl_ori = self._tokenizer_corruption(smpl_ori, 0.05)
        wrist_q = future["smpl_joint_pos"][:, :, SONIC_WRIST_JOINT_INDICES]
        actor_obs = pack_sonic_observation_terms(actor_terms, SONIC_ACTOR_OBSERVATION_TERMS)
        critic_obs = pack_sonic_observation_terms(
            {
                "command_multi_future": command,
                "motion_anchor_pos_b": anchor_pos_b,
                "motion_anchor_ori_b": anchor_ori_b,
                "body_pos": body_pos_b,
                "body_ori": body_ori_b,
                **critic_history_terms,
            },
            SONIC_CRITIC_OBSERVATION_TERMS,
        )
        tokenizer = pack_sonic_observation_terms(
            {
                "encoder_index": self._encoder_index[env_ids],
                "command_multi_future_nonflat": command.reshape(len(env_ids), 10, 58),
                "command_z_multi_future_nonflat": command_z_multi,
                "command_z": command_z_multi[:, 0],
                "motion_anchor_ori_b": anchor_ori_token,
                "motion_anchor_ori_b_mf_nonflat": future_ori_token.reshape(len(env_ids), 10, 6),
                "command_multi_future_lower_body": lower,
                "vr_3point_local_target": vr_pos,
                "vr_3point_local_orn_target": vr_quat,
                "smpl_joints_multi_future_local_nonflat": smpl_local.reshape(len(env_ids), 10, 72),
                "smpl_root_ori_b_multi_future": smpl_ori,
                "joint_pos_multi_future_wrist_for_smpl": wrist_q,
            },
            SONIC_TOKENIZER_OBSERVATION_TERMS,
        )
        return {
            "actor_obs": actor_obs,
            "critic_obs": critic_obs,
            "tokenizer": tokenizer,
        }


@registry.env("SonicG1Tracking", sim_backend="mujoco")
class RegisteredSonicG1TrackingEnv(SonicG1TrackingEnv):
    """Registry binding kept separate so the implementation remains testable."""


__all__ = [
    "SONIC_ACTION_SCALE",
    "SONIC_BODY_ORDER",
    "SONIC_JOINT_ORDER",
    "SONIC_LOWER_BODY_POLICY_INDICES",
    "SONIC_MUJOCO_TO_POLICY",
    "SONIC_POLICY_JOINT_ORDER",
    "SONIC_POLICY_TO_MUJOCO",
    "SONIC_RELEASE_REVISION",
    "SONIC_TOKENIZER_OBSERVATION_TERMS",
    "SONIC_WRIST_JOINT_INDICES",
    "SonicObservationTerm",
    "SonicG1TrackingCfg",
    "SonicG1TrackingEnv",
    "SonicG1TrackingEnvCfg",
    "pack_sonic_observation_terms",
    "sonic_action_scale",
]
