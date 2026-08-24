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
from typing import TYPE_CHECKING, Any, Sequence

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

# These positions are in the SONIC policy joint order.  Keep the ordering
# explicit because the wrist feature is consumed by the SMPL encoder.
SONIC_WRIST_JOINT_INDICES: tuple[int, ...] = (19, 20, 21, 26, 27, 28)


def sonic_action_scale() -> np.ndarray:
    """Return the release model-12 scale in MuJoCo actuator order.

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
    return np.asarray([values[name] for name in SONIC_JOINT_ORDER], dtype=np.float32)


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
    use_release_action_scale: bool = True
    # SONIC clips are materialized in the configured 14-body order. The
    # shared tracking engine otherwise assumes MuJoCo body-id indexing.
    motion_data_body_indices: tuple[int, ...] = tuple(range(len(SONIC_BODY_ORDER)))

    def __post_init__(self) -> None:
        self.sim_dt = 0.005
        self.ctrl_dt = 0.02
        self.sampling_mode = "adaptive"
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
        self._history = np.zeros((num_envs, self.cfg.history_length, 93), dtype=np.float32)
        self._critic_history = np.zeros_like(self._history)
        self._history_valid = np.zeros((num_envs,), dtype=bool)
        self._encoder_index = np.zeros((num_envs, len(self.cfg.encoder_names)), dtype=np.float32)
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

    def _resolve_actuator_permutation(self) -> np.ndarray:
        names = tuple(self._backend.get_actuator_names())
        normalized = tuple(name.removesuffix("_dof") for name in names)
        if set(normalized) != set(SONIC_JOINT_ORDER) or len(normalized) != len(SONIC_JOINT_ORDER):
            raise ValueError(
                "SONIC actuator names do not match the 29-DoF release order: "
                f"expected={SONIC_JOINT_ORDER}, actual={normalized}"
            )
        return np.asarray([SONIC_JOINT_ORDER.index(name) for name in normalized], dtype=np.int32)

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
        probabilities = np.asarray(self.cfg.encoder_sample_probs, dtype=np.float64)
        if probabilities.shape != (len(self.cfg.encoder_names),) or np.any(probabilities < 0):
            raise ValueError("encoder_sample_probs must match encoder_names and be non-negative")
        if probabilities.sum() <= 0:
            raise ValueError("encoder_sample_probs must contain a positive mass")
        probabilities /= probabilities.sum()
        choices = np.random.choice(len(probabilities), size=len(env_ids), p=probabilities)
        self._encoder_index[env_ids] = 0.0
        self._encoder_index[env_ids, choices] = 1.0

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
            if self.cfg.control_config.simulate_action_latency
            else actions
        )
        target_policy = (
            delayed * np.asarray(self.cfg.control_config.action_scale) + self._policy_default_angles
        )
        target_backend = target_policy[:, self._policy_to_backend]
        bias = state.info.get("default_dof_pos_bias")
        if isinstance(bias, np.ndarray):
            target_backend = target_backend + bias[:, self._policy_to_backend]
        return target_backend

    def _policy_joint_values(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)[:, self._backend_to_policy]

    def _future_reference(self, frame_indices: np.ndarray) -> dict[str, np.ndarray]:
        if self._sonic_store is None:
            n = len(frame_indices)
            frames = self.cfg.num_future_frames
            smpl_frames = self.cfg.smpl_num_future_frames
            return {
                "joint_pos": np.zeros((n, frames, 29), dtype=np.float32),
                "joint_vel": np.zeros((n, frames, 29), dtype=np.float32),
                "body_pos": np.zeros((n, frames, len(self.cfg.body_names), 3), dtype=np.float32),
                "body_quat": np.broadcast_to(
                    np.asarray([1, 0, 0, 0], dtype=np.float32),
                    (n, frames, len(self.cfg.body_names), 4),
                ).copy(),
                "smpl_joints": np.zeros((n, smpl_frames, 24, 3), dtype=np.float32),
                "smpl_root_quat": np.broadcast_to(
                    np.asarray([1, 0, 0, 0], dtype=np.float32), (n, smpl_frames, 4)
                ).copy(),
            }
        sim_fps = self._sonic_fps
        offsets = np.rint(
            np.arange(self.cfg.num_future_frames) * self.cfg.dt_future_ref_frames * sim_fps
        ).astype(int)
        smpl_offsets = np.rint(
            np.arange(self.cfg.smpl_num_future_frames)
            * self.cfg.smpl_dt_future_ref_frames
            * sim_fps
        ).astype(int)
        indices = self._sonic_store.future_indices(frame_indices, offsets)
        smpl_indices = self._sonic_store.future_indices(frame_indices, smpl_offsets)
        flat = indices.reshape(-1)
        smpl_flat = smpl_indices.reshape(-1)
        future_fields = self._sonic_store.gather_fields(
            ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w"), flat
        )
        smpl_fields = self._sonic_store.gather_fields(
            ("smpl_joints", "smpl_root_quat_w"), smpl_flat
        )
        return {
            "joint_pos": future_fields["joint_pos"].reshape(len(frame_indices), -1, 29),
            "joint_vel": future_fields["joint_vel"].reshape(len(frame_indices), -1, 29),
            "body_pos": future_fields["body_pos_w"].reshape(
                len(frame_indices), -1, self._sonic_num_bodies, 3
            )[:, :, : len(self.cfg.body_names)],
            "body_quat": future_fields["body_quat_w"].reshape(
                len(frame_indices), -1, self._sonic_num_bodies, 4
            )[:, :, : len(self.cfg.body_names)],
            "smpl_joints": smpl_fields["smpl_joints"].reshape(len(frame_indices), -1, 24, 3),
            "smpl_root_quat": smpl_fields["smpl_root_quat_w"].reshape(len(frame_indices), -1, 4),
        }

    @staticmethod
    def _body_row(body_names: Sequence[str], name: str, fallback: int) -> int:
        try:
            return tuple(body_names).index(name)
        except ValueError:
            return fallback

    def _build_history(
        self,
        env_ids: np.ndarray,
        linvel: np.ndarray,
        gyro: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        robot_body_quat_w: np.ndarray,
        last_actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        dof_pos = self._policy_joint_values(dof_pos)
        dof_vel = self._policy_joint_values(dof_vel)
        anchor_quat = robot_body_quat_w[:, self.anchor_body_idx]
        gravity = np.broadcast_to(np.asarray([0.0, 0.0, -1.0], dtype=np.float32), (len(env_ids), 3))
        gravity = np.asarray(np_quat_apply_inverse(anchor_quat, gravity), dtype=np.float32)
        clean = np.concatenate(
            [gravity, gyro, dof_pos - self._policy_default_angles, dof_vel, last_actions], axis=1
        )
        noise_cfg = self.cfg.noise_config
        actor_current = clean.copy()
        if noise_cfg.level > 0:
            begin = 0
            actor_current[:, begin : begin + 3] = self._obs_noise(
                actor_current[:, begin : begin + 3], noise_cfg.scale_gravity
            )
            begin += 3
            actor_current[:, begin : begin + 3] = self._obs_noise(
                actor_current[:, begin : begin + 3], noise_cfg.scale_gyro
            )
            begin += 3
            actor_current[:, begin : begin + 29] = self._obs_noise(
                actor_current[:, begin : begin + 29], noise_cfg.scale_joint_angle
            )
            begin += 29
            actor_current[:, begin : begin + 29] = self._obs_noise(
                actor_current[:, begin : begin + 29], noise_cfg.scale_joint_vel
            )
        reset_ids = self._sonic_reset_ids
        if reset_ids is not None:
            self._history[env_ids] = actor_current[:, None, :]
            self._critic_history[env_ids] = clean[:, None, :]
            self._history_valid[env_ids] = True
        else:
            self._history[env_ids, :-1] = self._history[env_ids, 1:]
            self._history[env_ids, -1] = actor_current
            self._critic_history[env_ids, :-1] = self._critic_history[env_ids, 1:]
            self._critic_history[env_ids, -1] = clean
        return self._history[env_ids].reshape(len(env_ids), -1), self._critic_history[
            env_ids
        ].reshape(len(env_ids), -1)

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
        current_actions = info.get("current_actions")
        if not isinstance(current_actions, np.ndarray):
            current_actions = np.zeros((len(env_ids), 29), dtype=np.float32)
        last_actions = np.asarray(current_actions, dtype=np.float32)
        actor_obs, history_clean = self._build_history(
            env_ids, linvel, gyro, dof_pos, dof_vel, robot_body_quat_w, last_actions
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

        critic_obs = np.concatenate(
            [
                future["joint_pos"].reshape(len(env_ids), -1),
                future["joint_vel"].reshape(len(env_ids), -1),
                anchor_pos_b,
                anchor_ori_b,
                body_pos_b.reshape(len(env_ids), -1),
                body_ori_b.reshape(len(env_ids), -1),
                history_clean,
            ],
            axis=1,
        ).astype(np.float32)
        if critic_obs.shape[1] != SONIC_CRITIC_OBS_DIM:
            raise RuntimeError(f"SONIC critic observation width drifted to {critic_obs.shape[1]}")

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
        command = np.concatenate([future["joint_pos"], future["joint_vel"]], axis=-1)
        command_z = future["body_pos"][:, :, 0, 2] - robot_anchor_pos[:, None, 2]
        lower = np.concatenate(
            [future["joint_pos"][:, :, :12], future["joint_vel"][:, :, :12]], axis=-1
        )
        body_names = tuple(self.cfg.body_names)
        wrist_l = self._body_row(body_names, "left_wrist_yaw_link", max(0, len(body_names) - 2))
        wrist_r = self._body_row(body_names, "right_wrist_yaw_link", max(0, len(body_names) - 1))
        torso = self._body_row(body_names, "torso_link", self.anchor_body_idx)
        vr_pos = future["body_pos"][:, 0, (wrist_l, wrist_r, torso)] - anchor_pos[:, None, :]
        vr_pos = np_quat_apply_batched(
            np_quat_conjugate_batched(anchor_quat)[:, None, :], vr_pos
        ).reshape(len(env_ids), -1)
        vr_quat = future["body_quat"][:, 0, (wrist_l, wrist_r, torso)]
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
        wrist_q = future["joint_pos"][:, :, SONIC_WRIST_JOINT_INDICES][
            :, : self.cfg.smpl_num_future_frames
        ]
        tokenizer = np.concatenate(
            [
                self._encoder_index[env_ids],
                command.reshape(len(env_ids), -1),
                command_z.reshape(len(env_ids), -1),
                future_ori,
                lower.reshape(len(env_ids), -1),
                vr_pos,
                vr_quat,
                anchor_ori_b,
                command_z[:, :1],
                smpl_local.reshape(len(env_ids), -1),
                smpl_ori.reshape(len(env_ids), -1),
                wrist_q.reshape(len(env_ids), -1),
            ],
            axis=1,
        ).astype(np.float32)
        if tokenizer.shape[1] != SONIC_TOKENIZER_OBS_DIM:
            raise RuntimeError(f"SONIC tokenizer observation width drifted to {tokenizer.shape[1]}")
        return {
            "actor_obs": actor_obs.astype(np.float32),
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
    "SONIC_WRIST_JOINT_INDICES",
    "SonicG1TrackingCfg",
    "SonicG1TrackingEnv",
    "SonicG1TrackingEnvCfg",
    "sonic_action_scale",
]
