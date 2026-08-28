"""Task-owned SONIC Manager-Based command and compact manifest loader.

The Bones-Seed SONIC corpus stores only the named tracking bodies.  Those
compact body columns are a dataset contract and must not be indexed with
backend body IDs, which can be sparse and backend-specific.  This module
resolves the manifest name order once during command construction and then
uses the ordinary in-memory :class:`MotionLoader` hot path.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, Literal, Sequence, cast

import numpy as np

from unilab.assets.hub import resolve_motion_files
from unilab.tasks.motion_tracking.common.manager_terms import (
    MotionCommand,
    MotionCommandCfg,
    MotionCommandParamsCfg,
)
from unilab.tasks.motion_tracking.common.motion_loader import MotionLoader

from .observations import SonicTokenizerObservationCache, SonicTokenizerObservationProvider

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


SONIC_MOTION_SCHEMA = "unilab.sonic.motion"
SONIC_MOTION_VERSION = 1

# Bones-Seed materialization order.  The policy-specific interleaved order is
# a separate learner concern; motion references enter the manager in this
# entity/actuator order.
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


class SonicMotionManifestError(ValueError):
    """Raised when a compact SONIC manifest violates its owner contract."""


@dataclass(frozen=True)
class _CompactManifest:
    path: Path
    joint_order: tuple[str, ...]
    body_order: tuple[str, ...]
    clip_paths: tuple[str, ...]
    clip_lengths: tuple[int, ...]
    fps: int
    field_names: frozenset[str]


def _name_order(value: Any, *, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise SonicMotionManifestError(f"{location} must be a non-empty list")
    if any(not isinstance(name, str) or not name for name in value):
        raise SonicMotionManifestError(f"{location} must contain non-empty strings")
    names = tuple(value)
    if len(set(names)) != len(names):
        raise SonicMotionManifestError(f"{location} contains duplicate names")
    return names


def _order_permutation(
    source: tuple[str, ...],
    expected: Sequence[str],
    *,
    field: str,
) -> np.ndarray | None:
    expected_names = tuple(expected)
    if not expected_names or any(not isinstance(name, str) or not name for name in expected_names):
        raise SonicMotionManifestError(f"expected {field} must contain non-empty strings")
    if len(set(expected_names)) != len(expected_names):
        raise SonicMotionManifestError(f"expected {field} contains duplicate names")
    if len(source) != len(expected_names) or set(source) != set(expected_names):
        missing = sorted(set(expected_names).difference(source))
        extra = sorted(set(source).difference(expected_names))
        raise SonicMotionManifestError(
            f"manifest {field} differs from the expected names; missing={missing}, extra={extra}"
        )
    if source == expected_names:
        return None
    permutation = np.asarray([source.index(name) for name in expected_names], dtype=np.intp)
    permutation.setflags(write=False)
    return permutation


def _validate_required_fields(
    value: Any,
    *,
    num_joints: int,
    num_bodies: int,
) -> frozenset[str]:
    if not isinstance(value, list):
        raise SonicMotionManifestError("manifest.fields must be a list")
    fields: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise SonicMotionManifestError(f"manifest.fields[{index}] must be an object")
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise SonicMotionManifestError(f"manifest.fields[{index}].name must be non-empty")
        if name in fields:
            raise SonicMotionManifestError(f"manifest.fields contains duplicate field {name!r}")
        fields[name] = raw

    expected_shapes: dict[str, tuple[Any, ...]] = {
        "joint_pos": ("num_frames", num_joints),
        "joint_vel": ("num_frames", num_joints),
        "body_pos_w": ("num_frames", num_bodies, 3),
        "body_quat_w": ("num_frames", num_bodies, 4),
        "body_lin_vel_w": ("num_frames", num_bodies, 3),
        "body_ang_vel_w": ("num_frames", num_bodies, 3),
    }
    for name, expected_shape in expected_shapes.items():
        raw = fields.get(name)
        if raw is None:
            raise SonicMotionManifestError(f"manifest.fields is missing required field {name!r}")
        if raw.get("dtype") != "float32":
            raise SonicMotionManifestError(f"manifest field {name!r} must use dtype 'float32'")
        shape = raw.get("shape")
        if not isinstance(shape, list) or tuple(shape) != expected_shape:
            raise SonicMotionManifestError(
                f"manifest field {name!r} has shape {shape!r}, expected {list(expected_shape)!r}"
            )
    optional_shapes: dict[str, tuple[Any, ...]] = {
        "smpl_joints": ("num_frames", 24, 3),
        "smpl_root_quat_w": ("num_frames", 4),
    }
    for name, expected_shape in optional_shapes.items():
        raw = fields.get(name)
        if raw is None:
            continue
        if raw.get("dtype") != "float32":
            raise SonicMotionManifestError(f"manifest field {name!r} must use dtype 'float32'")
        shape = raw.get("shape")
        if not isinstance(shape, list) or tuple(shape) != expected_shape:
            raise SonicMotionManifestError(
                f"manifest field {name!r} has shape {shape!r}, expected {list(expected_shape)!r}"
            )
    return frozenset(fields)


def _resolve_clip_path(manifest_path: Path, value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise SonicMotionManifestError(f"{location} must be a non-empty relative path")
    windows_path = PureWindowsPath(value)
    if Path(value).is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise SonicMotionManifestError(f"{location} must be relative")
    if any(part == ".." for part in windows_path.parts):
        raise SonicMotionManifestError(f"{location} must not contain parent traversal")
    path = (manifest_path.parent / value).resolve()
    if not path.is_relative_to(manifest_path.parent):
        raise SonicMotionManifestError(f"{location} escapes the manifest directory")
    if not path.is_file():
        raise SonicMotionManifestError(f"{location} does not exist: {path}")
    return str(path)


def _positive_int(value: Any, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SonicMotionManifestError(f"{location} must be a positive integer")
    return int(value)


def _positive_fps(value: Any, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SonicMotionManifestError(f"{location} must be a positive integer frame rate")
    number = float(value)
    if not math.isfinite(number) or number < 1 or not number.is_integer():
        raise SonicMotionManifestError(f"{location} must be a positive integer frame rate")
    return int(number)


def _load_compact_manifest(manifest_file: str) -> _CompactManifest:
    resolved = cast(str, resolve_motion_files(manifest_file))
    manifest_path = Path(resolved).resolve()
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise SonicMotionManifestError(
            f"could not read SONIC manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise SonicMotionManifestError("SONIC manifest root must be an object")
    if raw.get("schema") != SONIC_MOTION_SCHEMA:
        raise SonicMotionManifestError(
            f"manifest.schema must be {SONIC_MOTION_SCHEMA!r}, got {raw.get('schema')!r}"
        )
    version = raw.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != SONIC_MOTION_VERSION:
        raise SonicMotionManifestError(
            f"manifest.version must be {SONIC_MOTION_VERSION}, got {version!r}"
        )

    joint_order = _name_order(raw.get("joint_order"), location="manifest.joint_order")
    body_order = _name_order(raw.get("body_order"), location="manifest.body_order")
    field_names = _validate_required_fields(
        raw.get("fields"),
        num_joints=len(joint_order),
        num_bodies=len(body_order),
    )

    raw_clips = raw.get("clips")
    if not isinstance(raw_clips, list) or not raw_clips:
        raise SonicMotionManifestError("manifest.clips must be a non-empty list")
    clip_paths: list[str] = []
    clip_lengths: list[int] = []
    clip_fps: list[int] = []
    for index, clip in enumerate(raw_clips):
        location = f"manifest.clips[{index}]"
        if not isinstance(clip, dict):
            raise SonicMotionManifestError(f"{location} must be an object")
        if "joint_order" in clip:
            order = _name_order(clip["joint_order"], location=f"{location}.joint_order")
            if order != joint_order:
                raise SonicMotionManifestError(f"{location}.joint_order differs from manifest")
        if "body_order" in clip:
            order = _name_order(clip["body_order"], location=f"{location}.body_order")
            if order != body_order:
                raise SonicMotionManifestError(f"{location}.body_order differs from manifest")
        clip_paths.append(
            _resolve_clip_path(manifest_path, clip.get("path"), location=f"{location}.path")
        )
        clip_lengths.append(
            _positive_int(clip.get("num_frames"), location=f"{location}.num_frames")
        )
        clip_fps.append(_positive_fps(clip.get("fps"), location=f"{location}.fps"))
    if len(set(clip_fps)) != 1:
        raise SonicMotionManifestError("manifest clips must use one frame rate")

    return _CompactManifest(
        path=manifest_path,
        joint_order=joint_order,
        body_order=body_order,
        clip_paths=tuple(clip_paths),
        clip_lengths=tuple(clip_lengths),
        fps=clip_fps[0],
        field_names=field_names,
    )


class CompactSonicMotionLoader(MotionLoader):
    """Array-backed Manager loader for a compact SONIC v1 manifest.

    This first owner slice intentionally materializes its selected clips in
    memory.  Bounded lazy caching, mmap publication, and rank sharding belong
    to a later store-level change and do not alter this name-order seam.
    """

    def __init__(
        self,
        manifest_file: str,
        *,
        expected_joint_order: Sequence[str],
        expected_body_order: Sequence[str],
    ) -> None:
        manifest = _load_compact_manifest(manifest_file)
        joint_permutation = _order_permutation(
            manifest.joint_order,
            expected_joint_order,
            field="joint_order",
        )
        body_permutation = _order_permutation(
            manifest.body_order,
            expected_body_order,
            field="body_order",
        )

        # These are dataset column indices.  Backend body IDs are deliberately
        # absent from this constructor and from the manifest schema.
        super().__init__(manifest.clip_paths, body_indices=body_permutation)
        if joint_permutation is not None:
            self.joint_pos = np.ascontiguousarray(self.joint_pos[:, joint_permutation])
            self.joint_vel = np.ascontiguousarray(self.joint_vel[:, joint_permutation])

        expected_lengths = np.asarray(manifest.clip_lengths, dtype=np.int32)
        if not np.array_equal(self.clip_lengths, expected_lengths):
            raise SonicMotionManifestError(
                f"clip frame counts {self.clip_lengths.tolist()} differ from manifest "
                f"{expected_lengths.tolist()}"
            )
        if self.fps != manifest.fps:
            raise SonicMotionManifestError(
                f"motion files use fps={self.fps}, but manifest declares fps={manifest.fps}"
            )

        optional_fields = tuple(
            name for name in ("smpl_joints", "smpl_root_quat_w") if name in manifest.field_names
        )
        optional_chunks: dict[str, list[np.ndarray]] = {name: [] for name in optional_fields}
        for clip_path, expected_length in zip(
            manifest.clip_paths, manifest.clip_lengths, strict=True
        ):
            if not optional_fields:
                break
            with np.load(clip_path, allow_pickle=False) as archive:
                for name in optional_fields:
                    if name not in archive:
                        raise SonicMotionManifestError(
                            f"motion file {clip_path!r} is missing manifest field {name!r}"
                        )
                    values = np.asarray(archive[name], dtype=np.float32)
                    if values.shape[0] != expected_length:
                        raise SonicMotionManifestError(
                            f"motion file {clip_path!r} field {name!r} has "
                            f"{values.shape[0]} frames, expected {expected_length}"
                        )
                    optional_chunks[name].append(values)
        self.smpl_joints = (
            np.concatenate(optional_chunks["smpl_joints"], axis=0)
            if "smpl_joints" in optional_chunks
            else None
        )
        self.smpl_root_quat_w = (
            np.concatenate(optional_chunks["smpl_root_quat_w"], axis=0)
            if "smpl_root_quat_w" in optional_chunks
            else None
        )
        required_fields = {
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        }
        self.available_fields = frozenset((*required_fields, *optional_fields))
        expected_frames = int(expected_lengths.sum())
        expected_joints = len(tuple(expected_joint_order))
        expected_bodies = len(tuple(expected_body_order))
        expected_shapes = {
            "joint_pos": (expected_frames, expected_joints),
            "joint_vel": (expected_frames, expected_joints),
            "body_pos_w": (expected_frames, expected_bodies, 3),
            "body_quat_w": (expected_frames, expected_bodies, 4),
            "body_lin_vel_w": (expected_frames, expected_bodies, 3),
            "body_ang_vel_w": (expected_frames, expected_bodies, 3),
        }
        for name, expected_shape in expected_shapes.items():
            actual_shape = getattr(self, name).shape
            if actual_shape != expected_shape:
                raise SonicMotionManifestError(
                    f"motion field {name!r} has shape {actual_shape}, expected {expected_shape}"
                )

        self.manifest_path = manifest.path
        self.joint_order = tuple(expected_joint_order)
        self.body_order = tuple(expected_body_order)

    def gather_fields(
        self,
        fields: Sequence[str],
        frame_indices: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Gather already-materialized fields without reopening motion assets."""

        names = tuple(dict.fromkeys(str(name) for name in fields))
        unknown = sorted(set(names).difference(self.available_fields))
        if unknown:
            raise KeyError(f"compact SONIC loader does not expose fields {unknown}")
        indices = np.asarray(frame_indices, dtype=np.int64)
        result: dict[str, np.ndarray] = {}
        for name in names:
            source = getattr(self, name)
            if not isinstance(source, np.ndarray):
                raise RuntimeError(f"compact SONIC field {name!r} was not materialized")
            result[name] = np.take(source, indices, axis=0)
        return result


@dataclass
class SonicMotionCommandParamsCfg(MotionCommandParamsCfg):
    """SONIC parameters whose ``motion_file`` names a v1 manifest.

    The future sampling contract is deliberately attached to the motion
    command instead of an observation term.  A command owns clip-local frame
    state, so it is the only layer that can clamp reference frames without
    either reopening a motion asset or relying on backend details.
    """

    num_future_frames: int = 10
    dt_future_ref_frames: float = 0.1
    smpl_num_future_frames: int = 10
    smpl_dt_future_ref_frames: float = 0.02
    encoder_sample_probs: tuple[float, float, float] = (1.0, 1.0, 1.0)
    teleop_sample_prob_when_smpl: float = 0.5
    tokenizer_enable_corruption: bool = True
    # Lazy store controls are owner-local and optional.  The compact eager
    # loader remains the default for tiny contract fixtures; full training
    # enables the bounded clip cache explicitly and requests SMPL columns via
    # ``motion_optional_fields``.
    use_lazy_motion_loader: bool = False
    # ``auto`` keeps one active clip per environment up to a fixed owner-level
    # ceiling.  This avoids the pathological two-entry default without making
    # the resident set grow without bound with rollout size.
    motion_cache_size: int | Literal["auto"] = "auto"
    motion_cache_max_size: int = 128
    motion_cache_max_bytes: int = 512 * 1024 * 1024
    motion_optional_fields: tuple[str, ...] = ()
    motion_rank: int = 0
    motion_world_size: int = 1
    motion_shard_clips: bool = False
    # Optional fixed-layout Numba tokenizer assembly.  NumPy assembly remains
    # the default because it is the reference path and preserves the public
    # term contract for existing experiments.
    use_numba_observation_assembly: bool = False


@dataclass(kw_only=True)
class SonicMotionCommandCfg(MotionCommandCfg):
    """Task-owned command config for compact Bones-Seed motion data."""

    params: SonicMotionCommandParamsCfg  # pyright: ignore[reportIncompatibleVariableOverride]

    def build(self, env: ManagerBasedRlEnv) -> SonicMotionCommand:
        return SonicMotionCommand(self, env)


class SonicMotionCommand(MotionCommand, SonicTokenizerObservationProvider):
    """SONIC motion owner with clip-local future and tokenizer state.

    The public manager emits only actor and critic observations.  The release
    PPO additionally needs a tokenizer input, so this command owns its cache
    and implements :class:`SonicTokenizerObservationProvider`.  Observation
    terms only assemble numeric values from this owner; no term reads an asset
    or a private observation-manager buffer on the rollout path.
    """

    cfg: SonicMotionCommandCfg  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, cfg: SonicMotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # Import here to avoid the package-level action/command import cycle.
        # This is construction-only configuration binding, never a rollout
        # path import or backend capability probe.
        from .actions import SONIC_POLICY_TO_JOINT

        if tuple(self.robot.joint_names) != SONIC_JOINT_ORDER:
            raise ValueError(
                "SONIC motion command requires Entity joint_names in canonical "
                f"order {SONIC_JOINT_ORDER}, got {tuple(self.robot.joint_names)}"
            )
        # Values read from the Entity/motion loader are in canonical joint
        # order.  Selecting ``SONIC_POLICY_TO_JOINT`` therefore emits release
        # policy order (the inverse mapping is used by the action term).
        self._policy_to_joint = np.asarray(SONIC_POLICY_TO_JOINT, dtype=np.intp)
        self._policy_to_joint.setflags(write=False)
        try:
            vr_indices = [self.cfg.body_names.index(name) for name in (
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
                "torso_link",
            )]
        except ValueError as exc:
            raise ValueError(
                "SONIC motion command body_names must include the three VR bodies"
            ) from exc
        self._vr_body_indices = np.asarray(vr_indices, dtype=np.intp)
        self._vr_body_indices.setflags(write=False)
        self._future_offsets = self._future_frame_offsets(
            cfg.params.num_future_frames,
            cfg.params.dt_future_ref_frames,
            self.motion.fps,
            name="G1",
        )
        # The Bones-Seed benchmark subset is 30 Hz and intentionally omits
        # the SMPL root quaternion.  In that case SMPL observations are the
        # documented zero reference, so do not reject an otherwise valid G1
        # owner merely because 0.02 s is not an integral 30 Hz frame step.
        if {"smpl_joints", "smpl_root_quat_w"}.issubset(self.motion.available_fields):
            self._smpl_future_offsets = self._future_frame_offsets(
                cfg.params.smpl_num_future_frames,
                cfg.params.smpl_dt_future_ref_frames,
                self.motion.fps,
                name="SMPL",
            )
        else:
            self._smpl_future_offsets = np.zeros(cfg.params.smpl_num_future_frames, dtype=np.int64)
            self._smpl_future_offsets.setflags(write=False)
        self._validate_encoder_sampling(cfg.params)
        self._encoder_index = np.zeros((self.num_envs, 3), dtype=self.motion.joint_pos.dtype)
        # A manager constructs observation terms before the first reset.  Keep
        # that construction-only probe deterministic; normal encoder sampling
        # happens in reset() using env.rng.
        self._encoder_index[:, 0] = 1.0
        self._tokenizer_cache = SonicTokenizerObservationCache(
            self.num_envs, dtype=self.motion.joint_pos.dtype
        )
        self._use_numba_observation_assembly = bool(
            cfg.params.use_numba_observation_assembly
        )
        self._tokenizer_assembly_buffer = (
            np.empty((self.num_envs, 1761), dtype=self.motion.joint_pos.dtype)
            if self._use_numba_observation_assembly
            else None
        )
        self._observation_update_env_ids: np.ndarray | None = None
        self._future_reference_cache: Any | None = None

    @staticmethod
    def _future_frame_offsets(
        count: int,
        spacing: float,
        fps: int,
        *,
        name: str,
    ) -> np.ndarray:
        if isinstance(count, bool) or not isinstance(count, int) or count != 10:
            raise ValueError(
                f"SONIC {name} future count must be the release value 10, got {count!r}"
            )
        if isinstance(spacing, bool) or not isinstance(spacing, (int, float)):
            raise TypeError(f"SONIC {name} future spacing must be a finite real number")
        steps = float(spacing) * int(fps)
        rounded = int(round(steps))
        if not math.isfinite(steps) or rounded < 1 or not math.isclose(steps, rounded):
            raise ValueError(
                f"SONIC {name} future spacing={spacing} is not an integral positive frame "
                f"interval at motion fps={fps}"
            )
        offsets = np.arange(count, dtype=np.int64) * rounded
        offsets.setflags(write=False)
        return offsets

    @staticmethod
    def _validate_encoder_sampling(params: SonicMotionCommandParamsCfg) -> None:
        probabilities = np.asarray(params.encoder_sample_probs, dtype=np.float64)
        if probabilities.shape != (3,) or not np.isfinite(probabilities).all():
            raise ValueError("SONIC encoder_sample_probs must be three finite values")
        if np.any(probabilities < 0.0) or float(probabilities.sum()) <= 0.0:
            raise ValueError(
                "SONIC encoder_sample_probs must have positive total non-negative mass"
            )
        probability = params.teleop_sample_prob_when_smpl
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
        ):
            raise ValueError("SONIC teleop_sample_prob_when_smpl must be within [0, 1]")
        if not isinstance(params.tokenizer_enable_corruption, bool):
            raise TypeError("SONIC tokenizer_enable_corruption must be bool")
        if not isinstance(params.use_numba_observation_assembly, bool):
            raise TypeError("SONIC use_numba_observation_assembly must be bool")

    @property
    def policy_to_joint(self) -> np.ndarray:
        """Immutable policy-column to canonical joint-column mapping."""

        return self._policy_to_joint

    @property
    def vr_body_indices(self) -> np.ndarray:
        """Fixed dataset body columns used by the release VR targets."""

        return self._vr_body_indices

    @property
    def use_numba_observation_assembly(self) -> bool:
        """Whether the opt-in fixed-layout tokenizer kernel is enabled."""

        return self._use_numba_observation_assembly

    @property
    def tokenizer_assembly_buffer(self) -> np.ndarray:
        """Owner-owned output buffer used by the optional Numba assembler."""

        if self._tokenizer_assembly_buffer is None:
            raise RuntimeError("SONIC Numba tokenizer assembly is not enabled")
        return self._tokenizer_assembly_buffer

    @property
    def encoder_index(self) -> np.ndarray:
        """Current task-owned release encoder selection mask, shape ``(N, 3)``."""

        return self._encoder_index

    @property
    def future_offsets(self) -> np.ndarray:
        """G1 future offsets in frames, owned by the command's motion fps."""

        return self._future_offsets

    @property
    def smpl_future_offsets(self) -> np.ndarray:
        """SMPL future offsets in frames, owned by the command's motion fps."""

        return self._smpl_future_offsets

    @property
    def observation_update_env_ids(self) -> np.ndarray | None:
        """Rows whose tokenizer cache is being refreshed, or ``None`` for all rows."""

        return self._observation_update_env_ids

    @property
    def has_smpl_reference(self) -> bool:
        """Whether both optional SMPL fields are provided by the materialized loader."""

        return {"smpl_joints", "smpl_root_quat_w"}.issubset(self.motion.available_fields)

    def future_frame_indices(self, *, smpl: bool = False) -> np.ndarray:
        """Return clip-clamped future indices without touching motion storage."""

        offsets = self._smpl_future_offsets if smpl else self._future_offsets
        indices = self.time_steps[:, None].astype(np.int64, copy=False) + offsets[None, :]
        return np.minimum(indices, self.sampler.current_clip_end_frames[:, None])

    def get_future_reference_cache(self) -> Any | None:
        """Return the current numeric future-reference cache for observation terms."""

        return self._future_reference_cache

    def set_future_reference_cache(self, value: Any) -> None:
        """Publish one per-lifecycle numeric future-reference cache entry."""

        self._future_reference_cache = value

    def get_tokenizer_observations(self) -> np.ndarray:
        """Return the task-owned current tokenizer observation batch."""

        return self._tokenizer_cache.get_tokenizer_observations()

    def write_tokenizer_observations(self, values: np.ndarray) -> None:
        """Store a full or reset-row tokenizer batch through the typed cache."""

        env_ids = self._observation_update_env_ids
        if env_ids is None:
            self._tokenizer_cache.write(values)
        else:
            self._tokenizer_cache.write(values[env_ids], env_ids=env_ids)

    def _sample_encoder_indices(self, env_ids: np.ndarray) -> None:
        if not len(env_ids):
            return
        probabilities = np.asarray(self.cfg.params.encoder_sample_probs, dtype=np.float64)
        if not self.has_smpl_reference:
            probabilities[2] = 0.0
        if float(probabilities.sum()) <= 0.0:
            raise ValueError("SONIC motion data has no available encoder sampling mass")
        probabilities /= probabilities.sum()
        choices = self._env.rng.choice(3, size=len(env_ids), p=probabilities)
        self._encoder_index[env_ids] = 0.0
        self._encoder_index[env_ids, choices] = 1.0
        smpl_ids = env_ids[choices == 2]
        if len(smpl_ids):
            # Release training activates the paired G1 encoder for an SMPL
            # sample and optionally the teleop encoder as a second positive.
            self._encoder_index[smpl_ids, 0] = 1.0
            teleop = (
                self._env.rng.random(len(smpl_ids)) < self.cfg.params.teleop_sample_prob_when_smpl
            )
            self._encoder_index[smpl_ids[teleop], 1] = 1.0

    def reset(self, env_ids: np.ndarray | slice | None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if env_ids is None:
            ids = np.arange(self.num_envs, dtype=np.int32)
        elif isinstance(env_ids, slice):
            ids = np.arange(self.num_envs, dtype=np.int32)[env_ids]
        else:
            ids = np.asarray(env_ids, dtype=np.int32)
        self._sample_encoder_indices(ids)
        self._observation_update_env_ids = ids
        self._future_reference_cache = None
        return extras

    def _update_command(self, env_ids: np.ndarray | None) -> None:
        super()._update_command(env_ids)
        # Command frame state changes before observations are computed.  Clear
        # only the numeric cache; assets stay materialized in the loader.
        self._observation_update_env_ids = env_ids
        self._future_reference_cache = None

    def _make_motion_loader(
        self,
        motion_file: str | list[str],
        body_indices: np.ndarray,
    ) -> CompactSonicMotionLoader:
        # ``body_indices`` are MuJoCo/MJWarp/Motrix IDs used by Entity state
        # access.  Compact Bones-Seed columns are instead resolved by the
        # manifest's canonical names on this construction-only owner hook.
        del body_indices
        if not isinstance(motion_file, str):
            raise TypeError("SonicMotionCommand requires one manifest path")
        params = self.cfg.params
        if params.use_lazy_motion_loader:
            from .lazy_motion_loader import BoundedLazySonicMotionLoader

            return BoundedLazySonicMotionLoader(
                motion_file,
                expected_joint_order=SONIC_JOINT_ORDER,
                expected_body_order=self.cfg.body_names,
                cache_size=params.motion_cache_size,
                cache_max_size=params.motion_cache_max_size,
                cache_max_bytes=params.motion_cache_max_bytes,
                optional_fields=params.motion_optional_fields,
                num_envs=self.num_envs,
                rank=params.motion_rank,
                world_size=params.motion_world_size,
                shard_clips=params.motion_shard_clips,
            )  # type: ignore[return-value]
        return CompactSonicMotionLoader(
            motion_file,
            expected_joint_order=SONIC_JOINT_ORDER,
            expected_body_order=self.cfg.body_names,
        )


__all__ = [
    "CompactSonicMotionLoader",
    "SONIC_JOINT_ORDER",
    "SONIC_MOTION_SCHEMA",
    "SONIC_MOTION_VERSION",
    "SonicMotionCommand",
    "SonicMotionCommandCfg",
    "SonicMotionCommandParamsCfg",
    "SonicMotionManifestError",
]
