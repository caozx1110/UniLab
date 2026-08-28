"""Bounded lazy NPZ motion loader owned by the Manager-Based SONIC task.

The manifest and every name-to-column mapping are resolved during loader
construction.  Frame gathers operate on a rank-local global-frame space and
touch assets only through an explicit bounded clip-cache miss path; they never
reparse JSON or inspect backend metadata.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Literal, cast

import numpy as np

from unilab.assets.hub import resolve_motion_files
from unilab.tasks.motion_tracking.common.motion_loader import MotionData

from .manager_terms import (
    SONIC_MOTION_SCHEMA,
    SONIC_MOTION_VERSION,
    SonicMotionManifestError,
    _name_order,
    _order_permutation,
    _positive_fps,
    _positive_int,
    _resolve_clip_path,
    _validate_required_fields,
)

_REQUIRED_FIELDS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
_OPTIONAL_FIELDS = ("smpl_joints", "smpl_root_quat_w")
_JOINT_FIELDS = frozenset(("joint_pos", "joint_vel"))
_BODY_FIELDS = frozenset(("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"))

# The automatic policy follows the per-rank active working set up to a fixed
# ceiling.  It therefore grows for a small rollout (where keeping one clip per
# environment is useful), but never allocates cache entries proportional to an
# arbitrarily large environment count.  128 is deliberately above the 100-clip
# benchmark subset while keeping the full-corpus resident set bounded.
DEFAULT_AUTO_CACHE_SIZE = 128
DEFAULT_CACHE_MAX_BYTES = 512 * 1024 * 1024
CacheSize = int | Literal["auto"]


@dataclass(frozen=True)
class LazyMotionFieldSpec:
    """Shape/dtype surface used by ``MotionCommand`` without materialization."""

    shape: tuple[int, ...]
    dtype: np.dtype[Any]


@dataclass
class LazySonicMotionData(MotionData):
    """Motion batch with optional task-owned SMPL reference fields."""

    smpl_joints: np.ndarray | None = None
    smpl_root_quat_w: np.ndarray | None = None


@dataclass(frozen=True)
class _LazyManifest:
    path: Path
    joint_order: tuple[str, ...]
    body_order: tuple[str, ...]
    declared_fields: frozenset[str]
    clip_paths: tuple[str, ...]
    clip_lengths: tuple[int, ...]
    source_clip_indices: tuple[int, ...]
    fps: int


@dataclass(frozen=True)
class _DecodedClip:
    arrays: dict[str, np.ndarray]
    nbytes: int


def _relative_clip_path(value: Any, *, location: str) -> str:
    """Validate every manifest path without stat'ing clips owned by other ranks."""

    if not isinstance(value, str) or not value or "\x00" in value:
        raise SonicMotionManifestError(f"{location} must be a non-empty relative path")
    windows_path = PureWindowsPath(value)
    if Path(value).is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise SonicMotionManifestError(f"{location} must be relative")
    if any(part == ".." for part in windows_path.parts):
        raise SonicMotionManifestError(f"{location} must not contain parent traversal")
    return value


def _validate_shard_args(rank: int, world_size: int, shard_clips: bool) -> None:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError(f"rank must be a non-negative integer, got {rank!r}")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError(f"world_size must be a positive integer, got {world_size!r}")
    if rank >= world_size:
        raise ValueError(f"rank={rank} must be less than world_size={world_size}")
    if not isinstance(shard_clips, bool):
        raise TypeError("shard_clips must be bool")


def _validate_optional_field_specs(
    raw_fields: list[Any],
    *,
    num_joints: int,
    num_bodies: int,
) -> frozenset[str]:
    """Return supported declared fields after cold schema validation."""

    _validate_required_fields(raw_fields, num_joints=num_joints, num_bodies=num_bodies)
    by_name = {
        cast(dict[str, Any], item)["name"]: cast(dict[str, Any], item) for item in raw_fields
    }
    expected_optional_shapes = {
        "smpl_joints": ("num_frames", 24, 3),
        "smpl_root_quat_w": ("num_frames", 4),
    }
    for name, expected_shape in expected_optional_shapes.items():
        raw = by_name.get(name)
        if raw is None:
            continue
        if raw.get("dtype") != "float32":
            raise SonicMotionManifestError(f"manifest field {name!r} must use dtype 'float32'")
        shape = raw.get("shape")
        if not isinstance(shape, list) or tuple(shape) != expected_shape:
            raise SonicMotionManifestError(
                f"manifest field {name!r} has shape {shape!r}, expected {list(expected_shape)!r}"
            )
    return frozenset(name for name in (*_REQUIRED_FIELDS, *_OPTIONAL_FIELDS) if name in by_name)


def _load_lazy_manifest(
    manifest_file: str,
    *,
    rank: int,
    world_size: int,
    shard_clips: bool,
) -> _LazyManifest:
    """Parse and shard a SONIC v1 manifest exactly once on the cold path."""

    _validate_shard_args(rank, world_size, shard_clips)
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
    raw_fields = raw.get("fields")
    if not isinstance(raw_fields, list):
        raise SonicMotionManifestError("manifest.fields must be a list")
    declared_fields = _validate_optional_field_specs(
        raw_fields,
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
        clip_paths.append(_relative_clip_path(clip.get("path"), location=f"{location}.path"))
        clip_lengths.append(
            _positive_int(clip.get("num_frames"), location=f"{location}.num_frames")
        )
        clip_fps.append(_positive_fps(clip.get("fps"), location=f"{location}.fps"))
    if len(set(clip_fps)) != 1:
        raise SonicMotionManifestError("manifest clips must use one frame rate")

    if shard_clips and world_size > 1:
        selected = tuple(range(rank, len(raw_clips), world_size))
        if not selected:
            raise ValueError(
                "SONIC clip sharding requires at least one clip per rank; "
                f"got {len(raw_clips)} clips for rank={rank}, world_size={world_size}"
            )
    else:
        selected = tuple(range(len(raw_clips)))
    resolved_paths = tuple(
        _resolve_clip_path(
            manifest_path,
            clip_paths[index],
            location=f"manifest.clips[{index}].path",
        )
        for index in selected
    )
    return _LazyManifest(
        path=manifest_path,
        joint_order=joint_order,
        body_order=body_order,
        declared_fields=declared_fields,
        clip_paths=resolved_paths,
        clip_lengths=tuple(clip_lengths[index] for index in selected),
        source_clip_indices=selected,
        fps=clip_fps[0],
    )


def _normalize_optional_fields(
    value: Sequence[str] | None,
    *,
    declared_fields: frozenset[str],
) -> tuple[str, ...]:
    if value is None:
        return tuple(name for name in _OPTIONAL_FIELDS if name in declared_fields)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("optional_fields must be a sequence of field names or None")
    names = tuple(value)
    if len(set(names)) != len(names):
        raise ValueError("optional_fields contains duplicate names")
    unsupported = sorted(set(names).difference(_OPTIONAL_FIELDS))
    if unsupported:
        raise ValueError(f"unsupported optional SONIC motion fields: {unsupported}")
    missing = sorted(set(names).difference(declared_fields))
    if missing:
        raise SonicMotionManifestError(f"manifest does not declare optional fields {missing}")
    return tuple(name for name in _OPTIONAL_FIELDS if name in names)


class BoundedLazySonicMotionLoader:
    """MotionCommand-compatible lazy loader with a clip-count-bounded LRU.

    ``global_indices`` are relative to this loader's clip sequence.  With clip
    sharding enabled, that is the rank-local sequence formed by the rank's
    round-robin clip subset; no distributed runtime or IPC is created here.
    """

    def __init__(
        self,
        manifest_file: str,
        *,
        expected_joint_order: Sequence[str],
        expected_body_order: Sequence[str],
        cache_size: CacheSize = "auto",
        cache_max_size: int = DEFAULT_AUTO_CACHE_SIZE,
        cache_max_bytes: int = DEFAULT_CACHE_MAX_BYTES,
        optional_fields: Sequence[str] | None = (),
        num_envs: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        shard_clips: bool = False,
    ) -> None:
        if cache_size != "auto" and (
            isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0
        ):
            raise ValueError(
                "cache_size must be 'auto' or a non-negative integer, "
                f"got {cache_size!r}"
            )
        if (
            isinstance(cache_max_size, bool)
            or not isinstance(cache_max_size, int)
            or cache_max_size < 1
        ):
            raise ValueError(
                f"cache_max_size must be a positive integer, got {cache_max_size!r}"
            )
        if (
            isinstance(cache_max_bytes, bool)
            or not isinstance(cache_max_bytes, int)
            or cache_max_bytes < 1
        ):
            raise ValueError(
                f"cache_max_bytes must be a positive integer, got {cache_max_bytes!r}"
            )
        if num_envs is not None and (
            isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1
        ):
            raise ValueError(f"num_envs must be a positive integer or None, got {num_envs!r}")
        manifest = _load_lazy_manifest(
            manifest_file,
            rank=rank,
            world_size=world_size,
            shard_clips=shard_clips,
        )
        self._joint_permutation = _order_permutation(
            manifest.joint_order,
            expected_joint_order,
            field="joint_order",
        )
        self._body_permutation = _order_permutation(
            manifest.body_order,
            expected_body_order,
            field="body_order",
        )
        _normalize_optional_fields(
            optional_fields,
            declared_fields=manifest.declared_fields,
        )

        self.manifest_path = manifest.path
        self.motion_files = manifest.clip_paths
        self.source_clip_indices = manifest.source_clip_indices
        self.joint_order = tuple(expected_joint_order)
        self.body_order = tuple(expected_body_order)
        self.fps = manifest.fps
        self.num_joints = len(self.joint_order)
        self.num_bodies = len(self.body_order)
        num_frames = sum(manifest.clip_lengths)
        if num_frames > np.iinfo(np.int32).max:
            raise SonicMotionManifestError(
                "rank-local SONIC frame count exceeds the int32 manager index contract"
            )
        self.clip_lengths = np.asarray(manifest.clip_lengths, dtype=np.int32)
        self.num_clips = int(self.clip_lengths.size)
        self.clip_offsets = np.zeros(self.num_clips, dtype=np.int32)
        if self.num_clips > 1:
            self.clip_offsets[1:] = np.cumsum(self.clip_lengths[:-1], dtype=np.int32)
        # Explicit task-owned spelling consumed by the SONIC observation owner.
        # It is the same immutable array, not a second offsets allocation.
        self.clip_starts = self.clip_offsets
        self.clip_end_frames = self.clip_offsets + self.clip_lengths - 1
        self.num_frames = num_frames
        for array in (self.clip_lengths, self.clip_offsets, self.clip_end_frames):
            array.setflags(write=False)

        # Capabilities come from the manifest, not from the caller's optional
        # preselection.  Observation owners discover SMPL columns here and
        # explicitly gather them only when their terms need those references.
        self.available_fields = manifest.declared_fields
        # The constructor option remains validation-only compatibility input;
        # it never hides manifest-declared fields or expands the default six-
        # field MotionData gather.
        self._field_shapes: dict[str, tuple[int, ...]] = {
            "joint_pos": (self.num_joints,),
            "joint_vel": (self.num_joints,),
            "body_pos_w": (self.num_bodies, 3),
            "body_quat_w": (self.num_bodies, 4),
            "body_lin_vel_w": (self.num_bodies, 3),
            "body_ang_vel_w": (self.num_bodies, 3),
            "smpl_joints": (24, 3),
            "smpl_root_quat_w": (4,),
        }
        dtype = np.dtype(np.float32)
        self.joint_pos = LazyMotionFieldSpec((self.num_frames, self.num_joints), dtype)
        self.joint_vel = LazyMotionFieldSpec((self.num_frames, self.num_joints), dtype)
        self.body_pos_w = LazyMotionFieldSpec((self.num_frames, self.num_bodies, 3), dtype)
        self.body_quat_w = LazyMotionFieldSpec((self.num_frames, self.num_bodies, 4), dtype)
        self.body_lin_vel_w = LazyMotionFieldSpec((self.num_frames, self.num_bodies, 3), dtype)
        self.body_ang_vel_w = LazyMotionFieldSpec((self.num_frames, self.num_bodies, 3), dtype)

        if cache_size == "auto":
            # A gather can touch at most one current clip per environment.
            # Bound that working-set estimate by the owner-level ceiling and
            # by the number of clips available on this rank.
            active_envs = DEFAULT_AUTO_CACHE_SIZE if num_envs is None else num_envs
            self.cache_size = min(
                self.num_clips,
                cache_max_size,
                max(1, active_envs),
            )
        else:
            if cache_size > cache_max_size:
                raise ValueError(
                    f"cache_size={cache_size} exceeds cache_max_size={cache_max_size}; "
                    "raise cache_max_size explicitly to opt into a larger resident set"
                )
            self.cache_size = cache_size
        self.cache_max_size = cache_max_size
        self.cache_max_bytes = cache_max_bytes
        self.requested_cache_size = cache_size
        self._cache: OrderedDict[int, _DecodedClip] = OrderedDict()
        self._loaded_clip_count = 0
        self._cached_bytes = 0
        self._peak_cached_bytes = 0
        self._peak_cached_clip_count = 0

    @property
    def cached_clip_count(self) -> int:
        return len(self._cache)

    @property
    def loaded_clip_count(self) -> int:
        return self._loaded_clip_count

    @property
    def cached_clip_indices(self) -> tuple[int, ...]:
        """Cached rank-local clip indices in oldest-to-newest LRU order."""

        return tuple(self._cache)

    @property
    def cached_bytes(self) -> int:
        return self._cached_bytes

    @property
    def peak_cached_bytes(self) -> int:
        return self._peak_cached_bytes

    @property
    def peak_cached_clip_count(self) -> int:
        return self._peak_cached_clip_count

    def clear_cache(self) -> None:
        self._cache.clear()
        self._cached_bytes = 0

    def get_clip_indices(self, frame_idx: np.ndarray) -> np.ndarray:
        indices = self._normalize_global_indices(frame_idx)
        return np.asarray(
            np.searchsorted(self.clip_offsets, indices, side="right") - 1,
            dtype=np.int32,
        )

    def make_motion_data_buffer(self, num_frames: int) -> LazySonicMotionData:
        if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 0:
            raise ValueError(f"num_frames must be a non-negative integer, got {num_frames!r}")
        return LazySonicMotionData(
            joint_pos=np.empty((num_frames, self.num_joints), dtype=np.float32),
            joint_vel=np.empty((num_frames, self.num_joints), dtype=np.float32),
            body_pos_w=np.empty((num_frames, self.num_bodies, 3), dtype=np.float32),
            body_quat_w=np.empty((num_frames, self.num_bodies, 4), dtype=np.float32),
            body_lin_vel_w=np.empty((num_frames, self.num_bodies, 3), dtype=np.float32),
            body_ang_vel_w=np.empty((num_frames, self.num_bodies, 3), dtype=np.float32),
        )

    def get_motion_at_frame(
        self,
        frame_idx: np.ndarray,
        out: MotionData | None = None,
    ) -> MotionData:
        indices = self._normalize_global_indices(frame_idx)
        if out is None:
            out = self.make_motion_data_buffer(int(indices.size))
        gathered = self.gather_fields(_REQUIRED_FIELDS, indices)
        np.copyto(out.joint_pos, gathered["joint_pos"])
        np.copyto(out.joint_vel, gathered["joint_vel"])
        np.copyto(out.body_pos_w, gathered["body_pos_w"])
        np.copyto(out.body_quat_w, gathered["body_quat_w"])
        np.copyto(out.body_lin_vel_w, gathered["body_lin_vel_w"])
        np.copyto(out.body_ang_vel_w, gathered["body_ang_vel_w"])
        return out

    def gather_fields(
        self,
        fields: Sequence[str],
        global_indices: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Gather requested manifest fields from the rank-local frame space."""

        if isinstance(fields, (str, bytes)) or not isinstance(fields, Sequence):
            raise TypeError("fields must be a sequence of field names")
        names = tuple(fields)
        if len(set(names)) != len(names):
            raise ValueError("fields contains duplicate names")
        unavailable = sorted(set(names).difference(self.available_fields))
        if unavailable:
            raise KeyError(f"SONIC motion fields are not available: {unavailable}")
        indices = self._normalize_global_indices(global_indices)
        results = {
            name: np.empty((indices.size, *self._field_shapes[name]), dtype=np.float32)
            for name in names
        }
        if indices.size == 0 or not names:
            return results
        clip_indices = np.searchsorted(self.clip_offsets, indices, side="right") - 1
        for clip_index in np.unique(clip_indices):
            rows = np.flatnonzero(clip_indices == clip_index)
            local_indices = indices[rows] - self.clip_offsets[clip_index]
            clip = self._get_clip(int(clip_index), names)
            for name in names:
                results[name][rows] = clip.arrays[name][local_indices]
        return results

    def _normalize_global_indices(self, value: np.ndarray) -> np.ndarray:
        raw = np.asarray(value)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise TypeError("global frame indices must be a one-dimensional integer array")
        indices = raw.astype(np.int64, copy=False)
        if np.any(indices < 0):
            indices = np.where(indices < 0, indices + self.num_frames, indices)
        if np.any((indices < 0) | (indices >= self.num_frames)):
            raise IndexError("SONIC global frame index out of bounds")
        return indices

    def _get_clip(self, clip_index: int, fields: tuple[str, ...]) -> _DecodedClip:
        cached = self._cache.get(clip_index)
        if cached is not None:
            missing = tuple(name for name in fields if name not in cached.arrays)
            if missing:
                decoded = self._decode_clip(clip_index, missing)
                merged = _DecodedClip(
                    arrays={**cached.arrays, **decoded.arrays},
                    nbytes=cached.nbytes + decoded.nbytes,
                )
                # The merged field set is needed for this gather, but may not
                # fit the byte budget.  Keep the old entry in that case so the
                # resident cache remains bounded; ``merged`` is a transient
                # miss result and is released after the caller's gather.
                if merged.nbytes <= self.cache_max_bytes:
                    cached = merged
                    self._cache[clip_index] = cached
                    self._cached_bytes += decoded.nbytes
                    self._peak_cached_bytes = max(self._peak_cached_bytes, self._cached_bytes)
                else:
                    cached = merged
            self._cache.move_to_end(clip_index)
            return cached
        decoded = self._decode_clip(clip_index, fields)
        self._loaded_clip_count += 1
        if self.cache_size == 0 or decoded.nbytes > self.cache_max_bytes:
            return decoded
        while self._cache and (
            len(self._cache) >= self.cache_size
            or self._cached_bytes + decoded.nbytes > self.cache_max_bytes
        ):
            _, evicted = self._cache.popitem(last=False)
            self._cached_bytes -= evicted.nbytes
        self._cache[clip_index] = decoded
        self._cached_bytes += decoded.nbytes
        self._peak_cached_clip_count = max(self._peak_cached_clip_count, len(self._cache))
        self._peak_cached_bytes = max(self._peak_cached_bytes, self._cached_bytes)
        return decoded

    def _decode_clip(self, clip_index: int, fields: tuple[str, ...]) -> _DecodedClip:
        path = self.motion_files[clip_index]
        frame_count = int(self.clip_lengths[clip_index])
        arrays: dict[str, np.ndarray] = {}
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(set(fields).difference(archive.files))
            if missing:
                raise SonicMotionManifestError(f"clip {path!r} is missing fields {missing}")
            if "fps" not in archive.files:
                raise SonicMotionManifestError(f"clip {path!r} is missing scalar fps metadata")
            clip_fps = np.asarray(archive["fps"])
            if clip_fps.ndim != 0 or int(clip_fps) != self.fps:
                raise SonicMotionManifestError(
                    f"clip {path!r} fps metadata differs from manifest fps={self.fps}"
                )
            for name in fields:
                source = np.asarray(archive[name])
                expected_shape = (frame_count, *self._source_field_shape(name))
                if source.shape != expected_shape:
                    raise SonicMotionManifestError(
                        f"clip {path!r} field {name!r} has shape {source.shape}, "
                        f"expected {expected_shape}"
                    )
                if source.dtype != np.dtype(np.float32):
                    raise SonicMotionManifestError(
                        f"clip {path!r} field {name!r} has dtype {source.dtype}, expected float32"
                    )
                if name in _JOINT_FIELDS and self._joint_permutation is not None:
                    source = source[:, self._joint_permutation]
                elif name in _BODY_FIELDS and self._body_permutation is not None:
                    source = source[:, self._body_permutation]
                if not source.flags.c_contiguous:
                    source = np.ascontiguousarray(source)
                source.setflags(write=False)
                arrays[name] = source
        return _DecodedClip(arrays, sum(array.nbytes for array in arrays.values()))

    def _source_field_shape(self, name: str) -> tuple[int, ...]:
        if name in _JOINT_FIELDS:
            return (len(self.joint_order),)
        if name == "body_quat_w":
            return (len(self.body_order), 4)
        if name in _BODY_FIELDS:
            return (len(self.body_order), 3)
        return self._field_shapes[name]


__all__ = [
    "BoundedLazySonicMotionLoader",
    "LazyMotionFieldSpec",
    "LazySonicMotionData",
]
