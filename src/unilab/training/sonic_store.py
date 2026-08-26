"""Immutable, rank-local SONIC motion store.

The manifest is validated on the cold path. Fields can remain array-like lazy
views over a bounded clip cache, or a SONIC owner can cold-materialize its
rollout-critical fields into one contiguous rank-local array each. Both modes
keep PKL/JSON/XML handling out of ``reset`` and ``step``; the latter trades a
bounded, explicit host-RAM reservation for direct O(N) frame gathers.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence, cast

import numpy as np

from unilab.training.sonic_motion import (
    GLOBAL_MMAP_STORE_SCHEMA,
    GLOBAL_MMAP_STORE_VERSION,
    GLOBAL_MMAP_TRUSTED_RECEIPT_SCHEMA,
    GLOBAL_MMAP_TRUSTED_RECEIPT_VERSION,
    GlobalMmapMaterializationReport,
    MotionManifest,
    preflight_motion_manifest,
    resolve_manifest_clip_path,
    sha256_file,
    validate_motion_manifest,
)


@dataclass
class SonicMotionData:
    """MotionLoader-compatible frame batch backed by a materialized store."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray


_JOINT_FIELDS = {
    "joint_pos",
    "joint_vel",
    "joint_acc",
    "dof_pos",
    "dof_vel",
}
_BODY_FIELDS = {"body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"}


class _LazyClipArrays:
    """Bounded clip cache used by :class:`SonicMotionStore`.

    A manifest is validated before this object is created.  The cache therefore
    only opens an already validated clip and never reparses the manifest.  NPZ
    members are decoded one clip at a time; NPY members are opened as read-only
    memory maps by :func:`_load_clip`.  Keeping the cache at clip granularity is
    important because a SONIC frame gather usually touches a small number of
    clips while the complete corpus can be tens of gigabytes.
    """

    def __init__(
        self,
        paths: Sequence[Path],
        fields: set[str],
        *,
        joint_permutation: np.ndarray | None,
        body_permutation: np.ndarray | None,
        frame_counts: Sequence[int],
        cache_size: int,
    ) -> None:
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError(f"cache_size must be a non-negative integer, got {cache_size!r}")
        self.paths = tuple(paths)
        self.fields = frozenset(fields)
        self.joint_permutation = joint_permutation
        self.body_permutation = body_permutation
        self.frame_counts = tuple(int(count) for count in frame_counts)
        if len(self.frame_counts) != len(self.paths):
            raise ValueError("frame_counts must have one entry per clip")
        if any(count <= 0 for count in self.frame_counts):
            raise ValueError("frame_counts must contain positive values")
        self.cache_size = cache_size
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self.load_count = 0

    @property
    def cached_clip_count(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        """Release decoded clips while retaining the immutable metadata."""

        self._cache.clear()

    def _load(self, clip_index: int) -> dict[str, np.ndarray]:
        arrays = _load_clip(self.paths[clip_index], set(self.fields))
        expected_frames = self.frame_counts[clip_index]
        for name, array in arrays.items():
            if array.ndim == 0 or array.shape[0] != expected_frames:
                actual_frames = int(array.shape[0]) if array.ndim else 0
                raise ValueError(
                    f"clip {clip_index} field {name!r} has {actual_frames} frames; "
                    f"expected {expected_frames}"
                )
        for array in arrays.values():
            array.setflags(write=False)
        self.load_count += 1
        return arrays

    def get_clip(self, clip_index: int) -> dict[str, np.ndarray]:
        if clip_index < 0 or clip_index >= len(self.paths):
            raise IndexError(f"clip index out of range: {clip_index}")
        cached = self._cache.get(clip_index)
        if cached is not None:
            self._cache.move_to_end(clip_index)
            return cached
        arrays = self._load(clip_index)
        if self.cache_size:
            self._cache[clip_index] = arrays
            self._cache.move_to_end(clip_index)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return arrays

    def field_shape(self, field_name: str) -> tuple[int, ...]:
        """Return the post-permutation shape excluding the frame axis."""

        source = self.get_clip(0)[field_name]
        shape = tuple(int(extent) for extent in source.shape[1:])
        if field_name in _JOINT_FIELDS and self.joint_permutation is not None:
            if not shape:
                raise ValueError(f"joint field {field_name!r} has no joint axis")
            shape = (*shape[:-1], int(self.joint_permutation.size))
        if field_name in _BODY_FIELDS and self.body_permutation is not None:
            if len(shape) < 1:
                raise ValueError(f"body field {field_name!r} has no body axis")
            shape = (int(self.body_permutation.size), *shape[1:])
        return shape

    def rows(
        self,
        field_name: str,
        clip_index: int,
        local_indices: np.ndarray,
    ) -> np.ndarray:
        """Read rows and apply the cold-path joint/body permutation."""

        return self._rows_from_clip(self.get_clip(clip_index), field_name, local_indices)

    def _rows_from_clip(
        self,
        arrays: Mapping[str, np.ndarray],
        field_name: str,
        local_indices: np.ndarray,
    ) -> np.ndarray:
        source = arrays[field_name]
        result = np.asarray(source[local_indices])
        if field_name in _JOINT_FIELDS and self.joint_permutation is not None:
            result = np.take(result, self.joint_permutation, axis=-1)
        if field_name in _BODY_FIELDS and self.body_permutation is not None:
            result = np.take(result, self.body_permutation, axis=1)
        return result

    def gather(
        self, field_name: str, flat_indices: np.ndarray, clip_offsets: np.ndarray
    ) -> np.ndarray:
        """Gather global rows without constructing a concatenated field."""

        return self.gather_fields((field_name,), flat_indices, clip_offsets)[field_name]

    def gather_fields(
        self,
        field_names: Sequence[str],
        flat_indices: np.ndarray,
        clip_offsets: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Gather several fields while opening each touched clip only once."""

        names = tuple(dict.fromkeys(str(name) for name in field_names))
        unknown = set(names).difference(self.fields)
        if unknown:
            raise KeyError(f"unknown SONIC motion fields: {sorted(unknown)}")
        flat_indices = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
        if flat_indices.size == 0:
            first_clip = self.get_clip(0)
            return {
                name: np.empty((0, *self.field_shape(name)), dtype=first_clip[name].dtype)
                for name in names
            }
        clip_indices = np.searchsorted(clip_offsets, flat_indices, side="right") - 1
        clip_indices = np.clip(clip_indices, 0, len(self.paths) - 1)
        local_indices = flat_indices - clip_offsets[clip_indices]
        results: dict[str, np.ndarray] = {}
        # Grouping by clip minimizes cache churn and NPZ decompression when a
        # rollout batch contains many frames from the same motion clip.  Do
        # not find each group by re-scanning ``clip_indices``: a large
        # distributed rollout can touch thousands of clips, making that
        # approach quadratic in the number of touched clips.  A stable sort
        # produces contiguous groups in O(N log N), while indexing the result
        # by the original positions keeps the caller-visible row order intact.
        sorted_positions = np.argsort(clip_indices, kind="stable")
        sorted_clip_indices = clip_indices[sorted_positions]
        group_starts = np.flatnonzero(
            np.concatenate(
                (
                    np.ones(1, dtype=bool),
                    sorted_clip_indices[1:] != sorted_clip_indices[:-1],
                )
            )
        )
        for group_number, start in enumerate(group_starts):
            end = (
                int(group_starts[group_number + 1])
                if group_number + 1 < group_starts.size
                else sorted_positions.size
            )
            positions = sorted_positions[start:end]
            clip_index = sorted_clip_indices[start]
            arrays = self.get_clip(int(clip_index))
            local = np.asarray(local_indices[positions], dtype=np.int64)
            for name in names:
                rows = self._rows_from_clip(arrays, name, local)
                result = results.get(name)
                if result is None:
                    result = np.empty((flat_indices.size, *rows.shape[1:]), dtype=rows.dtype)
                    results[name] = result
                result[positions] = rows
        return results


class _LazyFieldView:
    """Array-like global field view backed by a bounded clip cache."""

    __array_priority__ = 1000

    def __init__(
        self,
        store: "SonicMotionStore | None",
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
    ) -> None:
        self._store = store
        self.name = name
        self.shape = shape
        self.dtype = dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    def __len__(self) -> int:
        return self.shape[0]

    def __array__(self, dtype: Any = None) -> np.ndarray:
        if self._store is None:  # pragma: no cover - construction guard
            raise RuntimeError("lazy motion field is not attached to a store")
        indices = np.arange(self.shape[0], dtype=np.int64)
        result = self._store.gather(self.name, indices)
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        return result

    def __getitem__(self, key: Any) -> Any:
        if self._store is None:  # pragma: no cover - construction guard
            raise RuntimeError("lazy motion field is not attached to a store")
        return self._store._field_getitem(self.name, key)

    def take(
        self,
        indices: Any,
        axis: int | None = None,
        out: np.ndarray | None = None,
        mode: str = "clip",
    ) -> np.ndarray:
        if axis is None:
            # ``np.take(..., axis=None)`` indexes the flattened array.  There
            # is no clip-local shortcut for that operation, so materialize
            # only when a caller explicitly requests NumPy's flattening
            # semantics.
            materialized = np.asarray(self)
            result = np.take(materialized, indices, axis=None, mode=cast(Any, mode))
            if out is not None:
                np.copyto(out, result, casting="unsafe")
                return out
            return result
        if axis != 0:
            # There is no efficient global view for a non-frame axis; callers
            # can explicitly request a materialized array with np.asarray.
            materialized = np.asarray(self)
            result = np.take(materialized, indices, axis=axis, mode=cast(Any, mode))
            if out is not None:
                np.copyto(out, result, casting="unsafe")
                return out
            return result
        index_array = np.asarray(indices, dtype=np.int64)
        if mode == "wrap":
            index_array = np.mod(index_array, self.shape[0])
        elif mode not in {"clip", "raise"}:
            raise ValueError(f"unsupported take mode: {mode!r}")
        if mode == "raise" and np.any((index_array < 0) | (index_array >= self.shape[0])):
            raise IndexError("SONIC motion frame index out of bounds")
        if self._store is None:  # pragma: no cover - construction guard
            raise RuntimeError("lazy motion field is not attached to a store")
        result = self._store.gather(self.name, index_array)
        if out is not None:
            np.copyto(out, result, casting="unsafe")
            return out
        return result

    def __array_function__(
        self,
        function: Any,
        types: tuple[Any, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if function is np.take:
            # ``args[0]`` is the array object itself; do not forward it as
            # the indices argument.
            return self.take(*args[1:], **kwargs)
        if function is np.unique:
            return np.unique(np.asarray(self), *args[1:], **kwargs)
        # Other NumPy functions retain the legacy array-like surface by
        # explicitly materializing this field.  Frame gathers and indexing do
        # not take this path, so the bounded cache remains effective in the
        # rollout hot path.
        converted_args = tuple(np.asarray(arg) if arg is self else arg for arg in args)
        converted_kwargs = {
            key: (np.asarray(value) if value is self else value) for key, value in kwargs.items()
        }
        return function(*converted_args, **converted_kwargs)

    def __repr__(self) -> str:
        return f"LazyMotionField(name={self.name!r}, shape={self.shape}, dtype={self.dtype})"


@dataclass(frozen=True)
class SonicMotionStore:
    """Read-only motion fields aligned with ``MotionSampler``.

    ``clip_offsets`` and ``clip_end_frames`` use the same global-frame layout
    as :class:`unilab.envs.motion_tracking.common.motion_loader.MotionLoader`.
    Optional fields are represented by ``None`` and are replaced by zeros by
    the environment's modality builder. ``arrays`` contains a mixture of lazy
    array-like views and cold-materialized read-only NumPy arrays for stores
    loaded by :func:`load_sonic_motion_store`; callers that construct this
    dataclass manually may still provide ordinary NumPy arrays.
    """

    manifest: MotionManifest
    arrays: Mapping[str, Any]
    clip_offsets: np.ndarray
    clip_lengths: np.ndarray
    _lazy_backend: _LazyClipArrays | None = dataclass_field(default=None, repr=False, compare=False)

    @property
    def num_frames(self) -> int:
        return int(self.clip_lengths.sum())

    @property
    def clip_end_frames(self) -> np.ndarray:
        return self.clip_offsets + self.clip_lengths - 1

    @property
    def num_joints(self) -> int:
        return len(self.manifest.joint_order)

    @property
    def num_bodies(self) -> int:
        return len(self.manifest.body_order)

    @property
    def cache_size(self) -> int | None:
        """Maximum number of decoded clips retained by this store."""

        return None if self._lazy_backend is None else self._lazy_backend.cache_size

    @property
    def cached_clip_count(self) -> int:
        """Number of clips currently resident in the bounded cache."""

        return 0 if self._lazy_backend is None else self._lazy_backend.cached_clip_count

    @property
    def loaded_clip_count(self) -> int:
        """Cumulative clip-open count, useful for cache/throughput diagnostics."""

        return 0 if self._lazy_backend is None else self._lazy_backend.load_count

    def clear_cache(self) -> None:
        """Release decoded clip arrays while keeping manifest metadata."""

        if self._lazy_backend is not None:
            self._lazy_backend.clear()

    def gather(self, field: str, indices: np.ndarray, *, default: int = 0) -> np.ndarray:
        """Gather a field at global frame indices, clipping invalid rows."""

        result = self.gather_fields((field,), indices)[field]
        if default and result.ndim == 1:
            result = result.reshape(-1, 1)
        return result

    def gather_fields(
        self,
        fields: Sequence[str],
        indices: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Gather multiple frame-aligned fields with one clip-cache traversal."""

        indices = np.asarray(indices, dtype=np.int64)
        original_shape = indices.shape
        flat_indices = indices.reshape(-1)
        names = tuple(dict.fromkeys(str(field) for field in fields))
        lazy_names = tuple(
            name for name in names if isinstance(self.arrays.get(name), _LazyFieldView)
        )
        gathered: dict[str, np.ndarray] = {}
        if self._lazy_backend is not None and lazy_names:
            safe = np.clip(flat_indices, 0, max(0, self.num_frames - 1))
            gathered.update(self._lazy_backend.gather_fields(lazy_names, safe, self.clip_offsets))
        results: dict[str, np.ndarray] = {}
        for name in names:
            source = self.arrays.get(name)
            if name in gathered:
                result = gathered[name]
            elif source is None:
                results[name] = np.zeros(
                    (*original_shape, *_field_default_shape(name, self)),
                    dtype=np.float32,
                )
                continue
            else:
                # Cold-materialized fields are already in effective
                # joint/body order. Do not route them through the lazy backend
                # (which would decode clips and apply its permutation again).
                # Flatten then reshape to preserve the lazy path's scalar and
                # N-D index semantics while using direct O(N) NumPy take.
                safe = np.clip(flat_indices, 0, max(0, source.shape[0] - 1))
                result = np.take(source, safe, axis=0)
            if original_shape == ():
                result = result[0]
            else:
                result_shape = (*original_shape, *result.shape[1:])
                result = result.reshape(result_shape)
            results[name] = result
        return results

    def future_indices(self, frame_indices: np.ndarray, offsets: Sequence[int]) -> np.ndarray:
        """Return clip-aware future indices with shape ``(N, F)``."""

        frames = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        offsets_array = np.asarray(tuple(int(item) for item in offsets), dtype=np.int64)
        if offsets_array.ndim != 1 or offsets_array.size == 0:
            raise ValueError("offsets must be a non-empty one-dimensional sequence")
        clip_idx = np.searchsorted(self.clip_offsets, frames, side="right") - 1
        clip_idx = np.clip(clip_idx, 0, len(self.clip_lengths) - 1)
        starts = self.clip_offsets[clip_idx]
        ends = self.clip_end_frames[clip_idx]
        return np.clip(frames[:, None] + offsets_array[None, :], starts[:, None], ends[:, None])

    def _field_getitem(self, name: str, key: Any) -> Any:
        """Implement ndarray-like indexing without materializing all clips."""

        if not isinstance(key, tuple):
            frame_key, trailing = key, ()
        else:
            if not key:
                return np.asarray(self.arrays[name])
            frame_key, trailing = key[0], key[1:]
        if frame_key is Ellipsis:
            frame_key = slice(None)
        if isinstance(frame_key, (int, np.integer)):
            index = int(frame_key)
            if index < 0:
                index += self.num_frames
            if index < 0 or index >= self.num_frames:
                raise IndexError("SONIC motion frame index out of bounds")
            result = self.gather(name, np.asarray([index], dtype=np.int64))[0]
            return result[trailing] if trailing else result
        if isinstance(frame_key, slice):
            frame_indices = np.arange(self.num_frames, dtype=np.int64)[frame_key]
        else:
            selector = np.asarray(frame_key)
            if selector.dtype == bool:
                if selector.shape != (self.num_frames,):
                    raise IndexError("SONIC boolean frame index has the wrong shape")
                frame_indices = np.flatnonzero(selector)
            else:
                frame_indices = selector.astype(np.int64, copy=False)
                frame_indices = np.where(
                    frame_indices < 0, frame_indices + self.num_frames, frame_indices
                )
                if np.any((frame_indices < 0) | (frame_indices >= self.num_frames)):
                    raise IndexError("SONIC motion frame index out of bounds")
        result = self.gather(name, frame_indices)
        if trailing:
            result = result[(..., *trailing)]
        return result


class SonicMotionLoader:
    """MotionLoader-compatible view over an already materialized SONIC store."""

    def __init__(self, store: SonicMotionStore) -> None:
        self.store = store
        self.fps = int(round(store.manifest.clips[0].fps))
        self.num_joints = store.num_joints
        self.num_bodies = store.num_bodies
        self.clip_lengths = np.asarray(store.clip_lengths, dtype=np.int32)
        self.num_clips = int(self.clip_lengths.size)
        self.clip_offsets = np.asarray(store.clip_offsets, dtype=np.int32)
        self.clip_end_frames = np.asarray(store.clip_end_frames, dtype=np.int32)
        self.num_frames = store.num_frames

    def get_clip_indices(self, frame_idx: np.ndarray) -> np.ndarray:
        return np.asarray(
            np.searchsorted(self.clip_offsets, np.asarray(frame_idx), side="right") - 1,
            dtype=np.int32,
        )

    def make_motion_data_buffer(self, num_frames: int) -> SonicMotionData:
        return SonicMotionData(
            joint_pos=np.empty((num_frames, self.num_joints), dtype=np.float32),
            joint_vel=np.empty((num_frames, self.num_joints), dtype=np.float32),
            body_pos_w=np.empty((num_frames, self.num_bodies, 3), dtype=np.float32),
            body_quat_w=np.empty((num_frames, self.num_bodies, 4), dtype=np.float32),
            body_lin_vel_w=np.empty((num_frames, self.num_bodies, 3), dtype=np.float32),
            body_ang_vel_w=np.empty((num_frames, self.num_bodies, 3), dtype=np.float32),
        )

    def get_motion_at_frame(
        self, frame_idx: np.ndarray, out: SonicMotionData | None = None
    ) -> SonicMotionData:
        # MotionLoader callers use a flat frame batch.  Flattening here also
        # keeps the preallocated ``out`` contract deterministic if a caller
        # supplies a one-dimensional view with an unusual stride.
        indices = np.asarray(frame_idx, dtype=np.int64).reshape(-1)
        if out is None:
            out = self.make_motion_data_buffer(int(indices.size))
        for name in (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ):
            if name not in self.store.arrays:
                raise ValueError(f"SONIC store is missing required field {name!r}")
        names = (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )
        gathered = self.store.gather_fields(names, indices)
        for name in names:
            np.copyto(getattr(out, name), gathered[name], casting="unsafe")
        return out


def _field_default_shape(field: str, store: SonicMotionStore) -> tuple[int, ...]:
    if field in {"joint_pos", "joint_vel", "joint_acc", "dof_pos", "dof_vel"}:
        return (store.num_joints,)
    if field in {"body_pos_w", "body_lin_vel_w", "body_ang_vel_w"}:
        return (store.num_bodies, 3)
    if field in {"body_quat_w", "smpl_root_quat_w"}:
        return (store.num_bodies if field == "body_quat_w" else 1, 4)
    if field in {"smpl_joints", "smpl_joints_w"}:
        return (24, 3)
    if field in {"teleop_pos_w"}:
        return (3, 3)
    if field in {"teleop_quat_w"}:
        return (3, 4)
    return (0,)


def _load_clip(path: Path, fields: set[str]) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".npy":
        if len(fields) != 1:
            raise ValueError("a .npy SONIC clip must declare exactly one field")
        return {next(iter(fields)): np.load(path, mmap_mode="r", allow_pickle=False)}
    if path.suffix.lower() != ".npz":
        raise ValueError(f"unsupported SONIC clip format: {path.suffix}")
    with np.load(path, mmap_mode="r", allow_pickle=False) as archive:
        archive_fields = set(archive.files)
        # ``fps`` is scalar clip metadata retained by the converter for
        # MotionLoader compatibility; it is intentionally absent from the
        # frame-aligned manifest field set.
        metadata_fields = {"fps"}
        if "fps" in archive_fields and np.asarray(archive["fps"]).ndim != 0:
            raise ValueError("SONIC clip metadata 'fps' must be scalar")
        if archive_fields - fields - metadata_fields or fields - archive_fields:
            missing = sorted(fields - archive_fields)
            extra = sorted(archive_fields - fields - metadata_fields)
            raise ValueError(f"clip fields disagree; missing={missing}, extra={extra}")
        # ``np.load`` returns lazy arrays for mmap-compatible members.  A copy
        # is not needed here; the archive remains valid after this context for
        # normal npz files only when arrays are materialized, so make that
        # ownership explicit.  Large NPY shards can still be mmap'ed directly.
        return {name: np.asarray(archive[name]) for name in archive.files if name in fields}


def _load_selected_clip_fields(path: Path, fields: Sequence[str]) -> dict[str, np.ndarray]:
    """Read selected cold-path fields without weakening the full clip contract."""

    selected = tuple(fields)
    if path.suffix.lower() == ".npy":
        if len(selected) != 1:
            raise ValueError("a .npy SONIC clip can materialize exactly one field")
        return {selected[0]: np.load(path, mmap_mode="r", allow_pickle=False)}
    if path.suffix.lower() != ".npz":
        raise ValueError(f"unsupported SONIC clip format: {path.suffix}")
    with np.load(path, mmap_mode="r", allow_pickle=False) as archive:
        missing = sorted(set(selected).difference(archive.files))
        if missing:
            raise ValueError(f"clip is missing selected global mmap fields: {missing}")
        return {name: np.asarray(archive[name]) for name in selected}


def _normalize_hot_fields(
    hot_fields: Sequence[str],
    *,
    available_fields: set[str],
) -> tuple[str, ...]:
    """Validate an explicit cold-materialization request before clip I/O."""

    if isinstance(hot_fields, (str, bytes)) or not isinstance(hot_fields, Sequence):
        raise ValueError("hot_fields must be a sequence of SONIC field names")
    names = tuple(hot_fields)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("hot_fields must contain non-empty string field names")
    if len(set(names)) != len(names):
        raise ValueError("hot_fields contains duplicate field names")
    unknown = sorted(set(names).difference(available_fields))
    if unknown:
        raise ValueError(f"unknown SONIC hot fields: {unknown}")
    return names


def _copy_hot_field(
    source: np.ndarray,
    destination: np.ndarray,
    *,
    name: str,
    joint_permutation: np.ndarray | None,
    body_permutation: np.ndarray | None,
) -> None:
    """Copy one checked clip member into final effective-order storage."""

    if name in _JOINT_FIELDS and joint_permutation is not None:
        np.take(source, joint_permutation, axis=-1, out=destination)
        return
    if name in _BODY_FIELDS and body_permutation is not None:
        np.take(source, body_permutation, axis=1, out=destination)
        return
    np.copyto(destination, source, casting="no")


def _validate_hot_source(
    source: np.ndarray,
    *,
    name: str,
    clip_index: int,
    frame_count: int,
    source_shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> None:
    """Defend a cold copy against a clip changed after its preflight."""

    if np.dtype(source.dtype) != dtype:
        raise ValueError(
            f"clip {clip_index} hot field {name!r} dtype {source.dtype} does not match {dtype}"
        )
    expected_shape = (frame_count, *source_shape)
    if tuple(source.shape) != expected_shape:
        raise ValueError(
            f"clip {clip_index} hot field {name!r} shape {tuple(source.shape)} "
            f"does not match {expected_shape}"
        )


def _materialize_hot_fields(
    *,
    paths: Sequence[Path],
    field_names: Sequence[str],
    clip_offsets: np.ndarray,
    frame_counts: Sequence[int],
    source_shapes: Mapping[str, tuple[int, ...]],
    destination_shapes: Mapping[str, tuple[int, ...]],
    dtypes: Mapping[str, np.dtype[Any]],
    joint_permutation: np.ndarray | None,
    body_permutation: np.ndarray | None,
    all_manifest_fields: set[str],
) -> dict[str, np.ndarray]:
    """Cold-materialize fields, opening each rank-local clip once.

    Individual NPZ members are decoded only while their archive is open, so
    source arrays never accumulate across clips. Outputs remain local until
    every copy succeeds; no partially initialized field is published.
    """

    names = tuple(field_names)
    if not names:
        return {}
    if len(paths) != len(frame_counts) or len(paths) != len(clip_offsets):
        raise ValueError("SONIC hot materialization clip metadata lengths disagree")
    total_frames = int(sum(int(count) for count in frame_counts))
    outputs = {
        name: np.empty((total_frames, *destination_shapes[name]), dtype=dtypes[name], order="C")
        for name in names
    }

    for clip_index, (path, offset, frame_count) in enumerate(
        zip(paths, clip_offsets, frame_counts, strict=True)
    ):
        start = int(offset)
        end = start + int(frame_count)

        def copy_member(name: str, member: Any) -> None:
            source = np.asarray(member)
            _validate_hot_source(
                source,
                name=name,
                clip_index=clip_index,
                frame_count=int(frame_count),
                source_shape=source_shapes[name],
                dtype=dtypes[name],
            )
            _copy_hot_field(
                source,
                outputs[name][start:end],
                name=name,
                joint_permutation=joint_permutation,
                body_permutation=body_permutation,
            )

        suffix = path.suffix.lower()
        if suffix == ".npy":
            if len(all_manifest_fields) != 1 or len(names) != 1:
                raise ValueError("a .npy SONIC clip must declare exactly one field")
            copy_member(names[0], np.load(path, mmap_mode="r", allow_pickle=False))
        elif suffix == ".npz":
            with np.load(path, mmap_mode="r", allow_pickle=False) as archive:
                for name in names:
                    if name not in archive.files:
                        raise ValueError(f"clip {clip_index} is missing hot field {name!r}")
                    copy_member(name, archive[name])
        else:
            raise ValueError(f"unsupported SONIC clip format: {path.suffix}")

    for output in outputs.values():
        # ``np.empty(..., order='C')`` establishes the layout; setting this
        # only after all copies preserves immutable store semantics.
        if not output.flags.c_contiguous:  # pragma: no cover - allocation invariant
            raise ValueError("SONIC hot materialization did not produce C-contiguous data")
        output.setflags(write=False)
    return outputs


def _normalize_global_mmap_fields(
    fields: Sequence[str],
    *,
    available_fields: set[str],
) -> tuple[str, ...]:
    """Validate the explicit fields selected for a global mmap publication."""

    if isinstance(fields, (str, bytes)) or not isinstance(fields, Sequence):
        raise ValueError("global mmap fields must be a sequence of SONIC field names")
    names = tuple(fields)
    if not names:
        raise ValueError("global mmap fields must not be empty")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("global mmap fields must contain non-empty string field names")
    if len(set(names)) != len(names):
        raise ValueError("global mmap fields contains duplicate field names")
    unknown = sorted(set(names).difference(available_fields))
    if unknown:
        raise ValueError(f"unknown SONIC global mmap fields: {unknown}")
    return names


def _effective_field_shape(
    name: str,
    source_shape: tuple[int, ...],
    *,
    joint_permutation: np.ndarray | None,
    body_permutation: np.ndarray | None,
) -> tuple[int, ...]:
    """Return one field's trailing shape after the declared order conversion."""

    if name in _JOINT_FIELDS and joint_permutation is not None:
        if not source_shape:
            raise ValueError(f"joint field {name!r} has no joint axis")
        return (*source_shape[:-1], int(joint_permutation.size))
    if name in _BODY_FIELDS and body_permutation is not None:
        if not source_shape:
            raise ValueError(f"body field {name!r} has no body axis")
        return (int(body_permutation.size), *source_shape[1:])
    return source_shape


def _sidecar_relative_array_path(metadata_path: Path, value: Any) -> Path:
    """Resolve a sidecar array path while rejecting publication-root escapes."""

    if not isinstance(value, str) or not value:
        raise ValueError("global mmap sidecar field path must be a non-empty string")
    relative = Path(value)
    windows_path = PureWindowsPath(value)
    if relative.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError(f"global mmap sidecar field path must be relative: {value!r}")
    if any(part == ".." for part in windows_path.parts):
        raise ValueError(f"global mmap sidecar field path must not traverse parents: {value!r}")
    root = metadata_path.parent
    path = (root / relative).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"global mmap sidecar field path escapes store: {value!r}") from exc
    return path


def _sidecar_order(value: Any, *, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{location} must be a non-empty string list")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{location} must not contain duplicates")
    return result


def _sidecar_shape(value: Any, *, location: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty integer list")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValueError(f"{location} must be a non-negative integer list")
    return tuple(value)


def _sidecar_checksum(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{location} must be a SHA256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{location} must be a SHA256 digest") from exc
    return value.lower()


def _global_mmap_metadata_path(sidecar: str | Path) -> Path:
    """Resolve the explicit sidecar file, allowing its publication directory."""

    candidate = Path(sidecar).expanduser().resolve()
    metadata_path = candidate / "metadata.json" if candidate.is_dir() else candidate
    if not metadata_path.is_file():
        raise ValueError(f"global mmap sidecar metadata does not exist: {metadata_path}")
    return metadata_path


_TRUSTED_RECEIPT_STAT_KEYS = frozenset({"st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"})


def _file_stat_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "st_dev": int(stat.st_dev),
        "st_ino": int(stat.st_ino),
        "st_size": int(stat.st_size),
        "st_mtime_ns": int(stat.st_mtime_ns),
        "st_ctime_ns": int(stat.st_ctime_ns),
    }


def _trusted_receipt_stat(value: Any, *, location: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _TRUSTED_RECEIPT_STAT_KEYS:
        raise ValueError(f"{location} must declare complete file stat identity")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value.values()):
        raise ValueError(f"{location} must contain non-negative integer stat values")
    return {key: int(value[key]) for key in _TRUSTED_RECEIPT_STAT_KEYS}


def _validate_global_mmap_trusted_receipt(
    receipt: str | Path,
    *,
    metadata_path: Path,
    manifest: MotionManifest,
) -> None:
    """Validate an explicit trusted receipt for the no-rehash fast path.

    The receipt binds current NPY inode/stat identity to the manifest and
    sidecar metadata produced by an offline complete audit. It is deliberately
    not a replacement for strict SHA256 verification after a filesystem's
    trust boundary has changed.
    """

    if manifest.manifest_path is None:
        raise ValueError("a file-backed manifest is required for a global mmap trusted receipt")
    receipt_path = Path(receipt).expanduser().resolve()
    if not receipt_path.is_file():
        raise ValueError(f"global mmap trusted receipt does not exist: {receipt_path}")
    try:
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read global mmap trusted receipt: {exc}") from exc
    if not isinstance(data, Mapping) or set(data) != {
        "schema",
        "version",
        "sidecar_metadata_sha256",
        "source_manifest_sha256",
        "field_sha256",
        "field_stat",
    }:
        raise ValueError("global mmap trusted receipt has an invalid schema")
    if data["schema"] != GLOBAL_MMAP_TRUSTED_RECEIPT_SCHEMA:
        raise ValueError("global mmap trusted receipt schema is unsupported")
    if data["version"] != GLOBAL_MMAP_TRUSTED_RECEIPT_VERSION:
        raise ValueError("global mmap trusted receipt version is unsupported")
    if _sidecar_checksum(
        data["sidecar_metadata_sha256"], location="sidecar_metadata_sha256"
    ) != sha256_file(metadata_path):
        raise ValueError("global mmap trusted receipt metadata digest differs")
    if _sidecar_checksum(
        data["source_manifest_sha256"], location="source_manifest_sha256"
    ) != sha256_file(manifest.manifest_path):
        raise ValueError("global mmap trusted receipt source manifest digest differs")
    fields = metadata.get("fields") if isinstance(metadata, Mapping) else None
    if not isinstance(fields, Mapping):
        raise ValueError("global mmap trusted receipt sidecar metadata is invalid")
    expected_checksums = {
        name: entry.get("sha256")
        for name, entry in fields.items()
        if isinstance(name, str) and isinstance(entry, Mapping)
    }
    if len(expected_checksums) != len(fields) or data["field_sha256"] != expected_checksums:
        raise ValueError("global mmap trusted receipt field digests differ")
    field_stats = data["field_stat"]
    if not isinstance(field_stats, Mapping) or set(field_stats) != set(fields):
        raise ValueError("global mmap trusted receipt field stat identities differ")
    for name, checksum in expected_checksums.items():
        _sidecar_checksum(checksum, location=f"field_sha256.{name}")
        entry = fields[name]
        assert isinstance(entry, Mapping)  # guarded while constructing checksums
        path = _sidecar_relative_array_path(metadata_path, entry.get("path"))
        expected_stat = _trusted_receipt_stat(field_stats[name], location=f"field_stat.{name}")
        if expected_stat != _file_stat_identity(path):
            raise ValueError(f"global mmap trusted receipt field {name!r} stat identity differs")


def _load_global_mmap_sidecar(
    sidecar: str | Path,
    *,
    manifest: MotionManifest,
    effective_joint_order: Sequence[str],
    effective_body_order: Sequence[str],
    verify_checksums: bool,
) -> dict[str, np.ndarray]:
    """Validate an immutable global sidecar and open its fields read-only.

    This is deliberately a cold-path operation.  Hashing every selected NPY
    file proves that an already-published store has not been replaced before
    NumPy maps it into a worker.  The source manifest digest binds the global
    frame order and all clip metadata to the sidecar without putting manifest
    parsing in rollout code.
    """

    if manifest.manifest_path is None:
        raise ValueError("a file-backed manifest is required for a global mmap sidecar")
    metadata_path = _global_mmap_metadata_path(sidecar)
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read global mmap sidecar {metadata_path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("global mmap sidecar metadata must be an object")
    allowed = {
        "schema",
        "version",
        "source_manifest",
        "frame_count",
        "effective_joint_order",
        "effective_body_order",
        "fields",
    }
    unknown = set(data).difference(allowed)
    if unknown:
        raise ValueError(f"global mmap sidecar has unknown keys: {sorted(unknown)}")
    if data.get("schema") != GLOBAL_MMAP_STORE_SCHEMA:
        raise ValueError("global mmap sidecar schema is unsupported")
    if data.get("version") != GLOBAL_MMAP_STORE_VERSION:
        raise ValueError("global mmap sidecar version is unsupported")
    source_manifest = data.get("source_manifest")
    if not isinstance(source_manifest, Mapping) or set(source_manifest) != {
        "schema",
        "version",
        "sha256",
    }:
        raise ValueError("global mmap sidecar source_manifest must declare schema/version/sha256")
    if source_manifest["schema"] != manifest.schema or source_manifest["version"] != manifest.version:
        raise ValueError("global mmap sidecar source manifest schema/version differs")
    if _sidecar_checksum(source_manifest["sha256"], location="source_manifest.sha256") != sha256_file(
        manifest.manifest_path
    ):
        raise ValueError("global mmap sidecar source manifest digest differs")
    frame_count = data.get("frame_count")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError("global mmap sidecar frame_count must be a positive integer")
    if frame_count != sum(clip.num_frames for clip in manifest.clips):
        raise ValueError("global mmap sidecar frame_count differs from the source manifest")
    if _sidecar_order(data.get("effective_joint_order"), location="effective_joint_order") != tuple(
        effective_joint_order
    ):
        raise ValueError("global mmap sidecar effective_joint_order differs from the loader")
    if _sidecar_order(data.get("effective_body_order"), location="effective_body_order") != tuple(
        effective_body_order
    ):
        raise ValueError("global mmap sidecar effective_body_order differs from the loader")
    entries = data.get("fields")
    if not isinstance(entries, Mapping) or not entries:
        raise ValueError("global mmap sidecar fields must be a non-empty object")
    manifest_fields = {field.name: field for field in manifest.fields}
    unknown_fields = set(entries).difference(manifest_fields)
    if unknown_fields:
        raise ValueError(f"global mmap sidecar has unknown fields: {sorted(unknown_fields)}")

    arrays: dict[str, np.ndarray] = {}
    for name, entry in entries.items():
        if not isinstance(name, str) or not isinstance(entry, Mapping):
            raise ValueError("global mmap sidecar fields must map field names to metadata objects")
        if set(entry) != {"path", "sha256", "dtype", "shape", "order"}:
            raise ValueError(f"global mmap sidecar field {name!r} has an invalid schema")
        path = _sidecar_relative_array_path(metadata_path, entry["path"])
        if not path.is_file():
            raise ValueError(f"global mmap sidecar field {name!r} does not exist: {path}")
        if path.suffix.lower() != ".npy":
            raise ValueError(f"global mmap sidecar field {name!r} must be an NPY file")
        checksum = _sidecar_checksum(entry["sha256"], location=f"fields.{name}.sha256")
        if verify_checksums and checksum != sha256_file(path):
            raise ValueError(f"global mmap sidecar field {name!r} SHA256 checksum mismatch")
        expected_dtype = np.dtype(manifest_fields[name].dtype)
        try:
            metadata_dtype = np.dtype(entry["dtype"])
        except TypeError as exc:
            raise ValueError(f"global mmap sidecar field {name!r} has an invalid dtype") from exc
        if metadata_dtype != expected_dtype:
            raise ValueError(f"global mmap sidecar field {name!r} dtype differs from manifest")
        shape = _sidecar_shape(entry["shape"], location=f"fields.{name}.shape")
        if shape[0] != frame_count:
            raise ValueError(f"global mmap sidecar field {name!r} frame axis differs")
        if entry["order"] != "C":
            raise ValueError(f"global mmap sidecar field {name!r} must use C order")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"could not mmap global sidecar field {name!r}: {exc}") from exc
        if (
            tuple(array.shape) != shape
            or np.dtype(array.dtype) != metadata_dtype
            or not array.flags.c_contiguous
            or array.flags.writeable
        ):
            raise ValueError(
                f"global mmap sidecar field {name!r} does not match its readonly C-order metadata"
            )
        if name in _JOINT_FIELDS and array.shape[-1] != len(effective_joint_order):
            raise ValueError(f"global mmap sidecar field {name!r} joint order width differs")
        if name in _BODY_FIELDS and array.shape[-2] != len(effective_body_order):
            raise ValueError(f"global mmap sidecar field {name!r} body order width differs")
        array.setflags(write=False)
        arrays[name] = array
    return arrays


def materialize_sonic_global_mmap_store(
    manifest: MotionManifest | str | Path,
    output_dir: str | Path,
    *,
    fields: Sequence[str],
    expected_joint_order: Sequence[str] | None = None,
    expected_body_order: Sequence[str] | None = None,
) -> GlobalMmapMaterializationReport:
    """Publish selected global-frame SONIC fields as immutable C-order NPY files.

    The source manifest is fully preflighted before any output is visible.
    Files are written under a sibling staging directory and the completed
    directory is renamed once metadata and checksums are complete.  The
    resulting ``metadata.json`` is the explicit input accepted by
    :func:`load_sonic_motion_store`'s ``motion_global_mmap_sidecar`` mode.
    """

    parsed = preflight_motion_manifest(manifest, verify_checksums=True, verify_shapes=True)
    if parsed.manifest_path is None:
        raise ValueError("a file-backed manifest is required to materialize a global mmap store")
    source_manifest_digest = sha256_file(parsed.manifest_path)
    field_names = {field.name for field in parsed.fields}
    selected_fields = _normalize_global_mmap_fields(fields, available_fields=field_names)
    joint_permutation = _resolve_order_permutation(
        parsed.joint_order,
        expected_joint_order,
        field="joint_order",
        allow_subset=False,
    )
    body_permutation = _resolve_order_permutation(
        parsed.body_order,
        expected_body_order,
        field="body_order",
        allow_subset=True,
    )
    effective_joint_order = (
        tuple(str(name) for name in expected_joint_order)
        if expected_joint_order is not None
        else parsed.joint_order
    )
    effective_body_order = (
        tuple(str(name) for name in expected_body_order)
        if expected_body_order is not None
        else parsed.body_order
    )
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"global mmap output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    arrays_dir = staging / "arrays"
    arrays_dir.mkdir()
    paths = tuple(
        resolve_manifest_clip_path(parsed.manifest_path, clip.path) for clip in parsed.clips
    )
    frame_counts = tuple(clip.num_frames for clip in parsed.clips)
    frame_count = sum(frame_counts)
    source_shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, np.dtype[Any]] = {}
    outputs: dict[str, np.memmap] = {}
    try:
        first = _load_selected_clip_fields(paths[0], selected_fields)
        for name in selected_fields:
            source = np.asarray(first[name])
            _validate_hot_source(
                source,
                name=name,
                clip_index=0,
                frame_count=frame_counts[0],
                source_shape=tuple(source.shape[1:]),
                dtype=np.dtype(source.dtype),
            )
            source_shapes[name] = tuple(int(size) for size in source.shape[1:])
            dtypes[name] = np.dtype(source.dtype)
            effective_shape = _effective_field_shape(
                name,
                source_shapes[name],
                joint_permutation=joint_permutation,
                body_permutation=body_permutation,
            )
            outputs[name] = np.lib.format.open_memmap(
                arrays_dir / f"{name}.npy",
                mode="w+",
                dtype=dtypes[name],
                shape=(frame_count, *effective_shape),
                fortran_order=False,
            )

        offset = 0
        for clip_index, (path, clip_frames) in enumerate(zip(paths, frame_counts, strict=True)):
            arrays = _load_selected_clip_fields(path, selected_fields)
            end = offset + clip_frames
            for name in selected_fields:
                source = np.asarray(arrays[name])
                _validate_hot_source(
                    source,
                    name=name,
                    clip_index=clip_index,
                    frame_count=clip_frames,
                    source_shape=source_shapes[name],
                    dtype=dtypes[name],
                )
                _copy_hot_field(
                    source,
                    outputs[name][offset:end],
                    name=name,
                    joint_permutation=joint_permutation,
                    body_permutation=body_permutation,
                )
            offset = end
        if offset != frame_count:  # pragma: no cover - manifest frame sum invariant
            raise ValueError("global mmap frame offsets do not cover the source manifest")
        if sha256_file(parsed.manifest_path) != source_manifest_digest:
            raise ValueError("source manifest changed during global mmap materialization")

        field_metadata: dict[str, dict[str, Any]] = {}
        total_bytes = 0
        for name in selected_fields:
            output = outputs[name]
            output.flush()
            path = arrays_dir / f"{name}.npy"
            field_metadata[name] = {
                "path": str(path.relative_to(staging)),
                "sha256": sha256_file(path),
                "dtype": str(output.dtype),
                "shape": [int(size) for size in output.shape],
                "order": "C",
            }
            total_bytes += path.stat().st_size
        outputs.clear()
        metadata_path = staging / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "schema": GLOBAL_MMAP_STORE_SCHEMA,
                    "version": GLOBAL_MMAP_STORE_VERSION,
                    "source_manifest": {
                        "schema": parsed.schema,
                        "version": parsed.version,
                        "sha256": source_manifest_digest,
                    },
                    "frame_count": frame_count,
                    "effective_joint_order": list(effective_joint_order),
                    "effective_body_order": list(effective_body_order),
                    "fields": field_metadata,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        receipt_path = staging / "trusted-receipt.json"
        receipt_path.write_text(
            json.dumps(
                {
                    "schema": GLOBAL_MMAP_TRUSTED_RECEIPT_SCHEMA,
                    "version": GLOBAL_MMAP_TRUSTED_RECEIPT_VERSION,
                    "sidecar_metadata_sha256": sha256_file(metadata_path),
                    "source_manifest_sha256": source_manifest_digest,
                    "field_sha256": {
                        name: metadata["sha256"] for name, metadata in field_metadata.items()
                    },
                    "field_stat": {
                        name: _file_stat_identity(arrays_dir / f"{name}.npy")
                        for name in field_metadata
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
        return GlobalMmapMaterializationReport(
            sidecar_path=destination / "metadata.json",
            trusted_receipt_path=destination / "trusted-receipt.json",
            frame_count=frame_count,
            fields=selected_fields,
            total_bytes=total_bytes,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_sonic_motion_store(
    manifest: MotionManifest | str | Path,
    *,
    verify_checksums: bool = True,
    verify_shapes: bool = True,
    expected_joint_order: Sequence[str] | None = None,
    expected_body_order: Sequence[str] | None = None,
    rank: int = 0,
    world_size: int = 1,
    shard_clips: bool = False,
    cache_size: int = 2,
    hot_fields: Sequence[str] = (),
    motion_global_mmap_sidecar: str | Path | None = None,
    motion_global_mmap_trusted_receipt: str | Path | None = None,
) -> SonicMotionStore:
    """Load and validate a deterministic motion shard on the cold path.

    With ``shard_clips=True`` each distributed rank exposes only its
    round-robin clip subset. ``arrays`` remains compatible with the historic
    mapping API. By default each value is a lazy field view backed by at most
    ``cache_size`` decoded clips. ``hot_fields`` explicitly selects rank-local
    C-contiguous arrays to build on this cold path; their later frame gathers
    use direct ``np.take`` without opening lazy clips. An explicit
    ``motion_global_mmap_sidecar`` opens prebuilt read-only C-order NPY fields
    over the complete manifest frame order. It cannot be combined with
    rank-local clip sharding or resident ``hot_fields``. By default workers
    re-hash the source and selected NPY files. Supplying an explicit immutable
    ``motion_global_mmap_trusted_receipt`` opts into an explicitly trusted
    fast path, which verifies manifest/metadata identity, NPY stat identity,
    and headers only. It is not a later cryptographic re-audit.
    """

    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError(f"rank must be a non-negative integer, got {rank!r}")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError(f"world_size must be a positive integer, got {world_size!r}")
    if rank >= world_size:
        raise ValueError(f"rank={rank} must be less than world_size={world_size}")
    if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
        raise ValueError(f"cache_size must be a non-negative integer, got {cache_size!r}")
    if motion_global_mmap_sidecar in (None, ""):
        motion_global_mmap_sidecar = None
    elif not isinstance(motion_global_mmap_sidecar, (str, Path)):
        raise ValueError("motion_global_mmap_sidecar must be a metadata path or publication directory")
    if motion_global_mmap_trusted_receipt in (None, ""):
        motion_global_mmap_trusted_receipt = None
    elif not isinstance(motion_global_mmap_trusted_receipt, (str, Path)):
        raise ValueError("motion_global_mmap_trusted_receipt must be a receipt path")
    if motion_global_mmap_trusted_receipt is not None and motion_global_mmap_sidecar is None:
        raise ValueError("motion_global_mmap_trusted_receipt requires motion_global_mmap_sidecar")
    if motion_global_mmap_sidecar is not None and shard_clips:
        raise ValueError("motion_global_mmap_sidecar requires shard_clips=False")

    # Parse the complete manifest and validate its immutable schema/order first,
    # but defer clip I/O until after rank-local sharding.  In distributed
    # training the rank shards form a disjoint cover of the corpus, so every
    # consumed clip is still hash/shape checked exactly once instead of once
    # per rank.
    parsed = validate_motion_manifest(manifest)
    if parsed.manifest_path is None:
        raise ValueError("a file-backed manifest is required to materialize a SONIC store")
    joint_permutation = _resolve_order_permutation(
        parsed.joint_order,
        expected_joint_order,
        field="joint_order",
        allow_subset=False,
    )
    body_permutation = _resolve_order_permutation(
        parsed.body_order,
        expected_body_order,
        field="body_order",
        allow_subset=True,
    )
    effective_manifest = parsed
    if expected_joint_order is not None:
        effective_manifest = replace(
            effective_manifest, joint_order=tuple(str(name) for name in expected_joint_order)
        )
    if expected_body_order is not None:
        effective_manifest = replace(
            effective_manifest, body_order=tuple(str(name) for name in expected_body_order)
        )
    field_names = {field.name for field in parsed.fields}
    selected_hot_fields = _normalize_hot_fields(hot_fields, available_fields=field_names)
    if motion_global_mmap_sidecar is not None and selected_hot_fields:
        raise ValueError("motion_global_mmap_sidecar cannot be combined with resident hot_fields")
    if motion_global_mmap_trusted_receipt is not None:
        assert motion_global_mmap_sidecar is not None  # validated above
        _validate_global_mmap_trusted_receipt(
            motion_global_mmap_trusted_receipt,
            metadata_path=_global_mmap_metadata_path(motion_global_mmap_sidecar),
            manifest=parsed,
        )
    global_mmap_arrays = (
        _load_global_mmap_sidecar(
            motion_global_mmap_sidecar,
            manifest=parsed,
            effective_joint_order=effective_manifest.joint_order,
            effective_body_order=effective_manifest.body_order,
            verify_checksums=motion_global_mmap_trusted_receipt is None,
        )
        if motion_global_mmap_sidecar is not None
        else {}
    )
    if shard_clips and world_size > 1:
        selected_indices = tuple(range(rank, len(parsed.clips), world_size))
        if not selected_indices:
            raise ValueError(
                "SONIC clip sharding requires at least one clip per rank; "
                f"got {len(parsed.clips)} clips for world_size={world_size}"
            )
    else:
        selected_indices = tuple(range(len(parsed.clips)))
    selected_clips = tuple(parsed.clips[index] for index in selected_indices)
    effective_manifest = replace(effective_manifest, clips=selected_clips)
    lengths = np.asarray([clip.num_frames for clip in selected_clips], dtype=np.int64)
    offsets = np.zeros_like(lengths)
    if len(offsets) > 1:
        offsets[1:] = np.cumsum(lengths[:-1], dtype=np.int64)
    if motion_global_mmap_trusted_receipt is not None:
        if set(global_mmap_arrays) != field_names:
            missing = sorted(field_names.difference(global_mmap_arrays))
            raise ValueError(
                "motion_global_mmap_trusted_receipt requires sidecar coverage of every "
                f"manifest field; missing={missing}"
            )
        # The explicitly trusted receipt records an offline full-corpus
        # verification. Do not re-hash or open source clips in every torchrun
        # child: all worker metadata and frame gathers come from the immutable
        # sidecar instead.
        return SonicMotionStore(effective_manifest, global_mmap_arrays, offsets, lengths)
    effective_manifest = preflight_motion_manifest(
        effective_manifest,
        verify_checksums=verify_checksums,
        verify_shapes=verify_shapes,
    )
    if global_mmap_arrays and set(global_mmap_arrays) == field_names:
        # Strict mode already completed the source audit above. No lazy field
        # metadata is needed when every field is supplied by the sidecar.
        return SonicMotionStore(effective_manifest, global_mmap_arrays, offsets, lengths)
    paths = tuple(
        resolve_manifest_clip_path(parsed.manifest_path, clip.path) for clip in selected_clips
    )
    lazy_backend = _LazyClipArrays(
        paths,
        field_names,
        joint_permutation=joint_permutation,
        body_permutation=body_permutation,
        frame_counts=tuple(int(item) for item in lengths),
        cache_size=cache_size,
    )
    arrays: dict[str, Any] = {}
    store = SonicMotionStore(effective_manifest, arrays, offsets, lengths, lazy_backend)
    # Read metadata from one clip only.  No field is concatenated and no other
    # clip is decoded until a caller gathers frames from it.
    source_shapes: dict[str, tuple[int, ...]] = {}
    destination_shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, np.dtype[Any]] = {}
    for name in sorted(field_names):
        shape = (int(lengths.sum()), *lazy_backend.field_shape(name))
        source = lazy_backend.get_clip(0)[name]
        source_shapes[name] = tuple(int(extent) for extent in source.shape[1:])
        destination_shapes[name] = tuple(int(extent) for extent in shape[1:])
        dtypes[name] = np.dtype(source.dtype)
        arrays[name] = _LazyFieldView(
            store,
            name,
            shape,
            dtypes[name],
        )
    arrays.update(global_mmap_arrays)
    if selected_hot_fields:
        # The metadata probe above may have retained one complete clip. Release
        # it before allocating multi-GiB resident fields, and leave the cache
        # empty after the cold copy. Unselected fields retain the bounded lazy
        # fallback without duplicating the full corpus in RAM.
        lazy_backend.clear()
        materialized = _materialize_hot_fields(
            paths=paths,
            field_names=selected_hot_fields,
            clip_offsets=offsets,
            frame_counts=lengths,
            source_shapes=source_shapes,
            destination_shapes=destination_shapes,
            dtypes=dtypes,
            joint_permutation=joint_permutation,
            body_permutation=body_permutation,
            all_manifest_fields=field_names,
        )
        arrays.update(materialized)
        lazy_backend.clear()
    return store


def _resolve_order_permutation(
    source: Sequence[str],
    expected: Sequence[str] | None,
    *,
    field: str,
    allow_subset: bool,
) -> np.ndarray | None:
    """Resolve a cold-path name permutation without weakening the hot path."""

    if expected is None:
        return None
    source_names = tuple(str(name) for name in source)
    expected_names = tuple(str(name) for name in expected)
    if len(set(expected_names)) != len(expected_names):
        raise ValueError(f"expected {field} contains duplicate names")
    if not allow_subset and set(source_names) != set(expected_names):
        raise ValueError(f"manifest {field} does not contain the expected names")
    if allow_subset and not set(expected_names).issubset(source_names):
        raise ValueError(f"manifest {field} does not contain all expected names")
    # An explicit canonical order is common for the SONIC owner.  Do not keep
    # an identity index array in that case: applying it with ``np.take`` to
    # every clip-local group would allocate an unnecessary copy in the
    # rollout hot path.  A strict prefix is intentionally *not* treated as
    # identity because it still selects a body subset.
    if source_names == expected_names:
        return None
    return np.asarray([source_names.index(name) for name in expected_names], dtype=np.int64)


__all__ = [
    "SonicMotionData",
    "SonicMotionLoader",
    "SonicMotionStore",
    "load_sonic_motion_store",
    "materialize_sonic_global_mmap_store",
]
