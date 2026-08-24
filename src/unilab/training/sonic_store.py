"""Immutable, rank-local SONIC motion store.

The manifest is validated on the cold path, but frame arrays are *not*
concatenated into one rank-local corpus.  Public fields are array-like lazy
views over a bounded clip cache.  A rollout gather therefore decodes only the
clips touched by that gather and evicts old clips according to the configured
cache limit.  This keeps PKL/JSON/XML handling out of ``reset`` and ``step``
without multiplying the complete corpus by the number of distributed ranks.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np

from unilab.training.sonic_motion import (
    MotionManifest,
    preflight_motion_manifest,
    resolve_manifest_clip_path,
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
        first_arrays = self.get_clip(int(clip_indices[0]))
        first_local = np.asarray([local_indices[0]], dtype=np.int64)
        results: dict[str, np.ndarray] = {}
        for name in names:
            first = self._rows_from_clip(first_arrays, name, first_local)
            results[name] = np.empty((flat_indices.size, *first.shape[1:]), dtype=first.dtype)
        # Grouping by clip minimizes cache churn and NPZ decompression when a
        # rollout batch contains many frames from the same motion clip.
        for clip_index in np.unique(clip_indices):
            positions = np.flatnonzero(clip_indices == clip_index)
            arrays = self.get_clip(int(clip_index))
            local = np.asarray(local_indices[positions], dtype=np.int64)
            for name in names:
                results[name][positions] = self._rows_from_clip(arrays, name, local)
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
    the environment's modality builder.  ``arrays`` contains lazy array-like
    views for stores loaded by :func:`load_sonic_motion_store`; callers that
    construct this dataclass manually may still provide ordinary NumPy arrays.
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
        present = tuple(name for name in names if name in self.arrays)
        gathered: dict[str, np.ndarray] = {}
        if self._lazy_backend is not None and present:
            safe = np.clip(flat_indices, 0, max(0, self.num_frames - 1))
            gathered.update(self._lazy_backend.gather_fields(present, safe, self.clip_offsets))
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
                safe = np.clip(indices, 0, max(0, source.shape[0] - 1))
                results[name] = np.asarray(source[safe])
                continue
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
) -> SonicMotionStore:
    """Load and validate a deterministic motion shard on the cold path.

    With ``shard_clips=True`` each distributed rank exposes only its
    round-robin clip subset.  ``arrays`` remains compatible with the historic
    mapping API, but each value is a lazy field view backed by at most
    ``cache_size`` decoded clips.  Calling ``np.asarray`` on one field is an
    explicit request to materialize that field; ordinary frame gathers stay
    bounded.
    """

    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError(f"rank must be a non-negative integer, got {rank!r}")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError(f"world_size must be a positive integer, got {world_size!r}")
    if rank >= world_size:
        raise ValueError(f"rank={rank} must be less than world_size={world_size}")
    if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
        raise ValueError(f"cache_size must be a non-negative integer, got {cache_size!r}")

    parsed = preflight_motion_manifest(
        manifest,
        verify_checksums=verify_checksums,
        verify_shapes=verify_shapes,
    )
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
    paths = tuple(
        resolve_manifest_clip_path(parsed.manifest_path, clip.path) for clip in selected_clips
    )
    lengths = np.asarray([clip.num_frames for clip in selected_clips], dtype=np.int64)
    offsets = np.zeros_like(lengths)
    if len(offsets) > 1:
        offsets[1:] = np.cumsum(lengths[:-1], dtype=np.int64)
    lazy_backend = _LazyClipArrays(
        paths,
        field_names,
        joint_permutation=joint_permutation,
        body_permutation=body_permutation,
        frame_counts=tuple(int(item) for item in lengths),
        cache_size=cache_size,
    )
    arrays: dict[str, _LazyFieldView] = {}
    store = SonicMotionStore(effective_manifest, arrays, offsets, lengths, lazy_backend)
    # Read metadata from one clip only.  No field is concatenated and no other
    # clip is decoded until a caller gathers frames from it.
    for name in sorted(field_names):
        shape = (int(lengths.sum()), *lazy_backend.field_shape(name))
        source = lazy_backend.get_clip(0)[name]
        arrays[name] = _LazyFieldView(
            store,
            name,
            shape,
            np.dtype(source.dtype),
        )
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
    return np.asarray([source_names.index(name) for name in expected_names], dtype=np.int64)


__all__ = [
    "SonicMotionData",
    "SonicMotionLoader",
    "SonicMotionStore",
    "load_sonic_motion_store",
]
