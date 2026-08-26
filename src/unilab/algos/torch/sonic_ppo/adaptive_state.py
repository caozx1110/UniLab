"""SONIC-private adaptive sampler state and distributed parity hooks.

The generic runner has no contract for task-owned curriculum state.  This
module keeps the release sampler's checkpoint/synchronization surface local to
SONIC and is called only at PPO iteration boundaries or checkpoint I/O.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

SONIC_ADAPTIVE_STATE_VERSION = 1
SONIC_ADAPTIVE_STATE_SCHEMA = "unilab.sonic.adaptive_sampling"
SONIC_ADAPTIVE_SYNC_INTERVAL = 200
SONIC_ADAPTIVE_CHECKPOINT_GLOBAL_MMAP = "shared_global_mmap.v1"
SONIC_ADAPTIVE_CHECKPOINT_RANK_LOCAL = "rank_local_uncheckpointed.v1"
_MISSING = object()


class SonicAdaptiveStateError(ValueError):
    """Raised when adaptive sampler state cannot satisfy its owner contract."""


def _sampler_from_env(env: Any) -> Any | None:
    sampler = getattr(env, "motion_sampler", None)
    if sampler is None:
        return None
    required = (
        "bin_count",
        "bin_size",
        "bin_episode_count",
        "bin_failed_count",
        "_bin_clip_indices",
        "_bin_local_starts",
        "_bin_local_ends",
        "_clip_offsets",
        "_clip_lengths",
    )
    if any(not hasattr(sampler, name) for name in required):
        return None
    return sampler


def _global_mmap_mode(env: Any) -> bool:
    """Return whether this owner has a legal common global-bin coordinate."""

    cfg = getattr(env, "_cfg", None)
    return bool(
        cfg is not None
        and getattr(cfg, "motion_global_mmap_sidecar", None)
        and not bool(getattr(cfg, "motion_shard_clips", True))
    )


def sampler_checkpoint_variant(env: Any) -> str | None:
    """Return the checkpoint ownership mode for the SONIC sampler.

    SONIC checkpoints are written by rank zero.  A rank-local shard therefore
    cannot own one checkpoint state: its bin coordinate is deliberately local
    to that rank and would be rejected by every other rank on resume.  Only
    the unsharded global-mmap layout is a common checkpoint coordinate.
    """

    if _sampler_from_env(env) is None:
        return None
    if _global_mmap_mode(env):
        return SONIC_ADAPTIVE_CHECKPOINT_GLOBAL_MMAP
    return SONIC_ADAPTIVE_CHECKPOINT_RANK_LOCAL


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def sampler_layout_identity(sampler: Any) -> dict[str, Any]:
    """Describe the immutable global bin coordinate used by a sampler."""

    try:
        arrays = tuple(
            np.asarray(getattr(sampler, name))
            for name in (
                "_clip_offsets",
                "_clip_lengths",
                "_bin_clip_indices",
                "_bin_local_starts",
                "_bin_local_ends",
            )
        )
        bin_size = int(sampler.bin_size)
        bin_count = int(sampler.bin_count)
    except (AttributeError, TypeError, ValueError) as exc:
        raise SonicAdaptiveStateError(
            "SONIC adaptive sampler does not expose a stable bin layout"
        ) from exc
    if bin_size < 1 or bin_count < 1:
        raise SonicAdaptiveStateError("SONIC adaptive sampler bin layout is empty or invalid")
    if any(array.ndim != 1 for array in arrays):
        raise SonicAdaptiveStateError(
            "SONIC adaptive sampler bin layout arrays must be one-dimensional"
        )
    if arrays[2].size != bin_count or arrays[3].size != bin_count or arrays[4].size != bin_count:
        raise SonicAdaptiveStateError("SONIC adaptive sampler bin layout lengths disagree")
    return {
        "version": SONIC_ADAPTIVE_STATE_VERSION,
        "bin_size": bin_size,
        "bin_count": bin_count,
        "clip_count": int(arrays[0].size),
        "layout_sha256": _array_digest(*arrays),
    }


def _validate_layout(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    keys = ("version", "bin_size", "bin_count", "clip_count", "layout_sha256")
    missing = [key for key in keys if key not in expected]
    if missing:
        raise SonicAdaptiveStateError(f"adaptive state layout is missing fields: {missing}")
    mismatch = {
        key: (expected[key], actual.get(key)) for key in keys if expected[key] != actual.get(key)
    }
    if mismatch:
        raise SonicAdaptiveStateError(f"adaptive sampler bin layout mismatch: {mismatch}")


def capture_sampler_state(env: Any) -> dict[str, Any] | None:
    """Capture release curriculum counters for a SONIC checkpoint."""

    sampler = _sampler_from_env(env)
    if sampler is None or sampler_checkpoint_variant(env) != SONIC_ADAPTIVE_CHECKPOINT_GLOBAL_MMAP:
        return None
    layout = sampler_layout_identity(sampler)
    episode = np.asarray(sampler.bin_episode_count, dtype=np.float64)
    failed = np.asarray(sampler.bin_failed_count, dtype=np.float64)
    if episode.shape != (layout["bin_count"],) or failed.shape != episode.shape:
        raise SonicAdaptiveStateError("adaptive sampler statistics shape disagrees with bin layout")
    if not np.all(np.isfinite(episode)) or not np.all(np.isfinite(failed)):
        raise SonicAdaptiveStateError("adaptive sampler statistics contain non-finite values")
    return {
        "schema": SONIC_ADAPTIVE_STATE_SCHEMA,
        "version": SONIC_ADAPTIVE_STATE_VERSION,
        "layout": layout,
        "bin_episode_count": episode.copy(),
        "bin_failed_count": failed.copy(),
        "steps_since_probability_refresh": int(
            getattr(sampler, "_steps_since_probability_refresh", 0)
        ),
    }


def restore_sampler_state(env: Any, state: Mapping[str, Any]) -> None:
    """Restore and strictly validate adaptive counters from a checkpoint."""

    sampler = _sampler_from_env(env)
    if sampler is None:
        raise SonicAdaptiveStateError(
            "checkpoint contains SONIC adaptive state but env has no sampler"
        )
    if sampler_checkpoint_variant(env) != SONIC_ADAPTIVE_CHECKPOINT_GLOBAL_MMAP:
        raise SonicAdaptiveStateError(
            "SONIC adaptive checkpoint restore requires global mmap sidecar and shard_clips=false"
        )
    if state.get("schema") != SONIC_ADAPTIVE_STATE_SCHEMA:
        raise SonicAdaptiveStateError("unsupported SONIC adaptive checkpoint schema")
    if state.get("version") != SONIC_ADAPTIVE_STATE_VERSION:
        raise SonicAdaptiveStateError("unsupported SONIC adaptive checkpoint version")
    layout = state.get("layout")
    if not isinstance(layout, Mapping):
        raise SonicAdaptiveStateError("SONIC adaptive checkpoint layout must be a mapping")
    _validate_layout(layout, sampler_layout_identity(sampler))
    count = int(sampler.bin_count)
    episode = np.asarray(state.get("bin_episode_count"), dtype=np.float64)
    failed = np.asarray(state.get("bin_failed_count"), dtype=np.float64)
    if episode.shape != (count,) or failed.shape != (count,):
        raise SonicAdaptiveStateError("SONIC adaptive checkpoint statistics shape mismatch")
    if not np.all(np.isfinite(episode)) or not np.all(np.isfinite(failed)):
        raise SonicAdaptiveStateError("SONIC adaptive checkpoint statistics are non-finite")
    if np.any(episode <= 0.0) or np.any(failed < 0.0):
        raise SonicAdaptiveStateError("SONIC adaptive checkpoint statistics are out of range")
    steps = state.get("steps_since_probability_refresh", 0)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise SonicAdaptiveStateError("SONIC adaptive checkpoint refresh counter is invalid")
    sampler.bin_episode_count[...] = episode
    sampler.bin_failed_count[...] = failed
    sampler._refresh_sampling_probabilities()
    sampler._steps_since_probability_refresh = steps
    reload_active_pool = getattr(sampler, "reload_active_motion_pool_after_restore", _MISSING)
    if reload_active_pool is not _MISSING:
        if not callable(reload_active_pool):
            raise SonicAdaptiveStateError(
                "SONIC adaptive sampler active-pool restore hook must be callable"
            )
        try:
            reload_active_pool()
        except Exception as exc:
            raise SonicAdaptiveStateError(
                "SONIC adaptive sampler active-pool restore failed"
            ) from exc


def sync_sampler_state(env: Any, *, device: torch.device) -> bool:
    """Mean adaptive counters across ranks only in global mmap mode.

    A distributed job that accidentally reaches this hook with rank-local
    shards fails before the statistics collective. This avoids a legal-looking
    but semantically invalid all-reduce over incompatible bin coordinates.
    """

    sampler = _sampler_from_env(env)
    distributed = dist.is_available() and dist.is_initialized()
    if sampler is None:
        if distributed:
            flags: list[bool | None] = [None] * dist.get_world_size()
            dist.all_gather_object(flags, False)
            if any(flags):
                raise SonicAdaptiveStateError(
                    "SONIC adaptive sampler presence differs across ranks"
                )
        return False
    if not distributed:
        return False
    supported: list[bool | None] = [None] * dist.get_world_size()
    dist.all_gather_object(supported, _global_mmap_mode(env))
    if not all(supported):
        raise SonicAdaptiveStateError(
            "SONIC adaptive statistics sync requires global mmap sidecar and shard_clips=false"
        )
    local_layout = sampler_layout_identity(sampler)
    layouts: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(layouts, local_layout)
    if any(layout != local_layout for layout in layouts):
        raise SonicAdaptiveStateError("SONIC adaptive bin layouts differ across ranks")
    for name in ("bin_episode_count", "bin_failed_count"):
        values = torch.as_tensor(getattr(sampler, name), dtype=torch.float64, device=device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values.div_(dist.get_world_size())
        getattr(sampler, name)[...] = values.cpu().numpy()
    sampler._refresh_sampling_probabilities()
    return True


__all__ = [
    "SONIC_ADAPTIVE_STATE_SCHEMA",
    "SONIC_ADAPTIVE_STATE_VERSION",
    "SONIC_ADAPTIVE_SYNC_INTERVAL",
    "SONIC_ADAPTIVE_CHECKPOINT_GLOBAL_MMAP",
    "SONIC_ADAPTIVE_CHECKPOINT_RANK_LOCAL",
    "SonicAdaptiveStateError",
    "capture_sampler_state",
    "restore_sampler_state",
    "sampler_layout_identity",
    "sampler_checkpoint_variant",
    "sync_sampler_state",
]
