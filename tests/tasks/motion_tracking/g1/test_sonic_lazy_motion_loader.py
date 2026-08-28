"""Focused contracts for the task-owned bounded lazy SONIC loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import unilab.tasks.motion_tracking.g1.sonic.lazy_motion_loader as lazy_module
from unilab.tasks.motion_tracking.g1.sonic.lazy_motion_loader import (
    BoundedLazySonicMotionLoader,
    LazySonicMotionData,
)
from unilab.tasks.motion_tracking.g1.sonic.manager_terms import (
    SONIC_MOTION_SCHEMA,
    SonicMotionManifestError,
)

JOINT_ORDER = ("j0", "j1")
BODY_ORDER = ("b0", "b1")


def _field(name: str, *tail: int) -> dict[str, Any]:
    return {"name": name, "dtype": "float32", "shape": ["num_frames", *tail]}


def _write_store(
    tmp_path: Path,
    *,
    clip_lengths: tuple[int, ...] = (2, 3, 4),
    source_joint_order: tuple[str, ...] = tuple(reversed(JOINT_ORDER)),
    source_body_order: tuple[str, ...] = tuple(reversed(BODY_ORDER)),
    optional_fields: tuple[str, ...] = ("smpl_joints", "smpl_root_quat_w"),
) -> Path:
    fields: list[dict[str, Any]] = [
        _field("joint_pos", len(source_joint_order)),
        _field("joint_vel", len(source_joint_order)),
        _field("body_pos_w", len(source_body_order), 3),
        _field("body_quat_w", len(source_body_order), 4),
        _field("body_lin_vel_w", len(source_body_order), 3),
        _field("body_ang_vel_w", len(source_body_order), 3),
    ]
    if "smpl_joints" in optional_fields:
        fields.append(_field("smpl_joints", 24, 3))
    if "smpl_root_quat_w" in optional_fields:
        fields.append(_field("smpl_root_quat_w", 4))

    clips = []
    for clip_index, frames in enumerate(clip_lengths):
        frame_value = np.arange(frames, dtype=np.float32)[:, None] * 100.0
        joint_columns = np.asarray(
            [JOINT_ORDER.index(name) for name in source_joint_order], dtype=np.float32
        )[None, :]
        joint_pos = clip_index * 1000.0 + frame_value + joint_columns
        joint_vel = joint_pos + 10.0
        body_columns = np.asarray(
            [BODY_ORDER.index(name) for name in source_body_order], dtype=np.float32
        )[None, :]
        body_base = clip_index * 1000.0 + frame_value + body_columns
        body_pos = np.zeros((frames, len(source_body_order), 3), dtype=np.float32)
        body_pos[..., 0] = body_base
        body_quat = np.zeros((frames, len(source_body_order), 4), dtype=np.float32)
        body_quat[..., 0] = 1.0
        body_lin_vel = body_pos + 20.0
        body_ang_vel = body_pos + 30.0
        arrays: dict[str, np.ndarray] = {
            "fps": np.asarray(30, dtype=np.int32),
            "joint_pos": joint_pos.astype(np.float32),
            "joint_vel": joint_vel.astype(np.float32),
            "body_pos_w": body_pos,
            "body_quat_w": body_quat,
            "body_lin_vel_w": body_lin_vel,
            "body_ang_vel_w": body_ang_vel,
        }
        if "smpl_joints" in optional_fields:
            arrays["smpl_joints"] = np.full((frames, 24, 3), clip_index + 0.25, dtype=np.float32)
        if "smpl_root_quat_w" in optional_fields:
            arrays["smpl_root_quat_w"] = np.full((frames, 4), clip_index + 0.5, dtype=np.float32)
        clip_path = tmp_path / f"clip_{clip_index}.npz"
        np.savez(clip_path, **arrays)
        clips.append(
            {
                "id": f"clip_{clip_index}",
                "path": clip_path.name,
                "num_frames": frames,
                "fps": 30,
                "joint_order": list(source_joint_order),
                "body_order": list(source_body_order),
            }
        )
    manifest = {
        "schema": SONIC_MOTION_SCHEMA,
        "version": 1,
        "joint_order": list(source_joint_order),
        "body_order": list(source_body_order),
        "fields": fields,
        "clips": clips,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _loader(
    manifest_path: Path,
    *,
    cache_size: int | str = 2,
    cache_max_size: int = 128,
    cache_max_bytes: int = 512 * 1024 * 1024,
    optional_fields: tuple[str, ...] | None = (),
    num_envs: int | None = None,
    rank: int = 0,
    world_size: int = 1,
    shard_clips: bool = False,
) -> BoundedLazySonicMotionLoader:
    return BoundedLazySonicMotionLoader(
        str(manifest_path),
        expected_joint_order=JOINT_ORDER,
        expected_body_order=BODY_ORDER,
        cache_size=cache_size,
        cache_max_size=cache_max_size,
        cache_max_bytes=cache_max_bytes,
        optional_fields=optional_fields,
        num_envs=num_envs,
        rank=rank,
        world_size=world_size,
        shard_clips=shard_clips,
    )


def test_motion_command_surface_and_optional_global_gather(tmp_path: Path) -> None:
    loader = _loader(_write_store(tmp_path), optional_fields=None)

    assert loader.num_clips == 3
    assert loader.num_frames == 9
    assert loader.fps == 30
    assert loader.num_joints == 2
    assert loader.num_bodies == 2
    np.testing.assert_array_equal(loader.clip_offsets, [0, 2, 5])
    assert loader.clip_lengths.dtype == np.dtype(np.int32)
    assert loader.clip_offsets.dtype == np.dtype(np.int32)
    assert loader.clip_starts is loader.clip_offsets
    assert not loader.clip_starts.flags.writeable
    np.testing.assert_array_equal(loader.clip_end_frames, [1, 4, 8])
    assert loader.clip_end_frames.dtype == np.dtype(np.int32)
    assert loader.source_clip_indices == (0, 1, 2)
    assert loader.joint_pos.shape == (9, 2)
    assert loader.joint_pos.dtype == np.dtype(np.float32)
    assert loader.available_fields == frozenset(
        {
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "smpl_joints",
            "smpl_root_quat_w",
        }
    )

    indices = np.asarray([0, 2, 8], dtype=np.int32)
    out = loader.make_motion_data_buffer(3)
    assert isinstance(out, LazySonicMotionData)
    assert loader.get_motion_at_frame(indices, out=out) is out
    np.testing.assert_array_equal(
        out.joint_pos,
        [[0.0, 1.0], [1000.0, 1001.0], [2300.0, 2301.0]],
    )
    np.testing.assert_array_equal(out.body_pos_w[..., 0], out.joint_pos)
    assert out.smpl_joints is None
    assert out.smpl_root_quat_w is None
    assert all(
        set(decoded.arrays)
        == {
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        }
        for decoded in loader._cache.values()
    )
    np.testing.assert_array_equal(loader.get_clip_indices(indices), [0, 1, 2])

    optional = loader.gather_fields(
        ("smpl_joints", "smpl_root_quat_w"),
        np.asarray([0, 2, 8], dtype=np.int64),
    )
    np.testing.assert_array_equal(optional["smpl_joints"][:, 0, 0], [0.25, 1.25, 2.25])
    np.testing.assert_array_equal(optional["smpl_root_quat_w"][:, 0], [0.5, 1.5, 2.5])

    optional = loader.gather_fields(("smpl_joints",), np.asarray([-1], dtype=np.int64))
    np.testing.assert_array_equal(optional["smpl_joints"][:, 0, 0], [2.25])


def test_optional_preselection_does_not_hide_manifest_capabilities(tmp_path: Path) -> None:
    loader = _loader(
        _write_store(tmp_path, clip_lengths=(2,)),
        optional_fields=("smpl_joints",),
    )

    assert {"smpl_joints", "smpl_root_quat_w"}.issubset(loader.available_fields)
    gathered = loader.gather_fields(
        ("smpl_root_quat_w",),
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(gathered["smpl_root_quat_w"][:, 0], [0.5])


def test_lru_clip_count_and_bytes_remain_bounded(tmp_path: Path) -> None:
    loader = _loader(
        _write_store(tmp_path, clip_lengths=(2, 2, 2, 2, 2)),
        cache_size=2,
    )
    loader.gather_fields(("joint_pos",), loader.clip_offsets.copy())

    assert loader.loaded_clip_count == 5
    assert loader.cached_clip_count == 2
    assert loader.peak_cached_clip_count == 2
    assert loader.cached_clip_indices == (3, 4)
    assert 0 < loader.cached_bytes <= loader.peak_cached_bytes

    loader.gather_fields(("joint_pos",), np.asarray([0], dtype=np.int32))
    assert loader.loaded_clip_count == 6
    assert loader.cached_clip_count == 2
    assert loader.cached_clip_indices == (4, 0)
    assert loader.peak_cached_clip_count == 2

    loader.clear_cache()
    assert loader.cached_clip_count == 0
    assert loader.cached_bytes == 0


def test_auto_cache_tracks_active_working_set_with_fixed_ceiling(tmp_path: Path) -> None:
    manifest = _write_store(tmp_path, clip_lengths=tuple([2] * 200))

    small = _loader(manifest, cache_size="auto", num_envs=3)
    assert small.cache_size == 3
    assert small.requested_cache_size == "auto"

    large = _loader(manifest, cache_size="auto", num_envs=512)
    assert large.cache_size == 128


def test_explicit_cache_size_must_fit_owner_ceiling(tmp_path: Path) -> None:
    manifest = _write_store(tmp_path, clip_lengths=(2, 2))
    with pytest.raises(ValueError, match="exceeds cache_max_size"):
        _loader(manifest, cache_size=9, cache_max_size=8)


def test_zero_cache_mode_never_retains_a_decoded_clip(tmp_path: Path) -> None:
    loader = _loader(_write_store(tmp_path, clip_lengths=(2,)), cache_size=0)
    indices = np.asarray([0], dtype=np.int32)
    loader.gather_fields(("joint_pos",), indices)
    loader.gather_fields(("joint_pos",), indices)

    assert loader.loaded_clip_count == 2
    assert loader.cached_clip_count == 0
    assert loader.peak_cached_clip_count == 0
    assert loader.cached_bytes == loader.peak_cached_bytes == 0


def test_rank_clip_shard_rebuilds_a_rank_local_global_frame_space(tmp_path: Path) -> None:
    manifest_path = _write_store(tmp_path, clip_lengths=(2, 3, 4, 5, 6, 7, 8))
    loader = _loader(
        manifest_path,
        rank=1,
        world_size=3,
        shard_clips=True,
    )

    assert loader.source_clip_indices == (1, 4)
    np.testing.assert_array_equal(loader.clip_lengths, [3, 6])
    np.testing.assert_array_equal(loader.clip_offsets, [0, 3])
    assert loader.num_frames == 9
    gathered = loader.gather_fields(("joint_pos",), np.asarray([0, 3], dtype=np.int32))
    np.testing.assert_array_equal(gathered["joint_pos"], [[1000.0, 1001.0], [4000.0, 4001.0]])

    with pytest.raises(ValueError, match="at least one clip per rank"):
        _loader(manifest_path, rank=7, world_size=8, shard_clips=True)


def test_manifest_is_not_reparsed_by_frame_gathers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loader = _loader(_write_store(tmp_path, clip_lengths=(2,)))

    def reject_manifest_parse(*args, **kwargs):
        raise AssertionError("frame gather must not parse the manifest")

    monkeypatch.setattr(lazy_module.json, "load", reject_manifest_parse)
    gathered = loader.gather_fields(("joint_pos",), np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(gathered["joint_pos"], [[100.0, 101.0]])


def test_requested_optional_field_must_be_declared_on_the_cold_path(tmp_path: Path) -> None:
    manifest_path = _write_store(tmp_path, optional_fields=("smpl_joints",))

    with pytest.raises(SonicMotionManifestError, match="does not declare.*smpl_root_quat_w"):
        _loader(manifest_path, optional_fields=("smpl_root_quat_w",))
