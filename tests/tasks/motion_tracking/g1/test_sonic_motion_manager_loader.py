"""Owner-contract tests for compact Bones-Seed motion manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from unilab.tasks.motion_tracking.g1.sonic.manager_terms import (
    SONIC_JOINT_ORDER,
    SONIC_MOTION_SCHEMA,
    CompactSonicMotionLoader,
    SonicMotionCommand,
    SonicMotionCommandCfg,
    SonicMotionCommandParamsCfg,
    SonicMotionManifestError,
)

SONIC_BODY_ORDER = (
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


def _write_store(
    tmp_path: Path,
    *,
    joint_order: tuple[str, ...] = SONIC_JOINT_ORDER,
    body_order: tuple[str, ...] = SONIC_BODY_ORDER,
) -> tuple[Path, dict[str, Any]]:
    frames = 3
    joint_values = np.asarray(
        [SONIC_JOINT_ORDER.index(name) for name in joint_order], dtype=np.float32
    )
    joint_pos = np.broadcast_to(joint_values, (frames, len(joint_order))).copy()
    joint_vel = joint_pos + 100.0
    body_values = np.asarray(
        [SONIC_BODY_ORDER.index(name) for name in body_order], dtype=np.float32
    )
    body_pos = np.zeros((frames, len(body_order), 3), dtype=np.float32)
    body_pos[..., 0] = body_values
    body_quat = np.zeros((frames, len(body_order), 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    body_lin_vel = body_pos + 200.0
    body_ang_vel = body_pos + 300.0
    clip_path = tmp_path / "clip.npz"
    np.savez(
        clip_path,
        fps=np.asarray(30, dtype=np.int32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel,
    )
    fields = [
        {"name": "joint_pos", "dtype": "float32", "shape": ["num_frames", len(joint_order)]},
        {"name": "joint_vel", "dtype": "float32", "shape": ["num_frames", len(joint_order)]},
        {
            "name": "body_pos_w",
            "dtype": "float32",
            "shape": ["num_frames", len(body_order), 3],
        },
        {
            "name": "body_quat_w",
            "dtype": "float32",
            "shape": ["num_frames", len(body_order), 4],
        },
        {
            "name": "body_lin_vel_w",
            "dtype": "float32",
            "shape": ["num_frames", len(body_order), 3],
        },
        {
            "name": "body_ang_vel_w",
            "dtype": "float32",
            "shape": ["num_frames", len(body_order), 3],
        },
    ]
    manifest: dict[str, Any] = {
        "schema": SONIC_MOTION_SCHEMA,
        "version": 1,
        "joint_order": list(joint_order),
        "body_order": list(body_order),
        "fields": fields,
        "clips": [
            {
                "id": "clip",
                "path": clip_path.name,
                "num_frames": frames,
                "fps": 30,
                "joint_order": list(joint_order),
                "body_order": list(body_order),
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_owner_hook_does_not_use_sparse_backend_body_ids_as_dataset_columns(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_store(tmp_path)
    params = SonicMotionCommandParamsCfg(
        motion_file=str(manifest_path),
        anchor_body_name="pelvis",
        body_names=SONIC_BODY_ORDER,
    )
    cfg = SonicMotionCommandCfg(
        entity_name="robot",
        params=params,
        resampling_time_range=(1.0e9, 1.0e9),
    )
    command = object.__new__(SonicMotionCommand)
    command.cfg = cfg

    sparse_backend_ids = np.asarray(
        [1, 3, 5, 7, 9, 11, 13, 16, 18, 20, 23, 25, 27, 30], dtype=np.int32
    )
    loader = command._make_motion_loader(str(manifest_path), sparse_backend_ids)

    assert loader.num_bodies == 14
    np.testing.assert_array_equal(loader.body_pos_w[0, :, 0], np.arange(14, dtype=np.float32))


def test_compact_loader_permutates_joint_and_body_names_once_on_the_cold_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_store(
        tmp_path,
        joint_order=tuple(reversed(SONIC_JOINT_ORDER)),
        body_order=tuple(reversed(SONIC_BODY_ORDER)),
    )
    loader = CompactSonicMotionLoader(
        str(manifest_path),
        expected_joint_order=SONIC_JOINT_ORDER,
        expected_body_order=SONIC_BODY_ORDER,
    )

    np.testing.assert_array_equal(loader.joint_pos[0], np.arange(29, dtype=np.float32))
    np.testing.assert_array_equal(loader.joint_vel[0], np.arange(29, dtype=np.float32) + 100.0)
    np.testing.assert_array_equal(loader.body_pos_w[0, :, 0], np.arange(14, dtype=np.float32))

    def reject_clip_io(*args, **kwargs):
        raise AssertionError("frame gather must not reopen a clip or manifest")

    monkeypatch.setattr(np, "load", reject_clip_io)
    gathered = loader.get_motion_at_frame(np.asarray([2, 0], dtype=np.int32))
    np.testing.assert_array_equal(
        gathered.body_pos_w[..., 0],
        np.broadcast_to(np.arange(14, dtype=np.float32), (2, 14)),
    )


@pytest.mark.parametrize("order_name", ["joint_order", "body_order"])
def test_clip_local_order_must_match_the_manifest(
    order_name: str,
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _write_store(tmp_path)
    manifest["clips"][0][order_name] = list(reversed(manifest[order_name]))
    _write_manifest(manifest_path, manifest)

    with pytest.raises(SonicMotionManifestError, match=rf"clips\[0\]\.{order_name} differs"):
        CompactSonicMotionLoader(
            str(manifest_path),
            expected_joint_order=SONIC_JOINT_ORDER,
            expected_body_order=SONIC_BODY_ORDER,
        )


def test_manifest_schema_and_expected_name_sets_fail_closed(tmp_path: Path) -> None:
    manifest_path, manifest = _write_store(tmp_path)
    manifest["schema"] = "unilab.sonic.wrong"
    _write_manifest(manifest_path, manifest)
    with pytest.raises(SonicMotionManifestError, match="manifest.schema"):
        CompactSonicMotionLoader(
            str(manifest_path),
            expected_joint_order=SONIC_JOINT_ORDER,
            expected_body_order=SONIC_BODY_ORDER,
        )

    manifest["schema"] = SONIC_MOTION_SCHEMA
    _write_manifest(manifest_path, manifest)
    wrong_joint_order = (*SONIC_JOINT_ORDER[:-1], "unexpected_joint")
    with pytest.raises(SonicMotionManifestError, match="joint_order differs"):
        CompactSonicMotionLoader(
            str(manifest_path),
            expected_joint_order=wrong_joint_order,
            expected_body_order=SONIC_BODY_ORDER,
        )


def test_manifest_required_field_shape_is_validated_before_clip_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _write_store(tmp_path)
    next(field for field in manifest["fields"] if field["name"] == "body_pos_w")["shape"] = [
        "num_frames",
        31,
        3,
    ]
    _write_manifest(manifest_path, manifest)

    def reject_clip_io(*args, **kwargs):
        raise AssertionError("invalid manifest must fail before clip I/O")

    monkeypatch.setattr(np, "load", reject_clip_io)
    with pytest.raises(SonicMotionManifestError, match="body_pos_w.*shape"):
        CompactSonicMotionLoader(
            str(manifest_path),
            expected_joint_order=SONIC_JOINT_ORDER,
            expected_body_order=SONIC_BODY_ORDER,
        )
