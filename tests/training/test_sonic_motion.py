from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pytest

from unilab.training.sonic_motion import (
    MotionManifestError,
    convert_sonic_motion,
    load_motion_manifest,
    materialize_motion_store,
    normalize_sonic_motion,
    preflight_motion_manifest,
)
from unilab.training.sonic_store import load_sonic_motion_store


def test_normalize_aliases_resamples_and_derives_velocities():
    # Upstream motion-lib aliases: root_trans_offset/root_rot/dof.  XYZW root
    # quaternions are converted to UniLab's WXYZ convention.
    source = {
        "fps": 25,
        "root_trans_offset": np.asarray([[0, 0, 0], [0.1, 0, 0]], dtype=np.float32),
        "root_rot": np.asarray([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float32),
        "dof": np.asarray([[0.0, 1.0], [1.0, 3.0]], dtype=np.float32),
        "joint_order": ["a", "b"],
        "body_order": ["pelvis"],
    }
    normalized = normalize_sonic_motion(
        source,
        target_fps=50,
        joint_order=["b", "a"],
        body_order=["pelvis"],
    )
    assert int(normalized["fps"]) == 50
    assert normalized["joint_pos"].shape == (3, 2)
    np.testing.assert_allclose(normalized["joint_pos"][:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        normalized["body_quat_w"][:, 0], np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
    )
    # x moves 0.1 m over 0.04 s, so the central/edge finite difference is 2.5 m/s.
    np.testing.assert_allclose(normalized["root_lin_vel_w"][:, 0], 2.5, atol=1e-5)
    np.testing.assert_allclose(normalized["joint_vel"][1:, 0], 50.0, atol=1e-5)


def test_normalize_nested_pickle_and_smpl_fields(tmp_path: Path):
    source = tmp_path / "raw.pkl"
    payload = {
        "clip_a": {
            "frame_rate": 50,
            "qpos": np.zeros((2, 2), dtype=np.float64),
            "root_pos": np.zeros((2, 3), dtype=np.float32),
            "body_pos": np.zeros((2, 1, 3), dtype=np.float32),
            "body_quat": np.tile(np.asarray([[1, 0, 0, 0]], dtype=np.float32), (2, 1, 1)),
            "smpl_joint_positions": np.arange(2 * 72, dtype=np.float32).reshape(2, 72),
        }
    }
    with source.open("wb") as stream:
        pickle.dump(payload, stream)
    normalized = normalize_sonic_motion(source, clip_id="clip_a")
    assert normalized["smpl_joints"].shape == (2, 24, 3)
    assert normalized["joint_pos"].dtype == np.float32
    assert normalized["body_ang_vel_w"].shape == (2, 1, 3)


def test_convert_sonic_motion_writes_checksum_report(tmp_path: Path):
    source = {
        "fps": 50,
        "joint_pos": np.zeros((2, 1), dtype=np.float32),
        "root_pos": np.zeros((2, 3), dtype=np.float32),
    }
    output = tmp_path / "clip.npz"
    report = convert_sonic_motion(source, output)
    assert report.output_path == output.resolve()
    assert len(report.checksum) == 64
    with np.load(output, allow_pickle=False) as archive:
        assert set(report.fields).issubset(archive.files)
        assert int(archive["fps"]) == 50
    with pytest.raises(MotionManifestError, match="already exists"):
        convert_sonic_motion(source, output)


def test_normalize_can_run_cold_path_mujoco_fk(tmp_path: Path):
    model = tmp_path / "tiny.xml"
    model.write_text(
        """
<mujoco model="tiny">
  <worldbody>
    <body name="root" pos="0 0 0">
      <freejoint name="root_free"/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body name="tip" pos="1 0 0">
        <joint name="hinge" type="hinge" axis="0 0 1"/>
        <geom type="sphere" size="0.1" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator><position name="hinge_act" joint="hinge" kp="1"/></actuator>
</mujoco>
""",
        encoding="utf-8",
    )
    normalized = normalize_sonic_motion(
        {
            "fps": 50,
            "dof": np.zeros((2, 1), dtype=np.float32),
            "root_trans_offset": np.zeros((2, 3), dtype=np.float32),
            "root_rot": np.tile([0, 0, 0, 1], (2, 1)).astype(np.float32),
            "joint_order": ["hinge"],
        },
        fk_model_path=model,
        body_order=["root", "tip"],
    )
    assert normalized["body_pos_w"].shape == (2, 2, 3)
    np.testing.assert_allclose(normalized["body_pos_w"][:, 1, 0], 1.0)


def _clip(path: Path, frames: int = 4) -> None:
    np.savez(
        path,
        joint_pos=np.zeros((frames, 2), dtype=np.float32),
        body_pos_w=np.zeros((frames, 1, 3), dtype=np.float32),
    )


def test_materializer_writes_checksums_and_validates_shapes(tmp_path: Path):
    source_a, source_b = tmp_path / "a.npz", tmp_path / "b.npz"
    _clip(source_a, 4)
    _clip(source_b, 6)
    report = materialize_motion_store(
        [source_a, source_b],
        tmp_path / "store",
        fps=50,
        joint_order=["j0", "j1"],
        body_order=["pelvis"],
    )
    assert report.clip_count == 2
    manifest = load_motion_manifest(report.manifest_path)
    assert manifest.clips[0].num_frames == 4
    assert manifest.fields[0].shape[0] == "num_frames"
    preflight_motion_manifest(report.manifest_path)
    digest = hashlib.sha256(manifest.manifest_path.read_bytes()).hexdigest()
    assert len(digest) == 64


def test_materializer_accepts_scalar_fps_metadata(tmp_path: Path):
    source = tmp_path / "with_fps.npz"
    np.savez(
        source,
        fps=np.asarray(50, dtype=np.int32),
        joint_pos=np.zeros((2, 1), dtype=np.float32),
        joint_vel=np.zeros((2, 1), dtype=np.float32),
        body_pos_w=np.zeros((2, 1, 3), dtype=np.float32),
        body_quat_w=np.tile(np.asarray([1, 0, 0, 0], dtype=np.float32), (2, 1, 1)),
        body_lin_vel_w=np.zeros((2, 1, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((2, 1, 3), dtype=np.float32),
    )
    report = materialize_motion_store(
        [source],
        tmp_path / "store_fps",
        fps=50,
        joint_order=["j0"],
        body_order=["pelvis"],
    )
    preflight_motion_manifest(report.manifest_path)
    store = load_sonic_motion_store(report.manifest_path)
    assert store.num_frames == 2


def test_store_can_load_disjoint_rank_local_clip_shards(tmp_path: Path):
    sources = []
    for index, frames in enumerate((2, 3, 4, 5)):
        source = tmp_path / f"clip_{index}.npz"
        np.savez(
            source,
            fps=np.asarray(50, dtype=np.int32),
            joint_pos=np.full((frames, 1), index, dtype=np.float32),
            joint_vel=np.zeros((frames, 1), dtype=np.float32),
            body_pos_w=np.zeros((frames, 1, 3), dtype=np.float32),
            body_quat_w=np.tile(np.asarray([1, 0, 0, 0], dtype=np.float32), (frames, 1, 1)),
            body_lin_vel_w=np.zeros((frames, 1, 3), dtype=np.float32),
            body_ang_vel_w=np.zeros((frames, 1, 3), dtype=np.float32),
        )
        sources.append(source)
    report = materialize_motion_store(
        sources,
        tmp_path / "sharded_store",
        fps=50,
        joint_order=["j0"],
        body_order=["pelvis"],
    )

    rank_zero = load_sonic_motion_store(
        report.manifest_path, rank=0, world_size=2, shard_clips=True
    )
    rank_one = load_sonic_motion_store(report.manifest_path, rank=1, world_size=2, shard_clips=True)

    assert [clip.id for clip in rank_zero.manifest.clips] == ["clip_000000", "clip_000002"]
    assert [clip.id for clip in rank_one.manifest.clips] == ["clip_000001", "clip_000003"]
    assert rank_zero.num_frames == 6
    assert rank_one.num_frames == 8
    assert set(np.unique(rank_zero.arrays["joint_pos"])) == {0.0, 2.0}
    assert set(np.unique(rank_one.arrays["joint_pos"])) == {1.0, 3.0}


def test_store_rejects_more_sharded_ranks_than_clips(tmp_path: Path):
    source = tmp_path / "one_clip.npz"
    np.savez(
        source,
        fps=np.asarray(50, dtype=np.int32),
        joint_pos=np.zeros((2, 1), dtype=np.float32),
        joint_vel=np.zeros((2, 1), dtype=np.float32),
        body_pos_w=np.zeros((2, 1, 3), dtype=np.float32),
        body_quat_w=np.tile(np.asarray([1, 0, 0, 0], dtype=np.float32), (2, 1, 1)),
        body_lin_vel_w=np.zeros((2, 1, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((2, 1, 3), dtype=np.float32),
    )
    report = materialize_motion_store(
        [source],
        tmp_path / "small_store",
        fps=50,
        joint_order=["j0"],
        body_order=["pelvis"],
    )

    with pytest.raises(ValueError, match="at least one clip per rank"):
        load_sonic_motion_store(report.manifest_path, rank=1, world_size=2, shard_clips=True)


def test_materializer_rejects_inconsistent_fields(tmp_path: Path):
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    _clip(first)
    np.savez(
        second,
        joint_pos=np.zeros((4, 3), dtype=np.float32),
        body_pos_w=np.zeros((4, 1, 3), dtype=np.float32),
    )
    with pytest.raises(MotionManifestError, match="disagree"):
        materialize_motion_store(
            [first, second],
            tmp_path / "store",
            fps=50,
            joint_order=["j0", "j1"],
            body_order=["pelvis"],
        )


def test_materializer_does_not_overwrite_nonempty_output_by_default(tmp_path: Path):
    source = tmp_path / "a.npz"
    _clip(source)
    output = tmp_path / "store"
    output.mkdir()
    (output / "keep").write_text("x", encoding="utf-8")
    with pytest.raises(MotionManifestError, match="not empty"):
        materialize_motion_store(
            [source],
            output,
            fps=50,
            joint_order=["j0", "j1"],
            body_order=["pelvis"],
        )
