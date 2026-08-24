from __future__ import annotations

import gc
import hashlib
import importlib.util
import pickle
import sys
import weakref
from pathlib import Path

import joblib
import numpy as np
import pytest

from unilab.training.sonic_motion import (
    MotionManifestError,
    convert_sonic_motion,
    load_motion_manifest,
    materialize_motion_store,
    materialize_paired_sonic_motion,
    normalize_sonic_motion,
    preflight_motion_manifest,
)
from unilab.training.sonic_store import load_sonic_motion_store


def test_normalize_aliases_resamples_and_derives_velocities():
    # Upstream motion-lib aliases: root_trans_offset/root_rot/dof.  XYZW root
    # quaternions are converted to UniLab's WXYZ convention.
    source = {
        "fps": 25,
        "root_trans_offset": np.asarray([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]], dtype=np.float32),
        # Alternating antipodal signs require cumulative shortest-path unwrap.
        "root_rot": np.asarray([[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 1]], dtype=np.float32),
        "dof": np.asarray([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]], dtype=np.float32),
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
    # Fixed upstream uses arange(0, duration, 1 / target_fps), so resampling
    # excludes the source endpoint instead of stretching a linspace over it.
    assert normalized["joint_pos"].shape == (4, 2)
    np.testing.assert_allclose(normalized["joint_pos"][:, 0], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(
        normalized["body_quat_w"][:, 0], np.tile([1.0, 0.0, 0.0, 0.0], (4, 1))
    )
    # x moves 0.1 m over 0.04 s, so the central/edge finite difference is 2.5 m/s.
    np.testing.assert_allclose(normalized["root_lin_vel_w"][:, 0], 2.5, atol=1e-5)
    np.testing.assert_allclose(normalized["joint_vel"][1:, 0], 50.0, atol=1e-5)


def test_normalize_joblib_compressed_pickle_and_smpl_fields(tmp_path: Path):
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
    joblib.dump(payload, source, compress=3)
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


def test_normalize_resamples_inputs_before_cold_path_mujoco_fk(tmp_path: Path, monkeypatch):
    import unilab.training.sonic_motion as sonic_motion

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
    original_fk = sonic_motion._mujoco_forward_kinematics
    seen: dict[str, np.ndarray] = {}

    def recording_fk(**kwargs):
        for name in ("joint_pos", "root_pos", "root_quat_wxyz"):
            seen[name] = np.asarray(kwargs[name]).copy()
        return original_fk(**kwargs)

    monkeypatch.setattr(sonic_motion, "_mujoco_forward_kinematics", recording_fk)
    normalized = normalize_sonic_motion(
        {
            "fps": 25,
            "dof": np.arange(3, dtype=np.float32)[:, None],
            "root_trans_offset": np.asarray(
                [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]], dtype=np.float32
            ),
            "root_rot": np.tile([0, 0, 0, 1], (3, 1)).astype(np.float32),
            "joint_order": ["hinge"],
        },
        fk_model_path=model,
        body_order=["root", "tip"],
    )
    assert normalized["body_pos_w"].shape == (4, 2, 3)
    np.testing.assert_allclose(seen["joint_pos"][:, 0], [0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(seen["root_pos"][:, 0], [0.0, 0.05, 0.1, 0.15])
    np.testing.assert_allclose(normalized["body_pos_w"][:, 0], seen["root_pos"])


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


def _write_paired_source(root: Path, key: str, *, frames: int = 3, fps: int = 50) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{key}.pkl"
    payload = {
        "fps": fps,
        "root_trans_offset": np.zeros((frames, 3), dtype=np.float32),
        "root_rot": np.tile(np.asarray([0, 0, 0, 1], dtype=np.float32), (frames, 1)),
        "dof": np.zeros((frames, 2), dtype=np.float32),
        "joint_order": ["j0", "j1"],
        "body_order": ["pelvis"],
    }
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    return path


def _write_smpl_source(root: Path, key: str, *, frames: int = 3, fps: int = 50) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{key}.pkl"
    payload = {
        "fps": fps,
        "pose_aa": np.zeros((frames, 72), dtype=np.float32),
        "smpl_joints": np.zeros((frames, 24, 3), dtype=np.float32),
        "transl": np.zeros((frames, 3), dtype=np.float32),
    }
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    return path


def _paired_kwargs(robot: Path, smpl: Path, output: Path) -> dict[str, object]:
    return {
        "robot_root": robot,
        "smpl_root": smpl,
        "output_dir": output,
        "fps": 50,
        "joint_order": ["j0", "j1"],
        "body_order": ["pelvis"],
    }


def test_paired_materializer_is_sorted_and_manifest_loadable(tmp_path: Path):
    robot = tmp_path / "robot"
    smpl = tmp_path / "smpl"
    _write_paired_source(robot, "b")
    _write_paired_source(robot, "a")
    _write_smpl_source(smpl, "a")
    _write_smpl_source(smpl, "b")

    report = materialize_paired_sonic_motion(**_paired_kwargs(robot, smpl, tmp_path / "store"))
    manifest = load_motion_manifest(report.manifest_path)
    assert [clip.id for clip in manifest.clips] == ["a", "b"]
    assert {field.name for field in manifest.fields} >= {
        "joint_pos",
        "smpl_joints",
        "smpl_pose",
        "smpl_root_quat_w",
    }
    preflight_motion_manifest(report.manifest_path)
    store = load_sonic_motion_store(report.manifest_path)
    assert store.num_frames == 6
    assert store.arrays["smpl_joints"].shape == (6, 24, 3)

    second = materialize_paired_sonic_motion(
        **_paired_kwargs(robot, smpl, tmp_path / "second_store")
    )
    assert load_motion_manifest(second.manifest_path).to_dict() == manifest.to_dict()


def test_paired_materializer_rejects_missing_and_duplicate_keys(tmp_path: Path):
    robot = tmp_path / "robot"
    smpl = tmp_path / "smpl"
    _write_paired_source(robot, "matched")
    _write_paired_source(robot, "missing")
    _write_smpl_source(smpl, "matched")
    _write_smpl_source(smpl, "extra")
    with pytest.raises(MotionManifestError, match="unmatched.*missing.*extra"):
        materialize_paired_sonic_motion(**_paired_kwargs(robot, smpl, tmp_path / "missing_store"))
    with pytest.warns(UserWarning, match="skipping unmatched"):
        allowed = materialize_paired_sonic_motion(
            **_paired_kwargs(robot, smpl, tmp_path / "allowed_store"), allow_unmatched=True
        )
    assert [clip.id for clip in load_motion_manifest(allowed.manifest_path).clips] == ["matched"]

    duplicate = robot / "nested"
    duplicate.mkdir()
    _write_paired_source(duplicate, "matched").rename(duplicate / "matched.joblib")
    with pytest.raises(MotionManifestError, match="duplicate robot basename.*matched"):
        materialize_paired_sonic_motion(**_paired_kwargs(robot, smpl, tmp_path / "duplicate_store"))


def test_paired_materializer_resamples_official_30hz_50hz_duration_pair(tmp_path: Path):
    robot = tmp_path / "robot"
    smpl = tmp_path / "smpl"
    robot_path = _write_paired_source(robot, "official", frames=1202, fps=30)
    smpl_path = _write_smpl_source(smpl, "official", frames=2002, fps=50)
    robot_payload = pickle.loads(robot_path.read_bytes())
    robot_times = np.arange(1202, dtype=np.float32) / 30.0
    robot_payload["dof"][:, 0] = robot_times
    robot_path.write_bytes(pickle.dumps(robot_payload))
    smpl_payload = pickle.loads(smpl_path.read_bytes())
    smpl_times = np.arange(2002, dtype=np.float32) / 50.0
    smpl_payload["smpl_joints"][:, 0, 0] = smpl_times
    smpl_path.write_bytes(pickle.dumps(smpl_payload))

    report = materialize_paired_sonic_motion(**_paired_kwargs(robot, smpl, tmp_path / "store"))
    manifest = load_motion_manifest(report.manifest_path)
    assert manifest.clips[0].num_frames == 2002
    with np.load(report.manifest_path.parent / manifest.clips[0].path) as archive:
        assert archive["joint_pos"].shape[0] == 2002
        assert archive["smpl_joints"].shape[0] == 2002
        np.testing.assert_allclose(archive["joint_pos"][:, 0], smpl_times, atol=2.0e-6)
        np.testing.assert_array_equal(archive["smpl_joints"][:, 0, 0], smpl_times)


def test_paired_materializer_slerps_smpl_axis_angle(tmp_path: Path):
    robot, smpl = tmp_path / "robot", tmp_path / "smpl"
    _write_paired_source(robot, "turn", frames=2, fps=50)
    smpl_path = _write_smpl_source(smpl, "turn", frames=2, fps=25)
    payload = pickle.loads(smpl_path.read_bytes())
    payload["pose_aa"][:, 2] = np.deg2rad([170.0, -170.0])
    smpl_path.write_bytes(pickle.dumps(payload))

    report = materialize_paired_sonic_motion(**_paired_kwargs(robot, smpl, tmp_path / "store"))
    manifest = load_motion_manifest(report.manifest_path)
    with np.load(report.manifest_path.parent / manifest.clips[0].path) as archive:
        assert abs(archive["smpl_pose"][1, 2]) == pytest.approx(np.pi, abs=1.0e-5)


def test_paired_materializer_fails_atomically_on_duration_grid_mismatch(tmp_path: Path):
    robot = tmp_path / "robot"
    smpl = tmp_path / "smpl"
    _write_paired_source(robot, "a")
    _write_smpl_source(smpl, "a")
    _write_paired_source(robot, "bad", frames=3, fps=50)
    _write_smpl_source(smpl, "bad", frames=4, fps=50)
    output = tmp_path / "store"
    with pytest.raises(MotionManifestError, match="duration/target-grid mismatch"):
        materialize_paired_sonic_motion(**_paired_kwargs(robot, smpl, output))
    assert not output.exists()
    assert not list(tmp_path.glob(".store.staging-*"))

    _write_smpl_source(smpl, "bad", frames=3, fps=60)
    with pytest.raises(MotionManifestError, match="duration/target-grid mismatch"):
        materialize_paired_sonic_motion(**_paired_kwargs(robot, smpl, output))
    assert not output.exists()
    assert not list(tmp_path.glob(".store.staging-*"))


def test_paired_materializer_restores_old_store_when_atomic_swap_fails(tmp_path: Path, monkeypatch):
    import unilab.training.sonic_motion as sonic_motion

    robot = tmp_path / "robot"
    smpl = tmp_path / "smpl"
    _write_paired_source(robot, "a")
    _write_smpl_source(smpl, "a")
    output = tmp_path / "store"
    output.mkdir()
    (output / "old").write_text("keep", encoding="utf-8")
    original_replace = sonic_motion.os.replace

    def fail_staging_publish(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path.name.startswith(".store.staging-") and destination_path == output:
            raise OSError("synthetic publish failure")
        return original_replace(source, destination)

    monkeypatch.setattr(sonic_motion.os, "replace", fail_staging_publish)
    with pytest.raises(OSError, match="synthetic publish failure"):
        materialize_paired_sonic_motion(**_paired_kwargs(robot, smpl, output), overwrite=True)
    assert (output / "old").read_text(encoding="utf-8") == "keep"
    assert not (output / "manifest.json").exists()
    assert not list(tmp_path.glob(".store.staging-*"))
    assert not list(tmp_path.glob(".store.backup-*"))


def test_paired_materializer_releases_previous_pair_before_loading_next(
    tmp_path: Path, monkeypatch
):
    import unilab.training.sonic_motion as sonic_motion

    robot = tmp_path / "robot"
    smpl = tmp_path / "smpl"
    for key in ("a", "b", "c"):
        _write_paired_source(robot, key)
        _write_smpl_source(smpl, key)
    original = sonic_motion._load_motion_source
    references: list[weakref.ReferenceType[np.ndarray]] = []

    def wrapped(source):
        if not isinstance(source, (str, Path)):
            return original(source)
        source_path = Path(source)
        if source_path.parent == robot and references:
            gc.collect()
            assert references[-1]() is None
        loaded = original(source)
        if source_path.parent == robot:
            references.append(weakref.ref(loaded["dof"]))
        return loaded

    monkeypatch.setattr(sonic_motion, "_load_motion_source", wrapped)
    materialize_paired_sonic_motion(**_paired_kwargs(robot, smpl, tmp_path / "store"))
    gc.collect()
    assert references[-1]() is None


def test_paired_materializer_accepts_singleton_wrapper_with_different_inner_key(tmp_path: Path):
    robot = tmp_path / "robot"
    smpl = tmp_path / "smpl"
    robot_path = _write_paired_source(robot, "pair_key")
    with robot_path.open("rb") as stream:
        payload = pickle.load(stream)
    with robot_path.open("wb") as stream:
        pickle.dump({"different_inner_key": payload}, stream)
    _write_smpl_source(smpl, "pair_key")

    report = materialize_paired_sonic_motion(**_paired_kwargs(robot, smpl, tmp_path / "store"))
    assert [clip.id for clip in load_motion_manifest(report.manifest_path).clips] == ["pair_key"]


@pytest.mark.parametrize("extra", [["--hardlink"], ["--clip-id", "nested"]])
def test_paired_cli_rejects_single_source_only_options(
    tmp_path: Path, monkeypatch, extra: list[str]
):
    script = Path(__file__).parents[2] / "scripts" / "materialize_sonic_motion.py"
    spec = importlib.util.spec_from_file_location("materialize_sonic_motion_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--robot-root",
            str(tmp_path / "robot"),
            "--smpl-root",
            str(tmp_path / "smpl"),
            "--output",
            str(tmp_path / "store"),
            "--joint-order",
            "j0",
            "--body-order",
            "pelvis",
            *extra,
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        module.main()
