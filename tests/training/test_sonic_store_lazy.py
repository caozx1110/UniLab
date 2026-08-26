from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unilab.training.sonic_motion import materialize_motion_store
from unilab.training.sonic_store import (
    load_sonic_motion_store,
    materialize_sonic_global_mmap_store,
)


def _write_clip(path: Path, value: float, frames: int) -> None:
    joint = np.full((frames, 2), value, dtype=np.float32)
    body = np.full((frames, 1, 3), value, dtype=np.float32)
    smpl_pose = np.full((frames, 3), value, dtype=np.float32)
    np.savez(path, joint_pos=joint, body_pos_w=body, smpl_pose=smpl_pose)


def _make_store(tmp_path: Path, *, cache_size: int = 2, hot_fields: tuple[str, ...] = ()):
    clips = []
    for index, frames in enumerate((3, 4, 5)):
        path = tmp_path / f"clip_{index}.npz"
        _write_clip(path, float(index), frames)
        clips.append(path)
    report = materialize_motion_store(
        clips,
        tmp_path / "store",
        fps=50,
        joint_order=["j0", "j1"],
        body_order=["pelvis"],
    )
    return load_sonic_motion_store(
        report.manifest_path,
        cache_size=cache_size,
        hot_fields=hot_fields,
    )


def test_loader_uses_bounded_clip_cache(tmp_path: Path) -> None:
    store = _make_store(tmp_path, cache_size=1)

    # Construction reads metadata from one clip, but never concatenates the
    # three frame arrays into one resident corpus.
    assert store.cache_size == 1
    assert store.cached_clip_count == 1
    assert store.loaded_clip_count == 1

    np.testing.assert_allclose(store.gather("joint_pos", np.asarray([0, 3, 7]))[:, 0], [0, 1, 2])
    assert store.cached_clip_count <= 1
    assert store.loaded_clip_count >= 3

    store.clear_cache()
    assert store.cached_clip_count == 0


def test_lazy_field_keeps_numpy_indexing_surface(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    field = store.arrays["joint_pos"]

    assert field.shape == (12, 2)
    np.testing.assert_allclose(field[0], [0, 0])
    np.testing.assert_allclose(field[-1], [2, 2])
    np.testing.assert_allclose(field[2:5, 0], [0, 1, 1])
    np.testing.assert_allclose(field[[0, 4, 11], 1], [0, 1, 2])
    mask = np.zeros((12,), dtype=bool)
    mask[[0, 4, 11]] = True
    np.testing.assert_allclose(field[mask], [[0, 0], [1, 1], [2, 2]])
    np.testing.assert_allclose(np.take(field, [11, 0], axis=0)[:, 0], [2, 0])
    assert set(np.unique(field)) == {0.0, 1.0, 2.0}


def test_multi_field_gather_opens_each_touched_clip_once(tmp_path: Path) -> None:
    store = _make_store(tmp_path, cache_size=1)
    store.clear_cache()
    before = store.loaded_clip_count

    gathered = store.gather_fields(
        ("joint_pos", "body_pos_w"),
        np.asarray([8, 0, 4, 11, 3, 7, 1, 6]),
    )

    np.testing.assert_allclose(gathered["joint_pos"][:, 0], [2, 0, 1, 2, 1, 2, 0, 1])
    np.testing.assert_allclose(gathered["body_pos_w"][:, 0, 0], [2, 0, 1, 2, 1, 2, 0, 1])
    assert store.loaded_clip_count == before + 3


def test_matching_explicit_orders_skip_identity_permutations(tmp_path: Path) -> None:
    initial = _make_store(tmp_path)
    manifest_path = initial.manifest.manifest_path
    assert manifest_path is not None

    store = load_sonic_motion_store(
        manifest_path,
        expected_joint_order=("j0", "j1"),
        expected_body_order=("pelvis",),
    )

    assert store._lazy_backend is not None
    assert store._lazy_backend.joint_permutation is None
    assert store._lazy_backend.body_permutation is None


def test_hot_fields_are_readonly_contiguous_and_bypass_lazy_cache(tmp_path: Path) -> None:
    store = _make_store(
        tmp_path,
        cache_size=1,
        hot_fields=("joint_pos", "body_pos_w"),
    )

    joint = store.arrays["joint_pos"]
    body = store.arrays["body_pos_w"]
    assert isinstance(joint, np.ndarray)
    assert isinstance(body, np.ndarray)
    assert joint.flags.c_contiguous and not joint.flags.writeable
    assert body.flags.c_contiguous and not body.flags.writeable
    assert store.cached_clip_count == 0
    assert type(store.arrays["smpl_pose"]).__name__ == "_LazyFieldView"

    before = store.loaded_clip_count
    indices = np.asarray([[7, 0], [3, 11]], dtype=np.int64)
    gathered = store.gather_fields(("joint_pos", "body_pos_w"), indices)
    np.testing.assert_allclose(gathered["joint_pos"][..., 0], [[2, 0], [1, 2]])
    np.testing.assert_allclose(gathered["body_pos_w"][..., 0, 0], [[2, 0], [1, 2]])
    assert store.loaded_clip_count == before

    with pytest.raises(ValueError, match="read-only"):
        joint[0, 0] = 1.0

    # An unselected field still has the bounded lazy fallback.
    store.gather("smpl_pose", np.asarray([0, 7]))
    assert store.loaded_clip_count > before
    assert store.cached_clip_count <= 1


def test_hot_fields_apply_nonidentity_permutations_only_on_cold_copy(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    np.savez(
        source,
        joint_pos=np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        body_pos_w=np.asarray(
            [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]],
            dtype=np.float32,
        ),
    )
    report = materialize_motion_store(
        [source],
        tmp_path / "store",
        fps=50,
        joint_order=("j0", "j1"),
        body_order=("pelvis", "torso"),
    )

    store = load_sonic_motion_store(
        report.manifest_path,
        expected_joint_order=("j1", "j0"),
        expected_body_order=("torso", "pelvis"),
        hot_fields=("joint_pos", "body_pos_w"),
    )

    assert store._lazy_backend is not None
    assert store._lazy_backend.joint_permutation is not None
    assert store._lazy_backend.body_permutation is not None
    gathered = store.gather_fields(("joint_pos", "body_pos_w"), np.asarray([0, 1]))
    np.testing.assert_allclose(gathered["joint_pos"], [[20.0, 10.0], [40.0, 30.0]])
    np.testing.assert_allclose(gathered["body_pos_w"][..., 0], [[2.0, 1.0], [4.0, 3.0]])


def test_global_mmap_sidecar_is_readonly_global_and_effectively_ordered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import unilab.training.sonic_store as sonic_store_module

    source = tmp_path / "source.npz"
    np.savez(
        source,
        joint_pos=np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        body_pos_w=np.asarray(
            [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]],
            dtype=np.float32,
        ),
        smpl_pose=np.zeros((2, 3), dtype=np.float32),
    )
    report = materialize_motion_store(
        [source],
        tmp_path / "store",
        fps=50,
        joint_order=("j0", "j1"),
        body_order=("pelvis", "torso"),
    )
    sidecar = materialize_sonic_global_mmap_store(
        report.manifest_path,
        tmp_path / "global_mmap",
        fields=("joint_pos", "body_pos_w", "smpl_pose"),
        expected_joint_order=("j1", "j0"),
        expected_body_order=("torso", "pelvis"),
    )

    def reject_source_rescan(*args, **kwargs):
        raise AssertionError("verified global mmap workers must not rescan source clips")

    monkeypatch.setattr(sonic_store_module, "preflight_motion_manifest", reject_source_rescan)

    store = load_sonic_motion_store(
        report.manifest_path,
        expected_joint_order=("j1", "j0"),
        expected_body_order=("torso", "pelvis"),
        rank=1,
        world_size=2,
        shard_clips=False,
        motion_global_mmap_sidecar=sidecar.sidecar_path,
        motion_global_mmap_trusted_receipt=sidecar.trusted_receipt_path,
    )

    # Shared mmap mode deliberately retains the complete manifest clip layout,
    # so global-bin samplers see the same offsets on every rank.
    np.testing.assert_array_equal(store.clip_lengths, [2])
    np.testing.assert_array_equal(store.clip_offsets, [0])
    joint = store.arrays["joint_pos"]
    body = store.arrays["body_pos_w"]
    assert isinstance(joint, np.memmap)
    assert joint.flags.c_contiguous and not joint.flags.writeable
    assert body.flags.c_contiguous and not body.flags.writeable
    before = store.loaded_clip_count
    gathered = store.gather_fields(("joint_pos", "body_pos_w"), np.asarray([1, 0]))
    np.testing.assert_allclose(gathered["joint_pos"], [[40.0, 30.0], [20.0, 10.0]])
    np.testing.assert_allclose(gathered["body_pos_w"][..., 0], [[4.0, 3.0], [2.0, 1.0]])
    assert store.loaded_clip_count == before

    with pytest.raises(ValueError, match="shard_clips"):
        load_sonic_motion_store(
            report.manifest_path,
            motion_global_mmap_sidecar=sidecar.sidecar_path,
            shard_clips=True,
        )
    with pytest.raises(ValueError, match="resident hot_fields"):
        load_sonic_motion_store(
            report.manifest_path,
            motion_global_mmap_sidecar=sidecar.sidecar_path,
            hot_fields=("joint_pos",),
        )


def test_global_mmap_sidecar_rejects_source_manifest_identity_change(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    manifest_path = store.manifest.manifest_path
    assert manifest_path is not None
    sidecar = materialize_sonic_global_mmap_store(
        manifest_path,
        tmp_path / "global_mmap",
        fields=("joint_pos",),
    )
    global_store = load_sonic_motion_store(
        manifest_path,
        rank=1,
        world_size=2,
        shard_clips=False,
        motion_global_mmap_sidecar=sidecar.sidecar_path,
    )
    np.testing.assert_array_equal(global_store.clip_lengths, [3, 4, 5])
    np.testing.assert_array_equal(global_store.clip_offsets, [0, 3, 7])
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="source manifest digest"):
        load_sonic_motion_store(
            manifest_path,
            shard_clips=False,
            motion_global_mmap_sidecar=sidecar.sidecar_path,
        )


def test_global_mmap_sidecar_rejects_field_checksum_change(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    manifest_path = store.manifest.manifest_path
    assert manifest_path is not None
    sidecar = materialize_sonic_global_mmap_store(
        manifest_path,
        tmp_path / "global_mmap",
        fields=("joint_pos",),
    )
    field_path = sidecar.sidecar_path.parent / "arrays" / "joint_pos.npy"
    field = np.load(field_path, mmap_mode="r+", allow_pickle=False)
    field[0, 0] = 123.0
    field.flush()

    with pytest.raises(ValueError, match="SHA256 checksum mismatch"):
        load_sonic_motion_store(
            manifest_path,
            shard_clips=False,
            motion_global_mmap_sidecar=sidecar.sidecar_path,
        )


def test_global_mmap_trusted_receipt_rejects_changed_npy_stat(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    manifest_path = store.manifest.manifest_path
    assert manifest_path is not None
    sidecar = materialize_sonic_global_mmap_store(
        manifest_path,
        tmp_path / "global_mmap",
        fields=("joint_pos", "body_pos_w", "smpl_pose"),
    )
    field_path = sidecar.sidecar_path.parent / "arrays" / "joint_pos.npy"
    field = np.load(field_path, mmap_mode="r+", allow_pickle=False)
    field[0, 0] = 123.0
    field.flush()

    with pytest.raises(ValueError, match="stat identity differs"):
        load_sonic_motion_store(
            manifest_path,
            shard_clips=False,
            motion_global_mmap_sidecar=sidecar.sidecar_path,
            motion_global_mmap_trusted_receipt=sidecar.trusted_receipt_path,
        )


def test_sonic_owner_defaults_and_forwards_global_mmap_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import unilab.training.sonic_store as sonic_store_module
    from unilab.envs.motion_tracking.g1.sonic import SonicG1TrackingCfg, SonicG1TrackingEnv

    cfg = SonicG1TrackingCfg()
    assert cfg.motion_global_mmap_sidecar is None
    assert cfg.motion_global_mmap_trusted_receipt is None
    cfg.motion_manifest = "/cold-path/manifest.json"
    cfg.motion_global_mmap_sidecar = "/cold-path/metadata.json"
    cfg.motion_global_mmap_trusted_receipt = "/cold-path/trusted-receipt.json"
    seen: dict[str, object] = {}

    def load_stub(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        return object()

    monkeypatch.setattr(sonic_store_module, "load_sonic_motion_store", load_stub)
    assert SonicG1TrackingEnv._resolve_store(cfg) is not None
    assert seen["kwargs"]["motion_global_mmap_sidecar"] == cfg.motion_global_mmap_sidecar
    assert seen["kwargs"]["motion_global_mmap_trusted_receipt"] == (
        cfg.motion_global_mmap_trusted_receipt
    )


@pytest.mark.parametrize(
    "hot_fields, match",
    [
        (("joint_pos", "joint_pos"), "duplicate"),
        (("not_a_field",), "unknown"),
    ],
)
def test_hot_fields_fail_closed_for_duplicate_or_unknown_names(
    tmp_path: Path, hot_fields: tuple[str, ...], match: str
) -> None:
    manifest_path = _make_store(tmp_path).manifest.manifest_path
    assert manifest_path is not None
    with pytest.raises(ValueError, match=match):
        load_sonic_motion_store(manifest_path, hot_fields=hot_fields)


def test_cache_size_rejects_invalid_values(tmp_path: Path) -> None:
    # Build a valid manifest first so the argument validation is exercised at
    # the public loader boundary rather than in a private cache constructor.
    store_dir = _make_store(tmp_path).manifest.manifest_path
    assert store_dir is not None
    with pytest.raises(ValueError, match="cache_size"):
        load_sonic_motion_store(store_dir, cache_size=-1)
    with pytest.raises(ValueError, match="cache_size"):
        load_sonic_motion_store(store_dir, cache_size=True)
