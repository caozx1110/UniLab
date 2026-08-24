from __future__ import annotations

from pathlib import Path

import numpy as np

from unilab.training.sonic_motion import materialize_motion_store
from unilab.training.sonic_store import load_sonic_motion_store


def _write_clip(path: Path, value: float, frames: int) -> None:
    joint = np.full((frames, 2), value, dtype=np.float32)
    body = np.full((frames, 1, 3), value, dtype=np.float32)
    np.savez(path, joint_pos=joint, body_pos_w=body)


def _make_store(tmp_path: Path, *, cache_size: int = 2):
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
    return load_sonic_motion_store(report.manifest_path, cache_size=cache_size)


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

    gathered = store.gather_fields(("joint_pos", "body_pos_w"), np.asarray([0, 3, 7]))

    np.testing.assert_allclose(gathered["joint_pos"][:, 0], [0, 1, 2])
    np.testing.assert_allclose(gathered["body_pos_w"][:, 0, 0], [0, 1, 2])
    assert store.loaded_clip_count == before + 3


def test_cache_size_rejects_invalid_values(tmp_path: Path) -> None:
    # Build a valid manifest first so the argument validation is exercised at
    # the public loader boundary rather than in a private cache constructor.
    store_dir = _make_store(tmp_path).manifest.manifest_path
    assert store_dir is not None
    import pytest

    with pytest.raises(ValueError, match="cache_size"):
        load_sonic_motion_store(store_dir, cache_size=-1)
    with pytest.raises(ValueError, match="cache_size"):
        load_sonic_motion_store(store_dir, cache_size=True)
