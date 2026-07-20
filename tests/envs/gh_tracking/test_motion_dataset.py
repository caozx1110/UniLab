"""Tests for the GH motion dataset loader (Phase 3).

Synthetic fixtures drive all schema / sampling / gather logic. Real
interx/lafan/amass schema acceptance and elementwise parity vs GH are deferred
to when the real data lands (DP2: marked deferred, not parity-passed).
"""

from __future__ import annotations

import numpy as np

from unilab.envs.gh_tracking.motion_dataset import (
    MotionDatasetSource,
    MotionSlice,
    WeightedMotionDataset,
    motion_to_init_qpos_qvel,
    write_synthetic_dataset,
)
from unilab.utils.rotation import np_quat_apply, np_yaw_to_quat

NUM_J, NUM_B = 29, 28


def _make_slice(
    root_quat: np.ndarray | None = None,
    root_ang_vel_w: np.ndarray | None = None,
    root_lin_vel_w: np.ndarray | None = None,
    root_pos_w: np.ndarray | None = None,
) -> MotionSlice:
    """Build a single-env, single-frame (N=1, T=1) MotionSlice for init tests."""
    def _f(v, feat):
        a = np.zeros((1, 1, *feat), dtype=np.float32)
        if v is not None:
            a[0, 0] = v
        return a

    quat = np.array([1.0, 0.0, 0.0, 0.0]) if root_quat is None else root_quat
    return MotionSlice(
        motion_id=np.zeros((1, 1), dtype=np.int32),
        step=np.zeros((1, 1), dtype=np.int32),
        root_pos_w=_f(root_pos_w, (3,)),
        root_quat_w=_f(quat, (4,)),
        root_lin_vel_w=_f(root_lin_vel_w, (3,)),
        root_ang_vel_w=_f(root_ang_vel_w, (3,)),
        joint_pos=_f(None, (NUM_J,)),
        joint_vel=_f(None, (NUM_J,)),
        body_pos_w=_f(None, (NUM_B, 3)),
        body_pos_b=_f(None, (NUM_B, 3)),
        body_quat_w=_f(None, (NUM_B, 4)),
    )


# --- 3.1 single-dataset memmap source + schema ----------------------------- #


def test_synthetic_dataset_roundtrip_schema(tmp_path) -> None:
    d = tmp_path / "interx"
    write_synthetic_dataset(str(d), clip_lengths=[100, 250, 40], seed=0)
    src = MotionDatasetSource(str(d))

    assert src.num_motions == 3
    np.testing.assert_array_equal(src.lengths, [100, 250, 40])
    np.testing.assert_array_equal(src.starts, [0, 100, 350])
    np.testing.assert_array_equal(src.ends, [100, 350, 390])
    assert len(src.joint_names) == NUM_J
    assert len(src.body_names) == NUM_B

    # float fields -> f16, int fields -> i32
    assert src.field("root_pos_w").dtype == np.float16
    assert src.field("step").dtype == np.int32

    # field feature shapes (total_frames, *feature)
    assert src.field("root_pos_w").shape == (390, 3)
    assert src.field("root_quat_w").shape == (390, 4)
    assert src.field("root_lin_vel_w").shape == (390, 3)
    assert src.field("root_ang_vel_w").shape == (390, 3)
    assert src.field("joint_pos").shape == (390, NUM_J)
    assert src.field("joint_vel").shape == (390, NUM_J)
    assert src.field("body_pos_w").shape == (390, NUM_B, 3)
    assert src.field("body_pos_b").shape == (390, NUM_B, 3)
    assert src.field("body_quat_w").shape == (390, NUM_B, 4)


def test_body_names_include_mimic_markers(tmp_path) -> None:
    d = tmp_path / "interx"
    write_synthetic_dataset(str(d), clip_lengths=[10], seed=1)
    src = MotionDatasetSource(str(d))
    for nm in ("head_mimic", "left_hand_mimic", "right_hand_mimic"):
        assert nm in src.body_names


def test_step_field_is_within_clip_frame_index(tmp_path) -> None:
    d = tmp_path / "interx"
    write_synthetic_dataset(str(d), clip_lengths=[5, 7], seed=2)
    src = MotionDatasetSource(str(d))
    step = src.field("step")
    np.testing.assert_array_equal(step[0:5], [0, 1, 2, 3, 4])  # clip 0 frames
    np.testing.assert_array_equal(step[5:12], [0, 1, 2, 3, 4, 5, 6])  # clip 1 frames


# --- 3.2 weighted multi-dataset sampling + sample_once fixed clip ----------- #


def test_weights_normalized_and_empirical_distribution(tmp_path) -> None:
    for nm in ("interx", "lafan", "amass"):
        write_synthetic_dataset(str(tmp_path / nm), clip_lengths=[50] * 20, seed=abs(hash(nm)) % 1000)
    ds = WeightedMotionDataset(
        [str(tmp_path / n) for n in ("interx", "lafan", "amass")],
        weights=[0.4, 0.2, 0.4],
        env_size=6000,
        max_step=1000,
        seed=0,
    )
    np.testing.assert_allclose(ds.probs, [0.4, 0.2, 0.4], atol=1e-6)
    frac = np.bincount(ds._env_dataset_idx, minlength=3) / 6000
    np.testing.assert_allclose(frac, [0.4, 0.2, 0.4], atol=0.03)  # empirical ~ weights


def test_sample_once_fixed_clip_reset_keeps_clip(tmp_path) -> None:
    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[80, 120, 200], seed=1)
    ds = WeightedMotionDataset([str(tmp_path / "interx")], [1.0], env_size=32, max_step=1000, seed=2)

    before = ds.clip_of(np.arange(32)).copy()
    lengths = ds.reset(np.arange(32))
    after = ds.clip_of(np.arange(32))

    np.testing.assert_array_equal(before, after)  # reset does NOT re-draw the clip
    np.testing.assert_array_equal(lengths, ds._len[np.arange(32)])


def test_len_is_clip_length_clamped_to_max_step(tmp_path) -> None:
    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[1500, 300], seed=3)
    ds = WeightedMotionDataset([str(tmp_path / "interx")], [1.0], env_size=64, max_step=1000, seed=4)
    # every env's _len is its clip length clamped to max_step (1000)
    ds_idx, mid = ds.clip_of(np.arange(64)).T
    src = ds._sources[0]
    expected = np.minimum(src.lengths[mid], 1000)
    np.testing.assert_array_equal(ds._len[np.arange(64)], expected)


# --- 3.3 future-frame gather (clamp-to-last) + z offset --------------------- #

FUTURE = np.array([0, 2, 4, 8, 16])


def test_future_gather_indices_and_clamp_to_last(tmp_path) -> None:
    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[20], seed=3)
    ds = WeightedMotionDataset([str(tmp_path / "interx")], [1.0], env_size=1, max_step=1000, seed=0)
    # start=10 -> requested frames 10,12,14,18,26; clamp 26 -> last valid (len-1=19)
    sl = ds.get_slice(np.array([0]), np.array([10]), FUTURE)
    np.testing.assert_array_equal(sl.step[0], [10, 12, 14, 18, 19])


def test_get_slice_applies_z_offset(tmp_path) -> None:
    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[30], seed=4, zero_pos=True)
    ds = WeightedMotionDataset([str(tmp_path / "interx")], [1.0], env_size=1, max_step=1000, seed=0)
    sl = ds.get_slice(np.array([0]), np.array([0]), np.array([0]))
    np.testing.assert_allclose(sl.root_pos_w[0, 0, 2], 0.035, atol=1e-3)
    np.testing.assert_allclose(sl.body_pos_w[0, 0, :, 2], 0.035, atol=1e-3)


def test_get_slice_returns_float32(tmp_path) -> None:
    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[30], seed=5)
    ds = WeightedMotionDataset([str(tmp_path / "interx")], [1.0], env_size=1, max_step=1000, seed=0)
    sl = ds.get_slice(np.array([0]), np.array([0]), FUTURE)
    assert sl.root_pos_w.dtype == np.float32
    assert sl.joint_pos.dtype == np.float32
    assert sl.root_pos_w.shape == (1, 5, 3)
    assert sl.joint_pos.shape == (1, 5, 29)


# --- 3.4 episode start-offset sampling ------------------------------------- #


def test_zero_init_prob_one_gives_zero_start(tmp_path) -> None:
    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[500] * 8, seed=6)
    ds = WeightedMotionDataset([str(tmp_path / "interx")], [1.0], env_size=8, max_step=1000, seed=0)
    t = ds.sample_start_offsets(np.arange(8), zero_init_prob=1.0, rng=np.random.default_rng(0))
    # rand > 1.0 is never true -> gate zeroes every start
    np.testing.assert_array_equal(t, np.zeros(8, dtype=t.dtype))


def test_start_offset_within_valid_bound(tmp_path) -> None:
    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[500] * 2000, seed=7)
    ds = WeightedMotionDataset([str(tmp_path / "interx")], [1.0], env_size=2000, max_step=1000, seed=0)
    lengths = ds._len[np.arange(2000)]
    t = ds.sample_start_offsets(np.arange(2000), zero_init_prob=0.0, rng=np.random.default_rng(1))
    max_start = lengths - 16 - 1  # future_steps[-1] = 16
    assert (t >= 0).all()
    assert (t <= 0.75 * max_start).all()  # offsets = floor(rand * 0.75 * max_start)
    assert t.max() > 0  # zero_init_prob=0 -> nonzero starts appear


def test_start_offset_leaves_future_horizon_in_clip(tmp_path) -> None:
    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[100] * 500, seed=8)
    ds = WeightedMotionDataset([str(tmp_path / "interx")], [1.0], env_size=500, max_step=1000, seed=0)
    lengths = ds._len[np.arange(500)]
    t = ds.sample_start_offsets(np.arange(500), zero_init_prob=0.0, rng=np.random.default_rng(2))
    # start + 0.75-bounded offset stays well within [0, len-17] so the [0..16]
    # horizon never needs clamping at reset time
    assert (t + 16 <= lengths - 1).all()


# --- 3.5 motion frame -> MuJoCo init qpos/qvel (world->local ang vel) ------- #


def test_qpos_qvel_layout_and_lift(tmp_path) -> None:
    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[30], seed=8)
    ds = WeightedMotionDataset([str(tmp_path / "interx")], [1.0], env_size=1, max_step=1000, seed=0)
    sl = ds.get_slice(np.array([0]), np.array([0]), np.array([0]))

    qpos, qvel = motion_to_init_qpos_qvel(sl, lift_height=0.04)

    assert qpos.shape == (1, 36)  # [root_pos(3), root_quat(4 wxyz), joint_pos(29)]
    assert qvel.shape == (1, 35)  # [root_lin_vel(3), root_ang_vel(3), joint_vel(29)]
    np.testing.assert_allclose(qpos[0, 3:7], sl.root_quat_w[0, 0], atol=1e-3)  # quat direct
    np.testing.assert_allclose(qpos[0, 7:], sl.joint_pos[0, 0], atol=1e-3)
    # slice z already has +0.035; init adds lift +0.04 on top
    np.testing.assert_allclose(qpos[0, 2], sl.root_pos_w[0, 0, 2] + 0.04, atol=1e-3)


def test_ang_vel_written_in_local_frame() -> None:
    # world ang vel + yaw90 root -> qvel[3:6] must be the LOCAL-frame vector,
    # which round-trips back to world via the base quat (get_base_ang_vel_world).
    quat = np_yaw_to_quat(np.array([np.pi / 2]))[0]  # (4,) wxyz
    ang_world = np.array([1.0, 0.0, 0.0])
    sl = _make_slice(root_quat=quat, root_ang_vel_w=ang_world)

    _, qvel = motion_to_init_qpos_qvel(sl, lift_height=0.04)
    local = qvel[0, 3:6]

    # re-applying the root quat recovers the world-frame vector
    np.testing.assert_allclose(np_quat_apply(quat[None], local[None])[0], ang_world, atol=1e-6)
    # and it is NOT the world vector written directly (yaw90 maps world x -> local -y here)
    assert not np.allclose(local, ang_world, atol=1e-3)


def test_lin_vel_written_in_world_frame() -> None:
    sl = _make_slice(root_lin_vel_w=np.array([0.3, -0.2, 0.1]))
    _, qvel = motion_to_init_qpos_qvel(sl, lift_height=0.04)
    np.testing.assert_allclose(qvel[0, 0:3], [0.3, -0.2, 0.1], atol=1e-6)  # world, direct
