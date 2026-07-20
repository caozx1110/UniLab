"""GH motion dataset loader for the MuJoCo migration (Phase 3).

Pure-numpy loader mirroring GH's ``MotionDataset`` / ``ProgressiveMultiMotionDataset``
semantics (source: gentle-humanoid-training ``utils/motion.py`` + ``utils/multimotion.py``):

- float16 memmap storage, one ``<field>.memmap`` per field + ``meta.json``
- weighted multi-dataset sampling (per-env fixed clip under ``sample_once``)
- future-frame gather ``[0,2,4,8,16]`` with clamp-to-last
- z offset (+0.035) on load; reset lift (+0.04) at init
- world->local base angular-velocity conversion for MuJoCo free-joint qvel

Real interx/lafan/amass schema acceptance + elementwise parity vs GH are done
when the real data lands; synthetic fixtures drive the logic tests here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

# GH joint/body order (motion.py:104-105). 29 joints, 28 bodies (incl. 3 mimic).
JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link",
    "head_mimic", "left_hand_mimic", "right_hand_mimic",
)
NUM_JOINTS = len(JOINT_NAMES)  # 29
NUM_BODIES = len(BODY_NAMES)  # 27 (GH real training: "world" is dropped by
# select_in_order(return_missing=False) since the GMR G1 model has no "world" body)

FUTURE_STEPS: np.ndarray = np.array([0, 2, 4, 8, 16], dtype=np.int64)
Z_OFFSET = 0.035
LIFT_HEIGHT = 0.04
MAX_STEP_SIZE = 1000

# field name -> (dtype, per-frame feature shape). float->f16, int->i32.
_FLOAT = np.float16
_INT = np.int32
FIELD_SPEC: dict[str, tuple[type, tuple[int, ...]]] = {
    "motion_id": (_INT, ()),
    "step": (_INT, ()),
    "root_pos_w": (_FLOAT, (3,)),
    "root_quat_w": (_FLOAT, (4,)),
    "root_lin_vel_w": (_FLOAT, (3,)),
    "root_ang_vel_w": (_FLOAT, (3,)),
    "joint_pos": (_FLOAT, (NUM_JOINTS,)),
    "joint_vel": (_FLOAT, (NUM_JOINTS,)),
    "body_pos_w": (_FLOAT, (NUM_BODIES, 3)),
    "body_pos_b": (_FLOAT, (NUM_BODIES, 3)),
    "body_quat_w": (_FLOAT, (NUM_BODIES, 4)),
}


class MotionDatasetSource:
    """Lazy per-field memmap view of one on-disk GH motion dataset directory.

    Layout (matches tensordict ``MotionData.memmap``): ``meta.json`` +
    one raw ``<field>.memmap`` per field.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        with open(os.path.join(path, "meta_motion.json")) as f:
            meta = json.load(f)
        self.body_names: list[str] = list(meta["body_names"])
        self.joint_names: list[str] = list(meta["joint_names"])
        self.starts = np.asarray(meta["starts"], dtype=np.int64)
        self.ends = np.asarray(meta["ends"], dtype=np.int64)
        self.lengths = self.ends - self.starts
        self._total = int(self.ends[-1]) if len(self.ends) else 0
        self._fields: dict[str, np.memmap] = {}

    @property
    def num_motions(self) -> int:
        return len(self.starts)

    def field(self, name: str) -> np.memmap:
        mm = self._fields.get(name)
        if mm is None:
            dtype, feat = FIELD_SPEC[name]
            mm = np.memmap(
                os.path.join(self.path, f"{name}.memmap"),
                dtype=dtype,
                mode="r",
                shape=(self._total, *feat),
            )
            self._fields[name] = mm
        return mm


def write_synthetic_dataset(
    path: str,
    clip_lengths: list[int],
    seed: int = 0,
    zero_pos: bool = False,
) -> None:
    """Write a GH-isomorphic synthetic dataset (meta_motion.json + memmaps).

    For tests/fixtures only. ``zero_pos`` zeros positional fields so the +0.035
    z offset is checkable in isolation.
    """
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(seed)
    starts, ends, cursor = [], [], 0
    for length in clip_lengths:
        starts.append(cursor)
        cursor += int(length)
        ends.append(cursor)
    total = cursor

    for name, (dtype, feat) in FIELD_SPEC.items():
        arr = np.zeros((total, *feat), dtype=dtype)
        if name == "step":
            for s, e in zip(starts, ends):
                arr[s:e] = np.arange(e - s, dtype=dtype)
        elif name == "motion_id":
            for i, (s, e) in enumerate(zip(starts, ends)):
                arr[s:e] = i
        elif name in ("root_pos_w", "body_pos_w", "body_pos_b") and zero_pos:
            pass  # leave zeros
        else:
            arr[:] = rng.standard_normal(arr.shape).astype(dtype)
        arr.tofile(os.path.join(path, f"{name}.memmap"))

    meta = {
        "body_names": list(BODY_NAMES),
        "joint_names": list(JOINT_NAMES),
        "starts": starts,
        "ends": ends,
        "info": [],
    }
    with open(os.path.join(path, "meta_motion.json"), "w") as f:
        json.dump(meta, f)


class WeightedMotionDataset:
    """Weighted multi-dataset sampler with per-env prefetched clip buffers.

    Mirrors GH ``ProgressiveMultiMotionDataset`` under ``sample_once=True``: each
    env is bound to one (dataset, motion) clip for the whole phase; ``reset`` only
    re-samples the start offset within that fixed clip (it does not re-draw the
    clip). Per-env clips are prefetched into float16 buffers of length ``max_step``.
    """

    def __init__(
        self,
        dirs: list[str],
        weights: list[float],
        env_size: int,
        max_step: int = MAX_STEP_SIZE,
        seed: int = 0,
        sample_once: bool = True,
    ) -> None:
        assert len(dirs) == len(weights)
        self._sources = [MotionDatasetSource(d) for d in dirs]
        b0, j0 = self._sources[0].body_names, self._sources[0].joint_names
        for s in self._sources[1:]:
            assert s.body_names == b0 and s.joint_names == j0
        self.body_names = b0
        self.joint_names = j0

        w = np.asarray(weights, dtype=np.float64)
        self.probs = (w / w.sum()).astype(np.float64)
        self.env_size = int(env_size)
        self.max_step = int(max_step)
        self.sample_once = sample_once
        self._rng = np.random.default_rng(seed)

        self._env_dataset_idx = np.zeros(env_size, dtype=np.int64)
        self._env_motion_id = np.zeros(env_size, dtype=np.int64)
        self._len = np.zeros(env_size, dtype=np.int64)
        self._buf: dict[str, np.ndarray] = {}
        for name, (dtype, feat) in FIELD_SPEC.items():
            buf_dtype = np.float16 if np.issubdtype(dtype, np.floating) else dtype
            self._buf[name] = np.zeros((env_size, max_step, *feat), dtype=buf_dtype)

        self._populate_buffer_full()

    def _populate_buffer_full(self) -> None:
        path_samples = self._rng.choice(len(self._sources), size=self.env_size, p=self.probs)
        self._env_dataset_idx[:] = path_samples
        steps = np.arange(self.max_step, dtype=np.int64)
        for pi, src in enumerate(self._sources):
            mask = path_samples == pi
            k = int(mask.sum())
            if k == 0:
                continue
            mids = np.floor(self._rng.random(k) * src.num_motions).astype(np.int64)
            self._env_motion_id[mask] = mids
            local_starts = src.starts[mids]
            local_ends = src.ends[mids] - 1
            local_idx = local_starts[:, None] + steps[None, :]
            local_idx = np.minimum(local_idx, local_ends[:, None])  # clamp to last frame
            for name in FIELD_SPEC:
                self._buf[name][mask] = src.field(name)[local_idx]
            self._len[mask] = np.minimum(src.lengths[mids], self.max_step)

    def clip_of(self, env_ids: np.ndarray) -> np.ndarray:
        """Return the (dataset_idx, motion_id) each env is bound to, shape (N, 2)."""
        env_ids = np.asarray(env_ids)
        return np.stack(
            [self._env_dataset_idx[env_ids], self._env_motion_id[env_ids]], axis=-1
        )

    def reset(self, env_ids: np.ndarray) -> np.ndarray:
        """Reset selected envs. Under sample_once, keeps the fixed clip and returns
        its length (start-offset re-sampling is done separately via
        :meth:`sample_start_offsets`)."""
        env_ids = np.asarray(env_ids)
        return self._len[env_ids]

    def sample_start_offsets(
        self,
        env_ids: np.ndarray,
        zero_init_prob: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample episode start frames within each env's fixed clip.

        Mirrors GH ``motion_tracking.sample_init``:
        ``max_start = len - future_steps[-1] - 1``,
        ``offsets = floor(rand * 0.75 * max_start)``,
        ``t = offsets * (rand > zero_init_prob)`` so a per-env draw below
        ``zero_init_prob`` starts at frame 0.
        """
        env_ids = np.asarray(env_ids)
        lengths = self._len[env_ids].astype(np.int64)
        max_start = np.maximum(lengths - int(FUTURE_STEPS[-1]) - 1, 0)
        offsets = np.floor(rng.random(len(env_ids)) * 0.75 * max_start).astype(np.int64)
        gate = rng.random(len(env_ids)) > zero_init_prob
        return offsets * gate

    def get_slice(
        self, env_ids: np.ndarray, starts: np.ndarray, steps: np.ndarray
    ) -> "MotionSlice":
        """Gather ``steps`` future frames from each env's fixed clip.

        Indices ``starts[:, None] + steps`` are clamped to the clip's last valid
        frame (repeat-last-frame semantics, GH ``multimotion.get_slice``). Values
        are returned as float32 with the +0.035 z offset applied to root/body
        positions (GH ``_offset_pos_z``).
        """
        env_ids = np.asarray(env_ids)
        starts = np.asarray(starts, dtype=np.int64)
        steps = np.asarray(steps, dtype=np.int64)
        idx = starts[:, None] + steps[None, :]
        last = (self._len[env_ids] - 1)[:, None]
        idx = np.minimum(idx, last)  # clamp-to-last

        out: dict[str, np.ndarray] = {}
        for name, (dtype, _) in FIELD_SPEC.items():
            gathered = self._buf[name][env_ids[:, None], idx]
            if np.issubdtype(dtype, np.floating):
                gathered = gathered.astype(np.float32)
            out[name] = gathered
        out["root_pos_w"][..., 2] += Z_OFFSET
        out["body_pos_w"][..., 2] += Z_OFFSET
        return MotionSlice(**out)


@dataclass
class MotionSlice:
    """One gather result. Float fields are float32; ``(N, T, *feature)``."""

    motion_id: np.ndarray
    step: np.ndarray
    root_pos_w: np.ndarray
    root_quat_w: np.ndarray
    root_lin_vel_w: np.ndarray
    root_ang_vel_w: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_pos_b: np.ndarray
    body_quat_w: np.ndarray


def motion_to_init_qpos_qvel(
    slice0: MotionSlice, lift_height: float = LIFT_HEIGHT
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a motion frame (T index 0) to MuJoCo free-joint qpos/qvel.

    Pure layout/frame conversion (no init noise, no sim write — those belong to
    the env). Layout:
      qpos = [root_pos(3, z + lift), root_quat(4, wxyz), joint_pos(29)]  -> (N, 36)
      qvel = [root_lin_vel(3, world), root_ang_vel(3, LOCAL), joint_vel(29)] -> (N, 35)

    ``root_lin_vel_w`` is world-frame and maps directly to free-joint qvel[0:3].
    ``root_ang_vel_w`` is world-frame but MuJoCo free-joint qvel[3:6] is in the
    root LOCAL frame, so it is rotated by ``quat_apply_inverse(root_quat, .)``.
    Read it back in world frame via the backend's ``get_base_ang_vel_world``
    (NOT the P1-2-afflicted ``get_base_ang_vel``).
    """
    from unilab.utils.rotation import np_quat_apply_inverse

    root_pos = slice0.root_pos_w[:, 0].astype(np.float32).copy()
    root_pos[:, 2] += float(lift_height)
    root_quat = slice0.root_quat_w[:, 0].astype(np.float32)
    joint_pos = slice0.joint_pos[:, 0].astype(np.float32)

    lin_vel_w = slice0.root_lin_vel_w[:, 0].astype(np.float32)  # world -> qvel[0:3]
    ang_vel_w = slice0.root_ang_vel_w[:, 0].astype(np.float32)
    ang_vel_local = np_quat_apply_inverse(root_quat, ang_vel_w).astype(np.float32)
    joint_vel = slice0.joint_vel[:, 0].astype(np.float32)

    qpos = np.concatenate([root_pos, root_quat, joint_pos], axis=-1)
    qvel = np.concatenate([lin_vel_w, ang_vel_local, joint_vel], axis=-1)
    return qpos, qvel
