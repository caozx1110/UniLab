"""Tests for the GH action pipeline (Phase 4).

Residual->target + communication delay + alpha lerp + boot protection, mirroring
GH ``JointPosition`` (action.py). Pure-numpy, unit-tested with explicit substeps;
the env wiring (apply_action + reset order for the noised boot-protect pose, D2)
and implicit-vs-explicit PD torque difference (B5) are deferred.
"""

from __future__ import annotations

import numpy as np

from unilab.envs.gh_tracking.action import GHActionPipeline, resolve_action_scaling
from unilab.envs.gh_tracking.motion_dataset import JOINT_NAMES

SCALING = {
    ".*elbow_joint": 1.0, ".*shoulder.*": 1.0, ".*wrist.*": 1.0,
    ".*hip_roll.*": 0.25, ".*hip_yaw.*": 0.25, ".*hip_pitch.*": 0.5,
    ".*knee.*": 0.5, ".*waist.*": 0.25, ".*ankle.*": 0.5,
}


def _pipe(n: int = 4) -> GHActionPipeline:
    return GHActionPipeline(
        list(JOINT_NAMES), SCALING,
        default_joint_pos=np.zeros(29), num_envs=n,
        decimation=4, max_delay=4, boot_protect=True, alpha_jit_scale=0.025,
    )


# --- 4.1 resolve + construction + reset ------------------------------------ #


def test_resolve_action_scaling_per_joint() -> None:
    ids, scale = resolve_action_scaling(list(JOINT_NAMES), SCALING)
    assert len(ids) == 29
    assert scale.shape == (29,)
    jn = list(JOINT_NAMES)
    assert scale[jn.index("left_hip_pitch_joint")] == 0.5
    assert scale[jn.index("left_hip_roll_joint")] == 0.25
    assert scale[jn.index("left_hip_yaw_joint")] == 0.25
    assert scale[jn.index("left_knee_joint")] == 0.5
    assert scale[jn.index("left_ankle_roll_joint")] == 0.5
    assert scale[jn.index("waist_yaw_joint")] == 0.25
    assert scale[jn.index("left_shoulder_pitch_joint")] == 1.0
    assert scale[jn.index("left_elbow_joint")] == 1.0
    assert scale[jn.index("left_wrist_yaw_joint")] == 1.0


def test_pipeline_shapes_and_hist() -> None:
    p = _pipe()
    assert p.hist == 3  # max((4-1)//4 + 1, 3)
    assert p.action_buf.shape == (4, 3, 29)
    assert p.applied_action.shape == (4, 29)


def test_reset_sets_delay_boot_alpha() -> None:
    p = _pipe()
    p.reset(np.arange(4), np.random.default_rng(0))
    assert ((p.delay >= 0) & (p.delay <= 4)).all()
    np.testing.assert_array_equal(p.boot_delay, p.delay)  # boot_delay == delay at reset
    np.testing.assert_allclose(p.alpha, 0.9)  # uniform(0.9, 0.9)
    assert not p.action_buf.any()
    assert not p.applied_action.any()


# --- 4.2 control-step ingest (raw clamp / roll / prev_actions / alpha) ------ #


def test_raw_action_clamped_and_stored_at_slot0() -> None:
    p = _pipe(1)
    p.reset(np.array([0]), np.random.default_rng(0))
    p.start_control_step(np.full((1, 29), 100.0), np.random.default_rng(1))
    np.testing.assert_allclose(p.action_buf[0, 0], 10.0)  # clamp(-10, 10)
    p.start_control_step(np.full((1, 29), -50.0), np.random.default_rng(2))
    np.testing.assert_allclose(p.action_buf[0, 0], -10.0)  # newest at slot 0
    np.testing.assert_allclose(p.action_buf[0, 1], 10.0)  # previous rolled to slot 1


def test_prev_actions_are_raw_history() -> None:
    p = _pipe(1)
    p.reset(np.array([0]), np.random.default_rng(0))
    for v in (1.0, 2.0, 3.0):
        p.start_control_step(np.full((1, 29), v), np.random.default_rng(0))
    pa = p.prev_actions()  # action_buf[:, :3]
    assert pa.shape == (1, 3, 29)
    np.testing.assert_allclose(pa[0, :, 0], [3.0, 2.0, 1.0])  # newest first, raw


def test_action_buf_only_updated_at_control_step_not_substep() -> None:
    p = _pipe(1)
    p.reset(np.array([0]), np.random.default_rng(0))
    p.start_control_step(np.ones((1, 29)), np.random.default_rng(0))
    snap = p.action_buf.copy()
    for s in range(4):
        p.substep_target(s)  # substeps must NOT touch action_buf
    np.testing.assert_array_equal(p.action_buf, snap)


def test_alpha_jitter_walks_within_wide_range() -> None:
    p = _pipe(1)
    p.reset(np.array([0]), np.random.default_rng(0))
    for i in range(200):
        p.start_control_step(np.zeros((1, 29)), np.random.default_rng(i))
        assert 0.8 - 1e-9 <= p.alpha[0, 0] <= 1.0 + 1e-9


# --- 4.3 delay gather + lerp + residual->target ---------------------------- #


def test_delay_index_per_substep_gather() -> None:
    p = _pipe(1)
    p.reset(np.array([0]), np.random.default_rng(0))
    p.delay[:] = 2  # delay=2 -> idx per substep [1,1,0,0]
    for v in (2.0, 4.0, 6.0):  # in raw clamp range; buf: slot0=6(newest), 1=4, 2=2
        p.start_control_step(np.full((1, 29), v), np.random.default_rng(0))
    p.alpha[:] = 1.0  # lerp identity so applied == delayed (set after start jitters alpha)
    p.boot_delay[:] = 0  # isolate delay gather from boot protection
    got = [p.substep_target(s)[0, 0] for s in range(4)]  # target = 0(default) + delayed*scale
    # idx [1,1,0,0] -> history [4,4,6,6]; left_hip_pitch scale 0.5 -> [2,2,3,3]
    np.testing.assert_allclose(got, [2.0, 2.0, 3.0, 3.0], atol=1e-5)


def test_residual_to_target_formula() -> None:
    default = np.arange(29, dtype=float)
    p = GHActionPipeline(
        list(JOINT_NAMES), SCALING, default_joint_pos=default, num_envs=1,
        decimation=4, max_delay=0, boot_protect=False, alpha_jit_scale=None,
    )
    p.reset(np.array([0]), np.random.default_rng(0))
    p.start_control_step(np.ones((1, 29)), np.random.default_rng(0))
    p.alpha[:] = 1.0
    tgt = p.substep_target(0)[0]
    _, scale = resolve_action_scaling(list(JOINT_NAMES), SCALING)
    np.testing.assert_allclose(tgt, default + 1.0 * scale, atol=1e-5)  # default + applied*scale


def test_lerp_smoothing_alpha_0p9() -> None:
    p = _pipe(1)
    p.reset(np.array([0]), np.random.default_rng(0))
    p.start_control_step(np.full((1, 29), 2.0), np.random.default_rng(0))  # raw=2 -> delayed=2
    # pin state AFTER start (which re-jitters alpha); isolate from delay/boot
    p.delay[:] = 0
    p.alpha[:] = 0.9
    p.boot_delay[:] = 0
    a = 0.0
    for s in range(4):
        p.substep_target(s)
        a = a + 0.9 * (2.0 - a)  # applied += 0.9*(delayed - applied)
        np.testing.assert_allclose(p.applied_action[0, 0], a, atol=1e-5)


# --- 4.4 boot protection (D2 noised pose) ---------------------------------- #


def test_boot_protect_uses_pose_then_releases() -> None:
    p = _pipe(1)
    p.reset(np.array([0]), np.random.default_rng(0))
    boot_pose = np.full((1, 29), 0.123)  # stand-in for the NOISED init pose (D2)
    p.set_boot_protect_pose(np.array([0]), boot_pose)
    p.boot_delay[:] = 2  # first 2 substeps clamped to boot pose
    p.delay[:] = 0
    p.alpha[:] = 1.0
    p.start_control_step(np.full((1, 29), 5.0), np.random.default_rng(0))
    t0 = p.substep_target(0)
    t1 = p.substep_target(1)
    t2 = p.substep_target(2)
    np.testing.assert_allclose(t0[0], 0.123, atol=1e-6)  # boot pose verbatim (no scale/residual)
    np.testing.assert_allclose(t1[0], 0.123, atol=1e-6)
    assert not np.allclose(t2[0], 0.123)  # released -> normal target
    np.testing.assert_array_equal(p.boot_delay[0], [0])  # decremented, clamp_min 0


def test_boot_delay_never_negative() -> None:
    p = _pipe(1)
    p.reset(np.array([0]), np.random.default_rng(0))
    p.set_boot_protect_pose(np.array([0]), np.zeros((1, 29)))
    p.boot_delay[:] = 0
    p.start_control_step(np.zeros((1, 29)), np.random.default_rng(0))
    for s in range(4):
        p.substep_target(s)
    assert (p.boot_delay >= 0).all()


# --- 4.5 set_pre_step_control integration ---------------------------------- #


class _RecordingPool:
    def __init__(self) -> None:
        self.controls: list[np.ndarray] = []

    def step(self, state, *, nstep, control, control_spec, return_sensor=False,
             post_step_forward_sensor=False, chunk_size=None):
        self.controls.append(np.array(control, copy=True))
        out = np.asarray(state) + 1.0
        return (out, out[:, :1]) if return_sensor else out


def _fake_backend(fn):
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    b = object.__new__(MuJoCoBackend)
    b._pre_step_control_fn = fn
    b._pre_step_wrench_fn = None
    b._post_step_callback_fn = None
    b._num_envs = 1
    b._np_dtype = np.float32
    b._physics_state = np.zeros((1, 1), dtype=np.float32)
    b._sensor_data = np.zeros((1, 1), dtype=np.float32)
    b._pending_xfrc_applied = np.zeros((1, 0), dtype=np.float64)
    b._post_step_forward_sensor = False
    b._chunk_size = None
    b._pool = _RecordingPool()
    return b


def _deterministic_pipe():
    # no jitter, no boot, delay=0 -> fully deterministic per-substep targets
    p = GHActionPipeline(
        list(JOINT_NAMES), SCALING, default_joint_pos=np.zeros(29), num_envs=1,
        decimation=4, max_delay=0, boot_protect=False, alpha_jit_scale=None,
    )
    p.reset(np.array([0]), np.random.default_rng(0))
    p.start_control_step(np.full((1, 29), 3.0), np.random.default_rng(0))
    return p


def test_pre_step_control_feeds_per_substep_targets() -> None:
    # reference per-substep target trajectory from an identical deterministic pipeline
    ref = _deterministic_pipe()
    expected = [ref.substep_target(s).copy() for s in range(4)]

    p = _deterministic_pipe()
    backend = _fake_backend(p.as_pre_step_control())
    backend.step(np.full((1, 29), 3.0, dtype=np.float32), nsteps=4)

    assert len(backend._pool.controls) == 4  # one call per physics substep
    assert p._substep == 4  # counter advanced 0..4
    for s in range(4):
        np.testing.assert_allclose(
            backend._pool.controls[s][0, 0, :], expected[s][0], atol=1e-5
        )
