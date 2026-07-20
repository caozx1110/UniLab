"""T10.2 (motion velocity reference targets) + T10.3 (teacher/student reward target).

T10.2: root_vel/root_ang_vel tracking errors are body-frame diffs (GH
motion_tracking.py:369-406); joint_vel_tracking uses (cur-last)/step_dt vs motion
joint_vel over the 17-subset (:433). T10.3: teacher = current motion frame; student =
StudentRootCache head with robot-anchored t->t+50 tail (update_reward_target :442-484).
"""
import numpy as np

from unilab.base.np_env import NpEnvState
from unilab.envs.gh_tracking import rewards as rwd
from unilab.envs.gh_tracking.motion_dataset import write_synthetic_dataset


def _minimal_state(n):
    return NpEnvState(obs={}, reward=np.zeros(n), terminated=np.zeros(n, bool),
                      truncated=np.zeros(n, bool), info={"steps": np.zeros(n, np.uint32)})


def _make_env(tmp_path, n=2, student_train=False):
    from unilab.envs.gh_tracking.config import GHTrackingCfg
    from unilab.envs.gh_tracking.env import GHTrackingEnv
    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[200, 300], seed=0)
    cfg = GHTrackingCfg()
    cfg.motion.dirs = [str(tmp_path / "interx")]
    cfg.motion.weights = [1.0]
    cfg.student_train = student_train
    env = GHTrackingEnv(cfg, num_envs=n, backend_type="mujoco")
    env._reset_idx(np.arange(n))
    return env, cfg


def _drive_one_step(env, cfg, n):
    state = _minimal_state(n)
    ctrl = env.apply_action(np.zeros((n, 29)), state)
    env._backend.step(ctrl, cfg.sim_substeps)
    return env.update_state(state)


# --- T10.2 ---------------------------------------------------------------- #

def test_body_frame_vel_err_identity_quat_reduces_to_world_norm():
    from unilab.envs.gh_tracking.env import GHTrackingEnv
    q = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))   # identity -> body == world
    cur = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    ref = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    err = GHTrackingEnv._body_frame_vel_err(q, cur, q, ref)
    np.testing.assert_allclose(err, np.linalg.norm(ref - cur, axis=-1, keepdims=True))


def test_velocity_reference_targets_wired(tmp_path):
    env, cfg = _make_env(tmp_path, 2)
    _drive_one_step(env, cfg, 2)
    nj = env._track_joint_ids.shape[0]
    assert env._rc["root_ang_vel_err"].shape == (2, 1)
    assert env._rc["track_joint_vel_target"].shape == (2, nj)
    assert env._rc["track_joint_vel_diff"].shape == (2, nj)
    # no longer the zero placeholders; motion has nonzero ang-vel/joint-vel reference
    assert np.any(env._rc["root_ang_vel_err"] > 0)
    assert np.any(env._rc["track_joint_vel_target"] != 0)
    assert np.isfinite(env._rc["track_joint_vel_diff"]).all()
    env.close()


# --- T10.3 ---------------------------------------------------------------- #

def test_teacher_target_is_current_motion_frame(tmp_path):
    env, cfg = _make_env(tmp_path, 2, student_train=False)
    assert env._reward_manager.student_train is False
    _drive_one_step(env, cfg, 2)
    m_pos0 = env._motion_slice.root_pos_w[:, 0].astype(np.float64)
    m_quat0 = env._motion_slice.root_quat_w[:, 0].astype(np.float64)
    exp_pos, exp_quat = rwd.teacher_reward_target(m_pos0, m_quat0, np.zeros((2, 3)))
    np.testing.assert_allclose(env._rc["reward_root_pos_w"], exp_pos)
    np.testing.assert_allclose(env._rc["reward_root_quat_w"], exp_quat)
    env.close()


def test_student_target_consumes_cache_head_and_rolls(tmp_path):
    env, cfg = _make_env(tmp_path, 2, student_train=True)
    assert env._reward_manager.student_train is True

    # step 1: _before_update fills the cache (GH last_reset_env_ids, post-increment t),
    # _compute_reward consumes+rolls it. Capture the head left after step 1.
    _drive_one_step(env, cfg, 2)
    assert np.isfinite(env._rc["reward_root_pos_w"]).all()
    head1 = env.student_cache.ts_root_pos_w[:, 0].copy()

    # step 2 (no reset -> no re-fill): reward target == step-1's post-head, cache rolls again
    _drive_one_step(env, cfg, 2)
    np.testing.assert_allclose(env._rc["reward_root_pos_w"], head1)
    assert not np.allclose(env.student_cache.ts_root_pos_w[:, 0], head1)
    env.close()
