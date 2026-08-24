from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unilab.algos.torch.sonic_ppo import SonicPPORunner
from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.envs.motion_tracking.g1.sonic import (
    SONIC_ACTION_SCALE,
    SONIC_BODY_ORDER,
    SONIC_JOINT_ORDER,
    SONIC_WRIST_JOINT_INDICES,
    SonicG1TrackingEnv,
)
from unilab.training.sonic_motion import materialize_motion_store
from unilab.training.sonic_store import SonicMotionLoader

mujoco = pytest.importorskip("mujoco")


def _write_clip(path: Path, *, body_order: list[str], joint_order: list[str]) -> None:
    frames = 4
    joints = (
        np.arange(len(joint_order), dtype=np.float32)[None, :]
        + np.arange(frames, dtype=np.float32)[:, None]
    )
    body_pos = np.zeros((frames, len(body_order), 3), dtype=np.float32)
    body_pos[:, :, 0] = np.arange(len(body_order), dtype=np.float32)[None, :] + 1.0
    body_pos[:, :, 2] = 1.0
    body_quat = np.zeros((frames, len(body_order), 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    zeros = np.zeros_like(body_pos)
    np.savez(
        path,
        fps=np.asarray(50, dtype=np.int32),
        joint_pos=joints,
        joint_vel=np.zeros_like(joints),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=zeros,
        body_ang_vel_w=zeros,
    )


def test_registry_and_scene_own_sonic_contract() -> None:
    registry.ensure_registries()
    listed = registry.list_registered_envs()
    assert set(listed["SonicG1Tracking"]["available_backends"]) == {"mujoco"}

    model = mujoco.MjModel.from_xml_path(
        str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_sonic.xml")
    )
    assert model.opt.timestep == pytest.approx(0.005)
    assert model.nu == 29
    assert model.nq == 36
    assert len(SONIC_ACTION_SCALE) == 29
    assert np.all(np.isfinite(SONIC_ACTION_SCALE))
    assert np.all(SONIC_ACTION_SCALE > 0)

    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(model.nu)
    )
    actuator_ids = {name: index for index, name in enumerate(actuator_names)}
    for name in ("left_hip_pitch_joint", "right_hip_pitch_joint"):
        actuator_id = actuator_ids[name]
        assert model.actuator_gainprm[actuator_id, 0] == pytest.approx(99.098)
        assert model.actuator_biasprm[actuator_id, 2] == pytest.approx(-6.309)
        np.testing.assert_allclose(model.actuator_forcerange[actuator_id], [-139.0, 139.0])
    # Ordinary G1 tasks retain their historical actuator contract.
    ordinary = mujoco.MjModel.from_xml_path(
        str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
    )
    ordinary_names = tuple(
        mujoco.mj_id2name(ordinary, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(ordinary.nu)
    )
    ordinary_pitch = ordinary_names.index("left_hip_pitch_joint")
    assert ordinary.actuator_gainprm[ordinary_pitch, 0] == pytest.approx(40.179)


def test_sonic_wrist_feature_uses_release_order() -> None:
    assert SONIC_WRIST_JOINT_INDICES == (19, 20, 21, 26, 27, 28)


def test_sonic_observation_noise_matches_release_scales() -> None:
    from unilab.envs.motion_tracking.g1.sonic import SonicG1TrackingCfg

    noise = SonicG1TrackingCfg().noise_config
    assert noise.level == pytest.approx(1.0)
    assert noise.scale_gravity == pytest.approx(0.05)
    assert noise.scale_gyro == pytest.approx(0.2)
    assert noise.scale_joint_angle == pytest.approx(0.01)
    assert noise.scale_joint_vel == pytest.approx(0.5)


def test_sonic_env_uses_reordered_materialized_store(tmp_path: Path) -> None:
    source_joint_order = list(reversed(SONIC_JOINT_ORDER))
    source_body_order = ["extra_body", *SONIC_BODY_ORDER]
    clip = tmp_path / "source.npz"
    _write_clip(clip, body_order=source_body_order, joint_order=source_joint_order)
    report = materialize_motion_store(
        [clip],
        tmp_path / "store",
        fps=50,
        joint_order=source_joint_order,
        body_order=source_body_order,
    )

    registry.ensure_registries()
    env = registry.make(
        "SonicG1Tracking",
        sim_backend="mujoco",
        num_envs=2,
        env_cfg_override={
            "motion_manifest": str(report.manifest_path),
            "sampling_mode": "start",
        },
    )
    try:
        assert isinstance(env.motion_loader, SonicMotionLoader)
        joint_values = env.motion_loader.store.arrays["joint_pos"][0]
        body_values = env.motion_loader.store.arrays["body_pos_w"][0]
        assert joint_values[0] == pytest.approx(float(len(SONIC_JOINT_ORDER) - 1))
        assert body_values[0, 0] == pytest.approx(2.0)

        obs, info = env.reset()
        assert set(obs) == {"actor_obs", "critic_obs", "tokenizer"}
        assert obs["actor_obs"].shape == (2, 930)
        assert obs["critic_obs"].shape == (2, 1645)
        assert obs["tokenizer"].shape == (2, 1761)
        assert isinstance(info, dict)

        state = env.step(np.zeros((2, 29), dtype=np.float32))
        assert state.obs["actor_obs"].shape == (2, 930)
        assert state.reward.shape == (2,)

        runner = SonicPPORunner(
            env,
            {
                "algo": {
                    "num_steps_per_env": 1,
                    "num_learning_epochs": 1,
                    "num_mini_batches": 1,
                    "save_interval": 99,
                    "learning_rate": 0.0,
                },
                "sonic": {
                    "microbatch_size": 2,
                    "model": {
                        "hidden_dims": [8],
                        "tokenizer_hidden_dim": 8,
                    },
                },
            },
            device="cpu",
            log_dir=tmp_path / "run",
        )
        metrics = runner.learn(1)
        assert runner.current_learning_iteration == 1
        assert np.isfinite(metrics["loss"])
        assert (tmp_path / "run" / "last.pt").is_file()
    finally:
        env.close()


def test_sonic_actuator_order_mismatch_fails_closed() -> None:
    instance = object.__new__(SonicG1TrackingEnv)

    class Backend:
        def get_actuator_names(self) -> tuple[str, ...]:
            return (*SONIC_JOINT_ORDER[:-1], "unexpected_joint")

    instance._backend = Backend()
    with pytest.raises(ValueError, match="29-DoF release order"):
        instance._resolve_actuator_permutation()
