from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from unilab.envs.locomotion.common.raycast_scan import (
    RaycastScanConfig,
    init_raycast_scan_sensor,
    raycast_height_scan_obs,
    raycast_scan_directions,
    raycast_scan_pattern,
)
from unilab.envs.locomotion.g1.joystick import G1WalkEnv, G1WalkFlatCfg
from unilab.envs.locomotion.g1.rough_raycast import (
    G1WalkRoughRaycastCfg,
    G1WalkRoughRaycastEnv,
)


def test_raycast_scan_fan_direction_count_and_norms() -> None:
    cfg = RaycastScanConfig(enabled=True, pattern="fan", num_rays=17, cutoff=5.0)

    directions = raycast_scan_directions(cfg)

    assert directions.shape == (17, 3)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert directions.flags.c_contiguous


def test_raycast_scan_grid_matches_mjlab_g1_pattern() -> None:
    cfg = RaycastScanConfig(
        enabled=True,
        pattern="grid",
        grid_size=[1.6, 1.0],
        resolution=0.1,
        direction=[0.0, 0.0, -1.0],
    )

    origin_offsets, directions = raycast_scan_pattern(cfg)

    assert origin_offsets.shape == (187, 3)
    assert directions.shape == (187, 3)
    np.testing.assert_allclose(
        np.unique(origin_offsets[:, 0]), np.linspace(-0.8, 0.8, 17), atol=1.0e-12
    )
    np.testing.assert_allclose(
        np.unique(origin_offsets[:, 1]), np.linspace(-0.5, 0.5, 11), atol=1.0e-12
    )
    np.testing.assert_allclose(origin_offsets[:, 2], 0.0)
    np.testing.assert_allclose(directions, np.tile([0.0, 0.0, -1.0], (187, 1)))


def test_raycast_scan_obs_uses_backend_sensor_once() -> None:
    class FakeRaycaster:
        def __init__(self) -> None:
            self.calls = 0

        def cast(self):
            self.calls += 1
            return SimpleNamespace(
                distances=np.asarray([[1.0, -1.0], [2.0, 5.5]], dtype=np.float32),
                geom_ids=np.zeros((2, 2), dtype=np.int32),
                normals=None,
            )

    class FakeBackend:
        def __init__(self) -> None:
            self.raycaster = FakeRaycaster()

        def get_body_id(self, name: str) -> int:
            assert name == "pelvis"
            return 3

        def get_base_pos(self) -> np.ndarray:
            return np.zeros((2, 3), dtype=np.float32)

        def create_raycaster(self, **kwargs):
            assert kwargs["frame_body_id"] == 3
            assert kwargs["alignment"] == "yaw"
            np.testing.assert_allclose(kwargs["origin_offsets"], np.zeros((2, 3)))
            return self.raycaster

    env = SimpleNamespace(_backend=FakeBackend())
    cfg = RaycastScanConfig(
        enabled=True,
        frame_body_name="pelvis",
        pattern="fan",
        num_rays=2,
        cutoff=5.0,
        scale=0.2,
    )
    init_raycast_scan_sensor(env, cfg, "pelvis")

    obs = raycast_height_scan_obs(env, cfg, 2)

    z_scale = -env._raycast_scan_directions[:, 2] * cfg.scale
    np.testing.assert_allclose(obs, [[1.0 * z_scale[0], 1.0], [2.0 * z_scale[0], 1.0]])
    assert env._backend.raycaster.calls == 1


def test_g1_raycast_cfg_obs_dims_include_scan() -> None:
    cfg = G1WalkRoughRaycastCfg()
    env = object.__new__(G1WalkRoughRaycastEnv)
    env._raycast_scan_dim = 187

    assert cfg.raycast_scan.enabled is True
    assert cfg.raycast_scan.pattern == "grid"
    assert cfg.raycast_scan.grid_size == [1.6, 1.0]
    assert cfg.raycast_scan.resolution == 0.1
    assert cfg.scene.terrain is not None
    assert cfg.scene.terrain.generator is not None
    assert cfg.scene.terrain.generator.curriculum is True
    assert cfg.scene.terrain.generator.size == (8.0, 8.0)
    assert cfg.scene.terrain.generator.num_rows == 10
    assert cfg.scene.terrain.generator.num_cols == 20
    assert cfg.scene.terrain.generator.border_width == 20.0
    assert cfg.scene.terrain.generator.sub_terrains["flat"].proportion == 0.2
    assert env.obs_groups_spec == {"obs": 285, "critic": 288}


def test_g1_raycast_env_updates_mjlab_style_velocity_commands() -> None:
    cfg = G1WalkRoughRaycastCfg()
    cfg.commands.heading_command = True
    cfg.commands.heading_control_stiffness = 0.5
    cfg.commands.resampling_time = 0.0
    env = object.__new__(G1WalkRoughRaycastEnv)
    env._cfg = cfg
    env._num_envs = 2
    env._backend = SimpleNamespace(
        get_base_quat=lambda: np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]]), (2, 1))
    )
    info = {
        "commands": np.asarray([[0.2, 0.1, 9.0], [0.3, -0.1, -9.0]], dtype=np.float32),
        "heading_commands": np.asarray([1.0, -1.0], dtype=np.float32),
    }

    env._update_velocity_commands(info)

    np.testing.assert_allclose(info["commands"][:, 2], [0.5, -0.5], atol=1.0e-6)


def test_g1_raycast_env_forward_command_subset_matches_mjlab_rule() -> None:
    cfg = G1WalkRoughRaycastCfg()
    cfg.commands.rel_forward_envs = 1.0
    env = object.__new__(G1WalkRoughRaycastEnv)
    env._cfg = cfg
    commands = np.asarray([[-0.1, 0.5, 0.2], [-2.0, -0.3, -0.4]], dtype=np.float32)

    env._apply_forward_command_subset(commands)

    np.testing.assert_allclose(commands[:, 0], [0.3, 2.0])
    np.testing.assert_allclose(commands[:, 1:], 0.0)


def test_g1_base_walk_env_obs_dims_unchanged() -> None:
    cfg = G1WalkFlatCfg()
    env = object.__new__(G1WalkEnv)

    assert not hasattr(cfg, "raycast_scan")
    assert env.obs_groups_spec == {"obs": 98, "critic": 101}
