"""G1 rough-terrain raycast locomotion task."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg, TerrainSceneCfg
from unilab.dr import DomainRandomizationManager
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.commands import (
    apply_heading_yaw_feedback,
    sample_heading_commands,
    zero_small_xy_commands,
)
from unilab.envs.locomotion.common.raycast_scan import (
    RaycastScanConfig,
    configured_raycast_scan_dim,
    init_raycast_scan_sensor,
    raycast_height_scan_obs,
)
from unilab.envs.locomotion.g1.joystick import (
    G1WalkDomainRandomizationProvider,
    G1WalkEnv,
    G1WalkFlatCfg,
)
from unilab.terrains import ROUGH_TERRAINS_CFG


class G1WalkRoughRaycastEnv(G1WalkEnv):
    """Mjlab G1 rough velocity task variant with backend-native raycast scan."""

    _cfg: "G1WalkRoughRaycastCfg"

    def __init__(self, cfg: "G1WalkRoughRaycastCfg", num_envs=1, backend_type="mujoco"):
        self._raycast_scan_dim = (
            configured_raycast_scan_dim(cfg.raycast_scan) if cfg.raycast_scan.enabled else 0
        )
        self._raycast_scan_sensor = None
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._install_raycast_domain_randomization_provider()
        init_raycast_scan_sensor(self, cfg.raycast_scan, cfg.asset.base_name)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        raycast_dim = int(getattr(self, "_raycast_scan_dim", 0))
        return {
            "obs": 98 + raycast_dim,
            "critic": 101 + raycast_dim,
        }

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        self._update_velocity_commands(info)
        obs = super()._compute_obs(info, linvel, gyro, gravity, dof_pos, dof_vel)
        raycast_scan = raycast_height_scan_obs(self, self._cfg.raycast_scan, self._num_envs)
        if raycast_scan.shape[1] == 0:
            return obs
        return {
            "obs": np.concatenate([obs["obs"], raycast_scan], axis=1, dtype=get_global_dtype()),
            "critic": np.concatenate(
                [obs["critic"], raycast_scan],
                axis=1,
                dtype=get_global_dtype(),
            ),
        }

    def build_symmetry_augmentation(self, *, device: str):
        del device
        return None

    def _update_velocity_commands(self, info: dict) -> None:
        commands = np.asarray(info["commands"], dtype=get_global_dtype())
        resampling_time = float(self._cfg.commands.resampling_time)
        if resampling_time > 0.0:
            interval_steps = max(int(round(resampling_time / self._cfg.ctrl_dt)), 1)
            steps = np.asarray(info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32)))
            resample_mask = (steps > 0) & ((steps % interval_steps) == 0)
            if np.any(resample_mask):
                num_resample = int(np.count_nonzero(resample_mask))
                low = np.asarray(self._cfg.commands.vel_limit[0], dtype=get_global_dtype())
                high = np.asarray(self._cfg.commands.vel_limit[1], dtype=get_global_dtype())
                sampled = np.random.uniform(low=low, high=high, size=(num_resample, 3)).astype(
                    get_global_dtype()
                )
                zero_small_xy_commands(sampled)
                self._apply_forward_command_subset(sampled)
                commands[resample_mask] = sampled
                if self._cfg.commands.heading_command:
                    heading_commands = self._ensure_heading_commands(info, commands.shape[0])
                    heading_commands[resample_mask] = sample_heading_commands(self, num_resample)
                    info["heading_commands"] = heading_commands

        if self._cfg.commands.heading_command:
            heading_commands = self._ensure_heading_commands(info, commands.shape[0])
            base_quat = np.asarray(self._backend.get_base_quat(), dtype=get_global_dtype())
            if base_quat.shape[0] == commands.shape[0]:
                apply_heading_yaw_feedback(
                    commands,
                    base_quat,
                    heading_commands,
                    stiffness=float(self._cfg.commands.heading_control_stiffness),
                )
        info["commands"] = commands

    def _ensure_heading_commands(self, info: dict, num_obs: int) -> np.ndarray:
        heading_commands = info.get("heading_commands")
        if heading_commands is None or np.asarray(heading_commands).shape != (num_obs,):
            heading_commands = sample_heading_commands(self, num_obs)
        heading_commands = np.asarray(heading_commands, dtype=get_global_dtype())
        info["heading_commands"] = heading_commands
        return heading_commands

    def _apply_forward_command_subset(self, commands: np.ndarray) -> None:
        rel_forward = float(getattr(self._cfg.commands, "rel_forward_envs", 0.0))
        if rel_forward <= 0.0 or commands.shape[0] == 0:
            return
        forward_mask = np.random.uniform(size=(commands.shape[0],)) <= min(rel_forward, 1.0)
        if not np.any(forward_mask):
            return
        commands[forward_mask, 0] = np.maximum(np.abs(commands[forward_mask, 0]), 0.3)
        commands[forward_mask, 1:] = 0.0

    def _install_raycast_domain_randomization_provider(self) -> None:
        base_kp, base_kd = (
            self._backend.get_actuator_gains()
            if self._cfg.domain_rand.randomize_kp or self._cfg.domain_rand.randomize_kd
            else (None, None)
        )
        self._dr_manager = DomainRandomizationManager(
            self,
            G1RoughRaycastDomainRandomizationProvider(base_kp=base_kp, base_kd=base_kd),
        )


class G1RoughRaycastDomainRandomizationProvider(G1WalkDomainRandomizationProvider):
    def _sample_commands(self, env: G1WalkRoughRaycastEnv, num_reset: int) -> np.ndarray:
        commands = super()._sample_commands(env, num_reset)
        env._apply_forward_command_subset(commands)
        return commands


@registry.envcfg("G1WalkRoughRaycast")
@dataclass
class G1WalkRoughRaycastCfg(G1WalkFlatCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "g1.xml"),
            fragment_files=[
                str(ASSETS_ROOT_PATH / "robots" / "g1" / "locomotion_task.xml"),
            ],
            terrain=TerrainSceneCfg(
                generator=replace(ROUGH_TERRAINS_CFG, curriculum=True),
                hfield_name="terrain_hfield",
                geom_name="floor",
            ),
        )
    )
    raycast_scan: RaycastScanConfig = field(
        default_factory=lambda: RaycastScanConfig(
            enabled=True,
            frame_body_name="pelvis",
            pattern="grid",
            num_rays=187,
            grid_size=[1.6, 1.0],
            resolution=0.1,
            forward_range=[-0.8, 0.8],
            lateral_range=[-0.5, 0.5],
            alignment="yaw",
            geom_groups=[0],
            cutoff=5.0,
            scale=0.2,
        )
    )


registry.register_env("G1WalkRoughRaycast", G1WalkRoughRaycastEnv, sim_backend="mujoco")
