"""Shared batch raycast scan helpers for rough locomotion envs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.dtype_config import get_global_dtype


@dataclass
class RaycastScanConfig:
    enabled: bool = True
    frame_body_name: str | None = None
    pattern: str = "grid"
    num_rays: int = 187
    grid_size: list[float] = field(default_factory=lambda: [1.6, 1.0])
    resolution: float = 0.1
    forward_range: list[float] = field(default_factory=lambda: [-0.8, 0.8])
    lateral_range: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    measured_points_x: list[float] | None = None
    measured_points_y: list[float] | None = None
    direction: list[float] = field(default_factory=lambda: [0.0, 0.0, -1.0])
    origin_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    alignment: str = "yaw"
    geom_groups: list[int] | None = field(default_factory=lambda: [0])
    bodyexclude: int | None = None
    cutoff: float = 5.0
    return_normal: bool = False
    miss_value: float | None = None
    offset: float = 0.0
    scale: float = 0.2


def raycast_scan_directions(scan_cfg: RaycastScanConfig) -> np.ndarray:
    _, directions = raycast_scan_pattern(scan_cfg)
    return directions


def raycast_scan_pattern(scan_cfg: RaycastScanConfig) -> tuple[np.ndarray, np.ndarray]:
    pattern = scan_cfg.pattern.lower()
    if pattern == "grid":
        xs, ys = _grid_points(scan_cfg)
        x_grid, y_grid = np.meshgrid(
            np.asarray(xs, dtype=np.float64),
            np.asarray(ys, dtype=np.float64),
            indexing="xy",
        )
        origin_offsets = np.zeros((x_grid.size, 3), dtype=np.float64)
        origin_offsets[:, 0] = x_grid.reshape(-1)
        origin_offsets[:, 1] = y_grid.reshape(-1)
        direction = np.asarray(scan_cfg.direction, dtype=np.float64)
        if direction.shape != (3,):
            raise ValueError(f"direction must have shape (3,), got {direction.shape}")
        directions = np.broadcast_to(direction[None, :], origin_offsets.shape).copy()
    elif pattern == "fan":
        num_rays = int(scan_cfg.num_rays)
        if num_rays <= 0:
            raise ValueError(f"raycast_scan.num_rays must be positive, got {num_rays}")
        points = _fan_points(num_rays, scan_cfg.forward_range, scan_cfg.lateral_range)
        origin_offsets = np.zeros((num_rays, 3), dtype=np.float64)
        directions = np.empty((num_rays, 3), dtype=np.float64)
        directions[:, 0] = points[:, 0]
        directions[:, 1] = points[:, 1]
        directions[:, 2] = -float(scan_cfg.cutoff)
    else:
        raise ValueError(f"Unsupported raycast_scan.pattern {scan_cfg.pattern!r}")

    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("raycast_scan generated a zero direction")
    origin_offset = np.asarray(scan_cfg.origin_offset, dtype=np.float64)
    if origin_offset.shape != (3,):
        raise ValueError(f"origin_offset must have shape (3,), got {origin_offset.shape}")
    origin_offsets = origin_offsets + origin_offset[None, :]
    return (
        np.ascontiguousarray(origin_offsets, dtype=np.float64),
        np.ascontiguousarray(directions / norms[:, None], dtype=np.float64),
    )


def configured_raycast_scan_dim(scan_cfg: RaycastScanConfig) -> int:
    return int(raycast_scan_directions(scan_cfg).shape[0])


def init_raycast_scan_sensor(env: Any, scan_cfg: RaycastScanConfig, base_body_name: str) -> None:
    env._raycast_scan_dim = 0 if not scan_cfg.enabled else configured_raycast_scan_dim(scan_cfg)
    env._raycast_scan_sensor = None
    env._raycast_scan_directions = None
    env._raycast_scan_frame_body_id = None
    if not scan_cfg.enabled:
        return
    if env._raycast_scan_dim <= 0:
        raise ValueError("raycast_scan must define at least one ray")

    frame_body_name = scan_cfg.frame_body_name or base_body_name
    frame_body_id = env._backend.get_body_id(frame_body_name)
    origin_offsets, directions = raycast_scan_pattern(scan_cfg)
    env._raycast_scan_frame_body_id = frame_body_id
    env._raycast_scan_directions = directions
    env._raycast_scan_origin_offsets = origin_offsets
    env._raycast_scan_sensor = env._backend.create_raycaster(
        frame_body_id=frame_body_id,
        directions=directions,
        origin_offsets=origin_offsets,
        alignment=scan_cfg.alignment,
        geomgroup=None if scan_cfg.geom_groups is None else np.asarray(scan_cfg.geom_groups),
        bodyexclude=scan_cfg.bodyexclude,
        return_normal=scan_cfg.return_normal,
        cutoff=float(scan_cfg.cutoff),
    )


def raw_raycast_scan_obs(env: Any, num_obs: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    sensor = getattr(env, "_raycast_scan_sensor", None)
    if sensor is None:
        return None, None
    base_pos = np.asarray(env._backend.get_base_pos(), dtype=get_global_dtype())
    if base_pos.shape[0] != num_obs:
        return None, None
    result = sensor.cast()
    distances = np.asarray(result.distances, dtype=get_global_dtype())
    if distances.shape != (num_obs, env._raycast_scan_dim):
        return None, None
    return distances, base_pos


def raycast_height_scan_obs(env: Any, scan_cfg: RaycastScanConfig, num_obs: int) -> np.ndarray:
    distances, _ = raw_raycast_scan_obs(env, num_obs)
    if distances is None:
        return np.zeros(
            (num_obs, int(getattr(env, "_raycast_scan_dim", 0))), dtype=get_global_dtype()
        )
    if str(scan_cfg.alignment).lower() not in {"yaw", "world", "none"}:
        raise ValueError("raycast height observations support only yaw/world-aligned scans")
    directions = np.asarray(getattr(env, "_raycast_scan_directions"), dtype=get_global_dtype())
    if directions.shape != (distances.shape[1], 3):
        raise ValueError(
            f"raycast directions must have shape ({distances.shape[1]}, 3), got {directions.shape}"
        )
    miss_value = float(scan_cfg.cutoff if scan_cfg.miss_value is None else scan_cfg.miss_value)
    heights = np.where(distances < 0.0, miss_value, -distances * directions[None, :, 2])
    heights = np.clip(heights - float(scan_cfg.offset), -float(scan_cfg.cutoff), float(scan_cfg.cutoff))
    return np.asarray(heights * float(scan_cfg.scale), dtype=get_global_dtype())


def _linspace_from_range(bounds: list[float], count: int) -> np.ndarray:
    if count <= 1:
        return np.asarray([(float(bounds[0]) + float(bounds[1])) * 0.5], dtype=np.float64)
    return np.linspace(float(bounds[0]), float(bounds[1]), count, dtype=np.float64)


def _grid_points(scan_cfg: RaycastScanConfig) -> tuple[np.ndarray, np.ndarray]:
    if scan_cfg.measured_points_x is not None or scan_cfg.measured_points_y is not None:
        if scan_cfg.measured_points_x is None or scan_cfg.measured_points_y is None:
            raise ValueError(
                "raycast_scan grid requires both measured_points_x and measured_points_y"
            )
        return (
            np.asarray(scan_cfg.measured_points_x, dtype=np.float64),
            np.asarray(scan_cfg.measured_points_y, dtype=np.float64),
        )

    if len(scan_cfg.grid_size) != 2:
        raise ValueError(f"raycast_scan.grid_size must contain 2 values, got {scan_cfg.grid_size}")
    resolution = float(scan_cfg.resolution)
    if resolution <= 0.0:
        raise ValueError(f"raycast_scan.resolution must be positive, got {resolution}")
    size_x, size_y = float(scan_cfg.grid_size[0]), float(scan_cfg.grid_size[1])
    return (
        _arange_inclusive(-size_x * 0.5, size_x * 0.5, resolution),
        _arange_inclusive(-size_y * 0.5, size_y * 0.5, resolution),
    )


def _arange_inclusive(start: float, stop: float, step: float) -> np.ndarray:
    return np.arange(start, stop + step * 0.5, step, dtype=np.float64)


def _fan_points(num_rays: int, forward_range: list[float], lateral_range: list[float]) -> np.ndarray:
    x_count = max(1, int(round(np.sqrt(num_rays * 1.55))))
    y_count = max(1, int(np.ceil(num_rays / x_count)))
    xs = _linspace_from_range(forward_range, x_count)
    ys = _linspace_from_range(lateral_range, y_count)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=1)
    return np.ascontiguousarray(points[:num_rays], dtype=np.float64)
