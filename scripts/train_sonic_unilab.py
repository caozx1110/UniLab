#!/usr/bin/env -S uv run --script
"""Launch or preflight the SONIC UniLab bridge.

Examples (run from the UniLab repository):

    UV_CACHE_DIR=/tmp/unilab-uv-cache uv run scripts/train_sonic_unilab.py
    UV_CACHE_DIR=/tmp/unilab-uv-cache uv run scripts/train_sonic_unilab.py \
        sonic.mode=train sonic.motion_manifest=/abs/path/to/manifest.json \
        sonic.require_motion_manifest=true
    UV_CACHE_DIR=/tmp/unilab-uv-cache uv run scripts/train_sonic_unilab.py \
        sonic.mode=train sonic.runtime_entrypoint=my_pkg.sonic_runtime:train

The default mode is ``preflight``.  Train mode selects UniLab's native
``SonicPPO`` owner unless an explicit compatible runtime is configured.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.ipc.dp_launcher import (  # noqa: E402
    current_torch_distributed_local_rank,
    current_torch_distributed_rank,
    current_torch_distributed_world_size,
    resolve_dp_topology,
    validate_dp_launchable,
)
from unilab.training.seed import apply_training_seed, resolve_training_seed  # noqa: E402
from unilab.training.sonic_bridge import (  # noqa: E402
    SonicBridgeError,
    apply_sonic_rank_resources,
    apply_sonic_torch_runtime,
    build_sonic_launch_plan,
    launch_sonic_workers,
    run_sonic_runtime,
    write_sonic_preflight,
)


def _mode(cfg: DictConfig) -> str:
    value = str(OmegaConf.select(cfg, "sonic.mode", default="preflight")).strip().lower()
    if value in {"plan", "check", "dry-run", "dry_run"}:
        return "preflight"
    if value not in {"preflight", "train"}:
        raise SonicBridgeError(f"sonic.mode must be 'preflight' or 'train', got {value!r}")
    return value


def run(cfg: DictConfig, *, argv: list[str] | None = None) -> dict[str, object] | None:
    """Run one Hydra job; split out for lightweight unit tests."""

    mode = _mode(cfg)
    devices = resolve_dp_topology(OmegaConf.select(cfg, "training.devices"))
    configured_device = OmegaConf.select(cfg, "training.device", default=None)
    if configured_device is not None and devices is not None:
        raise SonicBridgeError("Set either training.device or training.devices, not both")
    rank = current_torch_distributed_rank()
    local_rank = current_torch_distributed_local_rank()
    world_size = current_torch_distributed_world_size()
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise SonicBridgeError(
            f"invalid distributed rank/world_size: rank={rank}, world_size={world_size}"
        )
    if local_rank < 0 or local_rank >= world_size:
        raise SonicBridgeError(
            f"LOCAL_RANK={local_rank} is out of range for world_size={world_size}"
        )

    # The parent only composes config and starts torchrun.  No Torch or env is
    # imported before this branch, which keeps the CPU/NUMA setup effective in
    # every child rank.
    if mode == "train" and devices is not None and len(devices) > 1 and world_size == 1:
        launch_sonic_workers(
            cfg,
            script_path=Path(__file__),
            argv=list(sys.argv[1:] if argv is None else argv),
            root_dir=ROOT_DIR,
        )
        return None
    if mode == "train" and devices is not None and world_size == 1:
        validate_dp_launchable(devices)

    plan = build_sonic_launch_plan(
        cfg,
        root_dir=ROOT_DIR,
        rank=rank,
        world_size=world_size,
    )
    if plan.resources is not None:
        apply_sonic_rank_resources(
            plan.resources,
            pin_process=bool(OmegaConf.select(cfg, "sonic.resources.pin_process", default=False)),
        )
        # Import Torch and configure both pools before seeding or constructing
        # any learner/environment work that could initialize parallel kernels.
        apply_sonic_torch_runtime(plan.resources)
    seed_info = resolve_training_seed(cfg)
    effective_seed = (
        None if seed_info.effective_seed is None else int(seed_info.effective_seed) + int(rank)
    )
    apply_training_seed(effective_seed, torch_runtime=True, cuda=True)
    metadata_path = write_sonic_preflight(plan)

    if rank == 0:
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        print(f"SONIC preflight metadata: {metadata_path}")
    if mode == "preflight":
        return plan.to_dict()

    run_sonic_runtime(cfg, plan)
    return plan.to_dict()


@hydra.main(version_base="1.3", config_path="../conf/sonic", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
