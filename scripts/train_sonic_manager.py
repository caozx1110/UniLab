"""Train SONIC v1 PPO on the Manager-Based UniLab environment.

The script only assembles the existing registry, Manager-Based lifecycle,
rank-local backend device binding, and the task-owned SONIC PPO runner. One
torchrun process owns one simulator shard and one learner GPU.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.base.backend.process_device import configure_backend_process_device  # noqa: E402
from unilab.base.config_adapter import BackendAdapter, create_env  # noqa: E402
from unilab.ipc.dp_launcher import (  # noqa: E402
    UNILAB_DP_LOG_DIR,
    current_torch_distributed_local_rank,
    current_torch_distributed_rank,
    current_torch_distributed_world_size,
    launch_torchrun_workers,
    resolve_collector_cpu_ids,
    resolve_dp_topology,
    validate_dp_launchable,
)
from unilab.tasks.motion_tracking.g1.sonic.manager_terms import (  # noqa: E402
    SonicMotionCommand,
)
from unilab.tasks.motion_tracking.g1.sonic.observations import (  # noqa: E402
    SonicManagerObservationAdapter,
)
from unilab.tasks.motion_tracking.g1.sonic.runner import (  # noqa: E402
    SonicManagerPPORunner,
)
from unilab.training import ensure_registries  # noqa: E402


def _plain_mapping(cfg: DictConfig, path: str) -> dict[str, Any]:
    value = OmegaConf.select(cfg, path, default={})
    resolved = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    if not isinstance(resolved, Mapping):
        raise TypeError(f"{path} must resolve to a mapping")
    return {str(key): item for key, item in resolved.items()}


def _runner_config(cfg: DictConfig) -> dict[str, Any]:
    """Merge algorithm and model-owned sections for ``SonicPPO``."""

    result = _plain_mapping(cfg, "algo")
    for path in ("sonic.model", "sonic.algorithm"):
        result.update(_plain_mapping(cfg, path))
    return result


def _resolve_log_dir(cfg: DictConfig, *, world_size: int) -> Path:
    distributed = os.environ.get(UNILAB_DP_LOG_DIR)
    if distributed:
        return Path(distributed).expanduser().resolve()
    configured = OmegaConf.select(cfg, "training.log_dir", default=None)
    if configured:
        return Path(str(configured)).expanduser().resolve()
    root = OmegaConf.select(cfg, "training.log_root", default=None)
    log_root = Path(str(root)).expanduser() if root else ROOT_DIR / "logs" / "sonic"
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backend = str(OmegaConf.select(cfg, "training.sim_backend"))
    task = str(OmegaConf.select(cfg, "training.task_name"))
    return (log_root / task / f"{stamp}_{backend}_{world_size}gpu").resolve()


def _resolve_device(
    cfg: DictConfig,
    *,
    devices: tuple[int, ...] | None,
    local_rank: int,
    world_size: int,
) -> str:
    configured = OmegaConf.select(cfg, "training.device", default=None)
    if configured is not None and devices is not None:
        raise ValueError("Set either training.device or training.devices, not both")
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed SONIC PPO requires CUDA/NCCL")
        return f"cuda:{local_rank}"
    if configured is not None:
        return str(configured)
    if devices is not None:
        if len(devices) != 1:
            raise ValueError("single-process SONIC expects at most one training.devices entry")
        return f"cuda:{devices[0]}"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _rank_cpu_ids(cfg: DictConfig, *, rank: int, world_size: int) -> list[int] | None:
    explicit_cfg = OmegaConf.select(cfg, "training.cpu_ids_by_rank", default=None)
    explicit = (
        OmegaConf.to_container(explicit_cfg, resolve=True) if explicit_cfg is not None else None
    )
    cpu_count = os.cpu_count() or 1
    if world_size == 1:
        if explicit is None:
            return None
        if not isinstance(explicit, list) or len(explicit) != 1:
            raise ValueError("single-rank cpu_ids_by_rank must contain exactly one segment")
        return [int(value) for value in explicit[0]]
    return resolve_collector_cpu_ids(world_size, rank, cpu_count, explicit=explicit)


def _configure_torch_threads(cfg: DictConfig) -> None:
    intra = int(OmegaConf.select(cfg, "training.torch_num_threads", default=2))
    inter = int(OmegaConf.select(cfg, "training.torch_num_interop_threads", default=1))
    if intra < 1 or inter < 1:
        raise ValueError("SONIC torch thread counts must be positive")
    torch.set_num_threads(intra)
    torch.set_num_interop_threads(inter)


def _initialize_distributed(device: str, *, world_size: int) -> bool:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
    if world_size == 1:
        return False
    if dist.is_initialized():
        if dist.get_world_size() != world_size:
            raise RuntimeError("existing process group disagrees with WORLD_SIZE")
        return False
    dist.init_process_group(backend="nccl", init_method="env://")
    return True


def _global_iteration_metrics(
    metrics: dict[str, float],
    *,
    device: str,
    transitions_per_rank: int,
    world_size: int,
) -> dict[str, float]:
    if world_size == 1:
        return metrics
    time_names = (
        "time/collection_s",
        "time/env_step_s",
        "time/learning_s",
        "time/iteration_s",
    )
    times = torch.tensor([metrics[name] for name in time_names], device=device, dtype=torch.float64)
    dist.all_reduce(times, op=dist.ReduceOp.MAX)
    metrics.update(dict(zip(time_names, (float(value) for value in times.tolist()), strict=True)))
    transitions = transitions_per_rank * world_size
    metrics["throughput/collection_env_steps_s"] = transitions / max(
        metrics["time/collection_s"], 1.0e-12
    )
    metrics["throughput/iteration_env_steps_s"] = transitions / max(
        metrics["time/iteration_s"], 1.0e-12
    )
    return metrics


def _append_metrics(log_dir: Path, payload: Mapping[str, Any]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), sort_keys=True) + "\n")


@hydra.main(version_base="1.3", config_path="../conf/sonic", config_name="config")
def main(cfg: DictConfig) -> None:
    devices = resolve_dp_topology(OmegaConf.select(cfg, "training.devices", default=None))
    rank = current_torch_distributed_rank()
    local_rank = current_torch_distributed_local_rank()
    world_size = current_torch_distributed_world_size()

    if devices is not None and len(devices) > 1 and world_size == 1:
        log_dir = _resolve_log_dir(cfg, world_size=len(devices))
        launch_torchrun_workers(
            devices,
            script_path=Path(__file__),
            argv=sys.argv[1:],
            log_dir=str(log_dir),
            nccl_compat_defaults=bool(
                OmegaConf.select(cfg, "training.nccl_compat_defaults", default=False)
            ),
        )
        return
    if devices is not None and world_size == 1:
        validate_dp_launchable(devices)
    if world_size > 1 and (devices is None or len(devices) != world_size):
        raise ValueError("torchrun WORLD_SIZE must match training.devices")

    device = _resolve_device(
        cfg,
        devices=devices,
        local_rank=local_rank,
        world_size=world_size,
    )
    _configure_torch_threads(cfg)
    owns_process_group = _initialize_distributed(device, world_size=world_size)
    backend = str(OmegaConf.select(cfg, "training.sim_backend"))
    configure_backend_process_device(backend, device)

    base_seed = int(OmegaConf.select(cfg, "algo.seed", default=1))
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    env = None
    runner = None
    training_error: BaseException | None = None
    try:
        ensure_registries()
        env_override = BackendAdapter(
            cfg,
            root_dir=ROOT_DIR,
            algo_name="sonic",
        ).build_task_env_cfg_override()
        cpu_ids = _rank_cpu_ids(cfg, rank=rank, world_size=world_size)
        if cpu_ids is not None:
            env_override["cpu_ids"] = cpu_ids
        env_override["seed"] = base_seed + rank
        num_envs = int(OmegaConf.select(cfg, "algo.num_envs"))
        horizon = int(OmegaConf.select(cfg, "algo.num_steps_per_env"))
        env = create_env(cfg, num_envs=num_envs, env_cfg_override=env_override)
        command = env.command_manager.get_term("motion")
        if not isinstance(command, SonicMotionCommand):
            raise TypeError("SonicG1Tracking owner must expose SonicMotionCommand as 'motion'")
        adapter = SonicManagerObservationAdapter(command, num_envs=num_envs)
        runner = SonicManagerPPORunner(
            env,
            adapter,
            config=_runner_config(cfg),
            device=device,
            horizon=horizon,
        )
        # Model initialization is identical across ranks; subsequent action
        # sampling and PPO sequence shuffles are rank-local.
        torch.manual_seed(base_seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed + rank)

        resume = OmegaConf.select(cfg, "training.resume", default=None)
        if resume:
            runner.load(str(resume))
        max_iterations = int(OmegaConf.select(cfg, "algo.max_iterations"))
        save_interval = int(OmegaConf.select(cfg, "algo.save_interval", default=0))
        log_dir = _resolve_log_dir(cfg, world_size=world_size)
        if rank == 0:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "resolved_config.yaml").write_text(
                OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
            )
            print(
                f"SONIC Manager PPO: backend={backend} device={device} "
                f"world_size={world_size} envs_per_rank={num_envs}",
                flush=True,
            )

        while runner.current_learning_iteration < max_iterations:
            metrics = runner.learn(1)
            metrics = _global_iteration_metrics(
                metrics,
                device=device,
                transitions_per_rank=num_envs * horizon,
                world_size=world_size,
            )
            iteration = runner.current_learning_iteration
            if rank == 0:
                payload = {"iteration": iteration, "backend": backend, **metrics}
                _append_metrics(log_dir, payload)
                print(json.dumps(payload, sort_keys=True), flush=True)
                if save_interval > 0 and iteration % save_interval == 0:
                    runner.save(log_dir / f"model_{iteration}.pt")
        if rank == 0:
            runner.save(log_dir / "last.pt")
    except BaseException as exc:
        training_error = exc
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                if training_error is None:
                    raise
        if owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
