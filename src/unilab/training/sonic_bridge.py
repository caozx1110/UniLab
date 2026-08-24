"""SONIC-to-UniLab launch, preflight, and native-runtime bridge.

This module owns the boundary between Hydra/torchrun and a SONIC runtime.  The
built-in ``unilab.algos.torch.sonic_ppo`` owner supplies the native PPO loop,
UniversalToken bottleneck, auxiliary losses, normalizers, and checkpoint
handling; ``sonic.runtime_entrypoint`` remains an escape hatch for a separately
packaged compatible owner.  The bridge keeps launch arithmetic, manifest
validation, and CPU/NUMA setup outside the learner and simulator hot paths.

``build_sonic_launch_plan`` can be used by conversion tools and CI without
importing Torch or a simulator, while the command-line entrypoint's default
``preflight`` mode records the exact multi-GPU arithmetic and CPU affinity that
a later training run will use.
"""

from __future__ import annotations

import datetime as _datetime
import importlib
import inspect
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf

from unilab.ipc.dp_launcher import (
    UNILAB_DP_LOG_DIR,
    current_torch_distributed_local_rank,
    current_torch_distributed_rank,
    current_torch_distributed_world_size,
    launch_torchrun_workers,
    resolve_dp_topology,
    validate_dp_launchable,
)
from unilab.training.sonic_contract import (
    SONIC_RELEASE_SPEC,
    SonicPreflightReport,
    run_sonic_preflight,
    validate_sonic_owner,
)
from unilab.training.sonic_motion import (
    MotionManifest,
    load_motion_manifest,
    preflight_motion_manifest,
    resolve_manifest_clip_path,
)
from unilab.training.sonic_resources import (
    SONIC_8X4090_76C_PROFILE,
    SonicRankResources,
    apply_sonic_torch_threads,
    available_logical_cpu_ids,
    discover_gpu_numa_nodes,
    map_cuda_devices_to_numa_nodes,
    resolve_sonic_rank_resources,
)


class SonicBridgeError(RuntimeError):
    """Raised when the SONIC/UniLab boundary cannot be satisfied."""


@dataclass(frozen=True)
class SonicLaunchPlan:
    """Everything resolved on the cold path before env/model construction."""

    rank: int
    world_size: int
    local_rank: int
    devices: tuple[int, ...] | None
    log_dir: Path
    report: SonicPreflightReport
    resources: SonicRankResources | None
    resources_by_rank: tuple[SonicRankResources, ...]
    env_cfg_override: Mapping[str, Any]
    motion_manifest: MotionManifest | None = None
    runtime_device: str | None = None

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe launch metadata for tracker/checkpoint provenance."""

        return {
            "bridge_version": 1,
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "devices": list(self.devices) if self.devices is not None else None,
            "runtime_device": self.runtime_device,
            "log_dir": str(self.log_dir),
            "global_num_envs": self.report.global_num_envs,
            "num_envs_per_rank": self.report.num_envs_per_rank,
            "samples_per_rank": self.report.samples_per_rank,
            "global_samples": self.report.global_samples,
            "horizon": self.report.horizon,
            "num_learning_epochs": self.report.spec.num_learning_epochs,
            "num_mini_batches": self.report.num_mini_batches,
            "logical_optimizer_steps_per_iteration": (
                self.report.spec.num_learning_epochs * self.report.num_mini_batches
            ),
            "local_minibatch_size": self.report.local_minibatch_size,
            "microbatch_size": self.report.microbatch_size,
            "microbatches_per_minibatch": self.report.microbatches_per_minibatch,
            "resources": self.resources.to_dict() if self.resources is not None else None,
            "resources_by_rank": [item.to_dict() for item in self.resources_by_rank],
            "transport_environment": {
                name: os.environ.get(name)
                for name in (
                    "CUDA_VISIBLE_DEVICES",
                    "NCCL_P2P_DISABLE",
                    "NCCL_SHM_DISABLE",
                    "NCCL_DEBUG",
                )
            },
            "env_cfg_override": dict(self.env_cfg_override),
            "motion_manifest": (
                str(self.motion_manifest.manifest_path)
                if self.motion_manifest is not None and self.motion_manifest.manifest_path
                else None
            ),
        }


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _select(config: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def resolve_sonic_log_dir(
    cfg: DictConfig | Mapping[str, Any],
    *,
    root_dir: str | Path,
    world_size: int,
    timestamp: str | None = None,
) -> Path:
    """Resolve one rank-shared run directory.

    ``UNILAB_DP_LOG_DIR`` is set by the parent torchrun launcher and therefore
    wins over all config-derived paths.  A configured relative ``log_dir`` is
    rooted at ``root_dir`` to match the other UniLab entrypoints.
    """

    distributed = os.environ.get(UNILAB_DP_LOG_DIR)
    if distributed:
        return Path(distributed).expanduser().resolve()
    configured = _select(_plain(cfg), "training.log_dir")
    if configured:
        path = Path(str(configured)).expanduser()
        return (path if path.is_absolute() else Path(root_dir) / path).resolve()
    stamp = timestamp or _datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task = str(_select(_plain(cfg), "training.task_name", "SonicG1Tracking"))
    backend = str(_select(_plain(cfg), "training.sim_backend", "mujoco"))
    suffix = f"_gpux{world_size}" if world_size > 1 else ""
    return (Path(root_dir) / "logs" / "sonic" / task / f"{stamp}_{backend}{suffix}").resolve()


def resolve_sonic_device(
    cfg: DictConfig | Mapping[str, Any],
    *,
    devices: tuple[int, ...] | None,
    world_size: int,
    local_rank: int,
) -> str | None:
    """Resolve and validate the rank-local learner device.

    ``training.devices`` indexes the parent's visible CUDA list.  A singleton
    list therefore keeps its host-global index (``[3] -> cuda:3``), while a
    multi-rank torchrun child sees the selected list remapped into
    ``CUDA_VISIBLE_DEVICES`` and must use ``cuda:LOCAL_RANK``.  This helper is
    deliberately independent of CUDA probing so CPU-only preflight remains
    executable.
    """
    plain = _plain(cfg)
    configured = _select(plain, "training.device")
    if configured in (None, "", "null"):
        configured = _select(plain, "device")
    if configured in (None, "", "null"):
        configured = None
    if configured is not None and devices is not None:
        raise SonicBridgeError("Set either training.device or training.devices, not both")
    if world_size < 1:
        raise SonicBridgeError(f"world_size must be positive, got {world_size}")
    if local_rank < 0 or local_rank >= world_size:
        raise SonicBridgeError(
            f"LOCAL_RANK={local_rank} is out of range for world_size={world_size}"
        )
    if world_size > 1 and configured is not None:
        raise SonicBridgeError(
            "training.device cannot select one device in a distributed run; use training.devices"
        )
    if world_size > 1 and devices is not None and len(devices) != world_size:
        raise SonicBridgeError(
            f"training.devices has {len(devices)} entries but world_size={world_size}"
        )
    if devices is not None:
        if len(devices) == 1:
            if world_size != 1:
                raise SonicBridgeError("a singleton training.devices list requires world_size=1")
            return f"cuda:{devices[0]}"
        return f"cuda:{local_rank}"
    return None if configured is None else str(configured)


def _resource_plan(
    cfg: Mapping[str, Any],
    *,
    world_size: int,
    rank: int,
    devices: Sequence[int] | None = None,
) -> SonicRankResources:
    resources = _select(cfg, "sonic.resources", {})
    if not isinstance(resources, Mapping):
        raise SonicBridgeError("sonic.resources must be a mapping")
    profile_name = resources.get("profile", "auto")
    workers = int(resources.get("workers_per_rank", 6))
    torch_threads = int(resources.get("torch_num_threads", 2))
    interop_threads = int(resources.get("torch_num_interop_threads", 1))
    mode = str(_select(cfg, "sonic.mode", "preflight")).strip().lower()
    if mode == "train" and bool(resources.get("enabled", True)):
        # PCI BDF enumeration is not a CUDA ordinal contract.  A production
        # run must carry the rank-ordered topology obtained from nvidia-smi /
        # NVML explicitly; preflight may still use the deterministic profile
        # fallback for CPU-only planning.
        if resources.get("gpu_numa_nodes") is None:
            raise SonicBridgeError(
                "SONIC train mode requires rank-ordered "
                "sonic.resources.gpu_numa_nodes. PCI BDF order is not a CUDA "
                "ordinal contract; derive the list with nvidia-smi before launch."
            )

    # The explicit workstation profile supplies NUMA/core ordering, then gets
    # intersected with the process cpuset.  A scheduled container therefore
    # cannot receive CPU ids that its kernel affinity mask rejects.
    if profile_name in (SONIC_8X4090_76C_PROFILE.name, "8x4090", "4090x8") and world_size <= 8:
        profile = SONIC_8X4090_76C_PROFILE
        configured_nodes = resources.get("gpu_numa_nodes")
        rank_nodes: tuple[int | None, ...] | None = (
            tuple(int(item) for item in configured_nodes) if configured_nodes is not None else None
        )
        if rank_nodes is None and bool(resources.get("detect_gpu_numa", True)) and devices:
            discovered = discover_gpu_numa_nodes()
            if discovered:
                try:
                    rank_nodes = map_cuda_devices_to_numa_nodes(devices, discovered)
                except ValueError as exc:
                    raise SonicBridgeError(
                        f"could not map CUDA devices to NUMA nodes: {exc}"
                    ) from exc
        if rank_nodes is None:
            rank_nodes = tuple(profile.gpu_numa_nodes[:world_size])
        if rank_nodes is None or len(rank_nodes) != world_size:
            raise SonicBridgeError(
                f"sonic.resources.gpu_numa_nodes must have {world_size} entries, "
                f"got {len(rank_nodes)}"
            )
        configured_available = resources.get("available_cpu_ids")
        available_set = (
            set(int(item) for item in configured_available)
            if configured_available is not None
            else set(available_logical_cpu_ids())
        )
        node_masks: dict[int, tuple[int, ...]] = {}
        for node in set(item for item in rank_nodes if item is not None):
            node_masks[node] = tuple(
                cpu for cpu in dict(profile.numa_cpu_ids).get(node, ()) if cpu in available_set
            )
        configured_segments = resources.get("explicit_cpu_ids")
        if configured_segments is not None:
            rank_segments = [tuple(int(cpu) for cpu in segment) for segment in configured_segments]
        else:
            pools = {node: list(mask) for node, mask in node_masks.items()}
            rank_segments = []
            for segment_node in rank_nodes:
                if segment_node is None:
                    raise SonicBridgeError(
                        "automatic SONIC CPU allocation requires a NUMA node for every rank"
                    )
                pool = pools.get(segment_node, [])
                if len(pool) < workers:
                    raise SonicBridgeError(
                        f"NUMA node {segment_node} has fewer than {workers} available "
                        "CPUs for a SONIC rank"
                    )
                rank_segments.append(tuple(pool[:workers]))
                del pool[:workers]
        available = tuple(cpu for mask in node_masks.values() for cpu in mask)
        return resolve_sonic_rank_resources(
            world_size=world_size,
            rank=rank,
            available_cpu_ids=available,
            gpu_numa_nodes=rank_nodes,
            numa_cpu_ids=node_masks,
            explicit_cpu_ids=tuple(rank_segments),
            workers_per_rank=workers,
            torch_num_threads=torch_threads,
            torch_num_interop_threads=interop_threads,
        )

    available_cfg = resources.get("available_cpu_ids")
    available = (
        tuple(int(item) for item in available_cfg)
        if available_cfg is not None
        else available_logical_cpu_ids()
    )
    numa_nodes = resources.get("gpu_numa_nodes")
    numa_cpu_ids = resources.get("numa_cpu_ids")
    explicit = resources.get("explicit_cpu_ids")
    return resolve_sonic_rank_resources(
        world_size=world_size,
        rank=rank,
        available_cpu_ids=available,
        gpu_numa_nodes=numa_nodes,
        numa_cpu_ids=numa_cpu_ids,
        explicit_cpu_ids=explicit,
        workers_per_rank=workers,
        torch_num_threads=torch_threads,
        torch_num_interop_threads=interop_threads,
    )


def apply_sonic_rank_resources(
    resources: SonicRankResources,
    *,
    environ: dict[str, str] | None = None,
    pin_process: bool = False,
) -> dict[str, str]:
    """Apply rank-local BLAS/Torch thread defaults before importing Torch.

    Existing user environment values are preserved.  Optional process pinning
    is explicit because a caller may run several non-SONIC jobs in one process.
    """

    target = os.environ if environ is None else environ
    for key, value in resources.thread_env.items():
        target.setdefault(key, value)
    if pin_process and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, resources.cpu_ids)
    return dict(target)


def apply_sonic_torch_runtime(resources: SonicRankResources) -> dict[str, int]:
    """Apply the already-imported Torch thread pool settings."""
    return apply_sonic_torch_threads(resources)


def _load_manifest_from_cfg(
    cfg: Mapping[str, Any],
    *,
    require: bool,
) -> MotionManifest | None:
    raw_path = _select(cfg, "sonic.motion_manifest")
    if raw_path in (None, ""):
        if require:
            raise SonicBridgeError(
                "SONIC training requires sonic.motion_manifest; convert source PKL/SMPL "
                "data to the versioned UniLab manifest first"
            )
        return None
    path = Path(str(raw_path)).expanduser().resolve()
    if not path.is_file():
        raise SonicBridgeError(f"sonic.motion_manifest is not a file: {path}")
    try:
        manifest = load_motion_manifest(path)
        if bool(_select(cfg, "sonic.verify_motion_checksums", True)) or bool(
            _select(cfg, "sonic.verify_motion_shapes", True)
        ):
            manifest = preflight_motion_manifest(
                manifest,
                verify_checksums=bool(_select(cfg, "sonic.verify_motion_checksums", True)),
                verify_shapes=bool(_select(cfg, "sonic.verify_motion_shapes", True)),
            )
    except (ValueError, OSError) as exc:
        raise SonicBridgeError(f"SONIC motion manifest preflight failed: {exc}") from exc
    return manifest


def _build_env_override(
    cfg: Mapping[str, Any],
    resources: SonicRankResources | None,
    manifest: MotionManifest | None,
) -> dict[str, Any]:
    env_cfg = _select(cfg, "env", {})
    override = dict(env_cfg) if isinstance(env_cfg, Mapping) else {}
    reward_cfg = _select(cfg, "reward")
    if isinstance(reward_cfg, Mapping):
        override["reward_config"] = dict(reward_cfg)
    # SONIC's Isaac config names the physics step and decimation separately;
    # UniLab's EnvCfg consumes the derived control period.  Keep the source
    # fields in the contract config, but do not leak the Isaac-only
    # ``decimation`` key into the backend owner.
    sim_dt = float(override.get("sim_dt", SONIC_RELEASE_SPEC.sim_dt))
    decimation = int(override.pop("decimation", SONIC_RELEASE_SPEC.decimation))
    derived_ctrl_dt = sim_dt * decimation
    configured_ctrl_dt = override.get("ctrl_dt")
    if configured_ctrl_dt is not None and abs(float(configured_ctrl_dt) - derived_ctrl_dt) > 1e-9:
        raise SonicBridgeError(
            "env.ctrl_dt disagrees with env.sim_dt * env.decimation: "
            f"expected {derived_ctrl_dt}, got {configured_ctrl_dt}"
        )
    override["ctrl_dt"] = derived_ctrl_dt

    if resources is not None:
        # EnvCfg.cpu_ids is the declared backend contract; do not put this in
        # a trainer-only field where MuJoCo would silently ignore it.
        override["cpu_ids"] = list(resources.cpu_ids)
    configured_encoder_probs = _select(cfg, "sonic.encoder_sample_probs")
    if configured_encoder_probs is not None:
        if not isinstance(configured_encoder_probs, Sequence) or isinstance(
            configured_encoder_probs, (str, bytes)
        ):
            raise SonicBridgeError("sonic.encoder_sample_probs must be a sequence")
        override["encoder_sample_probs"] = tuple(float(value) for value in configured_encoder_probs)
    if manifest is not None:
        source = manifest.manifest_path
        if source is None:
            raise SonicBridgeError("manifest_path is required for a materialized SONIC store")
        override["motion_manifest"] = str(source)
        override["motion_rank"] = int(os.environ.get("RANK", "0"))
        override["motion_world_size"] = int(os.environ.get("WORLD_SIZE", "1"))
        override["motion_shard_clips"] = bool(_select(cfg, "sonic.motion_shard_clips", True))
        override["motion_cache_size"] = int(_select(cfg, "sonic.motion_cache_size", 2))
        if bool(_select(cfg, "sonic.use_manifest_clips", False)):
            override["motion_file"] = [
                str(resolve_manifest_clip_path(source, clip.path)) for clip in manifest.clips
            ]
    return override


def build_sonic_launch_plan(
    cfg: DictConfig | Mapping[str, Any],
    *,
    root_dir: str | Path = ".",
    rank: int | None = None,
    world_size: int | None = None,
    local_rank: int | None = None,
) -> SonicLaunchPlan:
    """Resolve and validate the complete rank-local launch plan."""

    plain = _plain(cfg)
    if not isinstance(plain, Mapping):
        raise SonicBridgeError("SONIC config must resolve to a mapping")
    try:
        validate_sonic_owner(plain, require_owner_marker=True)
    except ValueError as exc:
        raise SonicBridgeError(str(exc)) from exc
    resolved_rank = current_torch_distributed_rank() if rank is None else int(rank)
    resolved_world = (
        current_torch_distributed_world_size() if world_size is None else int(world_size)
    )
    resolved_local = (
        current_torch_distributed_local_rank() if local_rank is None else int(local_rank)
    )
    # Before torchrun starts, the parent has WORLD_SIZE=1 even though the
    # configured device list describes the intended worker topology.  Use that
    # topology for arithmetic/preflight; child ranks already expose WORLD_SIZE.
    devices_cfg = _select(plain, "training.devices")
    devices = resolve_dp_topology(devices_cfg)
    if world_size is None and devices is not None and len(devices) > resolved_world:
        resolved_world = len(devices)
    elif (
        world_size == 1
        and devices is not None
        and len(devices) > 1
        and os.environ.get("WORLD_SIZE") is None
    ):
        resolved_world = len(devices)
    if resolved_world < 1 or not 0 <= resolved_rank < resolved_world:
        raise SonicBridgeError(
            f"invalid distributed rank/world_size: rank={resolved_rank}, world_size={resolved_world}"
        )
    if not 0 <= resolved_local < resolved_world:
        raise SonicBridgeError(
            f"LOCAL_RANK={resolved_local} is out of range for world_size={resolved_world}"
        )
    if devices is not None and len(devices) != resolved_world and resolved_world > 1:
        raise SonicBridgeError(
            f"training.devices has {len(devices)} entries but torchrun world_size={resolved_world}"
        )
    runtime_device = resolve_sonic_device(
        plain,
        devices=devices,
        world_size=resolved_world,
        local_rank=resolved_local,
    )
    report = run_sonic_preflight(
        plain,
        world_size=resolved_world,
        num_envs_per_rank=_select(plain, "algo.num_envs"),
        microbatch_size=_select(plain, "sonic.microbatch_size"),
        dimensions=_select(plain, "sonic.dimensions"),
        require_dimensions=bool(_select(plain, "sonic.require_dimensions", True)),
    )
    manifest = _load_manifest_from_cfg(
        plain,
        require=bool(_select(plain, "sonic.require_motion_manifest", False)),
    )
    resources_by_rank = (
        tuple(
            _resource_plan(
                plain,
                world_size=resolved_world,
                rank=resource_rank,
                devices=devices,
            )
            for resource_rank in range(resolved_world)
        )
        if bool(_select(plain, "sonic.resources.enabled", True))
        else ()
    )
    resources = resources_by_rank[resolved_rank] if resources_by_rank else None
    log_dir = resolve_sonic_log_dir(
        plain,
        root_dir=root_dir,
        world_size=resolved_world,
    )
    return SonicLaunchPlan(
        rank=resolved_rank,
        world_size=resolved_world,
        local_rank=resolved_local,
        devices=devices,
        log_dir=log_dir,
        report=report,
        resources=resources,
        resources_by_rank=resources_by_rank,
        env_cfg_override=_build_env_override(plain, resources, manifest),
        motion_manifest=manifest,
        runtime_device=runtime_device,
    )


def write_sonic_preflight(plan: SonicLaunchPlan) -> Path:
    """Persist rank-0 launch metadata after all cold-path checks pass."""

    plan.log_dir.mkdir(parents=True, exist_ok=True)
    output = plan.log_dir / "sonic_preflight.json"
    if plan.rank == 0:
        output.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return output


def _resolve_callable(path: str) -> Callable[..., Any]:
    module_name, separator, attribute = path.partition(":")
    if not separator:
        module_name, _, attribute = path.rpartition(".")
    if not module_name or not attribute:
        raise SonicBridgeError(f"runtime_entrypoint must be 'module:callable' (got {path!r})")
    try:
        value = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise SonicBridgeError(f"could not import SONIC runtime {path!r}: {exc}") from exc
    if not callable(value):
        raise SonicBridgeError(f"SONIC runtime {path!r} is not callable")
    return value  # type: ignore[no-any-return]


def run_sonic_runtime(
    cfg: DictConfig | Mapping[str, Any],
    plan: SonicLaunchPlan,
    *,
    runtime: Callable[..., Any] | None = None,
) -> Any:
    """Invoke the built-in or explicitly selected native SONIC runtime.

    A runtime receives the resolved config, plan, and env override.  The
    signature is checked before invocation so a mistakenly configured RSL-RL
    entrypoint cannot run under the SONIC label.  Train mode defaults to the
    built-in ``unilab.algos.torch.sonic_ppo:train_sonic`` owner; preflight mode
    intentionally has no runtime to invoke.
    """

    plain = _plain(cfg)
    try:
        validate_sonic_owner(plain, require_owner_marker=True)
    except ValueError as exc:
        raise SonicBridgeError(str(exc)) from exc
    # A release SONIC rollout is driven by the versioned, checksummed motion
    # store.  Keep preflight useful without data, but fail at the train
    # boundary instead of letting registry construction fall through to the
    # historical G1 default motion path (which is absent on a fresh checkout).
    if (
        str(_select(plain, "sonic.mode", "preflight")).strip().lower() == "train"
        and plan.motion_manifest is None
    ):
        raise SonicBridgeError(
            "SONIC training requires sonic.motion_manifest; convert source PKL/SMPL "
            "data to the versioned UniLab manifest first"
        )
    if plan.report.microbatch_size != plan.report.local_minibatch_size and not bool(
        _select(plain, "sonic.allow_microbatch_change", False)
    ):
        raise SonicBridgeError(
            "SONIC microbatch_size differs from the release local minibatch. The upstream "
            "trainer steps/zeros each microbatch; implement and validate no_sync + frozen "
            "normalizer accumulation first, then set sonic.allow_microbatch_change=true."
        )
    runtime_path = _select(plain, "sonic.runtime_entrypoint")
    if not runtime_path and str(_select(plain, "sonic.mode", "preflight")).lower() == "train":
        runtime_path = "unilab.algos.torch.sonic_ppo:train_sonic"
    source_repo = _select(plain, "sonic.source_repo")
    if source_repo not in (None, ""):
        source_path = Path(str(source_repo)).expanduser().resolve()
        if source_path.is_dir() and str(source_path) not in sys.path:
            # Keep the upstream dependency isolated: this is a lazy import
            # path used only after preflight and rank CPU setup succeeded.
            sys.path.insert(0, str(source_path))
    callback = runtime or (_resolve_callable(str(runtime_path)) if runtime_path else None)
    if callback is None:
        raise SonicBridgeError(
            "No native SONIC runtime is configured for this mode. Use sonic.mode=train "
            "to select the built-in UniLab SonicPPO owner, or set "
            "sonic.runtime_entrypoint to a compatible external owner."
        )
    try:
        signature = inspect.signature(callback)
        if not any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            required = {"cfg", "plan", "env_cfg_override"}
            missing = required.difference(signature.parameters)
            if missing:
                raise SonicBridgeError(
                    f"SONIC runtime must accept keyword arguments {sorted(missing)}"
                )
    except (TypeError, ValueError):
        # Some extension callables do not expose signatures; invoking them is
        # still safe because the explicit keyword contract is documented.
        pass
    return callback(cfg=cfg, plan=plan, env_cfg_override=dict(plan.env_cfg_override))


def launch_sonic_workers(
    cfg: DictConfig | Mapping[str, Any],
    *,
    script_path: str | Path,
    argv: Sequence[str],
    root_dir: str | Path = ".",
) -> None:
    """Launch the configured multi-GPU SONIC workers through UniLab torchrun."""

    plain = _plain(cfg)
    devices = resolve_dp_topology(_select(plain, "training.devices"))
    if devices is None or len(devices) < 2:
        raise SonicBridgeError("launch_sonic_workers requires at least two training.devices")
    plan = build_sonic_launch_plan(
        plain,
        root_dir=root_dir,
        world_size=len(devices),
        rank=0,
    )
    validate_dp_launchable(devices)
    # torchrun otherwise injects OMP_NUM_THREADS=1 into every child.  Apply
    # the SONIC cold-path thread budget in the parent before spawning so the
    # launcher preserves the configured BLAS/MuJoCo budget; child ranks still
    # apply their own CPU affinity after LOCAL_RANK is known.
    if plan.resources is not None:
        apply_sonic_rank_resources(plan.resources, pin_process=False)
    launch_torchrun_workers(
        devices,
        script_path=script_path,
        argv=argv,
        log_dir=str(plan.log_dir),
        nccl_compat_defaults=bool(_select(plain, "sonic.resources.nccl_compat_defaults", False)),
    )


__all__ = [
    "SonicBridgeError",
    "SonicLaunchPlan",
    "apply_sonic_rank_resources",
    "apply_sonic_torch_runtime",
    "build_sonic_launch_plan",
    "launch_sonic_workers",
    "resolve_sonic_log_dir",
    "resolve_sonic_device",
    "run_sonic_runtime",
    "write_sonic_preflight",
]
