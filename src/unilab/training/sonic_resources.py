"""Cold-path CPU/NUMA resource planning for SONIC training ranks.

SONIC runs one CPU MuJoCo environment and one GPU learner in every distributed
rank.  This module keeps the host-side resource policy out of the environment,
backend, and generic RSL-RL launch path.  It deliberately does not import
Torch or query CUDA: callers provide the rank-to-NUMA layout discovered during
their preflight and pass the resulting ``cpu_ids`` to ``EnvCfg``.

CPU sequences are ordered masks, not counts.  Keeping their order lets a
preflight put one hardware thread from each physical core first while still
supporting non-contiguous container cpusets.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, cast

_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
)


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return int(value)


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}")
    return int(value)


def _normalize_cpu_ids(
    values: Sequence[int],
    *,
    field: str,
    cpu_count: int | None,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field} must be a sequence of logical CPU ids")
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{field} must be a sequence of logical CPU ids") from exc
    ids = tuple(_non_negative_int(value, field=f"{field} entry") for value in raw_values)
    if not ids and not allow_empty:
        raise ValueError(f"{field} must be non-empty")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{field} contains duplicate logical CPU ids: {list(ids)!r}")
    if cpu_count is not None:
        outside = [cpu_id for cpu_id in ids if cpu_id >= cpu_count]
        if outside:
            raise ValueError(f"{field} contains ids {outside} outside cpu_count={cpu_count}")
    return ids


def available_logical_cpu_ids(*, cpu_count: int | None = None) -> tuple[int, ...]:
    """Return the current process CPU mask, falling back to ``cpu_count``.

    ``os.sched_getaffinity`` is preferred so a container or job scheduler's
    non-contiguous cpuset is respected.  Supplying ``cpu_count`` is useful for
    deterministic planning/tests and produces ``range(cpu_count)``.
    """
    if cpu_count is not None:
        return tuple(range(_positive_int(cpu_count, field="cpu_count")))
    sched_getaffinity = cast(
        Callable[[int], Iterable[int]] | None,
        getattr(os, "sched_getaffinity", None),
    )
    if callable(sched_getaffinity):
        try:
            affinity = tuple(sorted(sched_getaffinity(0)))
        except (OSError, TypeError, ValueError):
            affinity = ()
        if affinity:
            return affinity
    return tuple(range(_positive_int(os.cpu_count() or 1, field="cpu_count")))


def discover_gpu_numa_nodes(
    *,
    sysfs_root: str | os.PathLike[str] = "/sys/bus/pci/devices",
    nvidia_vendor: str = "0x10de",
    graphics_class: str = "0x030000",
) -> tuple[int, ...]:
    """Discover NVIDIA graphics-device NUMA nodes from PCI sysfs.

    Entries are returned in deterministic PCI BDF order.  PCI order is a
    useful cold-path fallback when NVML is unavailable (for example inside a
    container without the NVIDIA driver), but it is not asserted to be CUDA
    ordinal order.  Callers that know ``CUDA_VISIBLE_DEVICES`` should reorder
    the result with :func:`map_cuda_devices_to_numa_nodes` or provide an
    explicit ``gpu_numa_nodes`` config value.

    An empty tuple means that topology could not be discovered; no exception
    is raised for a missing sysfs mount so CPU-only preflight remains usable.
    """

    root = Path(sysfs_root)
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError:
        return ()
    nodes: list[int] = []
    for entry in entries:
        try:
            vendor = (entry / "vendor").read_text(encoding="ascii").strip().lower()
            device_class = (entry / "class").read_text(encoding="ascii").strip().lower()
            if vendor != nvidia_vendor.lower() or device_class != graphics_class.lower():
                continue
            node = int((entry / "numa_node").read_text(encoding="ascii").strip())
        except (OSError, UnicodeError, ValueError):
            continue
        if node >= 0:
            nodes.append(node)
    return tuple(nodes)


def map_cuda_devices_to_numa_nodes(
    devices: Sequence[int],
    discovered_nodes: Sequence[int],
) -> tuple[int, ...]:
    """Map configured logical CUDA indices to a discovered NUMA sequence."""

    normalized_devices = tuple(
        _non_negative_int(device, field="CUDA device index") for device in devices
    )
    if len(set(normalized_devices)) != len(normalized_devices):
        raise ValueError(f"CUDA devices must be unique, got {list(normalized_devices)!r}")
    nodes = tuple(_non_negative_int(node, field="GPU NUMA node") for node in discovered_nodes)
    missing = [device for device in normalized_devices if device >= len(nodes)]
    if missing:
        raise ValueError(
            f"CUDA device indices {missing} are absent from discovered NUMA topology "
            f"with {len(nodes)} entries"
        )
    return tuple(nodes[device] for device in normalized_devices)


@dataclass(frozen=True)
class SonicRankResources:
    """Resolved host resources owned by one SONIC distributed rank."""

    world_size: int
    rank: int
    numa_node: int | None
    cpu_ids: tuple[int, ...]
    torch_num_threads: int
    torch_num_interop_threads: int

    @property
    def worker_count(self) -> int:
        """MuJoCo worker count implied by ``EnvCfg.cpu_ids``."""
        return len(self.cpu_ids)

    @property
    def thread_env(self) -> dict[str, str]:
        """Environment recommendations to apply before importing Torch/BLAS."""
        value = str(self.torch_num_threads)
        return {key: value for key in _THREAD_ENV_KEYS}

    def to_dict(self) -> dict[str, object]:
        """Return a logging/config-friendly resource manifest."""
        return {
            "world_size": self.world_size,
            "rank": self.rank,
            "numa_node": self.numa_node,
            "cpu_ids": list(self.cpu_ids),
            "worker_count": self.worker_count,
            "torch_num_threads": self.torch_num_threads,
            "torch_num_interop_threads": self.torch_num_interop_threads,
            "thread_env": self.thread_env,
        }


def apply_sonic_torch_threads(
    resources: SonicRankResources,
    *,
    torch_runtime: Any | None = None,
) -> dict[str, int]:
    """Apply the rank's Torch intra/inter-op thread budget on the cold path.

    ``apply_sonic_rank_resources`` only prepares BLAS environment variables so
    it can run before importing Torch.  This helper is intentionally separate:
    callers invoke it *after* Torch has been imported, but before constructing
    the learner or starting any parallel work.  PyTorch refuses to change the
    inter-op pool after work has started; surface that condition with a
    diagnostic instead of silently claiming that the configured budget was
    applied.

    ``torch_runtime`` is injectable for tests and for launchers that already
    imported Torch.  Passing ``None`` performs the lazy import here.
    """
    runtime = torch_runtime
    if runtime is None:
        import torch as imported_torch  # noqa: PLC0415

        runtime = imported_torch

    requested_intra = int(resources.torch_num_threads)
    requested_inter = int(resources.torch_num_interop_threads)
    set_intra = getattr(runtime, "set_num_threads", None)
    set_inter = getattr(runtime, "set_num_interop_threads", None)
    get_inter = getattr(runtime, "get_num_interop_threads", None)
    if not callable(set_intra):
        raise RuntimeError("SONIC Torch runtime does not expose set_num_threads")
    if not callable(set_inter):
        raise RuntimeError("SONIC Torch runtime does not expose set_num_interop_threads")

    try:
        # Set inter-op first: this setter must run before the first parallel
        # operator, while intra-op can safely be changed afterwards.
        current_inter = int(cast(Callable[[], int], get_inter)()) if callable(get_inter) else None
        if current_inter != requested_inter:
            set_inter(requested_inter)
        set_intra(requested_intra)
    except RuntimeError as exc:
        raise RuntimeError(
            "SONIC Torch thread budget must be configured before parallel work starts "
            f"(requested intra/inter={requested_intra}/{requested_inter})"
        ) from exc

    return {
        "torch_num_threads": requested_intra,
        "torch_num_interop_threads": requested_inter,
    }


def _normalize_rank_numa_nodes(
    values: Sequence[int | None] | None,
    *,
    world_size: int,
) -> tuple[int | None, ...] | None:
    if values is None:
        return None
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise ValueError("gpu_numa_nodes must be a sequence") from exc
    nodes = tuple(
        None if value is None else _non_negative_int(value, field="gpu_numa_nodes entry")
        for value in raw_values
    )
    if len(nodes) != world_size:
        raise ValueError(
            f"gpu_numa_nodes must provide world_size={world_size} entries, got {len(nodes)}"
        )
    return nodes


def _normalize_numa_cpu_ids(
    values: Mapping[int, Sequence[int]] | None,
    *,
    available: tuple[int, ...],
    cpu_count: int | None,
) -> dict[int, tuple[int, ...]] | None:
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise ValueError("numa_cpu_ids must be a mapping from NUMA node to CPU ids")
    available_set = set(available)
    normalized: dict[int, tuple[int, ...]] = {}
    seen: set[int] = set()
    for raw_node, raw_ids in values.items():
        node = _non_negative_int(raw_node, field="numa_cpu_ids node")
        ids = _normalize_cpu_ids(
            raw_ids,
            field=f"numa_cpu_ids[{node}]",
            cpu_count=cpu_count,
        )
        overlap = seen.intersection(ids)
        if overlap:
            raise ValueError(f"numa_cpu_ids masks overlap on logical CPU ids {sorted(overlap)}")
        seen.update(ids)
        # A system NUMA mask may be wider than the process cpuset.  Filtering
        # here retains the caller-provided (for example physical-core-first)
        # order while never assigning a CPU unavailable to this process.
        normalized[node] = tuple(cpu_id for cpu_id in ids if cpu_id in available_set)
    if not normalized:
        raise ValueError("numa_cpu_ids must contain at least one NUMA node")
    return normalized


def _normalize_explicit_cpu_ids(
    values: Sequence[Sequence[int]],
    *,
    world_size: int,
    available: tuple[int, ...],
    cpu_count: int | None,
    rank_numa_nodes: tuple[int | None, ...] | None,
    numa_cpu_ids: dict[int, tuple[int, ...]] | None,
) -> tuple[tuple[int, ...], ...]:
    try:
        raw_segments = tuple(values)
    except TypeError as exc:
        raise ValueError("explicit_cpu_ids must be a sequence of rank segments") from exc
    segments = tuple(
        _normalize_cpu_ids(
            segment,
            field=f"explicit_cpu_ids[{rank}]",
            cpu_count=cpu_count,
        )
        for rank, segment in enumerate(raw_segments)
    )
    if len(segments) != world_size:
        raise ValueError(
            f"explicit_cpu_ids must provide world_size={world_size} segments, got {len(segments)}"
        )

    available_set = set(available)
    seen: set[int] = set()
    for segment_rank, segment in enumerate(segments):
        unavailable = set(segment).difference(available_set)
        if unavailable:
            raise ValueError(
                f"explicit_cpu_ids[{segment_rank}] contains unavailable logical CPU ids "
                f"{sorted(unavailable)}"
            )
        overlap = seen.intersection(segment)
        if overlap:
            raise ValueError(
                f"explicit_cpu_ids rank segments overlap on logical CPU ids {sorted(overlap)}"
            )
        seen.update(segment)

        if rank_numa_nodes is not None and numa_cpu_ids is not None:
            node = rank_numa_nodes[segment_rank]
            if node is not None:
                if node not in numa_cpu_ids:
                    raise ValueError(
                        f"gpu_numa_nodes[{segment_rank}]={node} is missing from numa_cpu_ids"
                    )
                remote = set(segment).difference(numa_cpu_ids[node])
                if remote:
                    raise ValueError(
                        f"explicit_cpu_ids[{segment_rank}] contains CPUs {sorted(remote)} "
                        f"outside GPU-local NUMA node {node}"
                    )
    return segments


def _allocate_cpu_ids(
    *,
    world_size: int,
    workers_per_rank: int,
    available: tuple[int, ...],
    rank_numa_nodes: tuple[int | None, ...] | None,
    numa_cpu_ids: dict[int, tuple[int, ...]] | None,
) -> tuple[tuple[int, ...], ...]:
    allocations: list[tuple[int, ...] | None] = [None] * world_size
    if rank_numa_nodes is None:
        groups: tuple[tuple[int | None, tuple[int, ...], tuple[int, ...]], ...] = (
            (None, tuple(range(world_size)), available),
        )
    else:
        assert numa_cpu_ids is not None
        # Assign GPU-local ranks first.  Ranks whose GPU locality is unknown
        # may use the remainder, but must never consume a local rank's pool.
        ordered_nodes: list[int | None] = list(
            dict.fromkeys(node for node in rank_numa_nodes if node is not None)
        )
        if None in rank_numa_nodes:
            ordered_nodes.append(None)
        groups = tuple(
            (
                node,
                tuple(rank for rank, rank_node in enumerate(rank_numa_nodes) if rank_node == node),
                available if node is None else numa_cpu_ids.get(node, ()),
            )
            for node in ordered_nodes
        )

    already_assigned: set[int] = set()
    for node, ranks, raw_pool in groups:
        pool = tuple(cpu_id for cpu_id in raw_pool if cpu_id not in already_assigned)
        required = len(ranks) * workers_per_rank
        if len(pool) < required:
            location = "unassigned CPU pool" if node is None else f"NUMA node {node}"
            raise ValueError(
                f"{location} has {len(pool)} available logical CPUs but "
                f"{len(ranks)} rank(s) x {workers_per_rank} workers require {required}"
            )
        for offset, target_rank in enumerate(ranks):
            start = offset * workers_per_rank
            segment = pool[start : start + workers_per_rank]
            allocations[target_rank] = segment
            already_assigned.update(segment)

    if any(segment is None for segment in allocations):  # pragma: no cover - invariant guard.
        raise RuntimeError("internal error: not every SONIC rank received a CPU allocation")
    return tuple(segment for segment in allocations if segment is not None)


def resolve_sonic_rank_resources(
    *,
    world_size: int,
    rank: int,
    cpu_count: int | None = None,
    available_cpu_ids: Sequence[int] | None = None,
    gpu_numa_nodes: Sequence[int | None] | None = None,
    numa_cpu_ids: Mapping[int, Sequence[int]] | None = None,
    explicit_cpu_ids: Sequence[Sequence[int]] | None = None,
    workers_per_rank: int = 6,
    torch_num_threads: int = 2,
    torch_num_interop_threads: int = 1,
) -> SonicRankResources:
    """Resolve one rank's CPU affinity and thread budget.

    ``gpu_numa_nodes`` is ordered by distributed rank, not CUDA device number;
    therefore arbitrary ``training.devices`` order and any world size work.
    ``numa_cpu_ids`` values are ordered logical CPU masks.  When both are
    supplied, automatic and explicit assignments are required to be GPU-local.

    ``explicit_cpu_ids`` contains one non-overlapping segment per rank and
    overrides ``workers_per_rank``.  Otherwise each rank receives exactly
    ``workers_per_rank`` CPUs, either from its GPU-local node or from the
    process-wide available mask.
    """
    world_size = _positive_int(world_size, field="world_size")
    rank = _non_negative_int(rank, field="rank")
    if rank >= world_size:
        raise ValueError(f"rank={rank} is out of range for world_size={world_size}")
    resolved_cpu_count = None if cpu_count is None else _positive_int(cpu_count, field="cpu_count")
    workers_per_rank = _positive_int(workers_per_rank, field="workers_per_rank")
    torch_num_threads = _positive_int(torch_num_threads, field="torch_num_threads")
    torch_num_interop_threads = _positive_int(
        torch_num_interop_threads,
        field="torch_num_interop_threads",
    )

    if available_cpu_ids is None:
        available = available_logical_cpu_ids(cpu_count=resolved_cpu_count)
    else:
        available = _normalize_cpu_ids(
            available_cpu_ids,
            field="available_cpu_ids",
            cpu_count=resolved_cpu_count,
        )
    rank_numa_nodes = _normalize_rank_numa_nodes(gpu_numa_nodes, world_size=world_size)
    node_cpu_ids = _normalize_numa_cpu_ids(
        numa_cpu_ids,
        available=available,
        cpu_count=resolved_cpu_count,
    )
    if (rank_numa_nodes is None) != (node_cpu_ids is None):
        raise ValueError("gpu_numa_nodes and numa_cpu_ids must be provided together")
    if rank_numa_nodes is not None and node_cpu_ids is not None:
        missing_nodes = sorted(
            {node for node in rank_numa_nodes if node is not None}.difference(node_cpu_ids)
        )
        if missing_nodes:
            raise ValueError(
                "numa_cpu_ids is missing GPU-local NUMA node(s) "
                f"{missing_nodes} referenced by gpu_numa_nodes"
            )

    if explicit_cpu_ids is not None:
        assignments = _normalize_explicit_cpu_ids(
            explicit_cpu_ids,
            world_size=world_size,
            available=available,
            cpu_count=resolved_cpu_count,
            rank_numa_nodes=rank_numa_nodes,
            numa_cpu_ids=node_cpu_ids,
        )
    else:
        assignments = _allocate_cpu_ids(
            world_size=world_size,
            workers_per_rank=workers_per_rank,
            available=available,
            rank_numa_nodes=rank_numa_nodes,
            numa_cpu_ids=node_cpu_ids,
        )

    return SonicRankResources(
        world_size=world_size,
        rank=rank,
        numa_node=None if rank_numa_nodes is None else rank_numa_nodes[rank],
        cpu_ids=assignments[rank],
        torch_num_threads=torch_num_threads,
        torch_num_interop_threads=torch_num_interop_threads,
    )


@dataclass(frozen=True)
class SonicResourceProfile:
    """A named, explicit host profile suitable for config/preflight logging."""

    name: str
    cpu_count: int
    gpu_numa_nodes: tuple[int | None, ...]
    numa_cpu_ids: tuple[tuple[int, tuple[int, ...]], ...]
    cpu_ids_by_rank: tuple[tuple[int, ...], ...]
    torch_num_threads: int = 2
    torch_num_interop_threads: int = 1

    @property
    def world_size(self) -> int:
        return len(self.gpu_numa_nodes)

    def resolve(
        self,
        rank: int,
        *,
        available_cpu_ids: Sequence[int] | None = None,
    ) -> SonicRankResources:
        """Validate this profile against an optional runtime cpuset and resolve a rank."""
        runtime_cpu_ids = (
            available_logical_cpu_ids() if available_cpu_ids is None else available_cpu_ids
        )
        return resolve_sonic_rank_resources(
            world_size=self.world_size,
            rank=rank,
            cpu_count=self.cpu_count,
            available_cpu_ids=runtime_cpu_ids,
            gpu_numa_nodes=self.gpu_numa_nodes,
            numa_cpu_ids=dict(self.numa_cpu_ids),
            explicit_cpu_ids=self.cpu_ids_by_rank,
            torch_num_threads=self.torch_num_threads,
            torch_num_interop_threads=self.torch_num_interop_threads,
        )


# Explicit profile for the target workstation: 8 x RTX 4090, 76 physical
# cores / 152 logical CPUs, and a 3-GPU + 5-GPU split across two NUMA nodes.
# The first hardware thread of each selected physical core is used; sibling
# SMT threads and spare cores remain available to the OS, logging, and I/O.
SONIC_8X4090_76C_PROFILE = SonicResourceProfile(
    name="sonic-8x4090-76c-2numa",
    cpu_count=152,
    gpu_numa_nodes=(0, 0, 0, 1, 1, 1, 1, 1),
    numa_cpu_ids=(
        (0, tuple(range(0, 38)) + tuple(range(76, 114))),
        (1, tuple(range(38, 76)) + tuple(range(114, 152))),
    ),
    cpu_ids_by_rank=(
        tuple(range(0, 6)),
        tuple(range(6, 12)),
        tuple(range(12, 18)),
        tuple(range(38, 44)),
        tuple(range(44, 50)),
        tuple(range(50, 56)),
        tuple(range(56, 62)),
        tuple(range(62, 68)),
    ),
)


__all__ = [
    "SONIC_8X4090_76C_PROFILE",
    "SonicRankResources",
    "SonicResourceProfile",
    "available_logical_cpu_ids",
    "apply_sonic_torch_threads",
    "discover_gpu_numa_nodes",
    "map_cuda_devices_to_numa_nodes",
    "resolve_sonic_rank_resources",
]
