from __future__ import annotations

from typing import Any

import pytest

from unilab.training.sonic_resources import (
    SONIC_8X4090_76C_PROFILE,
    apply_sonic_torch_threads,
    available_logical_cpu_ids,
    discover_gpu_numa_nodes,
    map_cuda_devices_to_numa_nodes,
    resolve_sonic_rank_resources,
)


def test_available_cpu_ids_honors_explicit_logical_cpu_count() -> None:
    assert available_logical_cpu_ids(cpu_count=8) == tuple(range(8))


def test_available_cpu_ids_falls_back_when_affinity_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(_pid: int) -> set[int]:
        raise OSError("not available")

    monkeypatch.setattr(
        "unilab.training.sonic_resources.os.sched_getaffinity",
        _raise,
    )
    monkeypatch.setattr("unilab.training.sonic_resources.os.cpu_count", lambda: 3)
    assert available_logical_cpu_ids() == (0, 1, 2)


def test_auto_allocation_preserves_non_contiguous_process_cpuset() -> None:
    available = (2, 4, 8, 10, 12, 14, 20, 22)
    rank0 = resolve_sonic_rank_resources(
        world_size=2,
        rank=0,
        available_cpu_ids=available,
        workers_per_rank=4,
    )
    rank1 = resolve_sonic_rank_resources(
        world_size=2,
        rank=1,
        available_cpu_ids=available,
        workers_per_rank=4,
    )
    assert rank0.cpu_ids == available[:4]
    assert rank1.cpu_ids == available[4:]
    assert rank0.worker_count == 4
    assert rank0.thread_env["OMP_NUM_THREADS"] == "2"
    assert rank0.thread_env["MKL_NUM_THREADS"] == "2"
    assert rank0.torch_num_interop_threads == 1


def test_resource_manifest_exposes_worker_and_thread_settings() -> None:
    resource = resolve_sonic_rank_resources(
        world_size=1,
        rank=0,
        available_cpu_ids=(9, 11, 13),
        workers_per_rank=3,
        torch_num_threads=4,
        torch_num_interop_threads=2,
    )
    assert resource.to_dict() == {
        "world_size": 1,
        "rank": 0,
        "numa_node": None,
        "cpu_ids": [9, 11, 13],
        "worker_count": 3,
        "torch_num_threads": 4,
        "torch_num_interop_threads": 2,
        "thread_env": {
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
            "TORCH_NUM_THREADS": "4",
        },
    }


def test_apply_sonic_torch_threads_sets_both_pools() -> None:
    class FakeTorch:
        def __init__(self) -> None:
            self.intra: int | None = None
            self.inter: int = 76

        def set_num_threads(self, value: int) -> None:
            self.intra = value

        def get_num_interop_threads(self) -> int:
            return self.inter

        def set_num_interop_threads(self, value: int) -> None:
            self.inter = value

    fake = FakeTorch()
    resource = resolve_sonic_rank_resources(
        world_size=1,
        rank=0,
        available_cpu_ids=(0, 1, 2),
        workers_per_rank=3,
        torch_num_threads=2,
        torch_num_interop_threads=1,
    )
    assert apply_sonic_torch_threads(resource, torch_runtime=fake) == {
        "torch_num_threads": 2,
        "torch_num_interop_threads": 1,
    }
    assert fake.intra == 2
    assert fake.inter == 1


def test_apply_sonic_torch_threads_fails_if_interop_is_too_late() -> None:
    class FakeTorch:
        def set_num_threads(self, value: int) -> None:
            del value

        def get_num_interop_threads(self) -> int:
            return 76

        def set_num_interop_threads(self, value: int) -> None:
            del value
            raise RuntimeError("parallel work has started")

    resource = resolve_sonic_rank_resources(
        world_size=1,
        rank=0,
        available_cpu_ids=(0, 1, 2),
        workers_per_rank=3,
    )
    with pytest.raises(RuntimeError, match="before parallel work"):
        apply_sonic_torch_threads(resource, torch_runtime=FakeTorch())


def test_numa_local_allocation_follows_rank_gpu_layout() -> None:
    # Deliberately use a non-contiguous NUMA mask and a rank order that maps
    # ranks 0/2 to node 1 and rank 1 to node 0.
    node0 = (1, 3, 5, 7)
    node1 = (40, 42, 44, 46, 48, 50)
    rank0 = resolve_sonic_rank_resources(
        world_size=3,
        rank=0,
        available_cpu_ids=node0 + node1,
        gpu_numa_nodes=(1, 0, 1),
        numa_cpu_ids={0: node0, 1: node1},
        workers_per_rank=2,
    )
    rank1 = resolve_sonic_rank_resources(
        world_size=3,
        rank=1,
        available_cpu_ids=node0 + node1,
        gpu_numa_nodes=(1, 0, 1),
        numa_cpu_ids={0: node0, 1: node1},
        workers_per_rank=2,
    )
    rank2 = resolve_sonic_rank_resources(
        world_size=3,
        rank=2,
        available_cpu_ids=node0 + node1,
        gpu_numa_nodes=(1, 0, 1),
        numa_cpu_ids={0: node0, 1: node1},
        workers_per_rank=2,
    )
    assert rank0.cpu_ids == (40, 42)
    assert rank1.cpu_ids == (1, 3)
    assert rank2.cpu_ids == (44, 46)
    assert rank0.numa_node == 1
    assert rank1.numa_node == 0


def test_explicit_segments_are_checked_for_overlap_and_cpuset() -> None:
    with pytest.raises(ValueError, match="overlap"):
        resolve_sonic_rank_resources(
            world_size=2,
            rank=0,
            available_cpu_ids=(0, 1, 2, 3),
            explicit_cpu_ids=((0, 1), (1, 2)),
        )
    with pytest.raises(ValueError, match="unavailable"):
        resolve_sonic_rank_resources(
            world_size=2,
            rank=0,
            available_cpu_ids=(0, 1, 2, 3),
            explicit_cpu_ids=((0, 4), (1, 2)),
        )


def test_explicit_segments_must_be_gpu_local() -> None:
    with pytest.raises(ValueError, match="outside GPU-local NUMA node"):
        resolve_sonic_rank_resources(
            world_size=2,
            rank=0,
            available_cpu_ids=(0, 1, 2, 3),
            gpu_numa_nodes=(0, 1),
            numa_cpu_ids={0: (0, 1), 1: (2, 3)},
            explicit_cpu_ids=((0, 2), (1, 3)),
        )


def test_numa_layout_must_cover_every_rank_gpu_node() -> None:
    with pytest.raises(ValueError, match="missing GPU-local NUMA node"):
        resolve_sonic_rank_resources(
            world_size=2,
            rank=0,
            available_cpu_ids=(0, 1, 2, 3),
            gpu_numa_nodes=(0, 1),
            numa_cpu_ids={0: (0, 1)},
            workers_per_rank=1,
        )


def test_8x4090_profile_has_rank_local_six_cpu_segments() -> None:
    profile = SONIC_8X4090_76C_PROFILE
    assert profile.world_size == 8
    resources = [profile.resolve(rank) for rank in range(profile.world_size)]
    assert [resource.worker_count for resource in resources] == [6] * 8
    assert resources[0].cpu_ids == tuple(range(0, 6))
    assert resources[3].cpu_ids == tuple(range(38, 44))
    assert resources[-1].cpu_ids == tuple(range(62, 68))
    assert resources[0].numa_node == 0
    assert resources[3].numa_node == 1
    assert resources[0].thread_env == {
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
        "TORCH_NUM_THREADS": "2",
    }


def test_profile_fails_when_scheduler_cpuset_hides_a_rank_cpu() -> None:
    profile = SONIC_8X4090_76C_PROFILE
    with pytest.raises(ValueError, match="unavailable"):
        profile.resolve(0, available_cpu_ids=tuple(range(1, 152)))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"world_size": 0, "rank": 0}, "world_size"),
        ({"world_size": 2, "rank": 2}, "out of range"),
        ({"world_size": 2, "rank": 0, "workers_per_rank": 0}, "workers_per_rank"),
        (
            {
                "world_size": 1,
                "rank": 0,
                "gpu_numa_nodes": (0,),
                "numa_cpu_ids": {0: None},
            },
            "sequence",
        ),
        (
            {
                "world_size": 2,
                "rank": 0,
                "gpu_numa_nodes": (0, 1),
            },
            "provided together",
        ),
    ],
)
def test_resource_input_validation(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_sonic_rank_resources(**kwargs)


def test_discover_gpu_numa_nodes_reads_only_nvidia_graphics_entries(tmp_path) -> None:
    for name, vendor, class_code, node in (
        ("0000:02:00.0", "0x10de", "0x030000", "1"),
        ("0000:01:00.0", "0x10de", "0x030000", "0"),
        ("0000:03:00.0", "0x8086", "0x030000", "0"),
        ("0000:04:00.0", "0x10de", "0x040300", "1"),
    ):
        device = tmp_path / name
        device.mkdir()
        (device / "vendor").write_text(vendor, encoding="ascii")
        (device / "class").write_text(class_code, encoding="ascii")
        (device / "numa_node").write_text(node, encoding="ascii")
    assert discover_gpu_numa_nodes(sysfs_root=tmp_path) == (0, 1)


def test_cuda_device_to_numa_mapping_preserves_configured_order() -> None:
    assert map_cuda_devices_to_numa_nodes((2, 0, 1), (0, 1, 1)) == (1, 0, 1)
    with pytest.raises(ValueError, match="absent"):
        map_cuda_devices_to_numa_nodes((3,), (0, 1, 1))
