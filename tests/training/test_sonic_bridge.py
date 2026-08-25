from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.training.sonic_bridge import (
    SonicBridgeError,
    apply_sonic_rank_resources,
    build_sonic_launch_plan,
    resolve_sonic_device,
    resolve_sonic_log_dir,
    run_sonic_runtime,
    write_sonic_preflight,
)


def _cfg(overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(Path(__file__).parents[2] / "conf" / "sonic"), version_base="1.3"
    ):
        return compose("config", overrides=overrides or [])


def test_default_plan_uses_configured_eight_rank_budget(tmp_path: Path):
    cfg = _cfg(["training.log_dir=" + str(tmp_path), "sonic.resources.pin_process=false"])
    plan = build_sonic_launch_plan(cfg, root_dir=tmp_path, rank=0, world_size=1)

    assert plan.world_size == 8
    assert plan.report.global_num_envs == 32768
    assert plan.report.global_samples == 786432
    assert plan.report.local_minibatch_size == 1024
    assert plan.resources is not None
    assert plan.resources.worker_count == 6
    assert cfg.training.task_name == "SonicG1Tracking"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.sonic.target_recipe == "sonic_v1_1"
    assert cfg.sonic.target_revision == "a0732b642c0333077e127a2f56ab0014c196bca4"
    assert cfg.sonic.observation_profile == "unitoken_all_noz_heading"
    assert cfg.sonic.owner.target_recipe == cfg.sonic.target_recipe
    assert cfg.sonic.owner.target_revision == cfg.sonic.target_revision
    assert cfg.sonic.owner.observation_profile == cfg.sonic.observation_profile
    assert plan.env_cfg_override["observation_profile"] == "unitoken_all_noz_heading"
    assert plan.env_cfg_override["cpu_ids"] == [0, 1, 2, 3, 4, 5]
    assert plan.env_cfg_override["reward_config"]["scales"]["motion_body_pos"] == 1.0
    # IsaacLab calls this ``decimation``; the UniLab EnvCfg contract uses the
    # equivalent control period and rejects unknown fields at registry time.
    assert "decimation" not in plan.env_cfg_override
    assert plan.env_cfg_override["sim_dt"] == pytest.approx(0.005)
    assert plan.env_cfg_override["ctrl_dt"] == pytest.approx(0.02)


def test_plan_supports_two_gpu_scan_without_using_eight_rank_arithmetic(tmp_path: Path):
    cfg = _cfg(
        [
            "training.devices=[0,1]",
            "training.log_dir=" + str(tmp_path),
            "algo.num_envs=256",
            "sonic.microbatch_size=64",
        ]
    )
    plan = build_sonic_launch_plan(cfg, root_dir=tmp_path, rank=1, world_size=2)
    assert plan.world_size == 2
    assert plan.report.global_num_envs == 512
    assert plan.report.global_samples == 12288
    assert plan.resources is not None
    assert plan.resources.cpu_ids == (6, 7, 8, 9, 10, 11)


def test_single_training_device_keeps_host_cuda_ordinal(tmp_path: Path) -> None:
    cfg = _cfg(
        [
            "training.devices=[3]",
            "training.log_dir=" + str(tmp_path),
            "algo.num_envs=256",
            "sonic.microbatch_size=64",
        ]
    )
    plan = build_sonic_launch_plan(cfg, root_dir=tmp_path, rank=0, world_size=1, local_rank=0)
    assert plan.runtime_device == "cuda:3"
    assert resolve_sonic_device(cfg, devices=(3,), world_size=1, local_rank=0) == "cuda:3"


def test_distributed_local_rank_and_device_conflicts_fail_closed(tmp_path: Path) -> None:
    cfg = _cfg(
        [
            "training.devices=[0,1]",
            "training.log_dir=" + str(tmp_path),
            "algo.num_envs=256",
            "sonic.microbatch_size=64",
        ]
    )
    with pytest.raises(SonicBridgeError, match="LOCAL_RANK"):
        build_sonic_launch_plan(cfg, root_dir=tmp_path, rank=0, world_size=2, local_rank=2)

    conflict = _cfg(
        [
            "training.devices=[0,1]",
            "training.device=cuda:0",
            "training.log_dir=" + str(tmp_path),
            "algo.num_envs=256",
            "sonic.microbatch_size=64",
        ]
    )
    with pytest.raises(SonicBridgeError, match="either training.device"):
        build_sonic_launch_plan(conflict, root_dir=tmp_path, rank=0, world_size=2, local_rank=0)

    single_device_without_topology = _cfg(
        [
            "training.device=cuda:0",
            "training.log_dir=" + str(tmp_path),
            "algo.num_envs=256",
            "sonic.microbatch_size=64",
        ]
    )
    with pytest.raises(SonicBridgeError, match="cannot select one device"):
        resolve_sonic_device(
            single_device_without_topology,
            devices=None,
            world_size=2,
            local_rank=0,
        )


def test_thread_environment_preserves_explicit_values():
    cfg = {"OMP_NUM_THREADS": "99"}
    plan = build_sonic_launch_plan(_cfg(), root_dir="/tmp", rank=0, world_size=8)
    assert plan.resources is not None
    result = apply_sonic_rank_resources(plan.resources, environ=cfg)
    assert result["OMP_NUM_THREADS"] == "99"
    assert result["MKL_NUM_THREADS"] == "2"


def test_log_dir_env_is_shared_across_ranks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("UNILAB_DP_LOG_DIR", str(tmp_path / "shared"))
    assert (
        resolve_sonic_log_dir(_cfg(), root_dir="/unused", world_size=8)
        == (tmp_path / "shared").resolve()
    )


def test_preflight_metadata_is_rank_zero_owned(tmp_path: Path):
    cfg = _cfg(["training.log_dir=" + str(tmp_path)])
    plan = build_sonic_launch_plan(cfg, root_dir=tmp_path, rank=0, world_size=8)
    path = write_sonic_preflight(plan)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["global_samples"] == 786432
    assert loaded["resources"]["worker_count"] == 6
    assert len(loaded["resources_by_rank"]) == 8
    assert loaded["logical_optimizer_steps_per_iteration"] == 20


def test_runtime_fails_closed_without_native_owner(tmp_path: Path):
    cfg = _cfg(["training.log_dir=" + str(tmp_path)])
    plan = build_sonic_launch_plan(cfg, root_dir=tmp_path, rank=0, world_size=8)
    with pytest.raises(SonicBridgeError, match="No native SONIC runtime"):
        run_sonic_runtime(cfg, plan)


def test_train_runtime_requires_versioned_motion_manifest(tmp_path: Path):
    cfg = _cfg(
        [
            "training.log_dir=" + str(tmp_path),
            "sonic.mode=train",
            "sonic.resources.gpu_numa_nodes=[0,0,0,1,1,1,1,1]",
        ]
    )
    plan = build_sonic_launch_plan(cfg, root_dir=tmp_path, rank=0, world_size=8)
    with pytest.raises(SonicBridgeError, match="requires sonic.motion_manifest"):
        run_sonic_runtime(cfg, plan, runtime=lambda **_: None)


def test_train_plan_requires_rank_ordered_gpu_numa_nodes(tmp_path: Path):
    cfg = _cfg(
        [
            "training.log_dir=" + str(tmp_path),
            "sonic.mode=train",
        ]
    )
    with pytest.raises(SonicBridgeError, match="rank-ordered"):
        build_sonic_launch_plan(cfg, root_dir=tmp_path, rank=0, world_size=8)


def test_train_plan_requires_topology_even_with_auto_profile(tmp_path: Path):
    cfg = _cfg(
        [
            "training.log_dir=" + str(tmp_path),
            "sonic.mode=train",
            "sonic.resources.profile=auto",
        ]
    )
    with pytest.raises(SonicBridgeError, match="rank-ordered"):
        build_sonic_launch_plan(cfg, root_dir=tmp_path, rank=0, world_size=8)


def test_runtime_receives_explicit_bridge_contract(tmp_path: Path):
    cfg = _cfg(["training.log_dir=" + str(tmp_path)])
    plan = build_sonic_launch_plan(cfg, root_dir=tmp_path, rank=0, world_size=8)
    seen: dict[str, object] = {}

    def runtime(*, cfg, plan, env_cfg_override):
        seen.update(cfg=cfg, plan=plan, env_cfg_override=env_cfg_override)
        return "ok"

    assert run_sonic_runtime(cfg, plan, runtime=runtime) == "ok"
    assert seen["plan"] is plan
    assert seen["env_cfg_override"]["cpu_ids"] == [0, 1, 2, 3, 4, 5]


def test_runtime_rejects_rsl_rl_style_callable(tmp_path: Path):
    cfg = _cfg(["training.log_dir=" + str(tmp_path)])
    plan = build_sonic_launch_plan(cfg, root_dir=tmp_path, rank=0, world_size=8)

    def wrong(cfg):
        del cfg

    with pytest.raises(SonicBridgeError, match="must accept keyword arguments"):
        run_sonic_runtime(cfg, plan, runtime=wrong)


def test_decimation_translation_rejects_inconsistent_ctrl_dt(tmp_path: Path):
    cfg = _cfg(
        [
            "training.log_dir=" + str(tmp_path),
            "+env.ctrl_dt=0.01",
        ]
    )
    with pytest.raises(SonicBridgeError, match="disagrees"):
        build_sonic_launch_plan(cfg, root_dir=tmp_path, rank=0, world_size=1)


def test_owner_backend_cannot_be_switched_by_training_override(tmp_path: Path):
    cfg = _cfg(
        [
            "training.log_dir=" + str(tmp_path),
            "training.sim_backend=motrix",
        ]
    )
    with pytest.raises(SonicBridgeError, match="owner"):
        build_sonic_launch_plan(cfg, root_dir=tmp_path, rank=0, world_size=1)


@pytest.mark.parametrize(
    ("override", "field"),
    (
        ("sonic.target_recipe=sonic_release", "target_recipe"),
        ("sonic.target_revision=deadbeef", "target_revision"),
        ("sonic.observation_profile=unitoken_all_noz", "observation_profile"),
        ("sonic.owner.target_recipe=sonic_release", "target_recipe"),
        ("sonic.owner.target_revision=deadbeef", "target_revision"),
        ("sonic.owner.observation_profile=unitoken_all_noz", "observation_profile"),
        ("env.observation_profile=unitoken_all_noz", "observation_profile"),
    ),
)
def test_owner_provenance_overrides_fail_closed(tmp_path: Path, override: str, field: str) -> None:
    cfg = _cfg(["training.log_dir=" + str(tmp_path), override])
    with pytest.raises(SonicBridgeError, match=field):
        build_sonic_launch_plan(cfg, root_dir=tmp_path, rank=0, world_size=1)
