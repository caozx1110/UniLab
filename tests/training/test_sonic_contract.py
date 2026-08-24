from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from unilab.training.sonic_contract import (
    SonicConfigError,
    SonicManifestError,
    SonicPathError,
    SonicReleaseSpec,
    validate_motion_manifest,
    validate_sonic_config,
    validate_sonic_dimensions,
    validate_sonic_paths,
)


def _config(**overrides):
    config = {"algo": {"num_envs": 2048}}
    config.update(overrides)
    return config


def test_release_arithmetic_uses_per_rank_env_count_and_global_world_size():
    report = validate_sonic_config(_config(), world_size=8, microbatch_size=128)

    assert report.num_envs_per_rank == 2048
    assert report.samples_per_rank == 2048 * 24
    assert report.global_num_envs == 16_384
    assert report.global_samples == 393_216
    assert report.local_minibatch_size == 512
    assert report.local_minibatch_transitions == 12_288
    assert report.microbatches_per_minibatch == 4
    assert report.microbatch_transitions == 3_072


def test_official_eight_gpu_profile_has_sequence_minibatch_1024_and_accumulation_8():
    report = validate_sonic_config({}, world_size=8, microbatch_size=128)

    assert report.global_samples == 786_432
    assert report.local_minibatch_size == 1_024
    assert report.microbatches_per_minibatch == 8


def test_dictconfig_is_accepted():
    report = validate_sonic_config(
        OmegaConf.create({"algo": {"num_envs": 256}}), world_size=1, microbatch_size=64
    )
    assert report.samples_per_rank == 256 * 24


def test_release_horizon_mismatch_is_rejected():
    with pytest.raises(SonicConfigError, match="num_steps_per_env"):
        validate_sonic_config({"algo": {"num_steps_per_env": 32}})


def test_rollout_must_be_divisible_by_minibatches():
    spec = SonicReleaseSpec(num_mini_batches=5)
    with pytest.raises(SonicConfigError, match="divisible by num_mini_batches"):
        validate_sonic_config({}, spec=spec)


def test_local_minibatch_must_be_divisible_by_microbatch():
    with pytest.raises(SonicConfigError, match="local PPO minibatch"):
        validate_sonic_config({"algo": {"num_envs": 256}}, microbatch_size=65)


def test_gradient_accumulation_must_match_microbatch_arithmetic():
    with pytest.raises(SonicConfigError, match="gradient_accumulation_steps"):
        validate_sonic_config(
            {"algo": {"num_envs": 256}, "gradient_accumulation_steps": 2},
            microbatch_size=64,
        )


def test_global_totals_are_checked_when_present():
    with pytest.raises(SonicConfigError, match="global_num_envs"):
        validate_sonic_config({"algo": {"num_envs": 256}, "global_num_envs": 255}, world_size=1)


def test_dimensions_match_release_contract():
    result = validate_sonic_dimensions(
        {
            "action": [29],
            "actor": {"shape": [930]},
            "critic": 1645,
            "tokenizer": [1761],
        }
    )
    assert result == {
        "action_dim": 29,
        "actor_obs_dim": 930,
        "critic_obs_dim": 1645,
        "tokenizer_obs_dim": 1761,
    }


def test_dimensions_are_required_when_requested():
    with pytest.raises(SonicConfigError, match="action_dim"):
        validate_sonic_config({}, require_dimensions=True)


def test_dimension_mismatch_is_rejected():
    with pytest.raises(SonicConfigError, match="actor_obs_dim"):
        validate_sonic_dimensions(
            {
                "action_dim": 29,
                "actor_obs_dim": 929,
                "critic_obs_dim": 1645,
                "tokenizer_obs_dim": 1761,
            }
        )


def test_paths_are_resolved_and_type_checked(tmp_path: Path):
    motion_dir = tmp_path / "motions"
    motion_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    result = validate_sonic_paths(
        {"motion": motion_dir, "manifest": manifest},
        directories=("motion",),
        files=("manifest",),
    )
    assert result["motion"] == motion_dir.resolve()
    assert result["manifest"] == manifest.resolve()


def test_missing_path_has_clear_error(tmp_path: Path):
    with pytest.raises(SonicPathError, match="does not exist"):
        validate_sonic_paths({"motion": tmp_path / "missing"})


def _manifest(**overrides):
    value = {
        "schema_version": "1",
        "fps": 50,
        "fields": {"qpos": {"shape": [29], "dtype": "float32"}},
        "clip_count": 2,
        "clips": [{"id": "a"}, {"id": "b"}],
    }
    value.update(overrides)
    return value


def test_manifest_minimum_schema_is_accepted():
    assert validate_motion_manifest(_manifest())["schema_version"] == "1"


def test_manifest_materializer_shape_with_numeric_version_and_per_clip_fps():
    manifest = {
        "schema": "unilab.sonic.motion",
        "version": 1,
        "fields": [{"name": "qpos", "shape": [29], "dtype": "float32"}],
        "clips": [
            {
                "id": "clip-a",
                "path": "clip-a.npz",
                "fps": 50,
                "num_frames": 12,
            }
        ],
    }
    assert validate_motion_manifest(manifest)["version"] == 1


@pytest.mark.parametrize(
    "override, message",
    [
        ({"schema_version": ""}, "schema_version"),
        ({"fps": 0}, "fps"),
        ({"fields": {}}, "fields"),
        ({"clips": [], "clip_count": None}, "clips"),
        ({"clip_count": 3}, "clip_count"),
    ],
)
def test_manifest_schema_errors_are_explicit(override, message):
    with pytest.raises(SonicManifestError, match=message):
        validate_motion_manifest(_manifest(**override))


def test_manifest_json_and_shards_are_checked(tmp_path: Path):
    shard = tmp_path / "shard-0.bin"
    shard.write_bytes(b"data")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(shards=[{"path": shard.name}])), encoding="utf-8")

    assert validate_motion_manifest(manifest, check_shards=True)["shards"]


def test_manifest_missing_shard_is_rejected(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(shards=[{"path": "missing.bin"}])), encoding="utf-8")
    with pytest.raises(SonicManifestError, match="shard does not exist"):
        validate_motion_manifest(manifest, check_shards=True)
