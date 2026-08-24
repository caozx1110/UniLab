"""Static SONIC release contracts and cold-path preflight checks.

The module accepts plain nested mappings and OmegaConf ``DictConfig`` values,
but intentionally does not import a simulator, Torch, or the upstream SONIC
package.  It validates policy dimensions, rollout arithmetic, required paths,
and the minimum packed-motion manifest schema before an environment is built.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SonicContractError(ValueError):
    """Base exception for SONIC contract failures."""


class SonicConfigError(SonicContractError):
    """Invalid or inconsistent training configuration."""


class SonicPathError(SonicContractError):
    """Required asset or materialized-data path is unavailable."""


class SonicManifestError(SonicContractError):
    """Packed/mmap motion manifest is malformed."""


@dataclass(frozen=True)
class SonicReleaseSpec:
    """Policy-defining constants from the upstream ``sonic_release`` config."""

    action_dim: int = 29
    actor_obs_dim: int = 930
    critic_obs_dim: int = 1645
    tokenizer_obs_dim: int = 1761
    fsq_num_tokens: int = 2
    fsq_num_levels: int = 32
    horizon: int = 24
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    num_learning_iterations: int = 100_000
    num_envs_per_rank: int = 4096
    sim_dt: float = 0.005
    decimation: int = 4
    actor_prop_history_length: int = 10
    actor_actions_history_length: int = 10
    critic_prop_history_length: int = 10
    critic_actions_history_length: int = 10

    @property
    def ctrl_dt(self) -> float:
        return self.sim_dt * self.decimation

    @property
    def num_steps_per_env(self) -> int:
        return self.horizon


SONIC_RELEASE_SPEC = SonicReleaseSpec()
SONIC_ACTION_DIM = SONIC_RELEASE_SPEC.action_dim
SONIC_ACTOR_OBS_DIM = SONIC_RELEASE_SPEC.actor_obs_dim
SONIC_CRITIC_OBS_DIM = SONIC_RELEASE_SPEC.critic_obs_dim
SONIC_TOKENIZER_OBS_DIM = SONIC_RELEASE_SPEC.tokenizer_obs_dim
SONIC_ROLLOUT_HORIZON = SONIC_RELEASE_SPEC.horizon

# SONIC currently has one production owner.  Keep backend identity in the
# task/backend owner YAML and validate it at the bridge boundary; a standalone
# ``training.sim_backend`` override must not silently select another env.
SONIC_OWNER_TASK = "SonicG1Tracking"
SONIC_OWNER_BACKEND = "mujoco"


@dataclass(frozen=True)
class SonicPreflightReport:
    """Resolved distributed rollout and sequence-microbatch arithmetic.

    ``local_minibatch_size`` and ``microbatch_size`` count environment
    sequences, matching the upstream trainer.  Their ``*_transitions``
    properties multiply by ``horizon`` for throughput accounting.
    """

    world_size: int
    num_envs_per_rank: int
    horizon: int
    num_mini_batches: int
    samples_per_rank: int
    global_num_envs: int
    global_samples: int
    local_minibatch_size: int
    microbatch_size: int
    microbatches_per_minibatch: int
    paths: Mapping[str, Path] | None = None
    manifest: Mapping[str, Any] | None = None
    spec: SonicReleaseSpec = SONIC_RELEASE_SPEC

    @property
    def transitions_per_iteration(self) -> int:
        return self.global_samples

    @property
    def global_transitions_per_iteration(self) -> int:
        return self.global_samples

    @property
    def local_minibatch(self) -> int:
        return self.local_minibatch_size

    @property
    def local_minibatch_transitions(self) -> int:
        """Transitions represented by one local sequence minibatch."""
        return self.local_minibatch_size * self.horizon

    @property
    def local_batch_size(self) -> int:
        """Number of environment sequences collected by one rank."""
        return self.num_envs_per_rank

    @property
    def global_batch_size(self) -> int:
        """Number of environment sequences collected by all ranks."""
        return self.global_num_envs

    @property
    def microbatch_transitions(self) -> int:
        """Transitions represented by one learner microbatch."""
        return self.microbatch_size * self.horizon


def validate_sonic_owner(
    config: Mapping[str, Any], *, require_owner_marker: bool = False
) -> tuple[str, str]:
    """Validate the resolved task/backend owner identity.

    The owner YAML is selected with ``task=<task>/<backend>``.  This helper is
    deliberately small and pure so both the bridge and direct native-runtime
    callers enforce the same fail-closed boundary.
    """

    if not isinstance(config, Mapping):
        raise SonicConfigError(f"config must be a nested mapping, got {type(config).__name__}")
    training = config.get("training")
    training_map = training if isinstance(training, Mapping) else {}
    task = str(training_map.get("task_name", SONIC_OWNER_TASK))
    backend = str(training_map.get("sim_backend", SONIC_OWNER_BACKEND))
    owner = config.get("sonic")
    owner_map = owner if isinstance(owner, Mapping) else {}
    owner_identity = owner_map.get("owner")
    identity_map = owner_identity if isinstance(owner_identity, Mapping) else {}
    if require_owner_marker and not identity_map:
        raise SonicConfigError(
            "SONIC config must come from a task/backend owner YAML with sonic.owner metadata"
        )
    expected_task = str(identity_map.get("task_name", SONIC_OWNER_TASK))
    expected_backend = str(identity_map.get("sim_backend", SONIC_OWNER_BACKEND))
    if task != expected_task or backend != expected_backend:
        raise SonicConfigError(
            "SONIC task/backend must be selected by its owner YAML: "
            f"expected task={expected_task!r}, backend={expected_backend!r}; "
            f"resolved task={task!r}, backend={backend!r}"
        )
    if task != SONIC_OWNER_TASK or backend != SONIC_OWNER_BACKEND:
        raise SonicConfigError(
            "SONIC owner is not registered for this task/backend yet: "
            f"task={task!r}, backend={backend!r}"
        )
    return task, backend


_MISSING = object()


def _get(mapping: Mapping[str, Any], dotted: str) -> Any:
    current: Any = mapping
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _first(mapping: Mapping[str, Any], names: Iterable[str]) -> tuple[Any, str | None]:
    for name in names:
        value = _get(mapping, name)
        if value is not _MISSING and value is not None:
            return value, name
    return _MISSING, None


def _int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise SonicConfigError(f"{name} must be an integer >= {minimum}, got {value!r}")
    try:
        parsed, exact = int(value), float(value)
    except (TypeError, ValueError, OverflowError):
        raise SonicConfigError(f"{name} must be an integer >= {minimum}, got {value!r}") from None
    if not math.isfinite(exact) or exact != parsed or parsed < minimum:
        raise SonicConfigError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return parsed


def _number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise SonicConfigError(f"{name} must be a finite number >= {minimum}, got {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        raise SonicConfigError(
            f"{name} must be a finite number >= {minimum}, got {value!r}"
        ) from None
    if not math.isfinite(parsed) or parsed < minimum:
        raise SonicConfigError(f"{name} must be a finite number >= {minimum}, got {value!r}")
    return parsed


def _shape_product(value: Any, name: str) -> int:
    if isinstance(value, Mapping):
        value, _ = _first(value, ("dim", "size", "shape"))
        if value is _MISSING:
            raise SonicConfigError(f"{name} must contain dim, size, or shape")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            raise SonicConfigError(f"{name} shape must not be empty")
        result = 1
        for index, extent in enumerate(value):
            result *= _int(extent, f"{name}[{index}]")
        return result
    return _int(value, name)


def _same(actual: Any, expected: Any, name: str) -> None:
    try:
        ok = (
            math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-12)
            if isinstance(expected, float)
            else (
                not isinstance(actual, bool)
                and int(actual) == expected
                and float(actual) == expected
            )
        )
    except (TypeError, ValueError, OverflowError):
        ok = False
    if not ok:
        raise SonicConfigError(
            f"SONIC contract mismatch for {name}: expected {expected!r}, got {actual!r}"
        )


def validate_sonic_dimensions(
    dimensions: Mapping[str, Any], *, spec: SonicReleaseSpec = SONIC_RELEASE_SPEC
) -> dict[str, int]:
    """Require action, actor, critic and tokenizer dimensions."""
    if not isinstance(dimensions, Mapping):
        raise SonicConfigError(f"dimensions must be a mapping, got {type(dimensions).__name__}")
    aliases = {
        "action_dim": ("action_dim", "action_space", "action"),
        "actor_obs_dim": ("actor_obs_dim", "actor", "policy"),
        "critic_obs_dim": ("critic_obs_dim", "critic"),
        "tokenizer_obs_dim": ("tokenizer_obs_dim", "tokenizer"),
    }
    expected = {
        "action_dim": spec.action_dim,
        "actor_obs_dim": spec.actor_obs_dim,
        "critic_obs_dim": spec.critic_obs_dim,
        "tokenizer_obs_dim": spec.tokenizer_obs_dim,
    }
    result: dict[str, int] = {}
    for name, names in aliases.items():
        value, source = _first(dimensions, names)
        if value is _MISSING:
            raise SonicConfigError(f"missing required SONIC dimension: {name}")
        actual = _shape_product(value, source or name)
        _same(actual, expected[name], source or name)
        result[name] = actual
    return result


def validate_sonic_config(
    config: Mapping[str, Any],
    *,
    world_size: int | None = None,
    num_envs_per_rank: int | None = None,
    microbatch_size: int | None = None,
    dimensions: Mapping[str, Any] | None = None,
    spec: SonicReleaseSpec = SONIC_RELEASE_SPEC,
    require_dimensions: bool = False,
) -> SonicPreflightReport:
    """Validate release constants and distributed PPO arithmetic."""
    if not isinstance(config, Mapping):
        raise SonicConfigError(f"config must be a nested mapping, got {type(config).__name__}")
    value, source = _first(
        config, ("world_size", "training.world_size", "algo.world_size", "algo.config.world_size")
    )
    world = _int(
        world_size if world_size is not None else (1 if value is _MISSING else value), "world_size"
    )
    value, source = _first(
        config,
        (
            "num_envs_per_rank",
            "training.num_envs_per_rank",
            "algo.num_envs",
            "num_envs",
            "algo.config.num_envs",
        ),
    )
    envs = _int(
        num_envs_per_rank
        if num_envs_per_rank is not None
        else (spec.num_envs_per_rank if value is _MISSING else value),
        "num_envs_per_rank",
    )
    value, source = _first(
        config,
        (
            "horizon",
            "rollout_horizon",
            "num_steps_per_env",
            "algo.num_steps_per_env",
            "algo.config.num_steps_per_env",
        ),
    )
    horizon = spec.horizon if value is _MISSING else _int(value, source or "horizon")
    _same(horizon, spec.horizon, source or "horizon")
    value, source = _first(
        config,
        (
            "num_learning_epochs",
            "ppo_epochs",
            "algo.num_learning_epochs",
            "algo.config.num_learning_epochs",
        ),
    )
    epochs = (
        spec.num_learning_epochs
        if value is _MISSING
        else _int(value, source or "num_learning_epochs")
    )
    _same(epochs, spec.num_learning_epochs, source or "num_learning_epochs")
    value, source = _first(
        config,
        (
            "num_mini_batches",
            "num_minibatches",
            "ppo_minibatches",
            "algo.num_mini_batches",
            "algo.config.num_mini_batches",
        ),
    )
    minibatches = (
        spec.num_mini_batches if value is _MISSING else _int(value, source or "num_mini_batches")
    )
    _same(minibatches, spec.num_mini_batches, source or "num_mini_batches")
    aliases = {
        "action_dim": ("action_dim", "env.action_dim", "env.num_actions", "action_space"),
        "actor_obs_dim": (
            "actor_obs_dim",
            "obs_dims.actor",
            "observation_dims.actor",
            "env.actor_obs_dim",
        ),
        "critic_obs_dim": (
            "critic_obs_dim",
            "obs_dims.critic",
            "observation_dims.critic",
            "env.critic_obs_dim",
        ),
        "tokenizer_obs_dim": (
            "tokenizer_obs_dim",
            "obs_dims.tokenizer",
            "observation_dims.tokenizer",
            "env.tokenizer_obs_dim",
        ),
    }
    expected = {
        "action_dim": spec.action_dim,
        "actor_obs_dim": spec.actor_obs_dim,
        "critic_obs_dim": spec.critic_obs_dim,
        "tokenizer_obs_dim": spec.tokenizer_obs_dim,
    }
    for name, names in aliases.items():
        dim, dim_source = (
            _first(dimensions, (name, *names)) if dimensions is not None else (_MISSING, None)
        )
        if dim is _MISSING:
            dim, dim_source = _first(config, names)
        if dim is _MISSING:
            if require_dimensions:
                raise SonicConfigError(f"missing required SONIC dimension: {name}")
            continue
        _same(_shape_product(dim, dim_source or name), expected[name], dim_source or name)
    scalar_fields: tuple[tuple[str, float | int, tuple[str, ...]], ...] = (
        (
            "fsq_num_tokens",
            spec.fsq_num_tokens,
            ("fsq_num_tokens", "algo.config.actor.backbone.max_num_tokens"),
        ),
        (
            "fsq_num_levels",
            spec.fsq_num_levels,
            ("fsq_num_levels", "algo.config.actor.backbone.num_fsq_levels"),
        ),
        (
            "decimation",
            spec.decimation,
            (
                "decimation",
                "env.decimation",
                "env.control_config.decimation",
                "manager_env.decimation",
            ),
        ),
        (
            "sim_dt",
            spec.sim_dt,
            ("sim_dt", "env.sim_dt", "env.sim.dt", "manager_env.sim_dt"),
        ),
        (
            "actor_prop_history_length",
            spec.actor_prop_history_length,
            ("actor_prop_history_length",),
        ),
        (
            "actor_actions_history_length",
            spec.actor_actions_history_length,
            ("actor_actions_history_length",),
        ),
        (
            "critic_prop_history_length",
            spec.critic_prop_history_length,
            ("critic_prop_history_length",),
        ),
        (
            "critic_actions_history_length",
            spec.critic_actions_history_length,
            ("critic_actions_history_length",),
        ),
    )
    for name, expected_value, field_names in scalar_fields:
        scalar, scalar_source = _first(config, field_names)
        if scalar is _MISSING:
            continue
        actual = (
            _number(scalar, scalar_source or name)
            if isinstance(expected_value, float)
            else _int(scalar, scalar_source or name)
        )
        _same(actual, expected_value, scalar_source or name)
    samples = envs * horizon
    # SONIC batches environments (sequences), rather than flattening their
    # horizon into the PPO batch dimension.  Keep transition counts separate
    # for throughput accounting and global sample budgets.
    if envs % minibatches:
        raise SonicConfigError(
            f"rollout environments per rank ({envs}) must be divisible by "
            f"num_mini_batches={minibatches}"
        )
    local_minibatch = envs // minibatches
    value, source = _first(
        config,
        (
            "microbatch_size",
            "micro_batch_size",
            "per_device_train_batch_size",
            "training.microbatch_size",
            "training.per_device_train_batch_size",
            "algo.microbatch_size",
            "algo.per_device_train_batch_size",
            "algo.config.microbatch_size",
            "algo.config.per_device_train_batch_size",
        ),
    )
    micro = _int(
        microbatch_size
        if microbatch_size is not None
        else (local_minibatch if value is _MISSING else value),
        "microbatch_size",
    )
    if micro > local_minibatch:
        raise SonicConfigError(
            f"microbatch_size ({micro}) must not exceed local PPO minibatch ({local_minibatch})"
        )
    if local_minibatch % micro:
        raise SonicConfigError(
            f"local PPO minibatch ({local_minibatch}) must be divisible by microbatch_size ({micro})"
        )
    accumulation = local_minibatch // micro
    value, source = _first(
        config,
        (
            "gradient_accumulation_steps",
            "training.gradient_accumulation_steps",
            "algo.gradient_accumulation_steps",
        ),
    )
    if (
        value is not _MISSING
        and _int(value, source or "gradient_accumulation_steps") != accumulation
    ):
        raise SonicConfigError(
            f"gradient_accumulation_steps must equal {accumulation} for local minibatch={local_minibatch} and microbatch_size={micro}"
        )
    global_envs, global_samples = envs * world, samples * world
    _check_total(
        config, ("global_num_envs", "global_env_budget", "training.global_num_envs"), global_envs
    )
    _check_total(
        config,
        (
            "global_samples",
            "global_transitions",
            "global_transitions_per_iteration",
            "training.global_samples",
        ),
        global_samples,
    )
    return SonicPreflightReport(
        world,
        envs,
        horizon,
        minibatches,
        samples,
        global_envs,
        global_samples,
        local_minibatch,
        micro,
        accumulation,
        spec=spec,
    )


def _check_total(config: Mapping[str, Any], names: Iterable[str], expected: int) -> None:
    value, source = _first(config, names)
    if value is not _MISSING and _int(value, source or "global total") != expected:
        raise SonicConfigError(f"{source} is inconsistent: expected {expected}, got {value!r}")


def validate_sonic_paths(
    paths: Mapping[str, str | Path], *, files: Iterable[str] = (), directories: Iterable[str] = ()
) -> dict[str, Path]:
    """Resolve required paths and optionally enforce file/directory kinds."""
    if not isinstance(paths, Mapping):
        raise SonicPathError(f"paths must be a mapping, got {type(paths).__name__}")
    file_keys, directory_keys = set(files), set(directories)
    if file_keys & directory_keys:
        raise SonicPathError(
            f"path keys cannot be both files and directories: {sorted(file_keys & directory_keys)}"
        )
    result: dict[str, Path] = {}
    for key, raw in paths.items():
        name = str(key)
        if raw is None or not str(raw).strip():
            raise SonicPathError(f"required path {name!r} is empty")
        try:
            path = Path(raw).expanduser().resolve(strict=False)
        except (TypeError, ValueError, OSError):
            raise SonicPathError(f"required path {name!r} is invalid: {raw!r}") from None
        if not path.exists():
            raise SonicPathError(f"required path {name!r} does not exist: {path}")
        if name in file_keys and not path.is_file():
            raise SonicPathError(f"required file {name!r} is not a file: {path}")
        if name in directory_keys and not path.is_dir():
            raise SonicPathError(f"required directory {name!r} is not a directory: {path}")
        result[name] = path
    for name in file_keys | directory_keys:
        if name not in result:
            raise SonicPathError(f"path mapping is missing required key {name!r}")
    return result


def _manifest_data(
    manifest: Mapping[str, Any] | str | Path,
) -> tuple[Mapping[str, Any], Path | None]:
    if isinstance(manifest, Mapping):
        return manifest, None
    path = Path(manifest).expanduser().resolve(strict=False)
    if not path.is_file():
        raise SonicManifestError(f"motion manifest is not a file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SonicManifestError(f"could not read motion manifest {path}: {exc}") from None
    if not isinstance(data, Mapping):
        raise SonicManifestError("motion manifest root must be a JSON object")
    return data, path


def validate_motion_manifest(
    manifest: Mapping[str, Any] | str | Path, *, check_shards: bool = False
) -> dict[str, Any]:
    """Validate the minimum packed/mmap motion manifest schema."""
    data, manifest_path = _manifest_data(manifest)
    version = data.get("schema_version", data.get("version"))
    valid_version = (isinstance(version, str) and bool(version.strip())) or (
        isinstance(version, int) and not isinstance(version, bool) and version > 0
    )
    if not valid_version:
        raise SonicManifestError("manifest requires a non-empty schema_version/version")
    raw_fps = data.get("fps", data.get("sample_rate"))
    if raw_fps is None:
        # The strict materializer schema stores fps per clip.  Accept that
        # representation only when every clip supplies a positive value.
        candidate_clips = data.get("clips")
        clip_fps = (
            [clip.get("fps") for clip in candidate_clips if isinstance(clip, Mapping)]
            if isinstance(candidate_clips, Sequence)
            and not isinstance(candidate_clips, (str, bytes, bytearray))
            else []
        )
        if clip_fps and all(_positive_manifest_number(value) for value in clip_fps):
            raw_fps = clip_fps[0]
    if raw_fps is None or not _positive_manifest_number(raw_fps):
        raise SonicManifestError(f"manifest fps must be positive, got {raw_fps!r}")
    fields = data.get("fields")
    if isinstance(fields, Mapping):
        field_items = list(fields.items())
    elif isinstance(fields, Sequence) and not isinstance(fields, (str, bytes, bytearray)):
        field_items = [
            (item.get("name") if isinstance(item, Mapping) else None, item) for item in fields
        ]
    else:
        field_items = []
    if not field_items:
        raise SonicManifestError("manifest requires a non-empty fields mapping or list")
    for name, descriptor in field_items:
        if not isinstance(name, str) or not name.strip() or not isinstance(descriptor, Mapping):
            raise SonicManifestError("manifest fields must map non-empty names to objects")
        shape, dtype = descriptor.get("shape"), descriptor.get("dtype")
        if (
            not isinstance(shape, Sequence)
            or isinstance(shape, (str, bytes, bytearray))
            or not shape
        ):
            raise SonicManifestError(f"manifest fields[{name!r}].shape must be a non-empty list")
        for index, extent in enumerate(shape):
            if isinstance(extent, str) and extent.strip():
                # Materializer manifests may use symbolic extents such as
                # ``num_frames``; resolving them belongs to array validation,
                # not this metadata-only preflight.
                continue
            if isinstance(extent, bool):
                raise SonicManifestError(
                    f"manifest fields[{name!r}].shape[{index}] must be a non-negative integer"
                )
            try:
                integer, exact = int(extent), float(extent)
            except (TypeError, ValueError, OverflowError):
                raise SonicManifestError(
                    f"manifest fields[{name!r}].shape[{index}] must be a non-negative integer"
                ) from None
            if not math.isfinite(exact) or exact != integer or integer < 0:
                raise SonicManifestError(
                    f"manifest fields[{name!r}].shape[{index}] must be a non-negative integer"
                )
        if not isinstance(dtype, str) or not dtype.strip():
            raise SonicManifestError(f"manifest fields[{name!r}].dtype must be a non-empty string")
    for order_name in ("joint_order", "body_order"):
        order = data.get(order_name)
        if order is None:
            continue
        if (
            not isinstance(order, Sequence)
            or isinstance(order, (str, bytes, bytearray))
            or not order
        ):
            raise SonicManifestError(f"manifest {order_name} must be a non-empty list")
        names = []
        for index, item in enumerate(order):
            if not isinstance(item, str) or not item.strip():
                raise SonicManifestError(
                    f"manifest {order_name}[{index}] must be a non-empty string"
                )
            names.append(item)
        if len(names) != len(set(names)):
            raise SonicManifestError(f"manifest {order_name} must not contain duplicates")
    clips, count = data.get("clips"), data.get("clip_count", data.get("num_clips"))
    if clips is None and count is None:
        raise SonicManifestError("manifest requires clips or positive clip_count")
    if clips is not None and (
        not isinstance(clips, Sequence) or isinstance(clips, (str, bytes, bytearray)) or not clips
    ):
        raise SonicManifestError("manifest clips must be a non-empty list")
    if count is not None:
        try:
            count_int, exact = int(count), float(count)
        except (TypeError, ValueError, OverflowError):
            raise SonicManifestError("manifest clip_count must be a positive integer") from None
        if (
            isinstance(count, bool)
            or not math.isfinite(exact)
            or exact != count_int
            or count_int < 1
        ):
            raise SonicManifestError("manifest clip_count must be a positive integer")
        if clips is not None and len(clips) != count_int:
            raise SonicManifestError(
                f"manifest clip_count={count_int} does not match clips length={len(clips)}"
            )
    shards = data.get("shards")
    if shards is not None:
        if (
            not isinstance(shards, Sequence)
            or isinstance(shards, (str, bytes, bytearray))
            or not shards
        ):
            raise SonicManifestError("manifest shards must be a non-empty list")
        for index, shard in enumerate(shards):
            raw = shard.get("path") if isinstance(shard, Mapping) else shard
            if not isinstance(raw, str) or not raw.strip():
                raise SonicManifestError(f"manifest shards[{index}] requires a non-empty path")
            if check_shards:
                base = manifest_path.parent if manifest_path else Path.cwd()
                shard_path = (base / raw).resolve(strict=False)
                if not shard_path.exists():
                    raise SonicManifestError(f"manifest shard does not exist: {shard_path}")
    return dict(data)


def _positive_manifest_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(parsed) and parsed > 0


def run_sonic_preflight(
    config: Mapping[str, Any],
    *,
    paths: Mapping[str, str | Path] | None = None,
    path_files: Iterable[str] = (),
    path_directories: Iterable[str] = (),
    manifest: Mapping[str, Any] | str | Path | None = None,
    check_manifest_shards: bool = False,
    world_size: int | None = None,
    num_envs_per_rank: int | None = None,
    microbatch_size: int | None = None,
    dimensions: Mapping[str, Any] | None = None,
    spec: SonicReleaseSpec = SONIC_RELEASE_SPEC,
    require_dimensions: bool = False,
) -> SonicPreflightReport:
    """Run config, path, and optional manifest checks in one cold-path call."""
    report = validate_sonic_config(
        config,
        world_size=world_size,
        num_envs_per_rank=num_envs_per_rank,
        microbatch_size=microbatch_size,
        dimensions=dimensions,
        spec=spec,
        require_dimensions=require_dimensions,
    )
    checked_paths = (
        validate_sonic_paths(paths, files=path_files, directories=path_directories)
        if paths is not None
        else None
    )
    checked_manifest = (
        validate_motion_manifest(manifest, check_shards=check_manifest_shards)
        if manifest is not None
        else None
    )
    return SonicPreflightReport(
        report.world_size,
        report.num_envs_per_rank,
        report.horizon,
        report.num_mini_batches,
        report.samples_per_rank,
        report.global_num_envs,
        report.global_samples,
        report.local_minibatch_size,
        report.microbatch_size,
        report.microbatches_per_minibatch,
        checked_paths,
        checked_manifest,
        report.spec,
    )


validate_sonic_release_config = validate_sonic_config
validate_sonic_manifest = validate_motion_manifest

__all__ = [
    "SONIC_ACTION_DIM",
    "SONIC_ACTOR_OBS_DIM",
    "SONIC_CRITIC_OBS_DIM",
    "SONIC_RELEASE_SPEC",
    "SONIC_ROLLOUT_HORIZON",
    "SONIC_TOKENIZER_OBS_DIM",
    "SONIC_OWNER_BACKEND",
    "SONIC_OWNER_TASK",
    "SonicConfigError",
    "SonicContractError",
    "SonicManifestError",
    "SonicPathError",
    "SonicPreflightReport",
    "SonicReleaseSpec",
    "run_sonic_preflight",
    "validate_motion_manifest",
    "validate_sonic_config",
    "validate_sonic_dimensions",
    "validate_sonic_manifest",
    "validate_sonic_owner",
    "validate_sonic_paths",
    "validate_sonic_release_config",
]
