"""Cold-path conversion for the released SONIC v1.1 checkpoint."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .algorithm import SonicPPO
from .model import SonicActorCritic

OFFICIAL_SONIC_V11_FORMAT = "nvlabs.sonic_v1_1.trl"


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapped_policy_key(key: str) -> str:
    if key == "std":
        return key
    parts = key.split(".")
    if len(parts) >= 5 and parts[:2] == ["actor_module", "encoders"] and parts[3] == "module":
        return ".".join(("tokenizer", "encoders", parts[2], *parts[4:]))
    if len(parts) >= 5 and parts[:2] == ["actor_module", "decoders"] and parts[3] == "module":
        return ".".join(("tokenizer", "decoders", parts[2], *parts[4:]))
    raise ValueError(f"unsupported official SONIC v1.1 policy key: {key!r}")


def _mapped_value_key(key: str) -> str:
    prefix = "critic_module.module."
    if key.startswith(prefix):
        return f"critic.{key.removeprefix(prefix)}"
    normalizer_keys = {
        "running_mean_std.running_mean": "critic_rms.mean",
        "running_mean_std.running_var": "critic_rms.var",
        "running_mean_std.count": "critic_rms.count",
    }
    if key in normalizer_keys:
        return normalizer_keys[key]
    raise ValueError(f"unsupported official SONIC v1.1 value key: {key!r}")


def map_official_sonic_v11_model_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    policy_state = checkpoint.get("policy_state_dict")
    value_state = checkpoint.get("value_state_dict")
    if not isinstance(policy_state, Mapping) or not isinstance(value_state, Mapping):
        raise ValueError(
            "official SONIC v1.1 checkpoint requires policy_state_dict and value_state_dict"
        )
    mapped: dict[str, torch.Tensor] = {}
    for source_state, key_mapper in (
        (policy_state, _mapped_policy_key),
        (value_state, _mapped_value_key),
    ):
        for raw_key, raw_value in source_state.items():
            key = key_mapper(str(raw_key))
            if key in mapped:
                raise ValueError(f"official SONIC v1.1 keys collide at {key!r}")
            if not isinstance(raw_value, torch.Tensor):
                raise ValueError(f"official SONIC v1.1 state {raw_key!r} is not a tensor")
            mapped[key] = raw_value
    return mapped


def _trainer_iteration(checkpoint: Mapping[str, Any]) -> int:
    state = checkpoint.get("state")
    value = (
        state.get("global_step", 0)
        if isinstance(state, Mapping)
        else getattr(state, "global_step", 0)
    )
    iteration = int(value)
    if iteration < 0:
        raise ValueError(f"official SONIC v1.1 global_step must be non-negative, got {iteration}")
    return iteration


def _optimizer_update_count(optimizer_state: Mapping[str, Any], iteration: int) -> int:
    raw_state = optimizer_state.get("state")
    if isinstance(raw_state, Mapping):
        for parameter_state in raw_state.values():
            if isinstance(parameter_state, Mapping) and "step" in parameter_state:
                return int(torch.as_tensor(parameter_state["step"]).item())
    return iteration * 20


def _validate_optimizer_state(
    optimizer: torch.optim.Optimizer,
    optimizer_state: Mapping[str, Any],
) -> None:
    groups = optimizer_state.get("param_groups")
    states = optimizer_state.get("state")
    if not isinstance(groups, list) or not isinstance(states, Mapping):
        raise ValueError("official SONIC v1.1 optimizer_state_dict is malformed")
    if len(groups) != len(optimizer.param_groups):
        raise ValueError(
            "official SONIC v1.1 optimizer group count does not match the UniLab owner"
        )
    for group_index, (source_group, target_group) in enumerate(
        zip(groups, optimizer.param_groups, strict=True)
    ):
        if not isinstance(source_group, Mapping) or not isinstance(
            source_group.get("params"), list
        ):
            raise ValueError(f"official SONIC v1.1 optimizer group {group_index} is malformed")
        source_parameters = source_group["params"]
        target_parameters = target_group["params"]
        if len(source_parameters) != len(target_parameters):
            raise ValueError(
                f"official SONIC v1.1 optimizer group {group_index} has "
                f"{len(source_parameters)} parameters; expected {len(target_parameters)}"
            )
        for source_id, target_parameter in zip(source_parameters, target_parameters, strict=True):
            parameter_state = states.get(source_id)
            if not isinstance(parameter_state, Mapping):
                raise ValueError(f"official SONIC v1.1 optimizer has no state for {source_id}")
            for moment_name in ("exp_avg", "exp_avg_sq"):
                moment = parameter_state.get(moment_name)
                if not isinstance(moment, torch.Tensor) or moment.shape != target_parameter.shape:
                    actual_shape = (
                        None if not isinstance(moment, torch.Tensor) else tuple(moment.shape)
                    )
                    raise ValueError(
                        f"official SONIC v1.1 optimizer {moment_name} shape {actual_shape} "
                        f"does not match parameter shape {tuple(target_parameter.shape)}"
                    )


def convert_official_sonic_v11_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    source_sha256: str | None = None,
    horizon: int = 24,
    include_optimizer: bool = True,
) -> dict[str, Any]:
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")
    model = SonicActorCritic(model_profile="sonic_v1_1", critic_obs_normalization=True)
    mapped_model_state = map_official_sonic_v11_model_state(checkpoint)
    model.load_state_dict(mapped_model_state, strict=True)
    token_info = model.tokenizer.get_token_info()
    iteration = _trainer_iteration(checkpoint)

    converted: dict[str, Any] = {
        "model": mapped_model_state,
        "algorithm": {
            "update_count": iteration * 20,
            "last_optimizer_steps": 20 if iteration else 0,
        },
        "iteration": iteration,
        "token_info": token_info,
        "contract": {
            "model_contract_version": model.model_contract_version,
            "model_profile": model.model_profile,
            "actor_obs_dim": model.actor_obs_dim,
            "critic_obs_dim": model.critic_obs_dim,
            "tokenizer_obs_dim": model.tokenizer_obs_dim,
            "action_dim": model.action_dim,
            "horizon": int(horizon),
            "token_info": token_info,
        },
        "conversion": {
            "source_format": OFFICIAL_SONIC_V11_FORMAT,
            "source_sha256": source_sha256,
            "source_iteration": iteration,
        },
    }
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if include_optimizer:
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("official SONIC v1.1 checkpoint has no optimizer_state_dict")
        algorithm = SonicPPO(model, {"learning_rate": 2.0e-5})
        _validate_optimizer_state(algorithm.optimizer, optimizer_state)
        algorithm.optimizer.load_state_dict(dict(optimizer_state))
        converted["optimizer"] = algorithm.optimizer.state_dict()
        converted["algorithm"]["update_count"] = _optimizer_update_count(optimizer_state, iteration)
    return converted


def convert_official_sonic_v11_checkpoint_file(
    source: str | Path,
    output: str | Path,
    *,
    horizon: int = 24,
    include_optimizer: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not source_path.is_file():
        raise ValueError(f"official SONIC v1.1 checkpoint is not a file: {source_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"converted checkpoint already exists: {output_path}")
    source_sha256 = _checkpoint_sha256(source_path)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("official SONIC v1.1 checkpoint must contain a mapping")
    converted = convert_official_sonic_v11_checkpoint(
        checkpoint,
        source_sha256=source_sha256,
        horizon=horizon,
        include_optimizer=include_optimizer,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        torch.save(converted, temporary)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "source": str(source_path),
        "output": str(output_path),
        "source_sha256": source_sha256,
        "iteration": int(converted["iteration"]),
        "include_optimizer": include_optimizer,
        "model_tensors": len(converted["model"]),
    }


__all__ = [
    "OFFICIAL_SONIC_V11_FORMAT",
    "convert_official_sonic_v11_checkpoint",
    "convert_official_sonic_v11_checkpoint_file",
    "map_official_sonic_v11_model_state",
]
