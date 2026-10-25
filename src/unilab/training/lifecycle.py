"""Resolve task-owned off-policy lifecycle hooks outside the RL runtime."""

from __future__ import annotations

import importlib
from collections.abc import Mapping

from omegaconf import DictConfig, OmegaConf

from unilab.rl.runtime.offpolicy.lifecycle import LifecycleHook, validate_hook


def resolve_offpolicy_lifecycle(
    training: DictConfig,
) -> tuple[LifecycleHook | None, tuple[int, ...]]:
    """Read training.lifecycle: {factory: 'module:function', kwargs, checkpoint_iterations}."""
    configured = training.get("lifecycle")
    if configured is None:
        return None, ()
    data = (
        OmegaConf.to_container(configured, resolve=True)
        if OmegaConf.is_config(configured)
        else configured
    )
    if not isinstance(data, Mapping) or set(data) - {"factory", "kwargs", "checkpoint_iterations"}:
        raise ValueError("Unknown or invalid training.lifecycle configuration")
    iterations = data.get("checkpoint_iterations", [])
    if not isinstance(iterations, (tuple, list)) or any(
        type(x) is not int or x <= 0 for x in iterations
    ):
        raise ValueError(
            "training.lifecycle.checkpoint_iterations must be a list of positive integers"
        )
    if len(set(iterations)) != len(iterations):
        raise ValueError("training.lifecycle.checkpoint_iterations must be unique")
    kwargs = data.get("kwargs", {})
    if not isinstance(kwargs, Mapping):
        raise ValueError("training.lifecycle.kwargs must be a mapping")
    factory = data.get("factory")
    if factory is None:
        if kwargs:
            raise ValueError("Lifecycle kwargs require a factory")
        return None, tuple(iterations)
    if not isinstance(factory, str) or factory.count(":") != 1:
        raise ValueError("Lifecycle factory must use 'module:function'")
    module, name = factory.split(":")
    builder = getattr(importlib.import_module(module), name)
    hook = builder(**dict(kwargs))
    validate_hook(hook)
    if hook is None:
        raise TypeError("Lifecycle factory must return a callable, not None")
    return hook, tuple(iterations)
