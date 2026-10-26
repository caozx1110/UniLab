"""Cold-path, fail-closed manager continuation state.

Only value trees and explicitly owned child objects cross this boundary. In
particular, opaque backend handles and immutable motion tables are never
silently pickled. Tensor restoration copies into existing carriers so bound
views, compiled functions and manager aliases retain their storage.
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import torch


def class_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def copy_tree(value: Any, *, device=None):
    if isinstance(value, torch.Tensor):
        if value.numel() * value.element_size() > 256 * 1024**2:
            raise ValueError(
                "Continuation carrier exceeds 256 MiB; immutable tables need exclusion"
            )
        return value.detach().to(device=device or "cpu", copy=True)
    if isinstance(value, np.ndarray):
        if value.nbytes > 256 * 1024**2 or value.dtype.hasobject:
            raise ValueError("Continuation NumPy carrier is oversized or contains objects")
        return value.copy()
    if value is None or isinstance(value, (bool, int, float, str, bytes, slice, torch.device)):
        return value
    if isinstance(value, np.generic):
        return value.copy()
    if isinstance(value, dict):
        return {key: copy_tree(item, device=device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(copy_tree(item, device=device) for item in value)
    raise TypeError(f"Unsupported opaque continuation value: {class_name(value)}")


def restore_tree(target, saved, *, device):
    if isinstance(saved, torch.Tensor):
        if isinstance(target, torch.Tensor):
            if target.shape != saved.shape or target.dtype != saved.dtype:
                raise ValueError("Continuation tensor shape/dtype mismatch")
            target.copy_(saved.to(device=target.device))
            return target
        if target is not None:
            raise TypeError("Continuation tensor target changed type")
        return saved.to(device=device, copy=True)
    if isinstance(saved, np.ndarray):
        if isinstance(target, np.ndarray):
            if target.shape != saved.shape or target.dtype != saved.dtype:
                raise ValueError("Continuation NumPy shape/dtype mismatch")
            np.copyto(target, saved)
            return target
        if target is not None:
            raise TypeError("Continuation NumPy target changed type")
        return saved.copy()
    if isinstance(saved, dict):
        if target is not None and not isinstance(target, dict):
            raise TypeError("Continuation mapping target changed type")
        target = {} if target is None else target
        for key in tuple(target):
            if key not in saved:
                del target[key]
        for key, item in saved.items():
            target[key] = restore_tree(target.get(key), item, device=device)
        return target
    if isinstance(saved, (list, tuple)):
        if target is not None and not isinstance(target, type(saved)):
            raise TypeError("Continuation sequence target changed type")
        values = [
            restore_tree(
                target[i] if target is not None and i < len(target) else None, item, device=device
            )
            for i, item in enumerate(saved)
        ]
        if isinstance(saved, list) and target is not None:
            target[:] = values
            return target
        return type(saved)(values)
    return copy_tree(saved, device=device)


def capture_attributes(owner, *, exclude=(), children=()):
    excluded = set(exclude) | set(children)
    values = {key: copy_tree(value) for key, value in vars(owner).items() if key not in excluded}
    return {
        "class": class_name(owner),
        "values": values,
        "children": {key: capture_attributes(getattr(owner, key)) for key in children},
    }


def restore_attributes(owner, state, *, device, exclude=(), children=()):
    if set(state) != {"class", "values", "children"} or state["class"] != class_name(owner):
        raise ValueError("Continuation owner schema/class mismatch")
    excluded = set(exclude) | set(children)
    current = set(vars(owner)) - excluded
    if current != set(state["values"]) or set(children) != set(state["children"]):
        raise ValueError(f"Continuation attribute set changed for {class_name(owner)}")
    for key, value in state["values"].items():
        setattr(owner, key, restore_tree(getattr(owner, key), value, device=device))
    for key in children:
        restore_attributes(getattr(owner, key), state["children"][key], device=device)


def export_term(term):
    if isinstance(term, (types.FunctionType, types.BuiltinFunctionType)):
        if vars(term):
            raise TypeError("Stateful function term needs an explicit continuation API")
        return {"function": f"{term.__module__}.{term.__qualname__}"}
    method = getattr(term, "export_continuation_state", None)
    if not callable(method):
        raise TypeError(f"Term {class_name(term)} has no continuation API")
    return {"class": class_name(term), "state": method()}


def import_term(term, state):
    if "function" in state:
        if export_term(term) != state:
            raise ValueError("Continuation function term changed")
        return
    if state.get("class") != class_name(term):
        raise ValueError("Continuation manager term class changed")
    method = getattr(term, "import_continuation_state", None)
    if not callable(method):
        raise TypeError(f"Term {class_name(term)} has no continuation restore API")
    method(state["state"])


def _class_terms(manager):
    if hasattr(manager, "_terms"):
        return manager._terms
    if hasattr(manager, "_group_obs_term_cfgs"):
        return {
            f"{group}/{name}": cfg.func
            for group, configs in manager._group_obs_term_cfgs.items()
            for name, cfg in zip(manager._group_obs_term_names[group], configs, strict=True)
        }
    if hasattr(manager, "_mode_term_cfgs"):
        return {
            f"{mode}/{name}": cfg.func
            for mode, configs in manager._mode_term_cfgs.items()
            for name, cfg in zip(manager._mode_term_names[mode], configs, strict=True)
        }
    return {
        name: cfg.func for name, cfg in zip(manager._term_names, manager._term_cfgs, strict=True)
    }


_MANAGER_FIELDS = {
    "ActionManager": ("_action", "_prev_action", "_prev_prev_action", "_action_finite_valid"),
    "CommandManager": (),
    "NullCommandManager": (),
    "ObservationManager": (
        "_obs_buffer",
        "_group_term_noise_carriers",
        "_group_term_noise_generation",
        "_group_finite_valid",
    ),
    "EventManager": (
        "_interval_term_time_left",
        "_reset_term_last_triggered_step_id",
        "_reset_term_last_triggered_once",
    ),
    "TerminationManager": ("_term_dones", "_truncated_buf", "_terminated_buf", "_dones_buf"),
    "RewardManager": ("_reward_buf", "_step_reward", "_finite_valid"),
}


def manager_state(manager):
    name = type(manager).__name__
    if name not in _MANAGER_FIELDS:
        raise TypeError(f"Unsupported continuation manager {class_name(manager)}")
    result = {
        "class": class_name(manager),
        "values": {key: copy_tree(getattr(manager, key)) for key in _MANAGER_FIELDS[name]},
        "terms": {key: export_term(term) for key, term in _class_terms(manager).items()},
    }
    if name == "ObservationManager":
        if any(manager._group_obs_class_instances.values()):
            raise NotImplementedError("Noise-model continuation requires an owner implementation")
        if any(manager._group_obs_term_delay_buffer.values()):
            raise NotImplementedError("Delay-buffer continuation requires an owner implementation")
        result["histories"] = {
            group: {key: capture_attributes(buffer) for key, buffer in items.items()}
            for group, items in manager._group_obs_term_history_buffer.items()
        }
    return result


def restore_manager(manager, state, *, device):
    name = type(manager).__name__
    if state.get("class") != class_name(manager) or name not in _MANAGER_FIELDS:
        raise ValueError("Continuation manager class mismatch")
    if set(state["values"]) != set(_MANAGER_FIELDS[name]):
        raise ValueError("Continuation manager fields mismatch")
    terms = _class_terms(manager)
    if set(terms) != set(state["terms"]):
        raise ValueError("Continuation manager terms mismatch")
    for key, value in state["values"].items():
        setattr(manager, key, restore_tree(getattr(manager, key), value, device=device))
    for key, term in terms.items():
        import_term(term, state["terms"][key])
    if name == "ObservationManager":
        histories = manager._group_obs_term_history_buffer
        if set(histories) != set(state["histories"]):
            raise ValueError("Continuation history groups changed")
        for group, items in histories.items():
            if set(items) != set(state["histories"][group]):
                raise ValueError("Continuation history terms changed")
            for key, buffer in items.items():
                restore_attributes(buffer, state["histories"][group][key], device=device)
