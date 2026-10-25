"""Owner-injected hooks at off-policy iteration and checkpoint boundaries.

The runtime owns quiescence; hooks own task schedules and optional external work.
Checkpoint hooks run on every rank with the same canonical checkpoint path. A
long-running hook must coordinate completion on the host before returning on
any rank, and actively publish progress. No background heartbeat hides a stuck
hook. Hooks must not reset the collector, replay, or training RNG.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class LifecycleContext:
    phase: Literal["iteration_start", "checkpoint"]
    learner: Any
    iteration: int
    rank: int
    world_size: int
    log_dir: str
    checkpoint_path: str | None = None
    checkpoint_token: str | None = None
    progress_publisher: Callable[[Mapping[str, Any]], None] = field(
        repr=False, compare=False, default=lambda _: None
    )
    stop_request: Callable[[str], None] | None = field(repr=False, compare=False, default=None)
    retained_state: Mapping[str, Any] | None = None

    def publish_progress(self, details: Mapping[str, Any] | None = None) -> None:
        """Publish caller-observed progress; call periodically during long work."""
        self.progress_publisher({} if details is None else details)

    def request_stop(self, reason: str) -> None:
        """Request normal cleanup at this saved checkpoint on every rank."""
        if self.phase != "checkpoint" or self.stop_request is None:
            raise RuntimeError("Stopping is only supported inside a checkpoint hook")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 4096:
            raise ValueError("A stop request needs a nonempty reason of at most 4096 characters")
        self.stop_request(reason)


LifecycleHook = Callable[[LifecycleContext], None]


def validate_checkpoint_iterations(values: Sequence[int], max_iterations: int) -> frozenset[int]:
    if any(type(value) is not int or not 1 <= value <= max_iterations for value in values):
        raise ValueError("checkpoint_iterations must contain positive integers <= max_iterations")
    if len(set(values)) != len(values):
        raise ValueError("checkpoint_iterations must not contain duplicates")
    return frozenset(values)


def validate_hook(hook: LifecycleHook | None) -> None:
    if hook is None:
        return
    if not callable(hook):
        raise TypeError("Lifecycle factory must return a callable")
    save = getattr(hook, "state_dict", None)
    load = getattr(hook, "load_state_dict", None)
    if (save is not None or load is not None) and not (callable(save) and callable(load)):
        raise TypeError("Stateful lifecycle hooks require state_dict and load_state_dict")


def checkpoint_hook_state(hook: LifecycleHook | None, iteration: int) -> dict[str, Any] | None:
    if hook is None:
        return None
    save = getattr(hook, "state_dict", None)
    state = save() if callable(save) else None
    if callable(save) and not isinstance(state, Mapping):
        raise TypeError("Lifecycle state_dict must return a mapping")
    return {"schema": "unilab.offpolicy-lifecycle-state.v1", "iteration": iteration, "hook": state}


def restore_hook_state(hook: LifecycleHook | None, state: Any, iteration: int) -> None:
    if hook is None and state is None:
        return
    if hook is None or not isinstance(state, Mapping):
        raise ValueError("Resume requires the original lifecycle hook and checkpoint state")
    if (
        state.get("schema") != "unilab.offpolicy-lifecycle-state.v1"
        or type(state.get("iteration")) is not int
        or state.get("iteration") != iteration
    ):
        raise ValueError("Lifecycle checkpoint schema/iteration mismatch")
    load = getattr(hook, "load_state_dict", None)
    if callable(load):
        if not isinstance(state.get("hook"), Mapping):
            raise ValueError("Stateful lifecycle checkpoint is missing hook state")
        load(state["hook"], iteration=iteration)
    elif state.get("hook") is not None:
        raise ValueError("Checkpoint has hook state but hook cannot restore it")


def publish_lifecycle_state(
    context: LifecycleContext, phase: str, details: Mapping[str, Any] | None = None
) -> None:
    """Atomically replace one rank's status without touching training counters."""
    target = Path(context.log_dir) / f"lifecycle.rank{context.rank}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f".json.{os.getpid()}.tmp")
    record = {
        "schema": "unilab.offpolicy-lifecycle.v1",
        "phase": phase,
        "iteration": context.iteration,
        "rank": context.rank,
        "world_size": context.world_size,
        "pid": os.getpid(),
        "checkpoint_path": context.checkpoint_path,
        "checkpoint_token": context.checkpoint_token,
        "retained_state": context.retained_state,
        "updated_at": time.time(),
        "details": dict(details or {}),
    }
    temporary.write_text(json.dumps(record, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)


def publish_retention_check(context: LifecycleContext, after: Mapping[str, Any]) -> bool:
    """Persist one immutable-boundary audit before allowing training to resume."""
    verified = context.retained_state == after
    target = Path(context.log_dir) / (
        f"lifecycle.retention.iter{context.iteration}.rank{context.rank}.json"
    )
    temporary = target.with_suffix(f".json.{os.getpid()}.tmp")
    record = {
        "schema": "unilab.offpolicy-retention.v1",
        "iteration": context.iteration,
        "rank": context.rank,
        "pid": os.getpid(),
        "checkpoint_path": context.checkpoint_path,
        "checkpoint_token": context.checkpoint_token,
        "updated_at": time.time(),
        "before": context.retained_state,
        "after": dict(after),
        "verified": verified,
    }
    temporary.write_text(json.dumps(record, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    return verified
