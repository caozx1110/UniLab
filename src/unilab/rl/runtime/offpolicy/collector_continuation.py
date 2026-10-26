"""Collector checkpoint messages serviced only at the held inference boundary."""

from __future__ import annotations

import queue
from pathlib import Path

import torch


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if value is None or isinstance(value, (bool, int, float, str, bytes, torch.device)):
        return value
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    raise TypeError(f"Unsupported collector state value {type(value).__name__}")


def restore_metric_state(metrics, saved):
    metrics.restore_continuation_values(saved)


def service_checkpoint_request(request_queue, response_queue, *, tick, snapshot):
    """The caller is parked after publishing obs[t], before consuming action[t]."""
    if request_queue is None:
        return
    if response_queue is None:
        raise ValueError("Collector checkpoint request queue requires a response queue")
    try:
        request = request_queue.get_nowait()
    except queue.Empty:
        return
    request_id = request.get("request_id") if isinstance(request, dict) else None
    try:
        if (
            not isinstance(request, dict)
            or request.get("operation") != "snapshot"
            or type(request.get("tick")) is not int
            or request["tick"] != tick
            or not isinstance(request_id, str)
            or not request_id
            or not isinstance(request.get("path"), str)
            or not Path(request["path"]).is_absolute()
        ):
            raise ValueError("Invalid collector checkpoint request or held tick")
        from unilab.rl.runtime.offpolicy.continuation import save_state

        binding = save_state(request["path"], snapshot())
        response_queue.put({"request_id": request_id, "tick": tick, **binding}, timeout=30)
    except BaseException as exc:
        response_queue.put(
            {"request_id": request_id, "tick": tick, "error": f"{type(exc).__name__}: {exc}"},
            timeout=30,
        )
        raise
