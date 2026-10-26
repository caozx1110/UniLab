"""Durable, identity-bound process snapshots for stopped inference boundaries.

Only trusted files produced by this runtime may be loaded: the payload includes
Python/NumPy RNG state and therefore deliberately uses unrestricted torch load.
The manifest is committed last; an interrupted write is never a recovery point.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

_SAVE_LOCK = threading.Lock()
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_WRITE_POLL_SECONDS = 0.5
_HASH_PROGRESS_SECONDS = 1.0

Progress = Callable[[dict[str, Any]], None]


def capture_rng_state(device: str | torch.device = "cpu") -> dict[str, Any]:
    device = torch.device(device)
    return {
        "schema": "unilab.process-rng.v1",
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state().clone(),
        "device_type": device.type,
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        "mps": torch.mps.get_rng_state() if device.type == "mps" else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    if state.get("schema") != "unilab.process-rng.v1":
        raise ValueError("Unsupported process RNG snapshot")
    if state["device_type"] == "cuda" and len(state["cuda"]) != torch.cuda.device_count():
        raise ValueError("Continuation CUDA device count differs")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if state["device_type"] == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])
    elif state["device_type"] == "mps":
        torch.mps.set_rng_state(state["mps"])


def file_binding(path: str | Path, *, progress: Progress | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    size = 0
    reported = 0
    next_report = 0.0
    with path.open("rb") as stream:
        while block := stream.read(_HASH_CHUNK_BYTES):
            digest.update(block)
            size += len(block)
            now = time.monotonic()
            if progress is not None and now >= next_report:
                progress({"stage": "hash", "bytes": size})
                reported = size
                next_report = now + _HASH_PROGRESS_SECONDS
    if progress is not None and size > reported:
        progress({"stage": "hash", "bytes": size})
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def verify_binding(binding: dict[str, Any]) -> Path:
    actual = file_binding(binding["path"])
    if actual["sha256"] != binding["sha256"] or (
        "bytes" in binding and actual["bytes"] != binding["bytes"]
    ):
        raise ValueError(f"Continuation file identity differs: {binding['path']}")
    return Path(actual["path"])


def temporary_path(path: str | Path, *, pid: int | None = None) -> Path:
    path = Path(path).resolve()
    return path.with_name(f".{path.name}.{os.getpid() if pid is None else pid}.tmp")


def _save_native(path: Path, state: dict[str, Any]) -> None:
    # A Python file-like writer materializes an entire storage as Python bytes
    # before write() can run (the replay storage can exceed 50 GB). Native paths
    # stream directly. ZIP CRC32 also scans a whole storage before its first
    # write; the manifest's SHA256 provides stronger, chunk-observable integrity.
    # These snapshots are a paused cold path. Restore the process-local option,
    # even on failure, and serialize our own concurrent callers around it.
    with _SAVE_LOCK:
        crc32 = torch.serialization.get_crc32_options()
        try:
            torch.serialization.set_crc32_options(False)
            torch.save(state, str(path))
        finally:
            torch.serialization.set_crc32_options(crc32)


def _save_with_progress(path: Path, state: dict[str, Any], progress: Progress) -> None:
    done = threading.Event()
    errors: list[BaseException] = []

    def write() -> None:
        try:
            _save_native(path, state)
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    writer = threading.Thread(target=write, name="continuation-writer", daemon=True)
    writer.start()
    reported = 0

    def report_growth() -> None:
        nonlocal reported
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return
        if size > reported:
            progress({"stage": "write", "bytes": size})
            reported = size

    try:
        while not done.wait(_WRITE_POLL_SECONDS):
            # Observed bytes, not thread liveness, extend the health deadline.
            # A blocked write/CRC/pickle/fsync has no automatic heartbeat.
            report_growth()
        report_growth()
    finally:
        writer.join()
    if errors:
        raise errors[0]


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_state(
    path: str | Path, state: dict[str, Any], *, progress: Progress | None = None
) -> dict[str, Any]:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    try:
        if progress is None:
            _save_native(temporary, state)
        else:
            _save_with_progress(temporary, state, progress)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        binding = file_binding(temporary, progress=progress)
        os.replace(temporary, path)
        sync_directory(path.parent)
        return {**binding, "path": str(path)}
    finally:
        temporary.unlink(missing_ok=True)


def load_state(binding: dict[str, Any]) -> dict[str, Any]:
    path = verify_binding(binding)
    result = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(result, dict):
        raise TypeError("Continuation payload must be a dictionary")
    return result


def commit_manifest(path: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = Path(path).resolve()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return file_binding(path)


def read_manifest(checkpoint: str | Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    path = checkpoint.with_suffix(".continuation.json")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "unilab.offpolicy-continuation.v1":
        raise ValueError("Unsupported off-policy continuation manifest")
    if Path(manifest["checkpoint"]["path"]).resolve() != checkpoint:
        raise ValueError("Continuation manifest points at a different checkpoint")
    verify_binding(manifest["checkpoint"])
    # Validate every component before mutating the new process. Large replay
    # data stays mmap-backed when read by its owner after cold preparation.
    for name in ("collector", "replay", "runtime"):
        verify_binding(manifest[name])
    return manifest
