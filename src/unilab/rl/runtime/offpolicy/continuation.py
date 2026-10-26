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
from pathlib import Path
from typing import Any

import numpy as np
import torch


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


def file_binding(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def verify_binding(binding: dict[str, Any]) -> Path:
    actual = file_binding(binding["path"])
    if actual["sha256"] != binding["sha256"] or (
        "bytes" in binding and actual["bytes"] != binding["bytes"]
    ):
        raise ValueError(f"Continuation file identity differs: {binding['path']}")
    return Path(actual["path"])


class HashingWriter:
    """Hash the sequential torch zip stream without rereading large replay files."""

    def __init__(self, stream: Any):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, data: bytes) -> int:
        count = self.stream.write(data)
        if count != len(data):
            raise OSError("Short continuation write")
        self.digest.update(data)
        self.size += count
        return count

    def flush(self) -> None:
        self.stream.flush()


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_state(path: str | Path, state: dict[str, Any]) -> dict[str, Any]:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            writer = HashingWriter(stream)
            torch.save(state, writer)
            writer.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
        return {"path": str(path), "sha256": writer.digest.hexdigest(), "bytes": writer.size}
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
