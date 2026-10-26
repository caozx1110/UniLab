"""Durable snapshots, corruption rejection and cross-process RNG continuity."""

import multiprocessing
import random

import numpy as np
import pytest
import torch

from unilab.rl.runtime.offpolicy import continuation


def draws():
    return random.random(), np.random.random(4).tolist(), torch.rand(4).tolist()


def restore_in_new_process(binding, response):
    # Simulate construction and cold preparation consuming all default RNGs.
    draws()
    saved = continuation.load_state(binding)
    continuation.restore_rng_state(saved["rng"])
    response.put((draws(), saved["tensor"].tolist()))


def test_durable_rng_replays_the_next_draw_in_a_new_process(tmp_path):
    random.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    binding = continuation.save_state(
        tmp_path / "runtime.pt",
        {
            "rng": continuation.capture_rng_state(),
            "tensor": torch.arange(6).reshape(2, 3),
        },
    )
    assert binding == continuation.file_binding(binding["path"])
    expected = draws()
    context = multiprocessing.get_context("spawn")
    response = context.Queue()
    process = context.Process(target=restore_in_new_process, args=(binding, response))
    process.start()
    actual, tensor = response.get(timeout=30)
    process.join(timeout=30)
    assert process.exitcode == 0
    assert actual == expected
    assert tensor == [[0, 1, 2], [3, 4, 5]]


def test_corrupt_or_incomplete_snapshot_never_loads(tmp_path):
    path = tmp_path / "runtime.pt"
    binding = continuation.save_state(path, {"value": torch.ones(2)})
    with path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="identity differs"):
        continuation.load_state(binding)
    checkpoint = tmp_path / "model_50.pt"
    checkpoint.write_bytes(b"model")
    with pytest.raises(FileNotFoundError):
        continuation.read_manifest(checkpoint)


def test_manifest_binds_every_component_and_native_checkpoint(tmp_path):
    checkpoint = tmp_path / "model_50.pt"
    checkpoint.write_bytes(b"native")
    components = {
        name: continuation.save_state(tmp_path / f"{name}.pt", {"value": name})
        for name in ("collector", "replay", "runtime")
    }
    manifest = {
        "schema": "unilab.offpolicy-continuation.v1",
        "iteration": 50,
        "checkpoint": continuation.file_binding(checkpoint),
        **components,
    }
    continuation.commit_manifest(checkpoint.with_suffix(".continuation.json"), manifest)
    assert continuation.read_manifest(checkpoint) == manifest
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity differs"):
        continuation.read_manifest(checkpoint)
