"""Durable snapshots, corruption rejection and cross-process RNG continuity."""

import multiprocessing
import queue
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unilab.rl.runtime.offpolicy import continuation
from unilab.rl.runtime.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner


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


def test_full_snapshot_observes_and_clones_live_native_learner_not_copied_checkpoint(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "model_64.pt"
    torch.save({"actor": {"weight": torch.tensor([1.0])}, "update_count": 64}, checkpoint)
    original_binding = continuation.file_binding(checkpoint)
    live_weight = torch.nn.Parameter(torch.tensor([2.0]))
    live_moment = torch.tensor([3.0])
    live_native = {
        "actor": {"weight": live_weight},
        "actor_optimizer": {"state": {0: {"exp_avg": live_moment}}, "param_groups": [{"lr": 1e-4}]},
        "update_count": 64,
    }
    observed = []
    original_save = continuation.save_state

    def save_state(path, state, **kwargs):
        if Path(path).name == "runtime.pt":
            actual = state["learner_native"]
            weight = actual["actor"]["weight"]
            moment = actual["actor_optimizer"]["state"][0]["exp_avg"]
            assert weight.device.type == moment.device.type == "cpu"
            assert not weight.requires_grad
            assert weight.data_ptr() != live_weight.data_ptr()
            assert moment.data_ptr() != live_moment.data_ptr()
            assert (
                actual["actor_optimizer"]["param_groups"]
                is not live_native["actor_optimizer"]["param_groups"]
            )
            observed.append(weight.item())
        return original_save(path, state, **kwargs)

    response = queue.Queue()

    class Requests:
        def put(self, request):
            binding = original_save(request["path"], {"schema": "collector-fixture"})
            response.put({"request_id": request["request_id"], "tick": request["tick"], **binding})

    runner = object.__new__(DoubleBufferOffPolicyRunner)
    runner.device = "cpu"
    runner.learner = SimpleNamespace(
        get_state_dict=lambda: live_native,
        export_continuation_state=lambda: {"schema": "nonpersistent-fixture"},
    )
    runner._continuation_identity = lambda: {"probe": True}
    monkeypatch.setattr(continuation, "save_state", save_state)
    runner._save_continuation(
        iteration=64,
        held_tick=162,
        log_dir=str(tmp_path),
        replay_pipeline=SimpleNamespace(
            export_continuation_state=lambda: {"prepare_tick_id": None}
        ),
        request_queue=Requests(),
        response_queue=response,
        runner_state=lambda: {"prepared_tick": None},
        drain_metrics=lambda tick: None,
    )
    manifest = continuation.read_manifest(checkpoint)
    actual = continuation.load_state(manifest["runtime"])["learner_native"]
    assert observed == [2.0]
    assert actual["actor"]["weight"].item() == 2.0
    assert actual["actor_optimizer"]["state"][0]["exp_avg"].item() == 3.0
    assert continuation.file_binding(checkpoint) == original_binding
