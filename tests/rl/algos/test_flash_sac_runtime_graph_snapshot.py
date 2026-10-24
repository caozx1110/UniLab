"""CPU evidence tests; CUDA/NCCL correctness requires the separate real run."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from unilab.rl.algos.flash_sac.learner import FlashSACLearner
from unilab.rl.logging.runtime_manifest_schema import validate_runtime_manifest
from unilab.rl.runtime.distributed.sync import DpParameterSync
from unilab.rl.runtime.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner


def _runner(learner, rank=0):
    runner = object.__new__(DoubleBufferOffPolicyRunner)
    runner.learner = learner
    runner.dp_sync = DpParameterSync(
        world_size=2, rank=rank, rendezvous_path="unused-cpu-test", backend="gloo"
    )
    runner.runtime_manifest = {"schema_version": 2}
    return runner


def test_snapshot_distinguishes_requested_path_from_capture_failure(tmp_path):
    learner = SimpleNamespace(
        use_compile=True,
        use_update_cycle=True,
        _update_cycle_graph=None,
        _update_cycle_capture_failed=True,
        _update_cycle_replay_submitted=False,
    )
    runner = _runner(learner)
    graph = runner._cuda_graph_runtime_manifest()
    assert graph["scope"] == "update_cycle"
    assert not graph["whole_cycle_graph_present"]
    assert graph["whole_cycle_capture_failed"]
    assert not graph["whole_cycle_replay_submitted"]
    runner.runtime_manifest["cuda_graph"] = graph
    runner._persist_runtime_manifest_snapshot(str(tmp_path), phase="learner_prepared")
    data = json.loads((tmp_path / "runtime_manifest.rank0.json").read_text())
    assert data["runtime_manifest"]["cuda_graph"] == graph
    validate_runtime_manifest(data["runtime_manifest"])


def test_rank_snapshots_replace_atomically_without_clobbering_other_rank(tmp_path):
    for rank in (0, 1):
        runner = _runner(SimpleNamespace(), rank)
        runner._persist_runtime_manifest_snapshot(str(tmp_path), phase="learner_prepared")
        runner.runtime_manifest["cuda_graph"] = {"whole_cycle_replay_submitted": True}
        runner._persist_runtime_manifest_snapshot(
            str(tmp_path), phase="first_whole_cycle_replay_submitted", iteration=99
        )
    files = sorted(tmp_path.iterdir())
    assert [path.name for path in files] == [
        "runtime_manifest.rank0.json",
        "runtime_manifest.rank1.json",
    ]
    for rank, path in enumerate(files):
        data = json.loads(path.read_text())
        assert data["rank"] == rank
        assert data["iteration"] == 99
        assert data["runtime_manifest"]["cuda_graph"]["whole_cycle_replay_submitted"]
        validate_runtime_manifest(data["runtime_manifest"])


def test_uninstrumented_learner_does_not_claim_replay_state():
    runner = _runner(SimpleNamespace(_update_cycle_graph=object()))
    graph = runner._cuda_graph_runtime_manifest()
    assert graph["whole_cycle_graph_present"]
    assert "whole_cycle_replay_submitted" not in graph


@pytest.mark.parametrize("fail", [False, True])
def test_replay_evidence_is_set_only_after_replay_returns(fail):
    graph = SimpleNamespace(
        replay=Mock(side_effect=RuntimeError("replay failed") if fail else None)
    )
    learner = SimpleNamespace(
        use_update_cycle=True,
        _update_cycle_capture_failed=False,
        _update_cycle_replay_submitted=False,
        _ensure_update_cycle_graph=Mock(),
        _update_cycle_graph=graph,
        _fill_update_cycle_lr_specs=Mock(),
        _gradient_graph_hooks=None,
        _advance_update_cycle_schedulers=Mock(),
        _update_cycle_graph_metric_values=None,
    )
    kwargs = dict(
        updates_per_step=1,
        policy_frequency=1,
        target_frequency=1,
        policy_before_critic=True,
    )
    if fail:
        with pytest.raises(RuntimeError, match="replay failed"):
            FlashSACLearner.update_cycle(learner, {}, **kwargs)
    else:
        FlashSACLearner.update_cycle(learner, {}, **kwargs)
    assert learner._update_cycle_replay_submitted is (not fail)
    FlashSACLearner._reset_update_cycle_graph(learner)
    assert not learner._update_cycle_replay_submitted
