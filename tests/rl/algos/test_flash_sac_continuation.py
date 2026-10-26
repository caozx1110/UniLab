"""Exploration must continue across a new learner, including repeat history."""

from __future__ import annotations

import pytest
import torch

from unilab.rl.algos.flash_sac.learner import FlashSACLearner
from unilab.rl.algos.flash_sac.network import FlashSACActor


def _learner() -> FlashSACLearner:
    return FlashSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=6,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        actor_num_blocks=1,
        critic_num_blocks=1,
        num_atoms=5,
        device="cpu",
        use_compile=False,
    )


def test_learner_nonpersistent_state_and_exploration_roundtrip(tmp_path):
    torch.manual_seed(7)
    original = _learner()
    obs = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12
    original.actor.explore(obs)
    original.actor.explore(obs)
    original._pending_actor_metric_values = torch.tensor([1.0, 2.0])
    original._aux_loss_names = ("saved_aux",)
    original.target_entropy = -4.125
    original._lr_peak = 7e-5
    native = original.get_state_dict()
    assert "_noise" not in native["actor"]  # Evaluator ABI remains unchanged.
    path = tmp_path / "learner.pt"
    torch.save(
        {
            "native": native,
            "continuation": original.export_continuation_state(),
            "rng": torch.random.get_rng_state(),
        },
        path,
    )
    done = torch.tensor([0.0, 1.0, 0.0])
    expected = [original.actor.explore(obs, dones=done).clone() for _ in range(12)]
    expected_rng = torch.random.get_rng_state()

    restored = _learner()
    # Simulate actor warmup consuming noise/repeat state before the final restore.
    restored.actor.explore(torch.zeros_like(obs))
    state = torch.load(path, weights_only=False)
    restored.load_state_dict(state["native"])
    restored.import_continuation_state(state["continuation"])
    torch.random.set_rng_state(state["rng"])
    assert restored._aux_loss_names == ("saved_aux",)
    assert restored.target_entropy == -4.125
    assert restored._lr_peak == 7e-5
    assert torch.equal(restored._pending_actor_metric_values, torch.tensor([1.0, 2.0]))
    actual = [restored.actor.explore(obs, dones=done).clone() for _ in range(12)]
    assert all(torch.equal(left, right) for left, right in zip(expected, actual, strict=True))
    assert torch.equal(torch.random.get_rng_state(), expected_rng)


def test_exploration_restore_rejects_changed_action_shape():
    actor = FlashSACActor(1, 4, 8, 2)
    actor.explore(torch.zeros(3, 4))
    state = actor.export_continuation_state()
    state["noise"] = torch.zeros(3, 3)
    with pytest.raises(ValueError, match="shape"):
        actor.import_continuation_state(state)


def test_empty_exploration_roundtrip():
    actor = FlashSACActor(1, 4, 8, 2)
    state = actor.export_continuation_state()
    actor.explore(torch.zeros(3, 4))
    actor.import_continuation_state(state)
    assert actor._noise.numel() == 0
    assert actor._repeat_count.numel() == 0


@pytest.mark.parametrize(
    "graph_kind",
    ["whole_requested", "whole_active", "whole_instance", "dp", "inductor", "inductor_default"],
)
def test_full_continuation_rejects_unvalidated_graph_rng_paths(graph_kind):
    learner = _learner()
    saved = learner.export_continuation_state()
    if graph_kind == "whole_requested":
        learner._whole_cycle_cuda_graph_requested = True
    elif graph_kind == "whole_active":
        learner._compile_full_update_cycle = True
    elif graph_kind == "whole_instance":
        learner._update_cycle_graph = object()
    elif graph_kind == "dp":
        learner._dp_whole_cycle_cuda_graph = True
    else:
        learner.use_compile = True
        if graph_kind == "inductor":
            learner.compile_options["triton.cudagraphs"] = True
    with pytest.raises(ValueError, match="CUDA graphs disabled"):
        learner.export_continuation_state()
    with pytest.raises(ValueError, match="CUDA graphs disabled"):
        learner.import_continuation_state(saved)


def test_full_continuation_rejects_changed_training_configuration():
    learner = _learner()
    saved = learner.export_continuation_state()
    learner.gamma = 0.123
    with pytest.raises(ValueError, match="configuration differs"):
        learner.import_continuation_state(saved)
