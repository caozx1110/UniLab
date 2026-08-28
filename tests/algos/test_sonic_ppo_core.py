from __future__ import annotations

import pytest
import torch

from unilab.algos.torch.sonic_ppo.algorithm import SonicPPO
from unilab.algos.torch.sonic_ppo.checkpoint import map_official_sonic_release_model_state
from unilab.algos.torch.sonic_ppo.model import SonicActorCritic
from unilab.algos.torch.sonic_ppo.storage import SonicRolloutStorage


def test_value_is_critic_only_and_matches_distribution_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SonicActorCritic(
        actor_obs_dim=4,
        critic_obs_dim=5,
        tokenizer_obs_dim=6,
        action_dim=2,
        hidden_dims=(8,),
        tokenizer_hidden_dim=8,
    )
    actor = torch.randn(3, 4)
    critic = torch.randn(3, 5)
    tokens = torch.randn(3, 6)
    model.eval()
    _, distribution_value = model.distribution(actor, critic, tokens)
    monkeypatch.setattr(
        model.tokenizer,
        "forward",
        lambda _tokens: pytest.fail("critic-only value must not call tokenizer"),
    )
    assert torch.allclose(model.value(critic), distribution_value)


def test_storage_compute_returns_and_ppo_update() -> None:
    model = SonicActorCritic(
        actor_obs_dim=4,
        critic_obs_dim=5,
        tokenizer_obs_dim=6,
        action_dim=2,
        hidden_dims=(8,),
        tokenizer_hidden_dim=8,
    )
    storage = SonicRolloutStorage(2, 3, 4, 5, 6, 2)
    for _ in range(2):
        actor = torch.randn(3, 4)
        critic = torch.randn(3, 5)
        tokens = torch.randn(3, 6)
        distribution, values = model.distribution(actor, critic, tokens)
        actions = distribution.sample()
        storage.add(
            actor,
            critic,
            tokens,
            actions,
            torch.randn(3),
            torch.zeros(3),
            values,
            distribution.log_prob(actions).sum(-1),
            distribution.mean,
            distribution.stddev,
        )
    storage.compute_returns(torch.zeros(3))
    metrics = SonicPPO(model, num_learning_epochs=1, num_mini_batches=1).update(storage)
    assert metrics.keys() >= {"loss", "value_loss", "policy_loss"}


def test_checkpoint_mapper_preserves_policy_and_critic_ownership() -> None:
    g1_weight = torch.randn(2, 3)
    critic_bias = torch.randn(1)
    mapped = map_official_sonic_release_model_state(
        {
            "policy_state_dict": {"actor_module.encoders.g1.module.0.weight": g1_weight},
            "value_state_dict": {"critic_module.module.0.bias": critic_bias},
        }
    )
    assert set(mapped) == {"tokenizer.encoders.g1.0.weight", "critic.0.bias"}
    assert mapped["tokenizer.encoders.g1.0.weight"] is g1_weight
    assert mapped["critic.0.bias"] is critic_bias
