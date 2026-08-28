from __future__ import annotations

import pytest
import torch

from unilab.algos.torch.sonic_ppo.model import SonicActorCritic


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
