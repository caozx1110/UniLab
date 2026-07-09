# SPDX-License-Identifier: BSD-3-Clause
#
# CSE-PPO actor-critic for UniLab. The actor consumes the current single-step
# observation concatenated with the estimator latent (detached); the critic is
# asymmetric and consumes the full privileged observation.

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from unilab.algos.torch.cse_ppo.estimator import CSEEstimator, get_activation


class CSEActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_one_step_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        critic_hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        estimator: dict | None = None,
    ) -> None:
        super().__init__()
        if num_one_step_obs <= 0:
            raise ValueError("num_one_step_obs must be positive")
        if num_actor_obs % num_one_step_obs != 0:
            raise ValueError(
                "num_actor_obs must be an integer multiple of num_one_step_obs, "
                f"got {num_actor_obs} and {num_one_step_obs}"
            )
        if len(actor_hidden_dims) == 0 or len(critic_hidden_dims) == 0:
            raise ValueError("actor_hidden_dims and critic_hidden_dims must not be empty")

        self.history_size = int(num_actor_obs // num_one_step_obs)
        self.num_actor_obs = int(num_actor_obs)
        self.num_critic_obs = int(num_critic_obs)
        self.num_actions = int(num_actions)
        self.num_one_step_obs = int(num_one_step_obs)

        estimator_cfg = dict(estimator or {})
        self.estimator = CSEEstimator(
            temporal_steps=self.history_size,
            num_one_step_obs=self.num_one_step_obs,
            activation=activation,
            **estimator_cfg,
        )

        actor_input_dim = self.num_one_step_obs + self.estimator.num_latent
        self.actor = _build_mlp(actor_input_dim, self.num_actions, actor_hidden_dims, activation)
        self.critic = _build_mlp(self.num_critic_obs, 1, critic_hidden_dims, activation)

        self.std = nn.Parameter(float(init_noise_std) * torch.ones(self.num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def forward(self) -> torch.Tensor:
        raise NotImplementedError

    def _actor_input(self, obs_history: torch.Tensor) -> torch.Tensor:
        # The actor receives a NON-detached latent, so the PPO surrogate/value
        # gradient co-trains the estimator encoder through the shared optimizer.
        # Detaching here would sever the encoder<->RL coupling that defines the
        # concurrent state estimator.
        latent = self.estimator.encode(obs_history)
        # The history is stored newest-frame-LAST, so the current single-step obs
        # is the final block.
        return torch.cat((obs_history[:, -self.num_one_step_obs :], latent), dim=-1)

    def update_distribution(self, obs_history: torch.Tensor) -> None:
        mean = self.actor(self._actor_input(obs_history))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, obs_history: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        self.update_distribution(obs_history)
        assert self.distribution is not None
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history: torch.Tensor, observations=None) -> torch.Tensor:
        del observations
        return self.actor(self._actor_input(obs_history))

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return self.critic(critic_observations)


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int] | tuple[int, ...],
    activation: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers += [nn.Linear(last_dim, int(hidden_dim)), get_activation(activation)]
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)
