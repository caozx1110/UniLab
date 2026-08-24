"""SONIC policy modules and the release token-shape contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
import torch.distributed as dist
from torch import nn


class RunningMeanStd(nn.Module):
    """Numerically stable per-feature running normalization."""

    def __init__(self, width: int, epsilon: float = 1e-4) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("var", torch.ones(width))
        self.register_buffer("count", torch.tensor(float(epsilon)))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        flattened = values.detach().reshape(-1, values.shape[-1]).float()
        if flattened.numel() == 0:
            return
        batch_mean = flattened.mean(0)
        batch_var = flattened.var(0, unbiased=False)
        batch_count = torch.tensor(float(flattened.shape[0]), device=flattened.device)
        mean = cast(torch.Tensor, self.mean)
        var = cast(torch.Tensor, self.var)
        count = cast(torch.Tensor, self.count)
        delta = batch_mean - mean
        total = count + batch_count
        mean.add_(delta * batch_count / total)
        m_a = var * count
        m_b = batch_var * batch_count
        var.copy_((m_a + m_b + delta.square() * count * batch_count / total) / total)
        count.copy_(total)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        mean = cast(torch.Tensor, self.mean)
        var = cast(torch.Tensor, self.var)
        return (values - mean.to(values)) / torch.sqrt(var.to(values) + 1e-5)


class FSQ(nn.Module):
    """Straight-through finite scalar quantization for ``2 x 32`` tokens.

    SONIC calls the second axis the token count and the last axis the FSQ
    level dimension. The module also accepts a flattened ``(..., 64)`` tensor.
    """

    def __init__(
        self,
        num_tokens: int = 2,
        levels: int = 32,
        token_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_tokens < 1 or levels < 2:
            raise ValueError("num_tokens must be >=1 and levels must be >=2")
        self.num_tokens = int(num_tokens)
        self.levels = int(levels)
        self.token_dim = int(token_dim if token_dim is not None else levels)
        if self.token_dim < 1:
            raise ValueError("token_dim must be positive")
        self.level_list = (self.levels,) * self.token_dim

    def _reshape(self, values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], bool]:
        if values.ndim < 2:
            raise ValueError("FSQ input must have at least two dimensions")
        if values.shape[-2:] == (self.num_tokens, self.token_dim):
            return values, tuple(values.shape), False
        if values.shape[-1] == self.num_tokens * self.token_dim:
            shape = (*values.shape[:-1], self.num_tokens, self.token_dim)
            return values.reshape(shape), tuple(values.shape), True
        if values.shape[-1] == self.num_tokens:
            shape = (*values.shape[:-1], self.num_tokens, 1)
            return values.reshape(shape), tuple(values.shape), False
        raise ValueError(
            "FSQ expects (..., num_tokens, token_dim), (..., num_tokens*token_dim), "
            f"or (..., {self.num_tokens}); got {tuple(values.shape)}"
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        reshaped, original_shape, flattened = self._reshape(values)
        half_level = (self.levels - 1) * (1.0 - 1.0e-3) / 2.0
        offset = 0.5 if self.levels % 2 == 0 else 0.0
        shift = torch.atanh(
            torch.as_tensor(
                offset / half_level,
                dtype=reshaped.dtype,
                device=reshaped.device,
            )
        )
        bounded = torch.tanh(reshaped + shift) * half_level - offset
        rounded = bounded.round()
        quantized = bounded + (rounded - bounded).detach()
        quantized = quantized / (self.levels // 2)
        if flattened:
            return quantized.reshape(original_shape)
        if original_shape[-1] == self.num_tokens:
            return quantized.squeeze(-1)
        return quantized

    def indices(self, values: torch.Tensor) -> torch.Tensor:
        quantized = self.forward(values)
        return (quantized * (self.levels // 2) + self.levels // 2).round().long()


class UniversalToken(nn.Module):
    """MLP encoder with the SONIC UniversalToken ``2 x 32`` bottleneck."""

    def __init__(
        self,
        input_dim: int = 1761,
        num_tokens: int = 2,
        levels: int = 32,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.input_dim = int(input_dim)
        self.num_tokens = int(num_tokens)
        self.token_dim = int(levels)
        self.token_total_dim = self.num_tokens * self.token_dim
        self.fsq = FSQ(self.num_tokens, levels, self.token_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.token_total_dim),
        )
        self.reconstruction = nn.Linear(self.token_total_dim, self.input_dim)

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != self.input_dim:
            raise ValueError(
                f"tokenizer expects {self.input_dim} features, got {observations.shape[-1]}"
            )
        return self.encoder(observations).reshape(
            *observations.shape[:-1], self.num_tokens, self.token_dim
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.fsq(self.encode(observations))

    def get_token_info(self) -> dict[str, object]:
        return {
            "token_dim": self.token_dim,
            "total_dim": self.token_total_dim,
            "num_tokens": self.num_tokens,
            "num_levels": self.token_dim,
            "level_list": list(self.fsq.level_list),
        }

    def auxiliary_losses(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return differentiable tokenizer reconstruction/commitment terms."""

        latent = self.encode(observations)
        tokens = self.fsq(latent)
        flat_tokens = tokens.reshape(*tokens.shape[:-2], -1)
        reconstruction = self.reconstruction(flat_tokens)
        return {
            "token_reconstruction": (reconstruction - observations).square().mean(),
            "token_commitment": (latent - tokens.detach()).square().mean(),
        }


def _normalise_hidden_dims(
    hidden_dims: Sequence[int] | None,
    fallback: tuple[int, ...],
) -> tuple[int, ...]:
    values = tuple(int(width) for width in (fallback if hidden_dims is None else hidden_dims))
    if not values or any(width < 1 for width in values):
        raise ValueError("hidden_dims must contain positive widths")
    return values


class SonicActorCritic(nn.Module):
    """SONIC actor, critic and tokenizer with release I/O dimensions."""

    def __init__(
        self,
        actor_obs_dim: int = 930,
        critic_obs_dim: int = 1645,
        tokenizer_obs_dim: int = 1761,
        action_dim: int = 29,
        hidden_dims: Sequence[int] | None = None,
        actor_hidden_dims: Sequence[int] | None = None,
        critic_hidden_dims: Sequence[int] | None = None,
        tokenizer_hidden_dim: int = 512,
        token_levels: int = 32,
        token_count: int = 2,
        critic_obs_normalization: bool = False,
        init_noise_std: float = 0.05,
        std_clamp_min: float = 0.001,
        std_clamp_max: float = 0.5,
    ) -> None:
        super().__init__()
        self.actor_obs_dim = int(actor_obs_dim)
        self.critic_obs_dim = int(critic_obs_dim)
        self.tokenizer_obs_dim = int(tokenizer_obs_dim)
        self.action_dim = int(action_dim)
        if (
            min(
                self.actor_obs_dim,
                self.critic_obs_dim,
                self.tokenizer_obs_dim,
                self.action_dim,
            )
            < 1
        ):
            raise ValueError("SONIC model dimensions must be positive")
        if not 0.0 < float(std_clamp_min) <= float(init_noise_std) <= float(std_clamp_max):
            raise ValueError(
                "SONIC noise std requires 0 < std_clamp_min <= init_noise_std <= std_clamp_max"
            )
        self.tokenizer = UniversalToken(
            self.tokenizer_obs_dim,
            num_tokens=token_count,
            levels=token_levels,
            hidden_dim=tokenizer_hidden_dim,
        )
        self.critic_obs_normalization = bool(critic_obs_normalization)
        self.critic_rms = (
            RunningMeanStd(self.critic_obs_dim) if self.critic_obs_normalization else None
        )
        self._normalizer_start: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        token_dim = self.tokenizer.token_total_dim
        fallback = (512, 256) if hidden_dims is None else tuple(hidden_dims)
        actor_widths = _normalise_hidden_dims(actor_hidden_dims, fallback)
        critic_widths = _normalise_hidden_dims(critic_hidden_dims, actor_widths)

        def mlp(input_dim: int, output_dim: int, widths: Sequence[int]) -> nn.Sequential:
            layers: list[nn.Module] = []
            last = input_dim
            for width in widths:
                layers.extend((nn.Linear(last, int(width)), nn.SiLU()))
                last = int(width)
            layers.append(nn.Linear(last, output_dim))
            return nn.Sequential(*layers)

        self.actor = mlp(self.actor_obs_dim + token_dim, self.action_dim, actor_widths)
        # The release value head consumes privileged observations directly;
        # UniversalToken is an actor-side bottleneck and is not concatenated
        # into the critic input.
        self.critic = mlp(self.critic_obs_dim, 1, critic_widths)
        self.std_clamp_min = float(std_clamp_min)
        self.std_clamp_max = float(std_clamp_max)
        # The release actor owns a direct std parameter (not log_std) and
        # clamps it in-place to [0.001, 0.5] before constructing Normal.
        self.std = nn.Parameter(torch.full((self.action_dim,), float(init_noise_std)))

    def _features(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        token_obs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if actor_obs.shape[-1] != self.actor_obs_dim:
            raise ValueError(
                f"actor expects {self.actor_obs_dim} features, got {actor_obs.shape[-1]}"
            )
        if critic_obs.shape[-1] != self.critic_obs_dim:
            raise ValueError(
                f"critic expects {self.critic_obs_dim} features, got {critic_obs.shape[-1]}"
            )
        if token_obs is None:
            token_obs = torch.zeros(
                *actor_obs.shape[:-1],
                self.tokenizer_obs_dim,
                device=actor_obs.device,
                dtype=actor_obs.dtype,
            )
        tokens = self.tokenizer(token_obs)
        flat_tokens = tokens.reshape(*tokens.shape[:-2], -1)
        if self.critic_rms is not None:
            critic_obs = self.critic_rms(critic_obs)
        return (
            torch.cat((actor_obs, flat_tokens), dim=-1),
            critic_obs,
            tokens,
        )

    def distribution(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        token_obs: torch.Tensor | None = None,
    ) -> tuple[torch.distributions.Normal, torch.Tensor]:
        actor_features, critic_features, _ = self._features(actor_obs, critic_obs, token_obs)
        mean = self.actor(actor_features)
        with torch.no_grad():
            self.std.clamp_(self.std_clamp_min, self.std_clamp_max)
        std = self.std.expand_as(mean)
        return torch.distributions.Normal(mean, std), self.critic(critic_features).squeeze(-1)

    def act(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        token_obs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.distribution(actor_obs, critic_obs, token_obs)
        action = distribution.sample()
        return action, distribution.log_prob(action).sum(-1), value

    def evaluate(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        token_obs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.distribution(actor_obs, critic_obs, token_obs)
        return (
            distribution.log_prob(actions).sum(-1),
            value,
            distribution.entropy().sum(-1),
        )

    def auxiliary_losses(self, token_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.tokenizer.auxiliary_losses(token_obs)

    @torch.no_grad()
    def update_normalizers(self, critic_obs: torch.Tensor) -> None:
        if self.critic_rms is not None:
            self.critic_rms.update(critic_obs)

    @torch.no_grad()
    def begin_normalizer_update(self) -> None:
        """Snapshot RMS state before a rank-local rollout is collected."""

        if self.critic_rms is None:
            self._normalizer_start = None
            return
        rms = self.critic_rms
        self._normalizer_start = (
            cast(torch.Tensor, rms.mean).detach().clone(),
            cast(torch.Tensor, rms.var).detach().clone(),
            cast(torch.Tensor, rms.count).detach().clone(),
        )

    @torch.no_grad()
    def synchronize_normalizers(self) -> None:
        if self.critic_rms is None:
            return
        if not (dist.is_available() and dist.is_initialized()):
            self._normalizer_start = None
            return
        if self._normalizer_start is None:
            return
        rms = self.critic_rms
        start_mean, start_var, start_count = self._normalizer_start
        current_mean = cast(torch.Tensor, rms.mean)
        current_var = cast(torch.Tensor, rms.var)
        current_count = cast(torch.Tensor, rms.count)
        batch_count = (current_count - start_count).clamp_min(0.0)
        batch_first = current_mean * current_count - start_mean * start_count
        batch_second = (current_var + current_mean.square()) * current_count - (
            start_var + start_mean.square()
        ) * start_count
        dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_first, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_second, op=dist.ReduceOp.SUM)
        total = start_count + batch_count
        mean = (start_mean * start_count + batch_first) / total.clamp_min(1e-8)
        var = ((start_var + start_mean.square()) * start_count + batch_second) / total.clamp_min(
            1e-8
        ) - mean.square()
        current_mean.copy_(mean)
        current_var.copy_(var.clamp_min(1e-6))
        current_count.copy_(total)
        self._normalizer_start = None


__all__ = ["FSQ", "RunningMeanStd", "SonicActorCritic", "UniversalToken"]
