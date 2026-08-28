"""Rollout storage for sequence-based SONIC PPO."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist


class SonicRolloutStorage:
    def __init__(
        self,
        horizon: int,
        num_envs: int,
        actor_obs_dim: int = 930,
        critic_obs_dim: int = 1645,
        tokenizer_obs_dim: int = 1761,
        action_dim: int = 29,
        device: str | torch.device = "cpu",
    ) -> None:
        self.horizon = int(horizon)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        shape = (self.horizon, self.num_envs)
        self.actor_obs = torch.zeros(*shape, actor_obs_dim, device=self.device)
        self.critic_obs = torch.zeros(*shape, critic_obs_dim, device=self.device)
        self.tokenizer_obs = torch.zeros(*shape, tokenizer_obs_dim, device=self.device)
        self.actions = torch.zeros(*shape, action_dim, device=self.device)
        self.action_mean = torch.zeros(*shape, action_dim, device=self.device)
        self.action_std = torch.zeros(*shape, action_dim, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.dones = torch.zeros(*shape, dtype=torch.bool, device=self.device)
        self.values = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)
        self.returns = torch.zeros_like(self.rewards)
        self.advantages = torch.zeros_like(self.rewards)
        self.step = 0

    def add(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        tokenizer_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
    ) -> None:
        if self.step >= self.horizon:
            raise RuntimeError("SONIC rollout storage overflow")
        index = self.step
        tensors = (
            (self.actor_obs[index], actor_obs, "actor_obs"),
            (self.critic_obs[index], critic_obs, "critic_obs"),
            (self.tokenizer_obs[index], tokenizer_obs, "tokenizer_obs"),
            (self.actions[index], actions, "actions"),
            (self.action_mean[index], action_mean, "action_mean"),
            (self.action_std[index], action_std, "action_std"),
        )
        for target, value, name in tensors:
            if target.shape != value.shape:
                raise ValueError(
                    f"{name} shape mismatch: expected {tuple(target.shape)}, "
                    f"got {tuple(value.shape)}"
                )
            target.copy_(value.detach())
        self.rewards[index].copy_(rewards.reshape(-1).detach())
        self.dones[index].copy_(dones.reshape(-1).bool().detach())
        self.values[index].copy_(values.reshape(-1).detach())
        self.log_probs[index].copy_(log_probs.reshape(-1).detach())
        self.step += 1

    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> None:
        if self.step != self.horizon:
            raise ValueError(
                f"SONIC rollout is incomplete: got {self.step}, expected {self.horizon}"
            )
        advantage = torch.zeros_like(last_values.reshape(-1))
        for index in reversed(range(self.horizon)):
            not_done = (~self.dones[index]).float()
            next_value = (
                last_values.reshape(-1) if index == self.horizon - 1 else self.values[index + 1]
            )
            delta = self.rewards[index] + gamma * next_value * not_done - self.values[index]
            advantage = delta + gamma * lam * not_done * advantage
            self.advantages[index] = advantage
        self.returns = self.advantages + self.values
        moments = torch.stack(
            (
                self.advantages.sum(dtype=torch.float64),
                self.advantages.square().sum(dtype=torch.float64),
                torch.tensor(
                    float(self.advantages.numel()),
                    dtype=torch.float64,
                    device=self.device,
                ),
            )
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(moments, op=dist.ReduceOp.SUM)
        count = moments[2]
        mean = moments[0] / count.clamp_min(1.0)
        centered_sum = moments[1] - moments[0].square() / count.clamp_min(1.0)
        variance = centered_sum / (count - 1.0).clamp_min(1.0)
        self.advantages = (self.advantages - mean) / torch.sqrt(variance + 1e-8)

    def batches(
        self,
        mini_batches: int,
        epochs: int = 5,
        microbatch_size: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[dict[str, torch.Tensor]]:
        total = self.num_envs
        if mini_batches < 1 or total < mini_batches:
            raise ValueError("mini_batches must not exceed rollout size")
        if total % mini_batches:
            raise ValueError("num_envs must be divisible by mini_batches for SONIC sequences")
        mini_size = total // mini_batches
        indices = torch.randperm(total, generator=generator, device=self.device)
        tensors = (
            ("actor_obs", self.actor_obs),
            ("critic_obs", self.critic_obs),
            ("tokenizer_obs", self.tokenizer_obs),
            ("actions", self.actions),
            ("action_mean", self.action_mean),
            ("action_std", self.action_std),
            ("values", self.values),
            ("returns", self.returns),
            ("advantages", self.advantages),
            ("log_probs", self.log_probs),
        )
        for _ in range(int(epochs)):
            for start in range(0, mini_size * mini_batches, mini_size):
                selected = indices[start : start + mini_size]
                batch = {name: value[:, selected].transpose(0, 1) for name, value in tensors}
                if microbatch_size is None or microbatch_size >= selected.numel():
                    yield batch
                else:
                    for offset in range(0, selected.numel(), microbatch_size):
                        yield {
                            name: value[offset : offset + microbatch_size]
                            for name, value in batch.items()
                        }

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step}

    def clear(self) -> None:
        self.step = 0

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step = int(state.get("step", 0))
