"""Sequence-aware PPO optimization for the native SONIC owner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from .model import SonicActorCritic
from .storage import SonicRolloutStorage


def _mapping_value(config: Mapping[str, Any], name: str, default: Any) -> Any:
    """Read a scalar from flat, ``algo`` or upstream ``algo.config`` layouts."""

    candidates: list[Mapping[str, Any]] = [config]
    algo = config.get("algo")
    if isinstance(algo, Mapping):
        candidates.append(algo)
        nested = algo.get("config")
        if isinstance(nested, Mapping):
            candidates.append(nested)
    sonic = config.get("sonic")
    if isinstance(sonic, Mapping):
        candidates.append(sonic)
    for candidate in candidates:
        if name in candidate and candidate[name] is not None:
            return candidate[name]
    return default


class SonicPPO:
    """PPO with logical minibatches measured in complete environment sequences."""

    def __init__(
        self,
        model: SonicActorCritic,
        config: Mapping[str, Any] | None = None,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> None:
        cfg: dict[str, Any] = dict(config or {})
        cfg.update(kwargs)
        self.model = model
        self.device = torch.device(device)
        self.gamma = float(_mapping_value(cfg, "gamma", 0.99))
        self.lam = float(_mapping_value(cfg, "lam", 0.95))
        self.clip_param = float(_mapping_value(cfg, "clip_param", 0.2))
        # The SONIC release config uses vf_coef/value_loss_coef=1.0 and its
        # clipped value loss has no implicit 0.5 factor.  Keep that contract
        # even when the native owner is constructed without a full Hydra cfg.
        self.value_loss_coef = float(
            _mapping_value(cfg, "value_loss_coef", _mapping_value(cfg, "vf_coef", 1.0))
        )
        self.entropy_coef = float(_mapping_value(cfg, "entropy_coef", 0.01))
        self.max_grad_norm = float(_mapping_value(cfg, "max_grad_norm", 0.1))
        self.num_learning_epochs = int(
            _mapping_value(cfg, "num_learning_epochs", _mapping_value(cfg, "num_ppo_epochs", 5))
        )
        self.num_mini_batches = int(_mapping_value(cfg, "num_mini_batches", 4))
        self.microbatch_size = _mapping_value(cfg, "microbatch_size", None)
        if self.microbatch_size is None:
            sonic = cfg.get("sonic")
            if isinstance(sonic, Mapping):
                self.microbatch_size = sonic.get("microbatch_size")
        self.use_clipped_value_loss = bool(_mapping_value(cfg, "use_clipped_value_loss", True))
        self.learning_rate = float(
            _mapping_value(
                cfg,
                "learning_rate",
                _mapping_value(cfg, "actor_learning_rate", _mapping_value(cfg, "lr", 3e-4)),
            )
        )
        self.desired_kl = float(_mapping_value(cfg, "desired_kl", 0.0))
        self.schedule = str(_mapping_value(cfg, "schedule", "constant"))
        self.adaptive_lr_min = float(_mapping_value(cfg, "adaptive_lr_min", 1e-6))
        self.adaptive_lr_max = float(_mapping_value(cfg, "adaptive_lr_max", 1e-2))
        self.optimizer_step_per_microbatch = bool(
            _mapping_value(cfg, "optimizer_step_per_microbatch", True)
        )
        raw_aux = _mapping_value(cfg, "aux_loss_coef", {})
        self.aux_loss_coef = (
            {str(name): float(value) for name, value in raw_aux.items()}
            if isinstance(raw_aux, Mapping)
            else {}
        )
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.update_count = 0
        self.last_optimizer_steps = 0

    def _sync_gradients(self) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        world = dist.get_world_size()
        for parameter in self.model.parameters():
            if not parameter.requires_grad:
                continue
            gradient = parameter.grad
            if gradient is None:
                gradient = torch.zeros_like(parameter)
                parameter.grad = gradient
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            gradient.div_(world)

    @staticmethod
    def _distributed_mean(value: float, device: torch.device) -> float:
        if not (dist.is_available() and dist.is_initialized()):
            return value
        tensor = torch.tensor(value, dtype=torch.float64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(dist.get_world_size())
        return float(tensor.item())

    def _adjust_learning_rate(self, approx_kl: float) -> None:
        if self.schedule != "adaptive" or self.desired_kl <= 0:
            return
        rate = self.optimizer.param_groups[0]["lr"]
        if approx_kl > self.desired_kl * 2.0:
            rate = max(self.adaptive_lr_min, rate / 1.5)
        elif 0.0 < approx_kl < self.desired_kl / 2.0:
            rate = min(self.adaptive_lr_max, rate * 1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = rate

    def update(self, storage: SonicRolloutStorage) -> dict[str, float]:
        if storage.step != storage.horizon:
            raise ValueError(
                f"SONIC rollout is incomplete: got {storage.step}, expected {storage.horizon}"
            )
        if self.num_mini_batches < 1 or storage.num_envs % self.num_mini_batches:
            raise ValueError("num_envs must be divisible by num_mini_batches")
        self.model.train()
        mini_size = storage.num_envs // self.num_mini_batches
        micro = int(self.microbatch_size or mini_size)
        if micro < 1 or micro > mini_size:
            raise ValueError(
                f"microbatch_size must be in [1, {mini_size}] for a sequence minibatch"
            )
        totals = {
            "loss": 0.0,
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            **{name: 0.0 for name in self.aux_loss_coef},
        }
        observations = (
            ("actor_obs", storage.actor_obs),
            ("critic_obs", storage.critic_obs),
            ("tokenizer_obs", storage.tokenizer_obs),
            ("actions", storage.actions),
            ("action_mean", storage.action_mean),
            ("action_std", storage.action_std),
            ("values", storage.values),
            ("returns", storage.returns),
            ("advantages", storage.advantages),
            ("log_probs", storage.log_probs),
        )
        update_count = 0
        for _ in range(self.num_learning_epochs):
            # The release trainer reshuffles sequence ids at the beginning of
            # every PPO epoch (``ppo_shuffle_every_epoch=true``).  Reusing one
            # permutation across epochs subtly changes both optimizer noise
            # and adaptive-KL cadence.
            indices = torch.randperm(storage.num_envs, device=self.device)
            for start in range(0, storage.num_envs, mini_size):
                selected = indices[start : start + mini_size]
                if not self.optimizer_step_per_microbatch:
                    self.optimizer.zero_grad(set_to_none=True)
                logical_samples = 0
                for offset in range(0, mini_size, micro):
                    env_ids = selected[offset : offset + micro]
                    if self.optimizer_step_per_microbatch:
                        self.optimizer.zero_grad(set_to_none=True)
                    batch = {name: value[:, env_ids] for name, value in observations}
                    distribution, values = self.model.distribution(
                        batch["actor_obs"],
                        batch["critic_obs"],
                        batch["tokenizer_obs"],
                    )
                    new_log_prob = distribution.log_prob(batch["actions"]).sum(-1)
                    entropy = distribution.entropy().sum(-1)
                    with torch.no_grad():
                        new_mean = distribution.mean
                        new_std = distribution.stddev
                        old_mean = batch["action_mean"]
                        old_std = batch["action_std"]
                        gaussian_kl = torch.sum(
                            torch.log(new_std / old_std + 1.0e-5)
                            + (old_std.square() + (old_mean - new_mean).square())
                            / (2.0 * new_std.square())
                            - 0.5,
                            dim=-1,
                        ).mean()
                        kl_value = self._distributed_mean(float(gaussian_kl), self.device)
                        self._adjust_learning_rate(kl_value)
                    ratio = torch.exp(new_log_prob - batch["log_probs"])
                    advantages = batch["advantages"]
                    unclipped = ratio * advantages
                    clipped = (
                        torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                        * advantages
                    )
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    if self.use_clipped_value_loss:
                        value_delta = values - batch["values"]
                        clipped_values = batch["values"] + value_delta.clamp(
                            -self.clip_param, self.clip_param
                        )
                        value_loss = torch.maximum(
                            (values - batch["returns"]).square(),
                            (clipped_values - batch["returns"]).square(),
                        ).mean()
                    else:
                        value_loss = (values - batch["returns"]).square().mean()
                    entropy_mean = entropy.mean()
                    loss = (
                        policy_loss
                        + self.value_loss_coef * value_loss
                        - self.entropy_coef * entropy_mean
                    )
                    aux_losses = (
                        self.model.auxiliary_losses(batch["tokenizer_obs"])
                        if self.aux_loss_coef
                        else {}
                    )
                    for name, coefficient in self.aux_loss_coef.items():
                        if name in aux_losses:
                            loss = loss + coefficient * aux_losses[name]
                    sample_count = int(env_ids.numel() * storage.horizon)
                    total_samples = int(mini_size * storage.horizon)
                    scale = (
                        1.0 if self.optimizer_step_per_microbatch else sample_count / total_samples
                    )
                    (loss * scale).backward()
                    totals["loss"] += float(loss.detach()) * sample_count
                    totals["value_loss"] += float(value_loss.detach()) * sample_count
                    totals["policy_loss"] += float(policy_loss.detach()) * sample_count
                    totals["entropy"] += float(entropy_mean.detach()) * sample_count
                    totals["approx_kl"] += kl_value * sample_count
                    for name in self.aux_loss_coef:
                        if name in aux_losses:
                            totals[name] += float(aux_losses[name].detach()) * sample_count
                    logical_samples += sample_count
                    if self.optimizer_step_per_microbatch:
                        self._sync_gradients()
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        update_count += 1
                if logical_samples != mini_size * storage.horizon:
                    raise RuntimeError("SONIC microbatch accounting drifted")
                if not self.optimizer_step_per_microbatch:
                    self._sync_gradients()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    update_count += 1
        storage.clear()
        denominator = max(1, self.num_learning_epochs * storage.num_envs * storage.horizon)
        result = {
            name: self._distributed_mean(value / denominator, self.device)
            for name, value in totals.items()
        }
        self.update_count += update_count
        self.last_optimizer_steps = update_count
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "last_optimizer_steps": self.last_optimizer_steps,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.update_count = int(state.get("update_count", 0))
        self.last_optimizer_steps = int(state.get("last_optimizer_steps", 0))


__all__ = ["SonicPPO"]
