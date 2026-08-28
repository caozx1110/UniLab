"""Sequence-aware PPO optimization for the native SONIC owner."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .model import SonicActorCritic
from .storage import SonicRolloutStorage

_RELEASE_AUXILIARY_LOSS_NAMES = (
    "g1_recon",
    "g1_smpl_latent",
    "g1_teleop_latent",
    "teleop_smpl_latent",
    "reencoded_smpl_g1_latent",
)


class _SonicDistributedTrainingForward(nn.Module):
    """Tensor-only DDP entrypoint for the release training graph."""

    def __init__(self, model: SonicActorCritic) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        tokenizer_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values, auxiliary_losses = self.model.training_forward(
            actor_obs,
            critic_obs,
            tokenizer_obs,
        )
        return (
            distribution.mean,
            distribution.stddev,
            values,
            torch.stack(tuple(auxiliary_losses[name] for name in _RELEASE_AUXILIARY_LOSS_NAMES)),
        )


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
        self.distributed_gradient_overlap = bool(
            _mapping_value(cfg, "distributed_gradient_overlap", False)
        )
        self.ddp_bucket_cap_mb = int(_mapping_value(cfg, "ddp_bucket_cap_mb", 25))
        if self.ddp_bucket_cap_mb < 1:
            raise ValueError("ddp_bucket_cap_mb must be positive")
        self.use_amp = bool(_mapping_value(cfg, "use_amp", False))
        amp_dtype = str(_mapping_value(cfg, "amp_dtype", "bf16")).strip().lower()
        if amp_dtype in {"auto", "bf16", "bfloat16"}:
            self.amp_dtype = torch.bfloat16
        elif amp_dtype in {"fp16", "float16", "half"}:
            self.amp_dtype = torch.float16
        else:
            raise ValueError(f"unsupported SONIC amp_dtype: {amp_dtype!r}")
        raw_aux = _mapping_value(cfg, "aux_loss_coef", {})
        self.aux_loss_coef = (
            {str(name): float(value) for name, value in raw_aux.items()}
            if isinstance(raw_aux, Mapping)
            else {}
        )
        decay_parameters: list[nn.Parameter] = []
        bias_parameters: list[nn.Parameter] = []
        for name, parameter in model.named_parameters():
            (bias_parameters if name.endswith(".bias") else decay_parameters).append(parameter)
        self.optimizer = torch.optim.AdamW(
            (
                {"params": decay_parameters, "weight_decay": 0.0},
                {"params": bias_parameters, "weight_decay": 0.0},
            ),
            lr=self.learning_rate,
        )
        self._distributed_training_model: DistributedDataParallel | None = None
        if (
            self.distributed_gradient_overlap
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            if self.device.type != "cuda":
                raise RuntimeError("SONIC DDP gradient overlap requires one CUDA device per rank")
            if self.model.model_profile != "sonic_release":
                raise RuntimeError(
                    "SONIC DDP gradient overlap is owned by model_profile='sonic_release'"
                )
            missing_auxiliary_losses = sorted(
                set(_RELEASE_AUXILIARY_LOSS_NAMES).difference(self.aux_loss_coef)
            )
            if missing_auxiliary_losses:
                raise RuntimeError(
                    "SONIC DDP gradient overlap requires the complete release auxiliary-loss "
                    f"graph; missing {missing_auxiliary_losses}"
                )
            device_index = (
                self.device.index if self.device.index is not None else torch.cuda.current_device()
            )
            self._distributed_training_model = DistributedDataParallel(
                _SonicDistributedTrainingForward(self.model),
                device_ids=[device_index],
                output_device=device_index,
                broadcast_buffers=True,
                bucket_cap_mb=self.ddp_bucket_cap_mb,
                find_unused_parameters=False,
                gradient_as_bucket_view=False,
            )
        self.update_count = 0
        self.last_optimizer_steps = 0

    def _autocast(self):
        enabled = self.use_amp and self.device.type in {"cuda", "cpu"}
        return (
            torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=enabled,
            )
            if enabled
            else nullcontext()
        )

    def _sync_gradients(self) -> None:
        if getattr(self, "_distributed_training_model", None) is not None:
            # DDP finishes its averaged bucket reductions before backward()
            # returns. A second flat all-reduce would duplicate communication.
            return
        if not (dist.is_available() and dist.is_initialized()):
            return
        world = dist.get_world_size()
        buckets: dict[
            tuple[torch.device, torch.dtype], list[tuple[nn.Parameter, torch.Tensor]]
        ] = {}
        for parameter in self.model.parameters():
            if not parameter.requires_grad:
                continue
            gradient = parameter.grad
            if gradient is None:
                gradient = torch.zeros_like(parameter)
                parameter.grad = gradient
            key = (gradient.device, gradient.dtype)
            buckets.setdefault(key, []).append((parameter, gradient))

        for bucket in buckets.values():
            flat = torch.cat([gradient.reshape(-1) for _, gradient in bucket])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(world)
            offset = 0
            for _, gradient in bucket:
                size = gradient.numel()
                gradient.copy_(flat.narrow(0, offset, size).view_as(gradient))
                offset += size

    @staticmethod
    def _distributed_mean(value: float | torch.Tensor, device: torch.device) -> float:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(device=device, dtype=torch.float64, copy=True).reshape(())
        else:
            tensor = torch.tensor(value, dtype=torch.float64, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor.div_(dist.get_world_size())
        return float(tensor.item())

    @staticmethod
    def _distributed_mean_vector(value: torch.Tensor) -> list[float]:
        tensor = value.detach().to(dtype=torch.float64, copy=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor.div_(dist.get_world_size())
        return tensor.tolist()

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
        metric_names = (
            "loss",
            "value_loss",
            "policy_loss",
            "entropy",
            "approx_kl",
            *self.aux_loss_coef,
        )
        totals = torch.zeros(len(metric_names), dtype=torch.float64, device=self.device)
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
                    aux_losses: dict[str, torch.Tensor]
                    with self._autocast():
                        if self._distributed_training_model is not None:
                            mean, std, values, auxiliary_values = self._distributed_training_model(
                                batch["actor_obs"],
                                batch["critic_obs"],
                                batch["tokenizer_obs"],
                            )
                            distribution = torch.distributions.Normal(mean, std)
                            aux_losses = dict(
                                zip(
                                    _RELEASE_AUXILIARY_LOSS_NAMES,
                                    auxiliary_values.unbind(),
                                    strict=True,
                                )
                            )
                        elif self.aux_loss_coef:
                            distribution, values, aux_losses = self.model.training_forward(
                                batch["actor_obs"],
                                batch["critic_obs"],
                                batch["tokenizer_obs"],
                            )
                        else:
                            distribution, values = self.model.distribution(
                                batch["actor_obs"],
                                batch["critic_obs"],
                                batch["tokenizer_obs"],
                            )
                            aux_losses = {}
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
                        # Keep the release's per-microbatch adaptive-LR cadence, but
                        # reduce the detached CUDA scalar directly.  Converting to a
                        # Python float first would force D2H, followed immediately by
                        # an H2D copy into the FP64 collective buffer.
                        kl_value = self._distributed_mean(gaussian_kl, self.device)
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
                    for name, coefficient in self.aux_loss_coef.items():
                        if name in aux_losses:
                            loss = loss + coefficient * aux_losses[name]
                    sample_count = int(env_ids.numel() * storage.horizon)
                    total_samples = int(mini_size * storage.horizon)
                    scale = (
                        1.0 if self.optimizer_step_per_microbatch else sample_count / total_samples
                    )
                    (loss * scale).backward()
                    metric_values = torch.stack(
                        (
                            loss.detach(),
                            value_loss.detach(),
                            policy_loss.detach(),
                            entropy_mean.detach(),
                            gaussian_kl.detach(),
                            *(
                                aux_losses[name].detach()
                                if name in aux_losses
                                else loss.detach().new_zeros(())
                                for name in self.aux_loss_coef
                            ),
                        )
                    )
                    totals.add_(metric_values.to(dtype=torch.float64), alpha=sample_count)
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
        averaged_metrics = self._distributed_mean_vector(totals / denominator)
        result = dict(zip(metric_names, averaged_metrics, strict=True))
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
