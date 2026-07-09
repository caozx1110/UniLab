# SPDX-License-Identifier: BSD-3-Clause
#
# CSE-PPO algorithm: standard PPO with an asymmetric critic plus a concurrent
# state estimator updated by its own optimizer inside the PPO loop. Mirrors
# HIM-PPO but the estimator is supervised (no contrastive swap loss).

from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from unilab.algos.torch.cse_ppo.actor_critic import CSEActorCritic
from unilab.algos.torch.cse_ppo.storage import CSERolloutStorage


class CSEPPO:
    actor_critic: CSEActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float | None = 0.01,
        min_learning_rate: float = 1e-5,
        max_learning_rate: float = 1e-2,
        min_policy_std: float = 1e-2,
        max_policy_std: float | None = None,
        use_amp: bool = False,
        amp_dtype: str = "bfloat16",
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.device = device
        # Mixed-precision (AMP) for the update forward/loss only. bf16 needs no
        # GradScaler and only runs on CUDA; collection/rollout stays fp32 to avoid
        # any train/deploy mismatch.
        self._amp_enabled = bool(use_amp) and "cuda" in str(device)
        self._amp_dtype = torch.bfloat16 if str(amp_dtype) == "bfloat16" else torch.float16
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = float(learning_rate)
        self.min_learning_rate = float(min_learning_rate)
        self.max_learning_rate = float(max_learning_rate)
        self.min_policy_std = float(min_policy_std)
        self.max_policy_std = float(max_policy_std) if max_policy_std is not None else None

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage: CSERolloutStorage | None = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        self.transition = CSERolloutStorage.Transition()

        self.clip_param = float(clip_param)
        self.num_learning_epochs = int(num_learning_epochs)
        self.num_mini_batches = int(num_mini_batches)
        self.value_loss_coef = float(value_loss_coef)
        self.entropy_coef = float(entropy_coef)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.max_grad_norm = float(max_grad_norm)
        self.use_clipped_value_loss = bool(use_clipped_value_loss)

    def _amp_ctx(self):
        """Autocast context for the update forward/loss (no-op when AMP is off)."""
        if self._amp_enabled:
            return torch.autocast(device_type="cuda", dtype=self._amp_dtype)
        return contextlib.nullcontext()

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ) -> None:
        self.storage = CSERolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def test_mode(self) -> None:
        self.actor_critic.eval()

    def train_mode(self) -> None:
        self.actor_critic.train()

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(
        self,
        next_obs: TensorDict | torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor | TensorDict],
    ) -> None:
        del next_obs
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Timeout (time-limit) bootstrap correction on the reward. The estimator
        # uses the same-step privileged target, so no next-obs buffer is needed.
        timeouts = extras.get("time_outs")
        timeout_bootstrap_obs = extras.get("time_out_bootstrap_obs")
        if isinstance(timeouts, torch.Tensor):
            timeout_bool = timeouts.to(self.device).bool().view(-1)
            timeout_mask = timeout_bool.float()
            if timeout_bootstrap_obs is not None and torch.count_nonzero(timeout_bool) > 0:
                bootstrap_obs = timeout_bootstrap_obs.to(self.device)
                bootstrap_critic_obs = _critic_obs(bootstrap_obs)
                bootstrap_values = self.actor_critic.evaluate(bootstrap_critic_obs).detach()
                correction = self.gamma * torch.squeeze(
                    bootstrap_values * timeout_mask.unsqueeze(1), 1
                )
                if self.transition.rewards.ndim == 2 and self.transition.rewards.shape[-1] == 1:
                    correction = correction.unsqueeze(1)
                self.transition.rewards += correction
            else:
                transition_values = self.transition.values
                assert transition_values is not None
                correction = self.gamma * torch.squeeze(
                    transition_values * timeout_mask.unsqueeze(1), 1
                )
                if self.transition.rewards.ndim == 2 and self.transition.rewards.shape[-1] == 1:
                    correction = correction.unsqueeze(1)
                self.transition.rewards += correction

        assert self.storage is not None
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        assert self.storage is not None
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def _adapt_learning_rate(self, kl_mean: float) -> None:
        """KL-adaptive learning-rate step, clamped to the configured bounds.

        The clamp floor (``min_learning_rate``) is configurable so late training
        does not stall at a hardcoded 1e-5 once the KL controller drives it down.
        """
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(self.min_learning_rate, self.learning_rate / 1.5)
        elif 0.0 < kl_mean < self.desired_kl / 2.0:
            self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.5)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def _clamp_policy_std(self) -> None:
        """Keep the action-noise std within ``[min_policy_std, max_policy_std]``.

        ``actor_critic.std`` is a raw, unconstrained Parameter used directly as
        the Normal scale; a low ``entropy_coef`` plus an aggressive update can
        drive it <= 0, which crashes ``Normal()`` ("std >= 0.0"). Clamp it to a
        small positive floor after every optimizer step. ``max_policy_std`` (when
        set) additionally caps the std from above: under a non-stationary
        disturbance the entropy bonus can out-pull a noisy advantage signal and
        walk the std up unboundedly (exploration runaway -> policy collapse); the
        ceiling halts that positive feedback. ``clamp_`` treats a ``None`` bound
        as unset, so a null ceiling preserves the floor-only behaviour.
        """
        self.actor_critic.std.data.clamp_(min=self.min_policy_std, max=self.max_policy_std)

    def update(self) -> tuple[float, float, float]:
        assert self.storage is not None
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_estimation_loss = 0.0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs,
        )

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            with self._amp_ctx():
                self.actor_critic.act(obs_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch)
            # Cast the AMP forward outputs back to fp32 so the KL / loss math below
            # runs in full precision (only the matmul-heavy forwards use bf16).
            actions_log_prob_batch = actions_log_prob_batch.float()
            value_batch = value_batch.float()
            mu_batch = self.actor_critic.action_mean.float()
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)
                self._adapt_learning_rate(float(kl_mean))

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self._clamp_policy_std()

            # Concurrent state-estimator update runs AFTER the PPO step, so the
            # adaptation module trains on the just-updated encoder. The estimator
            # owns its optimizer / fixed LR; its supervised target is the
            # same-step privileged block of critic_obs_batch. The actor path above
            # feeds a NON-detached latent, so the PPO step already co-trained the
            # encoder through the shared optimizer.
            estimation_loss = self.actor_critic.estimator.update(
                obs_batch,
                critic_obs_batch,
                autocast_enabled=self._amp_enabled,
                autocast_dtype=self._amp_dtype,
            )

            mean_value_loss += float(value_loss.item())
            mean_surrogate_loss += float(surrogate_loss.item())
            mean_estimation_loss += float(estimation_loss)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates
        self.storage.clear()
        return mean_value_loss, mean_surrogate_loss, mean_estimation_loss


def _critic_obs(obs: TensorDict | torch.Tensor) -> torch.Tensor:
    if isinstance(obs, TensorDict):
        for key in ("critic", "policy", "actor"):
            if key in obs.keys():
                return obs[key]
        raise KeyError("CSE-PPO TensorDict obs must contain critic, policy, or actor")
    return obs
