"""Minimal synchronous SONIC PPO runner for a Manager-Based owner.

This adapter deliberately reuses the existing :class:`NpEnvState` lifecycle:
the manager owns reset, stepping, autoreset, and terminal observations while
this class only performs policy inference, rollout storage, timeout critic
bootstrap, and one PPO update.  It does not create collectors, queues, or a
second learner protocol.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from unilab.algos.torch.sonic_ppo.algorithm import SonicPPO
from unilab.algos.torch.sonic_ppo.model import SonicActorCritic
from unilab.algos.torch.sonic_ppo.storage import SonicRolloutStorage
from unilab.base.np_env import NpEnvState

from .observations import SonicManagerObservationAdapter, SonicObservationBatch


class ManagerBasedSonicEnv(Protocol):
    """Subset of the existing synchronous ``NpEnv`` lifecycle SONIC consumes."""

    @property
    def state(self) -> NpEnvState | None:
        """Current state published by the regular ``NpEnv`` lifecycle."""

    def reset(self) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        """Reset the manager environment using its standard public contract."""

    def step(self, actions: np.ndarray) -> NpEnvState:
        """Advance the manager environment using its standard public contract."""


class SonicManagerPPORunner:
    """Synchronous SONIC PPO loop consuming a typed manager observation batch."""

    def __init__(
        self,
        env: ManagerBasedSonicEnv,
        observation_adapter: SonicManagerObservationAdapter,
        *,
        model: SonicActorCritic | None = None,
        algorithm: SonicPPO | None = None,
        config: Mapping[str, Any] | None = None,
        device: str | torch.device = "cpu",
        horizon: int = 24,
    ) -> None:
        if not isinstance(observation_adapter, SonicManagerObservationAdapter):
            raise TypeError("SONIC manager runner requires SonicManagerObservationAdapter")
        if horizon < 1:
            raise ValueError(f"SONIC PPO horizon must be positive, got {horizon}")
        self.env: ManagerBasedSonicEnv = env
        self.observation_adapter = observation_adapter
        # The DP launcher establishes the process group and passes its
        # rank-local device (for example ``cuda:LOCAL_RANK``).  This runner
        # neither discovers physical GPU IDs nor creates a second lifecycle.
        self.device = torch.device(device)
        self.num_envs = int(observation_adapter.num_envs)
        self.horizon = int(horizon)
        cfg = dict(config or {})
        if model is None:
            model = SonicActorCritic(
                actor_obs_dim=930,
                critic_obs_dim=1645,
                tokenizer_obs_dim=1761,
                action_dim=29,
                hidden_dims=cfg.get("hidden_dims"),
                actor_hidden_dims=cfg.get("actor_hidden_dims"),
                critic_hidden_dims=cfg.get("critic_hidden_dims"),
                model_profile=str(cfg.get("model_profile", "dense_test")),
                tokenizer_hidden_dim=int(cfg.get("tokenizer_hidden_dim", 512)),
                encoder_hidden_dims=cfg.get("encoder_hidden_dims"),
                kinematic_hidden_dims=cfg.get("kinematic_hidden_dims"),
                tokenizer_fields=cfg.get("tokenizer_fields"),
                encoders=cfg.get("encoders"),
                decoders=cfg.get("decoders"),
                token_levels=int(cfg.get("token_levels", 32)),
                token_count=int(cfg.get("token_count", 2)),
                critic_obs_normalization=bool(cfg.get("critic_obs_normalization", False)),
                init_noise_std=float(cfg.get("init_noise_std", 0.05)),
                std_clamp_min=float(cfg.get("std_clamp_min", 0.001)),
                std_clamp_max=float(cfg.get("std_clamp_max", 0.5)),
            )
        if (
            model.actor_obs_dim != 930
            or model.critic_obs_dim != 1645
            or model.tokenizer_obs_dim != 1761
            or model.action_dim != 29
        ):
            raise ValueError("SONIC manager runner requires the 930/1645/1761/29 model ABI")
        self.model = model.to(self.device)
        self.algorithm = algorithm or SonicPPO(self.model, cfg, self.device)
        self.storage = SonicRolloutStorage(
            self.horizon,
            self.num_envs,
            930,
            1645,
            1761,
            29,
            self.device,
        )
        self.current_learning_iteration = 0
        self._actor_obs: torch.Tensor | None = None
        self._critic_obs: torch.Tensor | None = None
        self._tokenizer_obs: torch.Tensor | None = None

    def _reset_state(self) -> NpEnvState:
        self.env.reset()
        state = self.env.state
        if not isinstance(state, NpEnvState):
            raise RuntimeError("SONIC manager environment did not publish NpEnvState after reset")
        return state

    def _batch_to_torch(self, batch: SonicObservationBatch) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.as_tensor(value, device=self.device, dtype=torch.float32)
            for value in (batch.actor, batch.critic, batch.tokenizer)
        )

    def _synchronize_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def reset_rollout_state(self) -> None:
        """Reset the ordinary environment lifecycle before the next iteration."""

        state = self._reset_state()
        batch = self.observation_adapter.adapt(state)
        self._actor_obs, self._critic_obs, self._tokenizer_obs = self._batch_to_torch(batch)

    def state_dict(self) -> dict[str, Any]:
        """Return the checkpointed learner state at an iteration boundary."""

        if self.storage.step != 0:
            raise RuntimeError(
                "SONIC checkpoints require an empty rollout at an iteration boundary"
            )
        return {
            "model": self.model.state_dict(),
            "algorithm": self.algorithm.state_dict(),
            "iteration": self.current_learning_iteration,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore model, optimizer, and completed iteration count."""

        model_state = state.get("model")
        algorithm_state = state.get("algorithm")
        if not isinstance(model_state, Mapping) or not isinstance(algorithm_state, Mapping):
            raise ValueError("SONIC checkpoint requires model and algorithm mappings")
        self.model.load_state_dict(model_state, strict=True)
        self.algorithm.load_state_dict(algorithm_state)
        self.current_learning_iteration = int(state.get("iteration", 0))
        self.storage.clear()
        # Simulator and observation histories are intentionally reconstructed
        # through their existing reset lifecycle after a warm-start restore.
        self._actor_obs = None
        self._critic_obs = None
        self._tokenizer_obs = None

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically save one complete iteration-boundary checkpoint."""

        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_name(f".{checkpoint.name}.tmp-{os.getpid()}")
        try:
            torch.save(self.state_dict(), temporary)
            os.replace(temporary, checkpoint)
        finally:
            if temporary.exists():
                temporary.unlink()

    def load(self, path: str | os.PathLike[str]) -> None:
        """Restore a checkpoint onto this runner's rank-local learner device."""

        checkpoint = Path(path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SONIC checkpoint does not exist: {checkpoint}")
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if not isinstance(state, Mapping):
            raise ValueError("SONIC checkpoint root must be a mapping")
        self.load_state_dict(state)

    def _timeout_correction(
        self,
        state: NpEnvState,
        current_values: torch.Tensor,
    ) -> torch.Tensor:
        timeout = np.asarray(state.truncated, dtype=bool).reshape(-1) & ~np.asarray(
            state.terminated, dtype=bool
        ).reshape(-1)
        if timeout.shape != (self.num_envs,):
            raise ValueError(
                f"SONIC timeout mask expects {self.num_envs} values, got {timeout.shape}"
            )
        correction = torch.zeros(self.num_envs, device=self.device, dtype=current_values.dtype)
        if not np.any(timeout):
            return correction
        final = state.final_observation
        final_critic = final.get("critic") if isinstance(final, Mapping) else None
        if final_critic is None and isinstance(state.info, Mapping):
            info_final = state.info.get("final_observation")
            final_critic = info_final.get("critic") if isinstance(info_final, Mapping) else None
        if final_critic is None:
            correction[torch.as_tensor(timeout, device=self.device)] = (
                self.algorithm.gamma
                * current_values.detach()[torch.as_tensor(timeout, device=self.device)]
            )
            return correction
        rows = np.flatnonzero(timeout)
        values = np.asarray(final_critic)
        if values.ndim != 2 or values.shape[-1] != 1645:
            raise ValueError("SONIC timeout final_observation must contain critic shape (*, 1645)")
        if values.shape[0] == self.num_envs:
            values = values[rows]
        elif values.shape[0] != len(rows):
            raise ValueError("SONIC timeout final critic rows do not match timeout mask")
        with torch.no_grad():
            bootstrap = self.model.value(
                torch.as_tensor(values, device=self.device, dtype=torch.float32)
            )
        correction[torch.as_tensor(rows, device=self.device)] = (
            self.algorithm.gamma * bootstrap.detach()
        )
        return correction

    def learn(self, num_learning_iterations: int = 1) -> dict[str, float]:
        if num_learning_iterations < 1:
            raise ValueError("SONIC manager runner requires at least one learning iteration")
        if self._actor_obs is None:
            self.reset_rollout_state()
        actor_obs = self._actor_obs
        critic_obs = self._critic_obs
        tokenizer_obs = self._tokenizer_obs
        assert actor_obs is not None and critic_obs is not None and tokenizer_obs is not None
        metrics: dict[str, float] = {}
        for _ in range(int(num_learning_iterations)):
            self._synchronize_device()
            iteration_start = time.perf_counter()
            collection_start = iteration_start
            env_step_seconds = 0.0
            self.model.eval()
            with torch.no_grad():
                for _step in range(self.horizon):
                    rollout_actor, rollout_critic, rollout_token = (
                        actor_obs.clone(),
                        critic_obs.clone(),
                        tokenizer_obs.clone(),
                    )
                    distribution, values = self.model.distribution(
                        rollout_actor, rollout_critic, rollout_token
                    )
                    action = distribution.sample()
                    log_prob = distribution.log_prob(action).sum(-1)
                    env_step_start = time.perf_counter()
                    state = self.env.step(action.detach().cpu().numpy())
                    env_step_seconds += time.perf_counter() - env_step_start
                    if not isinstance(state, NpEnvState):
                        raise TypeError("SONIC manager environment step must return NpEnvState")
                    rewards = torch.as_tensor(
                        state.reward, device=self.device, dtype=torch.float32
                    ).reshape(-1)
                    if rewards.shape != (self.num_envs,):
                        raise ValueError("SONIC manager rewards must be a (num_envs,) vector")
                    rewards = rewards + self._timeout_correction(state, values)
                    dones = torch.as_tensor(
                        np.asarray(state.terminated, dtype=bool)
                        | np.asarray(state.truncated, dtype=bool),
                        device=self.device,
                    )
                    self.storage.add(
                        rollout_actor,
                        rollout_critic,
                        rollout_token,
                        action,
                        rewards,
                        dones,
                        values,
                        log_prob,
                        distribution.mean,
                        distribution.stddev,
                    )
                    next_batch = self.observation_adapter.adapt(state)
                    actor_obs, critic_obs, tokenizer_obs = self._batch_to_torch(next_batch)
                with torch.no_grad():
                    last_values = self.model.value(critic_obs)
            self._synchronize_device()
            collection_seconds = time.perf_counter() - collection_start
            learning_start = time.perf_counter()
            self.storage.compute_returns(last_values, self.algorithm.gamma, self.algorithm.lam)
            self.model.train()
            metrics = self.algorithm.update(self.storage)
            self._synchronize_device()
            learning_seconds = time.perf_counter() - learning_start
            self.current_learning_iteration += 1
            iteration_seconds = time.perf_counter() - iteration_start
            transitions = self.num_envs * self.horizon
            metrics.update(
                {
                    "time/collection_s": collection_seconds,
                    "time/env_step_s": env_step_seconds,
                    "time/learning_s": learning_seconds,
                    "time/iteration_s": iteration_seconds,
                    "throughput/collection_env_steps_s": transitions
                    / max(collection_seconds, 1.0e-12),
                    "throughput/iteration_env_steps_s": transitions
                    / max(iteration_seconds, 1.0e-12),
                }
            )
        self._actor_obs = actor_obs
        self._critic_obs = critic_obs
        self._tokenizer_obs = tokenizer_obs
        return metrics


__all__ = ["ManagerBasedSonicEnv", "SonicManagerPPORunner"]
