from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from unilab.algos.torch.sonic_ppo.model import SonicActorCritic
from unilab.base.np_env import NpEnvState
from unilab.tasks.motion_tracking.g1.sonic.observations import (
    SONIC_ACTOR_OBSERVATION_DIM,
    SONIC_CRITIC_OBSERVATION_DIM,
    SONIC_TOKENIZER_OBSERVATION_DIM,
    SonicManagerObservationAdapter,
    SonicTokenizerObservationCache,
)
from unilab.tasks.motion_tracking.g1.sonic.runner import SonicManagerPPORunner


@dataclass
class _FakeManagerEnv:
    """Synchronous ManagerBasedRlEnv-shaped fake with one timeout transition."""

    cache: SonicTokenizerObservationCache
    num_envs: int = 2
    step_count: int = 0
    state: NpEnvState | None = None

    def _obs(self, value: float) -> dict[str, np.ndarray]:
        return {
            "obs": np.full((self.num_envs, SONIC_ACTOR_OBSERVATION_DIM), value, dtype=np.float32),
            "critic": np.full(
                (self.num_envs, SONIC_CRITIC_OBSERVATION_DIM), value, dtype=np.float32
            ),
        }

    def _write_tokenizer(self, value: float) -> None:
        self.cache.write(
            np.full(
                (self.num_envs, SONIC_TOKENIZER_OBSERVATION_DIM), value, dtype=np.float32
            )
        )

    def reset(self) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        self.step_count = 0
        self._write_tokenizer(0.0)
        self.state = NpEnvState(
            obs=self._obs(0.0),
            reward=np.zeros(self.num_envs, dtype=np.float32),
            terminated=np.zeros(self.num_envs, dtype=bool),
            truncated=np.zeros(self.num_envs, dtype=bool),
            info={},
        )
        return self.state.obs, {}

    def step(self, actions: np.ndarray) -> NpEnvState:
        assert actions.shape == (self.num_envs, 29)
        self.step_count += 1
        next_value = float(self.step_count)
        self._write_tokenizer(next_value)
        timeout = np.array([self.step_count == 1, False])
        final_observation = (
            {
                "obs": np.full(
                    (self.num_envs, SONIC_ACTOR_OBSERVATION_DIM), 7.0, dtype=np.float32
                ),
                "critic": np.full(
                    (self.num_envs, SONIC_CRITIC_OBSERVATION_DIM), 7.0, dtype=np.float32
                ),
            }
            if timeout[0]
            else None
        )
        self.state = NpEnvState(
            obs=self._obs(next_value),
            reward=np.ones(self.num_envs, dtype=np.float32),
            terminated=np.zeros(self.num_envs, dtype=bool),
            truncated=timeout,
            info={},
            final_observation=final_observation,
        )
        return self.state


def test_manager_runner_rollout_timeout_bootstrap_and_ppo_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SonicTokenizerObservationCache(2, dtype=np.float32)
    env = _FakeManagerEnv(cache)
    model = SonicActorCritic(
        hidden_dims=(4,),
        tokenizer_hidden_dim=4,
        model_profile="dense_test",
    )
    original_value = model.value
    value_inputs: list[torch.Tensor] = []

    def critic_only_value(critic: torch.Tensor) -> torch.Tensor:
        value_inputs.append(critic.detach().clone())
        if torch.all(critic == 7.0):
            return torch.full((critic.shape[0],), 5.0, device=critic.device)
        return original_value(critic)

    monkeypatch.setattr(model, "value", critic_only_value)
    runner = SonicManagerPPORunner(
        env,
        SonicManagerObservationAdapter(cache, num_envs=2),
        model=model,
        config={"num_learning_epochs": 1, "num_mini_batches": 1, "gamma": 0.9},
        device="cpu",
        horizon=2,
    )

    metrics = runner.learn()

    assert runner.current_learning_iteration == 1
    assert env.step_count == 2
    assert metrics.keys() >= {"loss", "value_loss", "policy_loss"}
    assert runner.storage.rewards[0, 0].item() == pytest.approx(1.0 + 0.9 * 5.0)
    assert any(torch.all(values == 7.0) for values in value_inputs)


def test_manager_runner_loads_nested_and_legacy_optimizer_checkpoint() -> None:
    cache = SonicTokenizerObservationCache(2, dtype=np.float32)
    runner = SonicManagerPPORunner(
        _FakeManagerEnv(cache),
        SonicManagerObservationAdapter(cache, num_envs=2),
        model=SonicActorCritic(hidden_dims=(4,), tokenizer_hidden_dim=4, model_profile="dense_test"),
        config={"num_learning_epochs": 1, "num_mini_batches": 1},
        device="cpu",
        horizon=2,
    )
    checkpoint = runner.state_dict()
    nested_optimizer = checkpoint["algorithm"]["optimizer"]
    nested_optimizer["param_groups"][0]["lr"] = 0.123
    runner.algorithm.optimizer.param_groups[0]["lr"] = 0.456

    runner.load_state_dict(checkpoint)
    assert runner.algorithm.optimizer.param_groups[0]["lr"] == pytest.approx(0.123)

    legacy_algorithm = dict(checkpoint["algorithm"])
    legacy_optimizer = legacy_algorithm.pop("optimizer")
    legacy_checkpoint = dict(checkpoint)
    legacy_checkpoint["algorithm"] = legacy_algorithm
    legacy_checkpoint["optimizer"] = legacy_optimizer
    runner.algorithm.optimizer.param_groups[0]["lr"] = 0.456

    runner.load_state_dict(legacy_checkpoint)
    assert runner.algorithm.optimizer.param_groups[0]["lr"] == pytest.approx(0.123)
