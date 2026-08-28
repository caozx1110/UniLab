from __future__ import annotations

import numpy as np
import pytest

from unilab.base.np_env import NpEnvState
from unilab.tasks.motion_tracking.g1.sonic.observations import (
    SONIC_ACTOR_OBSERVATION_DIM,
    SONIC_CRITIC_OBSERVATION_DIM,
    SONIC_TOKENIZER_OBSERVATION_DIM,
    SonicManagerObservationAdapter,
    SonicTokenizerObservationCache,
)


def _state(num_envs: int) -> NpEnvState:
    return NpEnvState(
        obs={
            "obs": np.zeros((num_envs, SONIC_ACTOR_OBSERVATION_DIM), dtype=np.float32),
            "critic": np.zeros((num_envs, SONIC_CRITIC_OBSERVATION_DIM), dtype=np.float32),
        },
        reward=np.zeros(num_envs, dtype=np.float32),
        terminated=np.zeros(num_envs, dtype=bool),
        truncated=np.zeros(num_envs, dtype=bool),
        info={},
    )


def test_adapter_reads_only_public_manager_groups_and_task_tokenizer_cache() -> None:
    num_envs = 3
    tokenizer = np.arange(
        num_envs * SONIC_TOKENIZER_OBSERVATION_DIM, dtype=np.float32
    ).reshape(num_envs, SONIC_TOKENIZER_OBSERVATION_DIM)
    cache = SonicTokenizerObservationCache(num_envs, dtype=np.float32)
    cache.write(tokenizer)

    state = _state(num_envs)
    batch = SonicManagerObservationAdapter(cache, num_envs=num_envs).adapt(state)

    assert batch.actor is state.obs["obs"]
    assert batch.critic is state.obs["critic"]
    assert batch.tokenizer is tokenizer or np.array_equal(batch.tokenizer, tokenizer)
    assert set(state.obs) == {"obs", "critic"}


def test_adapter_fails_closed_for_unavailable_tokenizer_rows_and_extra_public_groups() -> None:
    cache = SonicTokenizerObservationCache(2, dtype=np.float32)
    cache.write(
        np.zeros((1, SONIC_TOKENIZER_OBSERVATION_DIM), dtype=np.float32),
        env_ids=np.array([1], dtype=np.int32),
    )
    adapter = SonicManagerObservationAdapter(cache, num_envs=2)
    with pytest.raises(RuntimeError, match="not available"):
        adapter.adapt(_state(2))

    cache.write(np.zeros((2, SONIC_TOKENIZER_OBSERVATION_DIM), dtype=np.float32))
    state = _state(2)
    state.obs["tokenizer"] = np.zeros((2, SONIC_TOKENIZER_OBSERVATION_DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="exactly public keys"):
        adapter.adapt(state)
