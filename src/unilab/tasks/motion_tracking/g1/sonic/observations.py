"""Task-local SONIC observation ABI for the Manager-Based runtime.

The generic :class:`~unilab.base.np_env.NpEnvState` contract intentionally
exports only ``obs`` and ``critic``.  SONIC has a third, tokenizer-only input
which is neither a generic policy group nor an IPC payload.  This module owns
the explicit task-local handoff: a future SONIC observation term writes its
1761D result to :class:`SonicTokenizerObservationCache`, and the SONIC PPO
adapter reads that public provider together with the two generic groups.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from unilab.base.np_env import NpEnvState

SONIC_ACTOR_OBSERVATION_DIM = 930
SONIC_CRITIC_OBSERVATION_DIM = 1645
SONIC_TOKENIZER_OBSERVATION_DIM = 1761
_PUBLIC_MANAGER_OBSERVATION_KEYS = frozenset(("obs", "critic"))


def _validate_batch_matrix(
    value: np.ndarray,
    *,
    name: str,
    rows: int,
    width: int,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"SONIC {name} must be a numpy array, got {type(value).__name__}")
    if value.shape != (rows, width):
        raise ValueError(
            f"SONIC {name} shape must be {(rows, width)}, got {tuple(value.shape)}"
        )
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"SONIC {name} must use a floating dtype, got {value.dtype}")
    return value


class SonicTokenizerObservationProvider(ABC):
    """Public task-owned source of the current tokenizer observation batch."""

    @abstractmethod
    def get_tokenizer_observations(self) -> np.ndarray:
        """Return the current full ``(num_envs, 1761)`` tokenizer batch."""


class SonicTokenizerObservationCache(SonicTokenizerObservationProvider):
    """Preallocated tokenizer-term cache with explicit row validity.

    A manager term may update all rows after a step or only reset rows.  Reads
    fail closed until every row has been populated, avoiding accidental use of
    stale or uninitialized values.  The cache does no asset or backend access;
    callers must provide already-computed numeric observations.
    """

    def __init__(self, num_envs: int, *, dtype: np.dtype | type[np.floating]) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError(f"SONIC tokenizer cache num_envs must be positive, got {num_envs!r}")
        resolved_dtype = np.dtype(dtype)
        if not np.issubdtype(resolved_dtype, np.floating):
            raise TypeError(f"SONIC tokenizer cache dtype must be floating, got {resolved_dtype}")
        self._num_envs = num_envs
        self._dtype = resolved_dtype
        self._values = np.empty((num_envs, SONIC_TOKENIZER_OBSERVATION_DIM), dtype=resolved_dtype)
        self._valid_rows = np.zeros(num_envs, dtype=bool)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def write(self, values: np.ndarray, *, env_ids: np.ndarray | None = None) -> None:
        """Store full-batch or positionally ordered reset-row tokenizer values."""

        if env_ids is None:
            source = _validate_batch_matrix(
                values,
                name="tokenizer observations",
                rows=self._num_envs,
                width=SONIC_TOKENIZER_OBSERVATION_DIM,
            )
            if source.dtype != self._dtype:
                raise TypeError(
                    f"SONIC tokenizer observations dtype must be {self._dtype}, got {source.dtype}"
                )
            self._values[...] = source
            self._valid_rows.fill(True)
            return

        ids = np.asarray(env_ids)
        if (
            ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer)
            or np.issubdtype(ids.dtype, np.bool_)
        ):
            raise TypeError("SONIC tokenizer cache env_ids must be a one-dimensional integer array")
        ids = ids.astype(np.intp, copy=False)
        if np.any(ids < 0) or np.any(ids >= self._num_envs):
            raise IndexError(f"SONIC tokenizer cache env_ids out of range: {ids.tolist()}")
        if np.unique(ids).size != ids.size:
            raise ValueError(f"SONIC tokenizer cache env_ids contain duplicates: {ids.tolist()}")
        source = _validate_batch_matrix(
            values,
            name="tokenizer reset observations",
            rows=len(ids),
            width=SONIC_TOKENIZER_OBSERVATION_DIM,
        )
        if source.dtype != self._dtype:
            raise TypeError(
                f"SONIC tokenizer observations dtype must be {self._dtype}, got {source.dtype}"
            )
        self._values[ids] = source
        self._valid_rows[ids] = True

    def get_tokenizer_observations(self) -> np.ndarray:
        if not bool(np.all(self._valid_rows)):
            missing = np.flatnonzero(~self._valid_rows)
            raise RuntimeError(
                "SONIC tokenizer observations are not available for all environments; "
                f"missing={missing[:10].tolist()}"
            )
        return self._values


@dataclass(frozen=True)
class SonicObservationBatch:
    """Typed three-group input accepted by the task-owned SONIC PPO adapter."""

    actor: np.ndarray
    critic: np.ndarray
    tokenizer: np.ndarray


class SonicManagerObservationAdapter:
    """Join generic manager observations with the task-owned tokenizer provider.

    This is purposefully not a generic environment wrapper: it reads the
    documented ``NpEnvState.obs`` mapping and a typed task provider only.  In
    particular, it never accesses ``ObservationManager._obs_buffer``.
    """

    def __init__(self, tokenizer_provider: SonicTokenizerObservationProvider, *, num_envs: int):
        if not isinstance(tokenizer_provider, SonicTokenizerObservationProvider):
            raise TypeError(
                "SONIC tokenizer_provider must implement SonicTokenizerObservationProvider"
            )
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError(f"SONIC observation adapter num_envs must be positive, got {num_envs!r}")
        self._tokenizer_provider = tokenizer_provider
        self._num_envs = num_envs

    @property
    def num_envs(self) -> int:
        """Number of rows expected from the manager and tokenizer provider."""

        return self._num_envs

    def adapt(self, state: NpEnvState) -> SonicObservationBatch:
        """Return zero-copy actor/critic/tokenizer arrays after ABI validation."""

        if not isinstance(state, NpEnvState):
            raise TypeError(f"SONIC observation adapter requires NpEnvState, got {type(state).__name__}")
        if set(state.obs) != _PUBLIC_MANAGER_OBSERVATION_KEYS:
            raise ValueError(
                "SONIC manager observation ABI requires exactly public keys "
                f"{sorted(_PUBLIC_MANAGER_OBSERVATION_KEYS)}, got {sorted(state.obs)}"
            )
        actor = _validate_batch_matrix(
            state.obs["obs"],
            name="actor observations",
            rows=self._num_envs,
            width=SONIC_ACTOR_OBSERVATION_DIM,
        )
        critic = _validate_batch_matrix(
            state.obs["critic"],
            name="critic observations",
            rows=self._num_envs,
            width=SONIC_CRITIC_OBSERVATION_DIM,
        )
        tokenizer = _validate_batch_matrix(
            self._tokenizer_provider.get_tokenizer_observations(),
            name="tokenizer observations",
            rows=self._num_envs,
            width=SONIC_TOKENIZER_OBSERVATION_DIM,
        )
        return SonicObservationBatch(actor=actor, critic=critic, tokenizer=tokenizer)


__all__ = [
    "SONIC_ACTOR_OBSERVATION_DIM",
    "SONIC_CRITIC_OBSERVATION_DIM",
    "SONIC_TOKENIZER_OBSERVATION_DIM",
    "SonicManagerObservationAdapter",
    "SonicObservationBatch",
    "SonicTokenizerObservationCache",
    "SonicTokenizerObservationProvider",
]
