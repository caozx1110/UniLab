"""GH tracking domain-randomization provider (Phase 9.5).

Wraps the env's D2 reset backbone into the DomainRandomizationProvider contract so
the NpEnv autoreset lifecycle drives GH resets:

- ``build_reset_plan``   -> env._build_reset_plan_state (sample motion -> noised init
  qpos/qvel, component resets, boot-protect noised pose, student-cache 50-fill,
  per-reset DR sampling). Returns a ResetPlan the manager writes via set_state.
- ``build_reset_observation`` -> env._build_reset_observation (post set_state: obs
  history reset + compute for the reset envs).

Reset-payload backend randomization is left None here: GH's per-reset DR (batch-shared
joint zero offset -> action-manager offset, motor params, startup-once material/COM) is
applied env-side; PhysX friction/restitution have no direct MuJoCo slot (approximation,
blueprint §九) and are tracked as TODO(parity).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from unilab.dr.provider import DomainRandomizationProvider
from unilab.dr.types import DomainRandomizationCapabilities, ResetPlan

if TYPE_CHECKING:
    from unilab.envs.gh_tracking.env import GHTrackingEnv


class GHTrackingDRProvider(DomainRandomizationProvider):
    """DR provider for GHTrackingEnv — D2 reset order, student-cache fill, force origin."""

    def __init__(self, env: "GHTrackingEnv") -> None:
        self._env = env

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        # Reset-payload terms are capability-filtered by the DR manager; GH per-reset DR
        # is applied env-side in _build_reset_plan_state, so nothing to validate here.
        return None

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        qpos, qvel, info_updates = self._env._build_reset_plan_state(env_ids)
        return ResetPlan(
            env_ids=np.asarray(env_ids),
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=None,
        )

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        return self._env._build_reset_observation(env_ids)
