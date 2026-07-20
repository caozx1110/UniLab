"""GHDistillPolicy — the 5-network policy container (GHDistillPPO, Phase 10.6).

Owns the 5 learnable networks as ``named_children`` (so the GH checkpoint schema can
save/load them by child name): ``encoder_priv``, ``adapt_module``, ``actor_teacher``,
``actor_student``, ``critic`` (GH ppo.py:128-166). Provides phase-aware rollout
(``get_rollout_policy`` / ``act``, GH ppo.py:270-282), critic evaluation, and the two
latent producers used by the reg/estimator alignment (GH ppo.py:336-379).

Input dims (GH): encoder_priv 717→256; adapt_module 450→256; actors cat(policy 450,
latent 256)=706→(loc 29, scale 29); critic cat(policy 450, priv 717, priv_critic 3)=1170→1.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from unilab.algos.gh_distill_ppo.distributions import IndependentNormal
from unilab.algos.gh_distill_ppo.networks import (
    build_actor,
    build_adapt_module,
    build_critic,
    build_encoder_priv,
    init_orthogonal,
)

_DEFAULT_OBS_DIMS = {"policy": 450, "priv": 717, "priv_critic": 3}


class GHDistillPolicy(nn.Module):
    def __init__(
        self,
        obs_dims: dict[str, int] | None = None,
        action_dim: int = 29,
        latent_dim: int = 256,
        init_noise_scale: float = 1.0,
    ) -> None:
        super().__init__()
        dims = dict(obs_dims or _DEFAULT_OBS_DIMS)
        self.obs_dims = dims
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)

        # 5 learnable networks registered as named_children (checkpoint keys)
        self.encoder_priv = build_encoder_priv(latent_dim)
        self.adapt_module = build_adapt_module(latent_dim)
        self.actor_teacher = build_actor(dims["policy"] + latent_dim, action_dim, init_noise_scale)
        self.actor_student = build_actor(dims["policy"] + latent_dim, action_dim, init_noise_scale)
        self.critic = build_critic(dims["policy"] + dims["priv"] + dims["priv_critic"])

        self._materialize(dims)          # realize LazyLinear params before init/checkpoint
        for net in (self.encoder_priv, self.adapt_module, self.actor_teacher,
                    self.actor_student, self.critic):
            init_orthogonal(net, gain=0.01)  # all Linear orthogonal gain=0.01 (GH ppo.py:178-183)

    def _materialize(self, dims: dict[str, int]) -> None:
        with torch.no_grad():
            p = torch.zeros(1, dims["policy"])
            pv = torch.zeros(1, dims["priv"])
            pc = torch.zeros(1, dims["priv_critic"])
            lt = self.encoder_priv(pv)
            ls = self.adapt_module(p)
            self.actor_teacher(torch.cat([p, lt], dim=-1))
            self.actor_student(torch.cat([p, ls], dim=-1))
            self.critic(torch.cat([p, pv, pc], dim=-1))

    # --- latents (reg/estimator alignment) --- #
    def latent_teacher(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """priv_feature = encoder_priv(priv) (GH: teacher latent, encoder trainable)."""
        return self.encoder_priv(obs["priv"])

    def latent_student(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """priv_pred = adapt_module(policy) (GH: student estimator latent)."""
        return self.adapt_module(obs["policy"])

    # --- actor / critic forward --- #
    def _actor_forward(self, actor: nn.Module, obs, latent):
        loc, scale = actor(torch.cat([obs["policy"], latent], dim=-1))
        return loc, scale

    def act(self, obs: dict[str, torch.Tensor], phase: str, sample: bool = True):
        """Return (action, loc, scale, log_prob) for the phase's rollout policy.

        train -> encoder_priv + actor_teacher; adapt/finetune -> adapt_module + actor_student.
        """
        if phase == "train":
            latent, actor = self.latent_teacher(obs), self.actor_teacher
        elif phase in ("adapt", "finetune"):
            latent, actor = self.latent_student(obs), self.actor_student
        else:
            raise ValueError(f"unknown phase {phase!r}")
        loc, scale = self._actor_forward(actor, obs, latent)
        dist = IndependentNormal(loc, scale)
        action = dist.sample() if sample else dist.deterministic_sample
        return action, loc, scale, dist.log_prob(action)

    def get_rollout_policy(self, phase: str):
        """Callable obs-dict -> action for the given phase (GH get_rollout_policy).

        ``eval`` maps to the finetune policy (GH: get_rollout_policy ignores mode and
        branches on phase; finetune eval = Seq(adapt_module, actor_student))."""
        rollout_phase = "finetune" if phase == "eval" else phase
        sample = phase != "eval"  # eval uses the deterministic (mean) action

        def policy(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            return self.act(obs, rollout_phase, sample=sample)[0]

        return policy

    def evaluate_critic(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """state_value over cat(policy, priv, priv_critic) = 1170 (GH critic)."""
        return self.critic(torch.cat([obs["policy"], obs["priv"], obs["priv_critic"]], dim=-1))
