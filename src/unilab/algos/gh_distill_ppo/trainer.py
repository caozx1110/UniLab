"""Phase-aware GHDistillPPO training step (Phase 10.8).

Wires Phase-8 losses/GAE/optimizers + the GHDistillPolicy + GH symmetry transforms into
GH's 3-phase ``train_op`` (ppo.py:292-462):

- **train**: ``_ppo_update(update_teacher)`` (GAE -> modewise adv -> 5x8 teacher _update) then
  ``train_estimator`` (2x8). Trainable: encoder_priv, actor_teacher, critic (PPO) + adapt_module
  (estimator).
- **adapt**: ``train_estimator`` only (2x8). Trainable: adapt_module. No GAE/PPO/KL.
- **finetune**: if ``progress>0.025`` -> ``_ppo_update(update_student)`` (adapt_module as encoder,
  reg=0). Trainable: actor_student, adapt_module, critic. Else no update.

Per-minibatch _update (ppo.py:336-427): symmetry-cat to 2B -> encoder+actor forward ->
surrogate/entropy on [:B] + critic/reg on 2B (valid-masked) + symmetry loss -> zero_grad
(opt_actor, opt_critic) -> backward -> grad-clip 1.0 (actor, critic, encoder) -> step. The KL
schedule runs once after all minibatches and only touches opt_teacher/opt_student.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from unilab.algos.gh_distill_ppo.distributions import IndependentNormal
from unilab.algos.gh_distill_ppo.gae import (
    adaptive_kl_lr_schedule,
    compute_gae_returns,
    normalize_advantages_modewise,
)
from unilab.algos.gh_distill_ppo.gh_symmetry import build_gh_symmetry
from unilab.algos.gh_distill_ppo.losses import (
    ppo_entropy_loss,
    ppo_surrogate_loss,
    symmetry_loss,
)
from unilab.algos.gh_distill_ppo.optimizers import (
    apply_lr_to_teacher_student,
    build_optimizers,
)
from unilab.algos.gh_distill_ppo.policy import GHDistillPolicy

_OBS_GROUPS = ("policy", "priv", "priv_critic")


@dataclass
class GHDistillTrainerCfg:
    lr: float = 3e-4
    ppo_epochs: int = 5
    num_minibatches: int = 8
    estimator_epochs: int = 2
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    lr_min: float = 1e-5
    lr_max: float = 5e-3
    entropy_start: float = 0.005
    entropy_end: float = 0.002
    reg_lambda_max: float = 0.2


class GHDistillTrainer:
    def __init__(self, policy: GHDistillPolicy, cfg: GHDistillTrainerCfg | None = None) -> None:
        self.policy = policy
        self.cfg = cfg or GHDistillTrainerCfg()
        self.opt_teacher, self.opt_student, self.opt_critic, self.opt_estimator = build_optimizers(
            policy.encoder_priv, policy.adapt_module, policy.actor_teacher,
            policy.actor_student, policy.critic, lr=self.cfg.lr,
        )
        self.sym = build_gh_symmetry()
        self.act_transform = self.sym["action"]
        self.current_lr = self.cfg.lr
        self.entropy_coef = self.cfg.entropy_start
        self.reg_lambda = 0.0
        self.progress = 0.0
        self.num_updates = 0

    # --- schedules (GH ppo.py:285-289) --- #
    def step_schedule(self, progress: float) -> None:
        c = self.cfg
        self.progress = float(progress)
        self.entropy_coef = c.entropy_start * (c.entropy_end / c.entropy_start) ** progress
        self.reg_lambda = progress * c.reg_lambda_max

    # --- dispatch (GH train_op ppo.py:292-306) --- #
    def train_op(self, rollout: dict, phase: str, progress: float) -> dict:
        info: dict = {}
        if phase == "train":
            info.update(self._ppo_update(rollout, "train", progress))
            info.update(self._train_estimator(rollout))
        elif phase == "finetune":
            if progress > 0.025:
                info.update(self._ppo_update(rollout, "finetune", progress))
        elif phase == "adapt":
            info.update(self._train_estimator(rollout))
        else:
            raise ValueError(f"unknown phase {phase!r}")
        self.num_updates += 1
        return info

    # --- PPO (GH _ppo_update ppo.py:307-334) --- #
    def _ppo_update(self, rollout: dict, phase: str, progress: float) -> dict:
        flat = self._compute_advantage(rollout)
        kls = []
        for _ in range(self.cfg.ppo_epochs):
            for mb in self._minibatches(flat):
                kls.append(self._update(mb, phase))
        kl = float(torch.stack(kls).mean())
        # KL lr schedule — only opt_teacher/opt_student (GH ppo.py:243-265)
        self.current_lr = adaptive_kl_lr_schedule(
            kl, self.cfg.desired_kl, self.current_lr, progress, self.cfg.lr_min, self.cfg.lr_max)
        apply_lr_to_teacher_student(
            self.opt_teacher, self.opt_student, self.opt_critic, self.opt_estimator, self.current_lr)
        return {"kl": kl, "lr": self.current_lr}

    def _compute_advantage(self, rollout: dict) -> dict:
        """GAE + modewise adv (GH _compute_advantage/_modewise_adv_norm). rollout groups
        are [T,N,·]; critic values computed here (no_grad)."""
        T, N = rollout["reward"].shape
        with torch.no_grad():
            obs_flat = {g: rollout[g].reshape(T * N, -1) for g in _OBS_GROUPS}
            values = self.policy.evaluate_critic(obs_flat).reshape(T, N)
        next_values = torch.cat([values[1:], values[-1:]], dim=0)  # bootstrap last with itself
        ret_nt, adv_nt = compute_gae_returns(
            rollout["reward"].transpose(0, 1), values.transpose(0, 1),
            rollout["done"].float().transpose(0, 1), next_values.transpose(0, 1),
            self.cfg.gamma, self.cfg.lam,
        )
        ret = ret_nt.transpose(0, 1).reshape(T * N)
        adv = adv_nt.transpose(0, 1).reshape(T * N)
        is_init = rollout["is_init"].reshape(T * N)
        adv = normalize_advantages_modewise(adv, is_init)
        flat = {g: rollout[g].reshape(T * N, -1) for g in _OBS_GROUPS}
        flat.update(
            action=rollout["action"].reshape(T * N, -1),
            loc=rollout["loc"].reshape(T * N, -1),
            scale=rollout["scale"].reshape(T * N, -1),
            sample_log_prob=rollout["sample_log_prob"].reshape(T * N),
            adv=adv, ret=ret, is_init=is_init,
        )
        return flat

    def _minibatches(self, flat: dict):
        M = flat["adv"].shape[0]
        perm = torch.randperm(M)
        for idx in perm.chunk(self.cfg.num_minibatches):
            yield {k: v[idx] for k, v in flat.items()}

    def _update(self, mb: dict, phase: str) -> torch.Tensor:
        """One teacher/student PPO minibatch (GH _update ppo.py:336-427)."""
        teacher = phase == "train"
        actor = self.policy.actor_teacher if teacher else self.policy.actor_student
        encoder = self.policy.encoder_priv if teacher else self.policy.adapt_module
        opt_actor = self.opt_teacher if teacher else self.opt_student

        b = mb["adv"].shape[0]
        action_old, logp_old = mb["action"], mb["sample_log_prob"]

        # symmetry augmentation -> 2B (GH ppo.py:341-353)
        # Move sym transforms to same device as obs (trainer is not nn.Module, sym dict won't auto-follow .to())
        device = mb[_OBS_GROUPS[0]].device
        obs2 = {g: torch.cat([mb[g], self.sym[g].to(device)(mb[g])], dim=0) for g in _OBS_GROUPS}
        adv2 = torch.cat([mb["adv"], mb["adv"]], dim=0)
        ret2 = torch.cat([mb["ret"], mb["ret"]], dim=0)
        valid2 = ~torch.cat([mb["is_init"], mb["is_init"]], dim=0)

        # forward: encoder (grad) -> latent; actor -> loc/scale on 2B
        latent = encoder(obs2["priv"]) if teacher else encoder(obs2["policy"])
        loc, scale = actor(torch.cat([obs2["policy"], latent], dim=-1))

        # PPO surrogate + entropy on [:B] (GH ppo.py:361-373)
        dist = IndependentNormal(loc[:b], scale[:b])
        logp = dist.log_prob(action_old)
        policy_loss = ppo_surrogate_loss(logp_old, logp, adv2[:b], valid2[:b], self.cfg.clip_param)
        entropy_loss = ppo_entropy_loss(loc, scale, self.entropy_coef)

        # critic on 2B, valid-masked (GH ppo.py:375-377)
        values = self.policy.evaluate_critic(obs2)[:, 0]
        value_loss = (F.mse_loss(values, ret2, reduction="none") * valid2).mean()

        # reg (train only): grad to encoder_priv via priv_feature; adapt_module frozen (GH:379-386)
        if teacher:
            with torch.no_grad():
                priv_pred = self.policy.adapt_module(obs2["policy"])
            reg = self.reg_lambda * (
                F.mse_loss(priv_pred, latent, reduction="none") * valid2.unsqueeze(-1)
            ).mean()
        else:
            reg = torch.zeros((), device=loc.device)

        # symmetry loss: [:B] vs act_transform(mirror [B:]) (GH ppo.py:388-389)
        loc_sym = self.act_transform.to(loc.device)(loc[b:])
        scale_sym = self.act_transform.to(scale.device)(scale[b:], sign=False)
        sym = symmetry_loss(loc[:b], scale[:b], loc_sym, scale_sym)

        loss = policy_loss + entropy_loss + value_loss + reg + sym

        opt_actor.zero_grad()
        self.opt_critic.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        opt_actor.step()
        self.opt_critic.step()

        with torch.no_grad():
            loc_o, scale_o = mb["loc"], mb["scale"]
            kl = torch.sum(
                torch.log(scale[:b]) - torch.log(scale_o)
                + (scale_o**2 + (loc_o - loc[:b]) ** 2) / (2.0 * scale[:b] ** 2) - 0.5,
                dim=-1,
            ).mean()
        return kl.detach()

    # --- estimator (GH train_estimator/_update2 ppo.py:430-462) --- #
    def _train_estimator(self, rollout: dict) -> dict:
        flat = {g: rollout[g].reshape(-1, rollout[g].shape[-1]) for g in _OBS_GROUPS}
        flat["is_init"] = rollout["is_init"].reshape(-1)
        losses = []
        for _ in range(self.cfg.estimator_epochs):
            for mb in self._minibatches_estimator(flat):
                losses.append(self._update2(mb))
        return {"estimator_loss": float(torch.stack(losses).mean())}

    def _minibatches_estimator(self, flat: dict):
        M = flat["is_init"].shape[0]
        perm = torch.randperm(M)
        for idx in perm.chunk(self.cfg.num_minibatches):
            yield {k: v[idx] for k, v in flat.items()}

    def _update2(self, mb: dict) -> torch.Tensor:
        """Estimator: no_grad encoder_priv, train adapt_module, 2B masked MSE, no grad-clip."""
        device = mb[_OBS_GROUPS[0]].device
        obs2 = {g: torch.cat([mb[g], self.sym[g].to(device)(mb[g])], dim=0) for g in _OBS_GROUPS}
        valid2 = ~torch.cat([mb["is_init"], mb["is_init"]], dim=0)
        with torch.no_grad():
            priv_feature = self.policy.encoder_priv(obs2["priv"])
        priv_pred = self.policy.adapt_module(obs2["policy"])
        loss = (
            F.mse_loss(priv_pred, priv_feature, reduction="none") * valid2.unsqueeze(-1)
        ).mean()
        self.opt_estimator.zero_grad()
        loss.backward()
        self.opt_estimator.step()
        return loss.detach()
