"""GH-schema checkpoint save/load for GHDistillPPO (Phase 10.9).

Reproduces GH's checkpoint contract (ppo.py:526-590 + train.py:110-121):

- outer dict = ``{wandb, policy, env, cfg, vecnorm}`` — **no optimizer, no runtime state**
  (motion/force/admittance/action-delay/episode buffers are intentionally absent).
- ``policy`` = per-child ``state_dict`` for the 5 learnable nets (named_children) +
  ``last_phase`` + ``_meta`` (current_lr/entropy_coef/reg_lambda/progress/num_updates/
  world_size).
- ``load``: on ``last_phase=="train"`` hard-copy actor_teacher -> actor_student
  (``soft_copy_`` tau=1.0); restore ``_meta`` ONLY when ``last_phase == target_phase``
  (a phase change resets progress to 0). VecNorm decay-EMA stats are inherited.

The deterministic symmetry transforms + GAE are rebuilt from source (they hold no learned
state), so unlike GH they are not serialized; the learnable-network schema is identical.
"""
from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from unilab.algos.gh_distill_ppo.policy import GHDistillPolicy

_META_KEYS = ("current_lr", "entropy_coef", "reg_lambda", "progress", "num_updates", "world_size")
_LEARNABLE = ("encoder_priv", "adapt_module", "actor_teacher", "actor_student", "critic")


# --- weight copy (GH ppo.py:572-589) -------------------------------------- #
def soft_copy_(src: nn.Module, dst: nn.Module, tau: float) -> None:
    with torch.no_grad():
        src_params = dict(src.named_parameters())
        for name, dst_param in dst.named_parameters():
            if name in src_params:
                dst_param.data.copy_(tau * src_params[name].data + (1.0 - tau) * dst_param.data)


def hard_copy_(src: nn.Module, dst: nn.Module) -> None:
    """dst := src (GH hard_copy_ = soft_copy_ tau=1.0)."""
    soft_copy_(src, dst, 1.0)


# --- meta (trainer <-> checkpoint) ---------------------------------------- #
def meta_from_trainer(trainer, world_size: int = 1) -> dict[str, Any]:
    return {
        "current_lr": getattr(trainer, "current_lr", 0.0),
        "entropy_coef": getattr(trainer, "entropy_coef", 0.0),
        "reg_lambda": getattr(trainer, "reg_lambda", 0.0),
        "progress": getattr(trainer, "progress", 0.0),
        "num_updates": getattr(trainer, "num_updates", 0),
        "world_size": world_size,
    }


def apply_meta_to_trainer(trainer, meta: dict[str, Any]) -> None:
    for k in ("current_lr", "entropy_coef", "reg_lambda", "progress", "num_updates"):
        if k in meta:
            setattr(trainer, k, meta[k])


# --- policy state (GH ppo.py:526-544) ------------------------------------- #
def gh_policy_state_dict(policy: GHDistillPolicy, last_phase: str, meta: dict) -> OrderedDict:
    state: OrderedDict = OrderedDict()
    for n, m in policy.named_children():
        state[n] = m.state_dict()
    state["last_phase"] = last_phase
    state["_meta"] = {k: meta.get(k) for k in _META_KEYS}
    return state


def load_gh_policy_state_dict(
    policy: GHDistillPolicy, state: dict, target_phase: str, strict: bool = True
) -> dict[str, Any]:
    """Load the 5 nets, hard-copy teacher->student if last_phase=='train', and return the
    meta to restore (only when last_phase == target_phase)."""
    for n, m in policy.named_children():
        try:
            m.load_state_dict(state.get(n, {}), strict=strict)
        except Exception as e:  # pragma: no cover - mirrors GH's tolerant load
            warnings.warn(f"Failed to load {n}: {e}")
    last_phase = state.get("last_phase", "train")
    if last_phase == "train":
        hard_copy_(policy.actor_teacher, policy.actor_student)
    meta_restored = state.get("_meta", {}) if last_phase == target_phase else None
    return {"last_phase": last_phase, "meta_restored": meta_restored}


# --- outer checkpoint (GH train.py:110-121) ------------------------------- #
def build_env_state(env) -> dict[str, Any]:
    """NpEnv-equivalent of GH env.state_dict (specs only; no runtime state)."""
    return {
        "obs_groups_spec": dict(env.obs_groups_spec),
        "action_dim": int(env.action_space.shape[0]),
        "num_envs": int(env.num_envs),
    }


def save_gh_checkpoint(
    path, *, policy, vecnorm, env_state, cfg, last_phase, meta, wandb=None
) -> None:
    ckpt = {
        "wandb": wandb,
        "policy": gh_policy_state_dict(policy, last_phase, meta),
        "env": env_state,
        "cfg": cfg,
        "vecnorm": vecnorm.state_dict() if vecnorm is not None else None,
    }
    # invariant: no optimizer / runtime state in the checkpoint
    assert not any("optim" in str(k).lower() for k in ckpt), "GH checkpoint must not hold optimizer state"
    torch.save(ckpt, path)


def load_gh_checkpoint(path, *, policy, vecnorm, target_phase, strict: bool = True) -> dict[str, Any]:
    ckpt = torch.load(path, weights_only=False)
    res = load_gh_policy_state_dict(policy, ckpt["policy"], target_phase, strict=strict)
    if vecnorm is not None and ckpt.get("vecnorm") is not None:
        vecnorm.load_state_dict(ckpt["vecnorm"])   # inherit decay-EMA stats (frozen in eval)
    res["cfg"] = ckpt.get("cfg")
    res["env"] = ckpt.get("env")
    return res
