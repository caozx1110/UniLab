"""T10.8: phase-aware train_op — per-phase trainable/frozen sets (grad dump) + update order.

Drives a synthetic mini-rollout through each phase and asserts WHICH networks received
gradients, against the blueprint table (§一:56-60):
  train    -> encoder_priv, actor_teacher, critic (PPO) + adapt_module (estimator)
  adapt    -> adapt_module only (no GAE/PPO/KL)
  finetune (progress>0.025) -> actor_student, adapt_module, critic
  finetune (progress<=0.025) -> no update
KL lr schedule only touches opt_teacher/opt_student (critic/estimator lr invariant).
"""
import torch

from unilab.algos.gh_distill_ppo.policy import GHDistillPolicy
from unilab.algos.gh_distill_ppo.trainer import GHDistillTrainer, GHDistillTrainerCfg

_CFG = GHDistillTrainerCfg(ppo_epochs=1, num_minibatches=2, estimator_epochs=1)


def _rollout(T=8, N=4):
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)
    return {
        "policy": r(T, N, 450), "priv": r(T, N, 717), "priv_critic": r(T, N, 3),
        "action": r(T, N, 29), "loc": r(T, N, 29),
        "scale": torch.rand(T, N, 29, generator=g) * 0.5 + 0.5,
        "sample_log_prob": r(T, N),
        "reward": r(T, N), "done": torch.zeros(T, N, dtype=torch.bool),
        "is_init": torch.zeros(T, N, dtype=torch.bool),
    }


def _has_grad(module) -> bool:
    return any(p.grad is not None and torch.any(p.grad != 0) for p in module.parameters())


def _grad_set(policy) -> set[str]:
    names = ("encoder_priv", "adapt_module", "actor_teacher", "actor_student", "critic")
    return {n for n in names if _has_grad(getattr(policy, n))}


def test_train_phase_trainable_set():
    p = GHDistillPolicy()
    tr = GHDistillTrainer(p, _CFG)
    tr.step_schedule(0.5)
    tr.train_op(_rollout(), phase="train", progress=0.5)
    assert _grad_set(p) == {"encoder_priv", "actor_teacher", "critic", "adapt_module"}


def test_adapt_phase_trains_only_adapt_module():
    p = GHDistillPolicy()
    tr = GHDistillTrainer(p, _CFG)
    lr0 = tr.opt_teacher.param_groups[0]["lr"]
    tr.train_op(_rollout(), phase="adapt", progress=0.5)
    assert _grad_set(p) == {"adapt_module"}
    # adapt does NOT run the KL schedule -> teacher/student lr untouched
    assert tr.opt_teacher.param_groups[0]["lr"] == lr0
    assert tr.opt_student.param_groups[0]["lr"] == lr0


def test_finetune_phase_trains_student_set():
    p = GHDistillPolicy()
    tr = GHDistillTrainer(p, _CFG)
    tr.step_schedule(0.5)
    tr.train_op(_rollout(), phase="finetune", progress=0.5)
    assert _grad_set(p) == {"actor_student", "adapt_module", "critic"}


def test_finetune_before_2p5pct_does_no_update():
    p = GHDistillPolicy()
    tr = GHDistillTrainer(p, _CFG)
    tr.train_op(_rollout(), phase="finetune", progress=0.01)
    assert _grad_set(p) == set()          # frozen policy region


def test_kl_schedule_never_touches_critic_or_estimator():
    p = GHDistillPolicy()
    tr = GHDistillTrainer(p, _CFG)
    lr0 = _CFG.lr
    tr.train_op(_rollout(), phase="train", progress=0.5)   # triggers KL schedule
    # apply_lr_to_teacher_student only mutates teacher/student optimizers
    assert tr.opt_critic.param_groups[0]["lr"] == lr0
    assert tr.opt_estimator.param_groups[0]["lr"] == lr0


def test_step_schedule_entropy_and_reg_endpoints():
    tr = GHDistillTrainer(GHDistillPolicy(), GHDistillTrainerCfg(entropy_start=0.005, entropy_end=0.002))
    tr.step_schedule(0.0)
    assert abs(tr.entropy_coef - 0.005) < 1e-9 and tr.reg_lambda == 0.0
    tr.step_schedule(1.0)
    assert abs(tr.entropy_coef - 0.002) < 1e-9 and abs(tr.reg_lambda - 0.2) < 1e-9
