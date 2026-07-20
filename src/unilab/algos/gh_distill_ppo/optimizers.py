"""Optimizers and learning rate schedule for GHDistillPPO."""
import torch
from torch.optim import Adam


def build_optimizers(
    encoder_priv: torch.nn.Module,
    adapt_module: torch.nn.Module,
    actor_teacher: torch.nn.Module,
    actor_student: torch.nn.Module,
    critic: torch.nn.Module,
    lr: float = 3e-4,
) -> tuple[Adam, Adam, Adam, Adam]:
    """Build 4 Adam optimizers with correct parameter ownership.

    GH ppo.py:170-191 — opt_teacher (actor_teacher+encoder_priv), opt_student
    (actor_student+adapt_module), opt_critic, opt_estimator (adapt_module).

    Args:
        encoder_priv: privileged encoder network
        adapt_module: adaptation module network
        actor_teacher: teacher actor network
        actor_student: student actor network
        critic: critic network
        lr: learning rate (3e-4)

    Returns:
        (opt_teacher, opt_student, opt_critic, opt_estimator)
    """
    opt_teacher = Adam(
        list(actor_teacher.parameters()) + list(encoder_priv.parameters()),
        lr=lr,
    )
    opt_student = Adam(
        list(actor_student.parameters()) + list(adapt_module.parameters()),
        lr=lr,
    )
    opt_critic = Adam(critic.parameters(), lr=lr)
    opt_estimator = Adam(adapt_module.parameters(), lr=lr)

    return opt_teacher, opt_student, opt_critic, opt_estimator


def apply_lr_to_teacher_student(
    opt_teacher: Adam,
    opt_student: Adam,
    opt_critic: Adam,
    opt_estimator: Adam,
    new_lr: float,
) -> None:
    """Apply learning rate schedule to teacher and student optimizers only.

    GH ppo.py:243-265 — adaptive KL schedule only affects teacher/student, not
    critic/estimator.

    Args:
        opt_teacher: teacher optimizer
        opt_student: student optimizer
        opt_critic: critic optimizer (unchanged)
        opt_estimator: estimator optimizer (unchanged)
        new_lr: new learning rate
    """
    for param_group in opt_teacher.param_groups:
        param_group["lr"] = new_lr
    for param_group in opt_student.param_groups:
        param_group["lr"] = new_lr
