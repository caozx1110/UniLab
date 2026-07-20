"""Test GHDistillPPO optimizers and KL schedule."""
import torch


def test_build_optimizers_four_adam():
    """4 Adam optimizers with correct parameter ownership."""
    from unilab.algos.gh_distill_ppo.optimizers import build_optimizers

    # Create dummy networks
    encoder_priv = torch.nn.Linear(10, 8)
    adapt_module = torch.nn.Linear(5, 8)
    actor_teacher = torch.nn.Linear(8, 3)
    actor_student = torch.nn.Linear(8, 3)
    critic = torch.nn.Linear(8, 1)

    opts = build_optimizers(
        encoder_priv=encoder_priv,
        adapt_module=adapt_module,
        actor_teacher=actor_teacher,
        actor_student=actor_student,
        critic=critic,
        lr=3e-4,
    )

    # Should return 4 optimizers
    assert len(opts) == 4
    opt_teacher, opt_student, opt_critic, opt_estimator = opts

    # opt_teacher: actor_teacher + encoder_priv
    teacher_params = set(opt_teacher.param_groups[0]["params"])
    expected_teacher = set(actor_teacher.parameters()) | set(encoder_priv.parameters())
    assert teacher_params == expected_teacher

    # opt_student: actor_student + adapt_module
    student_params = set(opt_student.param_groups[0]["params"])
    expected_student = set(actor_student.parameters()) | set(adapt_module.parameters())
    assert student_params == expected_student

    # opt_critic: critic only
    critic_params = set(opt_critic.param_groups[0]["params"])
    expected_critic = set(critic.parameters())
    assert critic_params == expected_critic

    # opt_estimator: adapt_module only
    estimator_params = set(opt_estimator.param_groups[0]["params"])
    expected_estimator = set(adapt_module.parameters())
    assert estimator_params == expected_estimator


def test_apply_lr_to_teacher_student_only():
    """KL schedule only affects teacher/student, not critic/estimator."""
    from unilab.algos.gh_distill_ppo.optimizers import apply_lr_to_teacher_student

    # Create dummy optimizers with different initial lr
    opt_teacher = torch.optim.Adam([torch.nn.Parameter(torch.randn(2, 2))], lr=3e-4)
    opt_student = torch.optim.Adam([torch.nn.Parameter(torch.randn(2, 2))], lr=3e-4)
    opt_critic = torch.optim.Adam([torch.nn.Parameter(torch.randn(2, 2))], lr=3e-4)
    opt_estimator = torch.optim.Adam([torch.nn.Parameter(torch.randn(2, 2))], lr=3e-4)

    # Apply lr schedule
    new_lr = 1e-4
    apply_lr_to_teacher_student(opt_teacher, opt_student, opt_critic, opt_estimator, new_lr)

    # teacher and student should have new_lr
    assert opt_teacher.param_groups[0]["lr"] == new_lr
    assert opt_student.param_groups[0]["lr"] == new_lr

    # critic and estimator should remain unchanged
    assert opt_critic.param_groups[0]["lr"] == 3e-4
    assert opt_estimator.param_groups[0]["lr"] == 3e-4
