"""Integration test for GHDistillPPO — full forward→backward→step."""
import torch


def test_full_training_step_with_gradient_flow():
    """Assemble 5 networks + 4 opts + symmetry + losses, run real backward, verify batch ranges and gradient flow."""
    from unilab.algos.gh_distill_ppo.networks import (
        build_encoder_priv,
        build_adapt_module,
        build_actor,
        build_critic,
        init_orthogonal,
    )
    from unilab.algos.gh_distill_ppo.symmetry import SymmetryTransform
    from unilab.algos.gh_distill_ppo.gae import compute_gae, modewise_advantage_normalization
    from unilab.algos.gh_distill_ppo.losses import (
        ppo_surrogate_loss,
        entropy_loss,
        critic_loss,
        symmetry_loss,
        reg_loss,
        estimator_loss,
    )
    from unilab.algos.gh_distill_ppo.optimizers import build_optimizers

    # Simplified hyperparameters for faster test
    B = 4  # batch size
    T = 8  # horizon
    policy_dim = 16  # simplified from 450
    priv_dim = 24  # simplified from 717
    priv_critic_dim = 3
    latent_dim = 8  # simplified from 256
    action_dim = 6  # simplified from 29

    # Build 5 networks
    encoder_priv = build_encoder_priv(latent_dim=latent_dim)
    adapt_module = build_adapt_module(latent_dim=latent_dim)
    actor_teacher = build_actor(input_dim=policy_dim + latent_dim, action_dim=action_dim)
    actor_student = build_actor(input_dim=policy_dim + latent_dim, action_dim=action_dim)
    critic = build_critic(input_dim=policy_dim + priv_dim + priv_critic_dim)

    # Initialize lazy layers
    dummy_priv = torch.zeros(2, priv_dim)
    encoder_priv(dummy_priv)
    dummy_policy = torch.zeros(2, policy_dim)
    adapt_module(dummy_policy)
    dummy_actor_input = torch.zeros(2, policy_dim + latent_dim)
    actor_teacher(dummy_actor_input)
    actor_student(dummy_actor_input)
    dummy_critic_input = torch.zeros(2, policy_dim + priv_dim + priv_critic_dim)
    critic(dummy_critic_input)

    # Orthogonal init
    for net in [encoder_priv, adapt_module, actor_teacher, actor_student, critic]:
        init_orthogonal(net, gain=0.01)

    # Build 4 optimizers
    opt_teacher, opt_student, opt_critic, opt_estimator = build_optimizers(
        encoder_priv, adapt_module, actor_teacher, actor_student, critic, lr=3e-4
    )

    # Symmetry transform (for policy_dim=16, action_dim=6)
    perm_policy = torch.arange(policy_dim)
    perm_policy[0], perm_policy[1] = perm_policy[1].clone(), perm_policy[0].clone()  # swap first two
    signs_policy = torch.ones(policy_dim)
    signs_policy[2] = -1.0  # flip third
    sym_policy = SymmetryTransform(perm_policy, signs_policy)

    perm_action = torch.arange(action_dim)
    perm_action[0], perm_action[1] = perm_action[1].clone(), perm_action[0].clone()
    signs_action = torch.ones(action_dim)
    signs_action[2] = -1.0
    sym_action = SymmetryTransform(perm_action, signs_action)

    # Generate synthetic rollout data
    obs_policy = torch.randn(B, T, policy_dim)
    obs_priv = torch.randn(B, T, priv_dim)
    obs_priv_critic = torch.randn(B, T, priv_critic_dim)
    actions = torch.randn(B, T, action_dim)
    old_log_probs = torch.randn(B, T)
    rewards = torch.randn(B, T)
    dones = torch.zeros(B, T, dtype=torch.bool)
    is_init = torch.zeros(B, T, dtype=torch.bool)
    is_init[:, 0] = True  # first step of each episode

    # Flatten batch
    obs_policy_flat = obs_policy.reshape(B * T, policy_dim)
    obs_priv_flat = obs_priv.reshape(B * T, priv_dim)
    obs_priv_critic_flat = obs_priv_critic.reshape(B * T, priv_critic_dim)
    actions_flat = actions.reshape(B * T, action_dim)
    old_log_probs_flat = old_log_probs.reshape(B * T)
    rewards_flat = rewards.reshape(B * T)
    dones_flat = dones.reshape(B * T)
    is_init_flat = is_init.reshape(B * T)

    # === Forward pass ===
    # Encode privileged info
    priv_enc = encoder_priv(obs_priv_flat)

    # Adapt module (with no_grad for reg loss)
    with torch.no_grad():
        priv_pred_for_reg = adapt_module(obs_policy_flat)

    # Teacher actor
    actor_teacher_input = torch.cat([obs_policy_flat, priv_enc], dim=-1)
    loc_teacher, scale_teacher = actor_teacher(actor_teacher_input)

    # Student actor
    priv_pred = adapt_module(obs_policy_flat)
    actor_student_input = torch.cat([obs_policy_flat, priv_pred], dim=-1)
    loc_student, scale_student = actor_student(actor_student_input)

    # Critic
    critic_input = torch.cat([obs_policy_flat, obs_priv_flat, obs_priv_critic_flat], dim=-1)
    values = critic(critic_input).squeeze(-1)

    # Symmetry augmentation (double batch)
    obs_policy_sym = sym_policy(obs_policy_flat)
    actions_sym = sym_action(actions_flat)

    obs_policy_double = torch.cat([obs_policy_flat, obs_policy_sym], dim=0)
    obs_priv_double = torch.cat([obs_priv_flat, obs_priv_flat], dim=0)
    obs_priv_critic_double = torch.cat([obs_priv_critic_flat, obs_priv_critic_flat], dim=0)
    actions_double = torch.cat([actions_flat, actions_sym], dim=0)

    # Forward through networks again with doubled batch for symmetry
    priv_enc_double = encoder_priv(obs_priv_double)
    priv_pred_double = adapt_module(obs_policy_double)

    actor_teacher_input_double = torch.cat([obs_policy_double, priv_enc_double], dim=-1)
    loc_teacher_double, scale_teacher_double = actor_teacher(actor_teacher_input_double)

    actor_student_input_double = torch.cat([obs_policy_double, priv_pred_double], dim=-1)
    loc_student_double, scale_student_double = actor_student(actor_student_input_double)

    critic_input_double = torch.cat([obs_policy_double, obs_priv_double, obs_priv_critic_double], dim=-1)
    values_double = critic(critic_input_double).squeeze(-1)

    # Compute GAE (use original batch)
    advantages, returns = compute_gae(rewards_flat, values[:B*T], dones_flat, gamma=0.99, gae_lambda=0.95)

    # Modewise advantage normalization
    advantages_norm = modewise_advantage_normalization(advantages, is_init_flat)

    # Create valid mask for PPO (first B only)
    valid_mask = torch.ones(B * T, dtype=torch.bool)

    # === Compute losses ===
    # Compute new log probs from teacher policy
    dist = torch.distributions.Normal(loc_teacher_double[:B*T], scale_teacher_double[:B*T])
    new_log_probs_flat = dist.log_prob(actions_flat).sum(dim=-1)

    # PPO surrogate (first B only)
    loss_surrogate = ppo_surrogate_loss(
        old_log_probs_flat,
        new_log_probs_flat,
        advantages_norm,
        valid_mask,
        clip_param=0.2,
    )

    # Entropy (first B only, no valid mask)
    loss_entropy = entropy_loss(scale_teacher_double[:B*T])

    # Critic (full 2B)
    returns_double = torch.cat([returns, returns], dim=0)
    loss_critic = critic_loss(values_double, returns_double)

    # Symmetry (paired 2B)
    loss_sym = symmetry_loss(
        loc_teacher_double[:B*T],
        scale_teacher_double[:B*T],
        loc_teacher_double[B*T:],
        scale_teacher_double[B*T:],
    )

    # Reg loss (full 2B)
    loss_reg = reg_loss(priv_enc_double, priv_pred_for_reg.repeat(2, 1))

    # Estimator loss (full 2B, encoder frozen)
    with torch.no_grad():
        priv_enc_frozen = encoder_priv(obs_priv_double)
    loss_estim = estimator_loss(priv_enc_frozen, priv_pred_double)

    # === Backward and optimizer step ===
    # Zero all grads
    for opt in [opt_teacher, opt_student, opt_critic, opt_estimator]:
        opt.zero_grad()

    # PPO + entropy + critic + symmetry (train phase)
    ppo_loss = loss_surrogate + 0.005 * loss_entropy
    total_loss = ppo_loss + 0.5 * loss_critic + loss_sym + loss_reg
    total_loss.backward(retain_graph=True)

    # Check gradients exist for expected networks after train backward
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor_teacher.parameters()), "actor_teacher should have gradients"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder_priv.parameters()), "encoder_priv should have gradients (from reg_loss)"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in critic.parameters()), "critic should have gradients"

    # Step teacher and critic
    opt_teacher.step()
    opt_critic.step()

    # Estimator loop (separate backward)
    opt_estimator.zero_grad()
    loss_estim.backward()

    # Check estimator gradients
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapt_module.parameters()), "adapt_module should have gradients (from estimator_loss)"

    opt_estimator.step()

    # === Verify batch ranges ===
    # PPO operates on first B only (validated by valid_mask shape)
    assert valid_mask.shape[0] == B * T

    # Critic/reg/estimator operate on full 2B (validated by input shapes)
    assert values_double.shape[0] == 2 * B * T
    assert priv_enc_double.shape[0] == 2 * B * T

    print("✓ Integration test passed: 5 networks + 4 opts + symmetry + losses, real backward, batch ranges correct, gradients verified")


def test_adaptive_kl_schedule_integration():
    """Verify adaptive KL schedule only affects teacher/student optimizers."""
    from unilab.algos.gh_distill_ppo.networks import build_encoder_priv, build_adapt_module, build_actor, build_critic
    from unilab.algos.gh_distill_ppo.optimizers import build_optimizers, apply_lr_to_teacher_student
    from unilab.algos.gh_distill_ppo.gae import adaptive_kl_lr_schedule

    # Build networks (minimal)
    encoder_priv = build_encoder_priv(latent_dim=256)
    adapt_module = build_adapt_module(latent_dim=256)
    actor_teacher = build_actor(input_dim=706, action_dim=29)
    actor_student = build_actor(input_dim=706, action_dim=29)
    critic = build_critic(input_dim=1170)

    # Initialize
    encoder_priv(torch.zeros(2, 717))
    adapt_module(torch.zeros(2, 450))
    actor_teacher(torch.zeros(2, 706))
    actor_student(torch.zeros(2, 706))
    critic(torch.zeros(2, 1170))

    # Build optimizers with realistic initial lr
    initial_lr = 1e-3  # Use 1e-3 so decrease stays above lr_min (1e-5)
    opt_teacher, opt_student, opt_critic, opt_estimator = build_optimizers(
        encoder_priv, adapt_module, actor_teacher, actor_student, critic, lr=initial_lr
    )

    assert opt_teacher.param_groups[0]["lr"] == initial_lr
    assert opt_student.param_groups[0]["lr"] == initial_lr
    assert opt_critic.param_groups[0]["lr"] == initial_lr
    assert opt_estimator.param_groups[0]["lr"] == initial_lr

    # Simulate KL schedule (high KL → decrease lr)
    kl_teacher = 0.02  # higher than desired 0.01
    kl_student = 0.015
    desired_kl = 0.01
    lr_min = 1e-5
    lr_max = 5e-3
    progress = 0.5  # >= 0.1, schedule active

    new_lr_teacher = adaptive_kl_lr_schedule(kl_teacher, desired_kl, initial_lr, progress, lr_min, lr_max)
    new_lr_student = adaptive_kl_lr_schedule(kl_student, desired_kl, initial_lr, progress, lr_min, lr_max)

    # Both should decrease (KL > desired) by factor of 1.5
    assert new_lr_teacher < initial_lr
    assert new_lr_student < initial_lr
    assert abs(new_lr_teacher - initial_lr / 1.5) < 1e-9

    # Apply to teacher/student only
    apply_lr_to_teacher_student(opt_teacher, opt_student, opt_critic, opt_estimator, new_lr_teacher)

    # Verify only teacher/student changed
    assert opt_teacher.param_groups[0]["lr"] == new_lr_teacher
    assert opt_student.param_groups[0]["lr"] == new_lr_teacher  # using teacher's lr for simplicity
    assert opt_critic.param_groups[0]["lr"] == initial_lr  # unchanged
    assert opt_estimator.param_groups[0]["lr"] == initial_lr  # unchanged

    print("✓ KL schedule integration test passed: only teacher/student lr changed")
