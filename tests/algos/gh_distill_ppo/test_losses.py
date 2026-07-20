"""Test PPO surrogate and symmetry losses."""
import torch


def test_ppo_surrogate_loss_clipped():
    """PPO clipped surrogate loss with valid_mask [:B]."""
    from unilab.algos.gh_distill_ppo.losses import ppo_surrogate_loss

    B = 4
    # Batch [4]: first 2 valid, last 2 invalid
    log_probs_old = torch.tensor([0.1, 0.2, 0.3, 0.4])
    log_probs_new = torch.tensor([0.15, 0.25, 0.35, 0.45])
    advantages = torch.tensor([1.0, -0.5, 2.0, -1.0])
    valid_mask = torch.tensor([True, True, False, False])

    loss = ppo_surrogate_loss(
        log_probs_old, log_probs_new, advantages, valid_mask, clip_param=0.2
    )

    # Loss should only use first 2 entries
    # ratio = exp(new - old) = exp(0.05) ≈ 1.051
    # surrogate = -min(ratio*adv, clipped_ratio*adv)
    # With positive advantage, surrogate is negative (we maximize objective by minimizing negative)
    assert torch.isfinite(loss)

    # Verify mask works: loss with all-invalid should be nan (mean of empty)
    loss_invalid = ppo_surrogate_loss(
        log_probs_old, log_probs_new, advantages, torch.zeros_like(valid_mask, dtype=torch.bool), clip_param=0.2
    )
    assert torch.isnan(loss_invalid)


def test_ppo_entropy_loss_no_mask():
    """PPO entropy loss on [:B] with no valid_mask."""
    from unilab.algos.gh_distill_ppo.losses import ppo_entropy_loss

    B = 4
    # Teacher/student distributions (2B total, but entropy only [:B])
    loc = torch.randn(2 * B, 3)
    scale = torch.ones(2 * B, 3) * 0.5

    loss = ppo_entropy_loss(loc, scale, coef=0.01)

    # Entropy = 0.5 * log(2πe) + log(scale) ≈ 0.5 + log(0.5) = 0.5 - 0.693 per dim
    # Should be negative (we want to maximize entropy, so minimize -entropy)
    assert loss.item() < 0


def test_symmetry_loss_paired():
    """Symmetry loss on paired 2B samples (0.2*MSE(loc)+10*MSE(scale,sign=False))."""
    from unilab.algos.gh_distill_ppo.losses import symmetry_loss

    B = 4
    # Original and symmetric distributions
    loc_orig = torch.randn(2 * B, 3)
    scale_orig = torch.ones(2 * B, 3) * 0.5
    loc_sym = loc_orig + torch.randn(2 * B, 3) * 0.1  # Small perturbation
    scale_sym = scale_orig + torch.randn(2 * B, 3) * 0.1

    loss = symmetry_loss(loc_orig, scale_orig, loc_sym, scale_sym)

    # Loss should be small (perturbed by 0.1)
    assert 0 < loss.item() < 1.0

    # Perfect match → zero loss
    loss_zero = symmetry_loss(loc_orig, scale_orig, loc_orig, scale_orig)
    assert torch.allclose(loss_zero, torch.tensor(0.0), atol=1e-6)

    # Verify sign=False: negative scale should match positive
    scale_neg = -scale_orig.abs()
    loss_sign = symmetry_loss(loc_orig, scale_orig, loc_orig, scale_neg)
    assert torch.allclose(loss_sign, torch.tensor(0.0), atol=1e-6)


def test_reg_loss_gradient_flow():
    """Reg loss: adapt_module no_grad → encoder_priv receives grad."""
    from unilab.algos.gh_distill_ppo.losses import reg_loss

    # Create dummy networks
    encoder_priv = torch.nn.Linear(10, 8)
    adapt_module = torch.nn.Linear(5, 8)

    priv_input = torch.randn(4, 10)
    policy_input = torch.randn(4, 5)

    # Forward
    priv_enc = encoder_priv(priv_input)
    with torch.no_grad():
        priv_pred = adapt_module(policy_input)

    loss = reg_loss(priv_enc, priv_pred)

    # Backward
    loss.backward()

    # encoder_priv should have gradients, adapt_module should not
    assert encoder_priv.weight.grad is not None
    assert adapt_module.weight.grad is None


def test_estimator_loss_gradient_flow():
    """Estimator loss: freeze encoder_priv → adapt_module receives grad."""
    from unilab.algos.gh_distill_ppo.losses import estimator_loss

    # Create dummy networks
    encoder_priv = torch.nn.Linear(10, 8)
    adapt_module = torch.nn.Linear(5, 8)

    priv_input = torch.randn(4, 10)
    policy_input = torch.randn(4, 5)

    # Forward with encoder frozen
    with torch.no_grad():
        priv_enc = encoder_priv(priv_input)
    priv_pred = adapt_module(policy_input)

    loss = estimator_loss(priv_enc, priv_pred)

    # Backward
    loss.backward()

    # adapt_module should have gradients, encoder_priv should not
    assert adapt_module.weight.grad is not None
    assert encoder_priv.weight.grad is None

