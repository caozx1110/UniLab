"""Test network construction and parameter counts for GHDistillPPO."""
import torch


def test_encoder_priv_param_count_is_250112():
    """Encoder_priv must have exactly 250112 parameters (GH design doc golden value)."""
    from unilab.algos.gh_distill_ppo.networks import build_encoder_priv

    enc = build_encoder_priv(latent_dim=256)
    # Lazy init with priv input 717
    dummy = torch.zeros(2, 717)
    enc(dummy)

    total = sum(p.numel() for p in enc.parameters())
    assert total == 250112, f"Expected 250112, got {total}"


def test_adapt_module_param_count_is_429568():
    """Adapt_module must have exactly 429568 parameters (GH design doc golden value)."""
    from unilab.algos.gh_distill_ppo.networks import build_adapt_module

    adapt = build_adapt_module(latent_dim=256)
    dummy = torch.zeros(2, 450)  # policy input
    adapt(dummy)
    total = sum(p.numel() for p in adapt.parameters())
    assert total == 429568, f"Expected 429568, got {total}"


def test_actor_param_count_is_766010():
    """Actor must have exactly 766010 parameters (GH design doc golden value)."""
    from unilab.algos.gh_distill_ppo.networks import build_actor

    actor = build_actor(input_dim=706, action_dim=29, init_noise_scale=1.0)
    dummy = torch.zeros(2, 706)  # policy 450 + priv 256
    actor(dummy)
    total = sum(p.numel() for p in actor.parameters())
    assert total == 766010, f"Expected 766010, got {total}"


def test_critic_param_count_is_996353():
    """Critic must have exactly 996353 parameters (GH design doc golden value)."""
    from unilab.algos.gh_distill_ppo.networks import build_critic

    critic = build_critic(input_dim=1170)
    dummy = torch.zeros(2, 1170)  # policy 450 + priv 717 + priv_critic 3
    critic(dummy)
    total = sum(p.numel() for p in critic.parameters())
    assert total == 996353, f"Expected 996353, got {total}"


def test_orthogonal_init_gain_0p01():
    """Orthogonal init with gain=0.01 produces small weights, zero bias."""
    from unilab.algos.gh_distill_ppo.networks import build_critic, init_orthogonal

    critic = build_critic(input_dim=1170)
    dummy = torch.zeros(2, 1170)
    critic(dummy)

    init_orthogonal(critic, gain=0.01)

    # Check first Linear has small weights
    first_linear = None
    for m in critic.modules():
        if isinstance(m, torch.nn.Linear):
            first_linear = m
            break
    assert first_linear is not None
    weight_norm = torch.linalg.matrix_norm(first_linear.weight).item()
    assert weight_norm < 50.0  # Orthogonal with gain=0.01 → small norm
    assert torch.allclose(first_linear.bias, torch.zeros_like(first_linear.bias))
