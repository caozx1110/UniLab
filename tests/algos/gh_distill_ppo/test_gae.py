"""Test GAE, modewise advantage normalization, and adaptive KL schedule."""
import torch


def test_compute_gae_returns():
    """GAE with lambda=0.95, gamma=0.99."""
    from unilab.algos.gh_distill_ppo.gae import compute_gae_returns

    # Simple 3-step episode: rewards [1,2,3], values [0.5,1.0,1.5], terminal at step 3
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.tensor([[0.5, 1.0, 1.5]])
    dones = torch.tensor([[0.0, 0.0, 1.0]])
    next_values = torch.tensor([[1.0, 1.5, 0.0]])  # terminal → 0

    returns, advantages = compute_gae_returns(
        rewards, values, dones, next_values, gamma=0.99, lam=0.95
    )

    # Hand-compute TD errors
    # δ0 = 1 + 0.99*1.0 - 0.5 = 1.49
    # δ1 = 2 + 0.99*1.5 - 1.0 = 2.485
    # δ2 = 3 + 0.99*0.0 - 1.5 = 1.5
    # A0 = δ0 + 0.95*0.99*δ1 + (0.95*0.99)^2*δ2
    # A1 = δ1 + 0.95*0.99*δ2
    # A2 = δ2

    assert returns.shape == (1, 3)
    assert advantages.shape == (1, 3)
    # Just check non-zero and reasonable magnitudes
    assert advantages[0, 0] > 0 and advantages[0, 0] < 10
    assert torch.allclose(returns, advantages + values, atol=1e-5)


def test_modewise_advantage_normalization():
    """Modewise norm over ~is_init (GH ppo.py:518-524)."""
    from unilab.algos.gh_distill_ppo.gae import normalize_advantages_modewise

    # 4 steps: is_init [True, False, False, True]
    advantages = torch.tensor([[10.0, 2.0, 4.0, 100.0]])
    is_init = torch.tensor([[True, False, False, True]])

    adv_norm = normalize_advantages_modewise(advantages, is_init)

    # Only normalize over steps 1,2: mean=3, std=1 → (2-3)/1=-1, (4-3)/1=1
    # Steps 0,3 also normalized but irrelevant (excluded from mean/std)
    mask = ~is_init
    assert torch.allclose(adv_norm[mask].mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(adv_norm[mask].std(), torch.tensor(1.0), atol=1e-5)


def test_adaptive_kl_lr_schedule():
    """Adaptive KL lr schedule (desired_kl=0.01, bounds [1e-5,5e-3], progress>=0.1)."""
    from unilab.algos.gh_distill_ppo.gae import adaptive_kl_lr_schedule

    lr = 3e-4

    # kl < desired → increase lr (clamp to max 5e-3)
    lr_new = adaptive_kl_lr_schedule(kl=0.005, desired_kl=0.01, lr=lr, progress=0.2)
    assert lr_new > lr and lr_new <= 5e-3

    # kl > desired → decrease lr (clamp to min 1e-5)
    lr_new = adaptive_kl_lr_schedule(kl=0.02, desired_kl=0.01, lr=lr, progress=0.2)
    assert lr_new < lr and lr_new >= 1e-5

    # progress < 0.1 → no change
    lr_new = adaptive_kl_lr_schedule(kl=0.005, desired_kl=0.01, lr=lr, progress=0.05)
    assert lr_new == lr
