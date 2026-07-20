"""GAE computation and advantage normalization for GHDistillPPO.

GH ppo.py:201-227 (GAE), 518-524 (modewise adv norm).
"""
import torch


def compute_gae_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE returns and advantages.

    GH ppo.py:201-227.

    Args:
        rewards: [B, T]
        values: [B, T]
        dones: [B, T] (1.0 if terminal)
        next_values: [B, T] (bootstrapped next value)
        gamma: discount factor
        lam: GAE lambda

    Returns:
        (returns [B, T], advantages [B, T])
    """
    # TD errors: δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
    td_errors = rewards + gamma * next_values * (1.0 - dones) - values

    # GAE: A_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
    # Backward pass to accumulate
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(rewards.shape[1])):
        gae = td_errors[:, t] + gamma * lam * (1.0 - dones[:, t]) * gae
        advantages[:, t] = gae

    returns = advantages + values
    return returns, advantages


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE for flattened batch (integration test helper).

    Args:
        rewards: [N] flattened rewards
        values: [N] flattened values
        dones: [N] flattened done flags
        gamma: discount factor
        gae_lambda: GAE lambda

    Returns:
        (advantages [N], returns [N])
    """
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1.0 - dones[t].float()) - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - dones[t].float()) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]

    return advantages, returns


def normalize_advantages_modewise(
    advantages: torch.Tensor, is_init: torch.Tensor
) -> torch.Tensor:
    """Modewise advantage normalization over ~is_init.

    GH ppo.py:518-524 — normalize only over non-init steps.

    Args:
        advantages: [B, ...]
        is_init: [B, ...] bool mask (True = init step, exclude from norm)

    Returns:
        normalized advantages [B, ...]
    """
    mask = ~is_init
    if mask.sum() == 0:
        return advantages  # All init, no normalization

    mean = advantages[mask].mean()
    std = advantages[mask].std()
    return (advantages - mean) / (std + 1e-8)


def modewise_advantage_normalization(
    advantages: torch.Tensor, is_init: torch.Tensor
) -> torch.Tensor:
    """Alias for normalize_advantages_modewise (integration test compatibility)."""
    return normalize_advantages_modewise(advantages, is_init)


def adaptive_kl_lr_schedule(
    kl: float, desired_kl: float, lr: float, progress: float, lr_min: float = 1e-5, lr_max: float = 5e-3
) -> float:
    """Adaptive KL learning rate schedule.

    GH ppo.py:243-265 — only adjust when progress >= 0.1.

    Args:
        kl: current KL divergence
        desired_kl: target KL (0.01)
        lr: current learning rate
        progress: training progress [0,1]
        lr_min: minimum lr (1e-5)
        lr_max: maximum lr (5e-3)

    Returns:
        adjusted learning rate
    """
    if progress < 0.1:
        return lr

    if kl < desired_kl:
        # Increase lr
        lr_new = lr * 1.5
    else:
        # Decrease lr
        lr_new = lr / 1.5

    return max(lr_min, min(lr_max, lr_new))
