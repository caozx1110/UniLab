"""PPO and symmetry loss functions."""
import torch
import torch.nn.functional as F


def ppo_surrogate_loss(
    log_probs_old: torch.Tensor,
    log_probs_new: torch.Tensor,
    advantages: torch.Tensor,
    valid_mask: torch.Tensor,
    clip_param: float = 0.2,
) -> torch.Tensor:
    """PPO clipped surrogate loss.

    GH ppo.py:392-415 — only uses [:B] with valid_mask.

    Args:
        log_probs_old: old log probabilities [B]
        log_probs_new: new log probabilities [B]
        advantages: advantage estimates [B]
        valid_mask: valid steps [B]
        clip_param: clipping parameter (0.2)

    Returns:
        scalar loss
    """
    ratio = torch.exp(log_probs_new - log_probs_old)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)

    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages

    loss = -torch.min(surrogate1, surrogate2)
    loss = loss[valid_mask].mean()

    return loss


def ppo_entropy_loss(loc: torch.Tensor, scale: torch.Tensor, coef: float = 0.01) -> torch.Tensor:
    """PPO entropy loss on [:B] with no valid_mask.

    GH ppo.py:488-498 — entropy computed on first B samples only.

    Args:
        loc: distribution means [2B, action_dim]
        scale: distribution scales [2B, action_dim]
        coef: entropy coefficient (0.01)

    Returns:
        scalar loss (negative for minimization)
    """
    B = loc.shape[0] // 2
    loc_b = loc[:B]
    scale_b = scale[:B]

    # Gaussian entropy: 0.5 * log(2πe) + log(scale)
    entropy = 0.5 * torch.log(2 * torch.pi * torch.e * scale_b**2)
    entropy = entropy.sum(dim=-1).mean()

    return -coef * entropy


def symmetry_loss(
    loc_orig: torch.Tensor,
    scale_orig: torch.Tensor,
    loc_sym: torch.Tensor,
    scale_sym: torch.Tensor,
) -> torch.Tensor:
    """Symmetry loss on paired 2B samples.

    GH ppo.py:488-498 — 0.2*MSE(loc) + 10*MSE(scale, sign=False), no valid_mask.

    Args:
        loc_orig: original distribution means [2B, action_dim]
        scale_orig: original distribution scales [2B, action_dim]
        loc_sym: symmetric distribution means [2B, action_dim]
        scale_sym: symmetric distribution scales [2B, action_dim]

    Returns:
        scalar loss
    """
    # MSE on loc
    loss_loc = F.mse_loss(loc_orig, loc_sym)

    # MSE on scale (sign=False → use abs)
    loss_scale = F.mse_loss(scale_orig.abs(), scale_sym.abs())

    return 0.2 * loss_loc + 10.0 * loss_scale


def reg_loss(priv_enc: torch.Tensor, priv_pred: torch.Tensor) -> torch.Tensor:
    """Regularization loss — align adapt_module prediction with encoder_priv.

    GH ppo.py:541-557 — adapt_module called with no_grad, encoder_priv receives gradient.

    Args:
        priv_enc: encoder_priv output [B, latent_dim] (requires_grad=True)
        priv_pred: adapt_module output [B, latent_dim] (requires_grad=False)

    Returns:
        scalar MSE loss
    """
    return F.mse_loss(priv_enc, priv_pred)


def estimator_loss(priv_enc: torch.Tensor, priv_pred: torch.Tensor) -> torch.Tensor:
    """Estimator loss — train adapt_module to match frozen encoder_priv.

    GH ppo.py:560-582 — encoder_priv called with no_grad, adapt_module receives gradient.

    Args:
        priv_enc: encoder_priv output [B, latent_dim] (requires_grad=False)
        priv_pred: adapt_module output [B, latent_dim] (requires_grad=True)

    Returns:
        scalar MSE loss
    """
    return F.mse_loss(priv_enc, priv_pred)


def entropy_loss(scale: torch.Tensor) -> torch.Tensor:
    """Entropy loss (simplified for integration test).

    Args:
        scale: distribution scales [B, action_dim]

    Returns:
        scalar entropy loss (negative for maximization)
    """
    entropy = 0.5 * torch.log(2 * torch.pi * torch.e * scale**2)
    entropy = entropy.sum(dim=-1).mean()
    return -entropy


def critic_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Critic MSE loss on full 2B.

    Args:
        values: predicted values [2B]
        returns: target returns [2B]

    Returns:
        scalar MSE loss
    """
    return F.mse_loss(values, returns)
