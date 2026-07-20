"""Network construction for GHDistillPPO.

GH design doc golden param counts:
- encoder_priv: 250112
- adapt_module: 429568
- actor (teacher/student): 766010 each
- critic: 996353
"""
import torch
import torch.nn as nn


def make_mlp(units, activation=nn.Mish, norm="before"):
    """GH common.py:45-60 pattern — Linear→LayerNorm→Mish."""
    layers = []
    for n in units:
        layers.append(nn.LazyLinear(n))
        if norm == "before":
            layers.append(nn.LayerNorm(n))
            layers.append(activation())
        else:
            layers.append(activation())
    return nn.Sequential(*layers)


def build_encoder_priv(latent_dim=256):
    """encoder_priv: priv[717]→256→256. GH ppo.py:129-131."""
    return nn.Sequential(
        make_mlp([256], norm="before"),
        nn.LazyLinear(latent_dim)
    )


def build_adapt_module(latent_dim=256):
    """adapt_module: policy[450]→512→256→256. GH ppo.py:134-141."""
    return nn.Sequential(
        make_mlp([512, 256], norm="before"),
        nn.LazyLinear(latent_dim)
    )


class Actor(nn.Module):
    """GH common.py:152-177 — state-independent std Parameter[action_dim]."""
    def __init__(self, action_dim, init_noise_scale=1.0):
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.ones(action_dim) * init_noise_scale)

    def forward(self, features):
        loc = self.actor_mean(features)
        scale = torch.ones_like(loc) * self.actor_std
        return loc, scale


def build_actor(input_dim, action_dim=29, init_noise_scale=1.0):
    """actor: [policy+priv][706]→512→512→256→Linear(29)+Parameter[29]. GH ppo.py:146-157."""
    return nn.Sequential(
        make_mlp([512, 512, 256], norm="before"),
        Actor(action_dim, init_noise_scale)
    )


def build_critic(input_dim):
    """critic: [policy+priv+priv_critic][1170]→512→512→256→1. GH ppo.py:163-166."""
    return nn.Sequential(
        make_mlp([512, 512, 256], norm="before"),
        nn.LazyLinear(1)
    )


def init_orthogonal(module, gain=0.01):
    """GH common.py:234-237, ppo.py:178-183."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            nn.init.zeros_(m.bias)
