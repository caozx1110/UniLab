"""Batch 2: T10.4 IndependentNormal + T10.6 GHDistillPolicy + T10.7 GH VecNorm."""
import numpy as np
import torch

from unilab.algos.gh_distill_ppo.distributions import IndependentNormal
from unilab.algos.gh_distill_ppo.policy import GHDistillPolicy
from unilab.algos.gh_distill_ppo.vecnorm import VecNorm


# --- T10.4 IndependentNormal --------------------------------------------- #

def test_independent_normal_clamps_scale_and_sums_event_dim():
    d = IndependentNormal(torch.zeros(4, 29), torch.full((4, 29), 1e-9))
    torch.testing.assert_close(d.scale, torch.full((4, 29), 1e-6))  # clamp_min 1e-6
    a = d.sample()
    assert a.shape == (4, 29)
    assert d.log_prob(a).shape == (4,)                     # summed over 29 action dims
    assert torch.isfinite(d.entropy()).all() and d.entropy().shape == (4,)
    torch.testing.assert_close(d.deterministic_sample, torch.zeros(4, 29))  # mean


# --- T10.6 GHDistillPolicy ----------------------------------------------- #

def _obs(n=4):
    return {"policy": torch.randn(n, 450), "priv": torch.randn(n, 717),
            "priv_critic": torch.randn(n, 3)}


def test_policy_five_named_children_and_phase8_param_counts():
    p = GHDistillPolicy()
    assert set(dict(p.named_children())) == {
        "encoder_priv", "adapt_module", "actor_teacher", "actor_student", "critic"}
    # exact Phase-8 golden param counts (materialized LazyLinear)
    n = lambda m: sum(w.numel() for w in m.parameters())
    assert n(p.encoder_priv) == 250112
    assert n(p.adapt_module) == 429568
    assert n(p.actor_teacher) == 766010
    assert n(p.actor_student) == 766010
    assert n(p.critic) == 996353


def test_policy_act_shapes_and_phase_routing():
    p = GHDistillPolicy()
    for phase in ("train", "adapt", "finetune"):
        action, loc, scale, logp = p.act(_obs(4), phase)
        assert action.shape == (4, 29) and loc.shape == (4, 29)
        assert scale.shape == (4, 29) and logp.shape == (4,)
    assert p.evaluate_critic(_obs(4)).shape == (4, 1)


def test_policy_get_rollout_policy_uses_phase_networks():
    p = GHDistillPolicy()
    obs = _obs(2)
    # train latent path uses encoder_priv (depends on priv); student path uses adapt_module (policy)
    train_grad = _grad_source(p, obs, "train")
    fine_grad = _grad_source(p, obs, "finetune")
    assert train_grad["encoder_priv"] and not train_grad["adapt_module"]
    assert fine_grad["adapt_module"] and not fine_grad["encoder_priv"]


def _grad_source(p, obs, phase):
    p.zero_grad(set_to_none=True)
    o = {k: v.clone() for k, v in obs.items()}
    action, loc, scale, logp = p.act(o, phase)
    loc.sum().backward()
    return {"encoder_priv": _any_grad(p.encoder_priv), "adapt_module": _any_grad(p.adapt_module)}


def _any_grad(m):
    return any(w.grad is not None and torch.any(w.grad != 0) for w in m.parameters())


# --- T10.7 GH VecNorm (decay-EMA, NOT Welford) --------------------------- #

def test_vecnorm_is_decay_ema_not_welford():
    """A second (zero) batch multiplies the running sum by ``decay`` — the exact
    decay-EMA signature. Welford (running-count) would leave the sum unchanged."""
    vn = VecNorm({"policy": 3}, decay=0.9999)
    vn.update({"policy": torch.ones(4, 3)})                # sum = 4 per dim
    sum1 = getattr(vn, "_policy__sum").clone()
    cnt1 = getattr(vn, "_policy__count").clone()
    vn.update({"policy": torch.zeros(4, 3)})               # decay applied, adds nothing
    sum2 = getattr(vn, "_policy__sum")
    cnt2 = getattr(vn, "_policy__count")
    torch.testing.assert_close(sum2, 0.9999 * sum1)        # decay-EMA (Welford: sum2==sum1)
    torch.testing.assert_close(cnt2, 0.9999 * cnt1 + 4.0)  # count decays too (Welford: cnt1+4)


def test_vecnorm_converges_to_constant_and_normalizes():
    vn = VecNorm({"policy": 3}, decay=0.9999)
    c = torch.full((256, 3), 5.0)
    for _ in range(500):                       # many batches of the constant
        vn.update({"policy": c})
    mean, std = vn._stats("policy")
    torch.testing.assert_close(mean, torch.full((3,), 5.0), atol=1e-3, rtol=0)
    out = vn.normalize({"policy": c})["policy"]
    assert torch.isfinite(out).all()


def test_vecnorm_eval_freezes_updates():
    vn = VecNorm({"policy": 3}, decay=0.9999)
    vn.update({"policy": torch.ones(8, 3)})
    before = getattr(vn, "_policy__count").clone()
    vn.eval()
    vn.update({"policy": torch.ones(8, 3)})    # frozen -> no-op
    assert torch.equal(getattr(vn, "_policy__count"), before)


def test_vecnorm_state_in_state_dict():
    vn = VecNorm({"policy": 450, "priv": 717, "priv_critic": 3})
    keys = set(vn.state_dict())
    for g in ("policy", "priv", "priv_critic"):
        assert {f"_{g}__sum", f"_{g}__ssq", f"_{g}__count"} <= keys
