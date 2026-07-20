"""Test symmetry involution transform for GHDistillPPO."""
import torch


def test_symmetry_involution():
    """Involution property: x[perm*signs][perm*signs] = x for arbitrary input.

    For involution, swapped positions must have matching signs (both +1 or both -1).
    """
    from unilab.algos.gh_distill_ppo.symmetry import SymmetryTransform

    # Proper involution: 0↔1 (both +1), 2↔3 (both -1), 4→4 (+1)
    perm = torch.tensor([1, 0, 3, 2, 4])
    signs = torch.tensor([1., 1., -1., -1., 1.])
    transform = SymmetryTransform(perm, signs)

    x = torch.randn(2, 5)
    x_sym = transform(x, sign=True)
    x_back = transform(x_sym, sign=True)
    assert torch.allclose(x, x_back, atol=1e-6), "Involution property violated"


def test_symmetry_sign_false():
    """sign=False permutes without flipping."""
    from unilab.algos.gh_distill_ppo.symmetry import SymmetryTransform

    perm = torch.tensor([1, 0, 3, 2, 4])
    signs = torch.tensor([1., 1., -1., -1., 1.])
    transform = SymmetryTransform(perm, signs)

    x = torch.randn(2, 5)
    x_perm = transform(x, sign=False)
    # sign=False: only permute, no sign flip
    assert torch.allclose(x_perm, x[..., perm], atol=1e-6)


def test_symmetry_cat():
    """SymmetryTransform.cat concatenates transforms with offset."""
    from unilab.algos.gh_distill_ppo.symmetry import SymmetryTransform

    t1 = SymmetryTransform(torch.tensor([1, 0]), torch.tensor([1., -1.]))
    t2 = SymmetryTransform(torch.tensor([0, 2, 1]), torch.tensor([1., 1., -1.]))
    t_cat = SymmetryTransform.cat([t1, t2])

    # Expect perm [1,0, 2,4,3], signs [1,-1, 1,1,-1]
    assert torch.equal(t_cat.perm, torch.tensor([1, 0, 2, 4, 3]))
    assert torch.equal(t_cat.signs, torch.tensor([1., -1., 1., 1., -1.]))

    x = torch.randn(2, 5)
    x_cat = t_cat(x, sign=True)
    assert torch.allclose(x_cat[..., :2], t1(x[..., :2], sign=True), atol=1e-6)
    assert torch.allclose(x_cat[..., 2:], t2(x[..., 2:], sign=True), atol=1e-6)
