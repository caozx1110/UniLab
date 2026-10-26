from types import SimpleNamespace

import pytest
import torch

from unilab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg
from unilab.managers._noise import UniformNoiseCfg
from unilab.managers.continuation import (
    capture_attributes,
    copy_tree,
    manager_state,
    restore_attributes,
    restore_manager,
)
from unilab.managers.manager_base import ManagerTermBase

from .conftest import FakeEnv


def observation(env):
    return env.obs


def make_manager(env):
    return ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={
                    "position": ObservationTermCfg(
                        func=observation,
                        history_length=10,
                        noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
                    ),
                },
                enable_corruption=True,
            )
        },
        env,
    )


def test_noisy_history_new_owner_continues_exactly_with_partial_resets():
    original = FakeEnv()
    manager = make_manager(original)
    for step in range(17):
        original.obs.add_(1)
        if step == 12:
            manager.reset(torch.tensor([1]))
        manager.compute()
    snapshot = manager_state(manager)
    rng = original.rng.get_state().clone()
    next_env = FakeEnv(seed=1337)
    restored = make_manager(next_env)
    restored.compute()  # Cold initialization; restore must erase its draws/history.
    restore_manager(restored, snapshot, device="cpu")
    next_env.rng.set_state(rng)
    next_env.obs.copy_(original.obs)
    for step in range(13):
        original.obs.add_(0.5)
        next_env.obs.add_(0.5)
        if step == 4:
            manager.reset(torch.tensor([0, 2]))
            restored.reset(torch.tensor([0, 2]))
        assert torch.equal(manager.compute()["policy"], restored.compute()["policy"])
        assert torch.equal(original.rng.get_state(), next_env.rng.get_state())


def test_attribute_roundtrip_preserves_carrier_identity_and_handles_optional_state():
    owner = SimpleNamespace(x=torch.arange(4.0), cache=None, counter=4)
    saved = capture_attributes(owner)
    pointer = owner.x.data_ptr()
    owner.x.zero_()
    owner.cache = {"temporary": torch.ones(2)}
    owner.counter = 0
    restore_attributes(owner, saved, device="cpu")
    assert owner.x.data_ptr() == pointer
    assert torch.equal(owner.x, torch.arange(4.0))
    assert owner.cache is None and owner.counter == 4


def test_snapshot_is_detached_and_unknown_opaque_state_fails_closed():
    value = torch.ones(4)
    saved = copy_tree({"value": value})
    value.zero_()
    assert torch.equal(saved["value"], torch.ones(4))
    with pytest.raises(TypeError, match="opaque"):
        copy_tree(object())
    term = ManagerTermBase(FakeEnv())
    term.future_backend_handle = object()
    with pytest.raises(TypeError, match="opaque"):
        term.export_continuation_state()


def test_changed_owner_schema_and_tensor_shape_are_rejected():
    owner = SimpleNamespace(x=torch.ones(4))
    saved = capture_attributes(owner)
    owner.extra = 1
    with pytest.raises(ValueError, match="attribute set"):
        restore_attributes(owner, saved, device="cpu")
    del owner.extra
    owner.x = torch.ones(3)
    with pytest.raises(ValueError, match="shape"):
        restore_attributes(owner, saved, device="cpu")
