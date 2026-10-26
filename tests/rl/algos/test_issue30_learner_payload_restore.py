"""Cold CPU payload audit; device execution compatibility stays fail-closed."""

import copy

import pytest
import torch

from unilab.rl.algos.flash_sac.learner import FlashSACLearner


def learner():
    return FlashSACLearner(
        obs_dim=4,
        critic_obs_dim=6,
        action_dim=2,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        actor_num_blocks=1,
        critic_num_blocks=1,
        num_atoms=5,
        device="cpu",
        use_compile=False,
    )


def test_payload_restores_lazy_metric_and_exploration_without_configuration_bypass():
    source = learner()
    state = source.export_continuation_state()
    state["actor"]["noise"] = torch.randn(7, 2)
    state["actor"]["repeat_count"] = torch.randint(0, 5, (7,), dtype=torch.int32)
    state["actor"]["repeat_target"] = torch.randint(5, 9, (7,), dtype=torch.int32)
    state["tensors"]["_pending_cycle_critic_metric_values"] = torch.randn(2)
    state["target_entropy"] = -2.75
    state["lr_peak"] = 0.125
    state["aux_loss_names"] = ("sentinel",)
    state["configuration"]["use_amp"] = True
    target = learner()
    with pytest.raises(ValueError, match="configuration differs"):
        target.import_continuation_state(state)
    before_config = copy.deepcopy(target._continuation_configuration())
    target._import_continuation_payload(state)
    actual = target.export_continuation_state()
    assert target._continuation_configuration() == before_config
    for name, expected in state["actor"].items():
        if isinstance(expected, torch.Tensor):
            assert torch.equal(expected, actual["actor"][name])
            assert expected.data_ptr() != actual["actor"][name].data_ptr()
        else:
            assert actual["actor"][name] == expected
    for name, expected in state["tensors"].items():
        if expected is not None:
            assert torch.equal(expected, actual["tensors"][name])
        else:
            assert actual["tensors"][name] is None
    for name in ("target_entropy", "lr_peak", "aux_loss_names", "scaler"):
        assert actual[name] == state[name]


def test_payload_rejects_scalar_layout_before_exploration_mutation():
    target = learner()
    state = target.export_continuation_state()
    state["tensors"]["_target_tau"] = torch.zeros(2)
    with pytest.raises(ValueError, match="scalar tensor differs"):
        target._import_continuation_payload(state)
    assert target._target_tau.ndim == 0


def test_public_import_keeps_schema_and_capability_checks():
    target = learner()
    state = target.export_continuation_state()
    state["schema"] = "unsupported"
    with pytest.raises(ValueError, match="schema"):
        target.import_continuation_state(state)
