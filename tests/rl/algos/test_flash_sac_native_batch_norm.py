"""Native BN preserves eager arithmetic across explicit compiled graph breaks."""

from __future__ import annotations

import copy

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from tests.rl.algos.test_flash_sac_learner import _make_small_learner
from tests.rl.algos.test_flash_sac_runtime_graph_snapshot import _runner
from unilab.rl.algos.flash_sac import learner as learner_module
from unilab.rl.algos.flash_sac.layers import UnitBatchNorm


class _BatchNormRegion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = UnitBatchNorm(4)

    def forward(self, x, training):
        # Exact power-of-two operations provide real compiled regions around BN.
        return self.norm(x * 2, training=training) * 2


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_batch_norm_compiled_regions_keep_outputs_gradients_and_stats(training, dtype):
    torch.manual_seed(31)
    eager = _BatchNormRegion()
    with torch.no_grad():
        eager.norm.weight.uniform_(0.5, 1.5)
        eager.norm.bias.uniform_(-0.5, 0.5)
        eager.norm.running_mean.normal_()
        eager.norm.running_var.uniform_(0.5, 2.0)
    native = copy.deepcopy(eager)
    native.norm.preserve_native_execution = True
    initial_mean = eager.norm.running_mean.clone()
    initial_var = eager.norm.running_var.clone()
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(native, backend=counter, fullgraph=False)
    values = torch.randn(16, 4).to(dtype).requires_grad_()
    compiled_values = values.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        expected = eager(values, training)
        actual = compiled(compiled_values, training)
    assert counter.frame_count >= 1
    assert actual.dtype == expected.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected.float().square().sum().backward()
    actual.float().square().sum().backward()
    torch.testing.assert_close(compiled_values.grad, values.grad, atol=0, rtol=0)
    for name, parameter in eager.named_parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad)
        torch.testing.assert_close(
            dict(native.named_parameters())[name].grad, parameter.grad, atol=0, rtol=0
        )
    for name, value in eager.state_dict().items():
        torch.testing.assert_close(native.state_dict()[name], value, atol=0, rtol=0)
    assert torch.equal(eager.norm.running_mean, initial_mean) is not training
    assert torch.equal(eager.norm.running_var, initial_var) is not training


@pytest.mark.parametrize("enabled", [False, True])
def test_learner_native_batch_norm_is_explicit_and_actor_only(monkeypatch, enabled):
    observed_counts = []

    def record_compile(fn, **kwargs):
        del kwargs
        owner = fn.__self__
        actor = getattr(owner, "actor", owner)
        observed_counts.append(
            sum(
                isinstance(module, UnitBatchNorm) and module.preserve_native_execution
                for module in actor.modules()
            )
        )
        return fn

    monkeypatch.setattr(
        learner_module, "get_torch_compile_for_device", lambda *_args, **_kwargs: record_compile
    )
    learner = _make_small_learner(
        use_compile=True,
        compile_full_objectives=True,
        compile_native_batch_norm=enabled,
        use_whole_cycle_cuda_graph=False,
    )
    expected_count = 3 if enabled else 0
    assert learner.native_batch_norm_module_count == expected_count
    assert observed_counts == [expected_count, expected_count]
    assert all(
        not module.preserve_native_execution
        for network in (learner.critic, learner.target_critic)
        for module in network.modules()
        if isinstance(module, UnitBatchNorm)
    )
    manifest = _runner(learner)._cuda_graph_runtime_manifest()
    assert manifest["compile_native_batch_norm"] is enabled
    assert manifest["native_batch_norm_module_count"] == expected_count
    assert manifest["scope"] == (
        "full_objectives_with_native_batch_norm" if enabled else "full_objectives"
    )
    assert bool(manifest["intentional_graph_breaks"]) is enabled


def test_eager_learner_does_not_claim_native_compiled_modules():
    default = _make_small_learner()
    assert not default.compile_native_batch_norm
    assert default.native_batch_norm_module_count == 0
    requested = _make_small_learner(compile_native_batch_norm=True)
    manifest = _runner(requested)._cuda_graph_runtime_manifest()
    assert manifest["compile_native_batch_norm"]
    assert manifest["native_batch_norm_module_count"] == 0
    assert manifest["intentional_graph_breaks"] == []


@pytest.mark.parametrize(
    "graph_option", ["use_whole_cycle_cuda_graph", "dp_whole_cycle_cuda_graph"]
)
def test_native_batch_norm_rejects_whole_cycle_requests(graph_option):
    with pytest.raises(ValueError, match="compile_native_batch_norm introduces explicit"):
        _make_small_learner(compile_native_batch_norm=True, **{graph_option: True})
