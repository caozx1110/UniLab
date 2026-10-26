from unittest.mock import MagicMock

import pytest
import torch

from unilab.training.tensor_diagnostics import NanGuard, NanGuardCfg

from .test_torch_env import _actions, _backend, _StubTorchEnv


def continuation_backend():
    backend = _backend()
    backend.export_continuation_state = MagicMock(return_value={"physics": torch.tensor([3.0])})
    backend.import_continuation_state = MagicMock()
    return backend


def test_completed_transition_restores_terminal_payload_and_guard_without_reset():
    original = _StubTorchEnv(backend=continuation_backend())
    original.init_state()
    original.set_nan_guard(NanGuard(NanGuardCfg(enabled=True), 3, False))
    original.step(_actions(original))
    original.state.info["completion"] = object()  # Settled IPC event is deliberately not durable.
    original._nan_guard.capture(torch.arange(12.0).reshape(3, 4))
    saved = original.export_continuation_state()
    restored = _StubTorchEnv(backend=continuation_backend())
    restored.init_state()
    restored.set_nan_guard(NanGuard(NanGuardCfg(enabled=True), 3, False))
    reset_count = len(restored.reset_ids)
    restored.import_continuation_state(saved)
    assert len(restored.reset_ids) == reset_count
    assert restored.step_counter == original.step_counter
    assert "completion" not in restored.state.info
    assert torch.equal(restored.state.info["steps"], original.state.info["steps"])
    for name in original.state.obs:
        assert torch.equal(restored.state.obs[name], original.state.obs[name])
        assert torch.equal(
            restored.state.final_observation[name], original.state.final_observation[name]
        )
    assert torch.equal(restored._nan_guard._buffer[-1], original._nan_guard._buffer[-1])
    restored.backend.import_continuation_state.assert_called_once()


def test_inflight_physics_or_changed_environment_identity_is_rejected():
    env = _StubTorchEnv(backend=continuation_backend())
    env.init_state()
    saved = env.export_continuation_state()
    env._pending_backend_step_completion = object()
    with pytest.raises(RuntimeError, match="pending physics"):
        env.export_continuation_state()
    env._pending_backend_step_completion = None
    saved["num_envs"] += 1
    with pytest.raises(ValueError, match="identity"):
        env.import_continuation_state(saved)
