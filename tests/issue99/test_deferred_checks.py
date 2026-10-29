import pytest
import torch
from unilab.base.deferred_checks import DeferredChecks

@pytest.mark.parametrize('failure_step', [0, 1, 15, 30, 31])
def test_transient_failure_is_latched_until_boundary(failure_step):
    checks = DeferredChecks('cpu', 32)
    for step in range(32):
        checks.record(torch.tensor(step != failure_step))
        if step == 31:
            with pytest.raises(ValueError, match='Deferred device check failed'):
                checks.end_step()
        else:
            checks.end_step()
    with pytest.raises(ValueError):
        checks.flush()

def test_partial_window_flush_and_multiple_checks():
    checks = DeferredChecks('cpu', 32)
    checks.record(torch.tensor(False))
    checks.record(torch.tensor(True))
    checks.end_step()
    with pytest.raises(ValueError):
        checks.flush()

def test_successful_windows_and_default_immediate():
    checks = DeferredChecks('cpu')
    for _ in range(64):
        checks.record(torch.tensor(True))
        checks.end_step()
        assert not checks.pending
    checks.record(torch.tensor(False))
    with pytest.raises(ValueError):
        checks.end_step()

@pytest.mark.parametrize('interval', [0, -1, True, 1.5])
def test_invalid_interval(interval):
    with pytest.raises(ValueError):
        DeferredChecks('cpu', interval)
