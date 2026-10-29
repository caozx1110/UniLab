"""Opt-in isolated 200-iteration CPU/CUDA profiler window after warmup."""
import os
from pathlib import Path
import torch

_ROOT = os.environ.get('UNILAB_PROFILE_WINDOW')
_PROFILERS = {}
_COUNTS = {}


def begin(role):
    if not _ROOT:
        return
    count = _COUNTS.get(role, 0) + 1
    _COUNTS[role] = count
    if count == 2001:
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=False, with_stack=False)
        profiler.start()
        _PROFILERS[role] = profiler


def end(role):
    if not _ROOT:
        return
    profiler = _PROFILERS.get(role)
    if profiler is None:
        return
    if _COUNTS[role] == 2200:
        profiler.stop()
        root = Path(_ROOT); root.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(root / f'{role}-{os.getpid()}.json'))
        (root / f'{role}-{os.getpid()}.txt').write_text(profiler.key_averages().table(sort_by='self_device_time_total', row_limit=100))
        del _PROFILERS[role]
