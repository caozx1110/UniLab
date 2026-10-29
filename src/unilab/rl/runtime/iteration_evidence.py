"""Opt-in engineering evidence; no tensor-to-host transfers in collection."""
import atexit
import json
import os
import time
from pathlib import Path

import torch

_ROOT = os.environ.get('UNILAB_ITERATION_EVIDENCE')
ENABLED = bool(_ROOT)
_ROWS = {}

def record(role, values):
    if not _ROOT:
        return
    rows = _ROWS.setdefault(role, [])
    limit = int(os.environ.get('UNILAB_EVIDENCE_LIMIT', '2000'))
    if len(rows) >= limit:
        return
    rows.append({k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in values.items()})
    if len(rows) == limit:
        flush(role)

def flush(role=None):
    if not _ROOT:
        return
    root = Path(_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    for name in ([role] if role else list(_ROWS)):
        rows = _ROWS[name]
        if not rows:
            continue
        # One bulk D2H per tensor column, outside the measured prefix.
        result = {}
        for key in rows[0]:
            values = [row[key] for row in rows]
            result[key] = torch.stack(values).cpu().tolist() if isinstance(values[0], torch.Tensor) else values
        path = root / f'{name}-{os.getpid()}.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'role': name, 'rows': len(rows), 'columns': result}))
        temporary.replace(path)

atexit.register(flush)
