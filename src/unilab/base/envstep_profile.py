"""Issue104 opt-in nested CUDA-event timings; never modifies tensors or RNG."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import json
import os
import time
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = os.environ.get('UNILAB_ENVSTEP_PROFILE')
ACTIVE = False
COUNT = 0
STACK = []
ROWS = []

class SyncCounts(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if STACK and any(x in name for x in ('_local_scalar_dense', 'nonzero', 'index.Tensor')):
            counts = STACK[-1]['operations']
            counts[name] = counts.get(name, 0) + 1
        return func(*args, **(kwargs or {}))

@contextmanager
def phase(name):
    if not ACTIVE:
        yield
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    row = {'name': name, 'iteration':COUNT, 'parent':(STACK[-1]['id'] if STACK else None), 'id':len(ROWS), 'operations':{}, 'start':start, 'end':end}
    ROWS.append(row); STACK.append(row)
    start.record(); host=time.perf_counter()
    try:
        yield
    finally:
        end.record(); row['host_ms']=(time.perf_counter()-host)*1000; STACK.pop()

def timed(name, group=False):
    def decorate(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            label = name
            if group:
                label += '/' + str(args[1] if len(args)>1 else kwargs['group_name'])
            with phase(label):
                return fn(*args, **kwargs)
        return wrapper
    return decorate

def window(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        global ACTIVE, COUNT
        if not ROOT:
            return fn(*args, **kwargs)
        COUNT += 1
        ACTIVE = 2001 <= COUNT <= 2200
        try:
            if ACTIVE:
                with SyncCounts(), phase('env.step'):
                    result = fn(*args, **kwargs)
            else:
                result = fn(*args, **kwargs)
        finally:
            ACTIVE = False
        if COUNT == 2200:
            torch.cuda.synchronize()
            records=[]
            for row in ROWS:
                records.append({k:v for k,v in row.items() if k not in ('start','end')} | {'gpu_elapsed_ms':row['start'].elapsed_time(row['end'])})
            for row in records:
                row['gpu_exclusive_ms']=row['gpu_elapsed_ms']-sum(x['gpu_elapsed_ms'] for x in records if x['parent']==row['id'])
            root=Path(ROOT);root.mkdir(parents=True,exist_ok=True)
            (root/f'events-{os.getpid()}.json').write_text(json.dumps({'window':[2001,2200],'device':torch.cuda.get_device_name(),'records':records},indent=2))
        return result
    return wrapper
