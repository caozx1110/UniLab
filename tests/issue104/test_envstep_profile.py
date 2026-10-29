import torch
from unilab.base import envstep_profile as p


def test_disabled_window_preserves_value_and_rng(monkeypatch):
    monkeypatch.setattr(p,'ROOT',None)
    @p.window
    @p.timed('value')
    def f(x):return x+1
    torch.manual_seed(8);before=torch.get_rng_state().clone()
    assert f(torch.tensor(2)).item()==3
    assert torch.equal(before,torch.get_rng_state())
    assert not p.ACTIVE


def test_nested_event_order_and_leaf_operation_ownership(monkeypatch):
    class Event:
        clock=0
        def __init__(self,**kwargs):pass
        def record(self,stream=None):self.time=Event.clock;Event.clock+=1
        def elapsed_time(self,other):return other.time-self.time
    class Stream:cuda_stream=123
    monkeypatch.setattr(torch.cuda,'Event',Event)
    monkeypatch.setattr(torch.cuda,'current_stream',lambda:Stream())
    monkeypatch.setattr(p,'ACTIVE',True);monkeypatch.setattr(p,'ROWS',[]);monkeypatch.setattr(p,'STACK',[])
    with p.SyncCounts(),p.phase('parent'):
        with p.phase('child'):
            assert torch.tensor(True).item()
    parent,child=p.ROWS
    assert parent['parent'] is None and child['parent']==parent['id']
    assert child['operations']=={'aten._local_scalar_dense.default':1}
    assert not parent['operations'] and not p.STACK
    assert parent['start'].elapsed_time(parent['end'])==3
    assert child['start'].elapsed_time(child['end'])==1
