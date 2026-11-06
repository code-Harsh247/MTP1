"""R4b: task-boundary consolidation between CMS copies (nlear.cms.TaskConsolidation)."""
import torch
import torch.nn as nn

from nlear.cms import TaskConsolidation
from nlear.models import CMSParallel


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(CMSParallel(nn.Linear(2, 2, bias=False), levels=3))


def _set(m, values):
    for c, v in zip(m.body[0].copies, values):
        c.weight.data.fill_(v)


def _get(m):
    return [float(c.weight[0, 0]) for c in m.body[0].copies]


def test_schedule_and_order():
    m = _Toy()
    _set(m, [2.0, 0.0, 0.0])
    tc = TaskConsolidation(m, every=[1, 2], beta=0.5)
    tc.end_task(0)                                   # copy 2 moves, copy 3 waits
    assert _get(m) == [2.0, 1.0, 0.0]
    tc.end_task(1)                                   # copy 3 takes copy 2's old value first
    assert _get(m) == [2.0, 1.5, 0.5]
    assert tc.n_updates == {1: 2, 2: 1}


def test_only_the_fast_copy_trains():
    m = _Toy()
    tc = TaskConsolidation(m, every=[1, 2], beta=0.5)
    before = [c.weight.detach().clone() for c in m.body[0].copies]
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=0.1)
    m.body(torch.randn(4, 2)).pow(2).sum().backward()
    opt.step()
    after = [c.weight for c in m.body[0].copies]
    assert not torch.equal(before[0], after[0])
    assert torch.equal(before[1], after[1]) and torch.equal(before[2], after[2])
    assert all(c.weight.grad is None for c in m.body[0].copies[1:])
    assert tc.mods == [m.body[0]]
