"""R5: the per-step slow update, the frozen snapshot copies and the task list."""
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

from nlear.cms import CMSScheduler, TaskSnapshot, _PerStepSlow
from nlear.models import CMSParallel

ROOT = Path(__file__).resolve().parents[1]


class _Toy(nn.Module):
    def __init__(self, levels=3):
        super().__init__()
        self.body = nn.Sequential(CMSParallel(nn.Linear(2, 2, bias=False), levels=levels))
        self.neck = nn.Linear(2, 2)


def _weights(m):
    return [c.weight.detach().clone() for c in m.body[0].copies]


def test_snapshot_copies_copy1_once_then_frozen():
    m = _Toy()
    for c, v in zip(m.body[0].copies, (2.0, 0.0, 0.0)):
        c.weight.data.fill_(v)
    snap = TaskSnapshot(m, after=1)
    assert all(not p.requires_grad for c in m.body[0].copies[1:] for p in c.parameters())
    snap.end_task(0)                                   # end of task 1: copies 2-3 take copy 1
    assert all(torch.equal(w, _weights(m)[0]) for w in _weights(m)[1:])
    m.body[0].copies[0].weight.data.fill_(5.0)
    snap.end_task(1)                                   # later tasks: nothing changes
    assert float(m.body[0].copies[1].weight[0, 0]) == 2.0
    assert snap.n_updates == {1: 1, 2: 1}


def test_per_step_equals_ordinary_adamw_applied_late():
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(3))
    ref = nn.Parameter(p.detach().clone())
    slow = _PerStepSlow([([p], 0.1)], weight_decay=0.05)
    ref_opt = torch.optim.AdamW([ref], lr=0.1, weight_decay=0.05)
    start = p.detach().clone()
    grads = [torch.randn(3) for _ in range(4)]
    for g in grads:
        p.grad = g.clone()
        slow.accumulate()
        assert p.grad is None
        assert torch.equal(p.detach(), start)          # live weights wait for the period
        ref.grad = g.clone()
        ref_opt.step()
    slow.apply()
    assert torch.allclose(p.detach(), ref.detach())


def test_scheduler_per_step_mode_updates_only_on_period():
    torch.manual_seed(0)
    m, head = _Toy(levels=2), nn.Linear(2, 2)
    t = {"lr_body": 0.01, "lr_head": 0.01, "weight_decay": 0.0}
    cms = CMSScheduler.for_body(m, head, t, [1, 4], slow_mode="per-step")
    opt = cms.fast_optimizer()
    scaler = torch.amp.GradScaler(enabled=False)
    before = _weights(m)[1]
    for step in range(4):
        opt.zero_grad(set_to_none=True)
        head(m.neck(m.body(torch.randn(8, 2)))).pow(2).sum().backward()
        cms.hook(step, scaler)
        opt.step()
        moved = not torch.equal(_weights(m)[1], before)
        assert moved == (step == 3)
    assert cms.n_updates == {4: 1}


def test_task_list():
    out = subprocess.run([sys.executable, "scripts/dgx/make_r5_tasks.py", "--kaggle"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert len(out) == 18
    assert sum("--body-cms-slow per-step" in ln for ln in out) == 6
    assert sum("--body-cms-lr-pow 1 " in ln for ln in out) == 6
    assert sum("--snapshot-after-task 1" in ln for ln in out) == 3
    assert sum("run_cms_diagnostics.py" in ln for ln in out) == 3
    assert all("runs_kaggle/" in ln for ln in out)
