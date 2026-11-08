"""R5b: the snapshot with fixed mixing weights, and the task list."""
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from nlear.cms import TaskSnapshot
from nlear.models import CMSParallel

ROOT = Path(__file__).resolve().parents[1]


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(CMSParallel(nn.Linear(2, 2, bias=False), levels=3))


def test_snapshot_fixes_mixing_share_at_the_snapshot():
    m = _Toy()
    mod = m.body[0]
    snap = TaskSnapshot(m, after=2, frozen_share=0.95)
    assert mod.agg_logits.requires_grad                      # learned until the snapshot
    snap.end_task(0)                                          # end of task 1: not yet
    assert mod.agg_logits.requires_grad
    mod.agg_logits.grad = torch.ones(3)
    snap.end_task(1)                                          # end of task 2: fixed now
    w = torch.softmax(mod.agg_logits, 0)
    assert torch.allclose(w, torch.tensor([0.05, 0.475, 0.475]), atol=1e-6)
    assert not mod.agg_logits.requires_grad and mod.agg_logits.grad is None
    out = mod(torch.randn(4, 2)).sum()
    out.backward()
    assert mod.agg_logits.grad is None and mod.copies[0].weight.grad is not None


def test_fixed_mixing_survives_optimizer_steps():
    m = _Toy()
    mod = m.body[0]
    opt = torch.optim.AdamW(m.parameters(), lr=0.1, weight_decay=0.05)
    snap = TaskSnapshot(m, after=1, frozen_share=0.8)
    for _ in range(3):                                        # task 1: logits learn
        opt.zero_grad(set_to_none=True)
        mod(torch.randn(4, 2)).pow(2).sum().backward()
        opt.step()
    snap.end_task(0)
    fixed = mod.agg_logits.detach().clone()
    for _ in range(3):                                        # later: logits stay put
        opt.zero_grad(set_to_none=True)
        mod(torch.randn(4, 2)).pow(2).sum().backward()
        opt.step()
    assert torch.equal(mod.agg_logits.detach(), fixed)


def test_share_must_be_inside_0_1():
    with pytest.raises(ValueError):
        TaskSnapshot(_Toy(), after=1, frozen_share=1.0)


def test_task_list():
    out = subprocess.run([sys.executable, "scripts/dgx/make_r5b_tasks.py", "--kaggle"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert len(out) == 48
    train = [ln for ln in out if "train_continual.py" in ln]
    assert sum("--snapshot-mix 0.95" in ln for ln in train) == 6
    assert sum("--snapshot-mix 0.8" in ln for ln in train) == 6
    assert sum("--snapshot-mix" not in ln for ln in train) == 32      # 4 EarVN x 3 + 4 small x 5
    assert sum("run_cms_diagnostics.py" in ln for ln in out) == 4
    assert all("--out runs_kaggle/" in ln for ln in out)
    assert len(set(out)) == len(out)
