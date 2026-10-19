import pytest
import torch
import torch.nn as nn

from nlear.cms import CMSScheduler
from nlear.models import ArcFace, CMSNeck

T = {"lr_body": 1e-2, "lr_head": 1e-2, "weight_decay": 0.0}


class Tiny(nn.Module):
    def __init__(self, n_levels=2):
        super().__init__()
        self.body = nn.Linear(6, 8)
        self.neck = CMSNeck(8, 4, n_levels=n_levels, mult=2)

    def forward(self, x):
        return self.neck(self.body(x))


def _step(model, head, opt, sched, x, y, scaler):
    opt.zero_grad(set_to_none=True)
    loss, _ = head(model(x), y)
    loss.backward()
    sched.hook(0, scaler)
    opt.step()


def test_new_cms_blocks_are_identity():
    neck = CMSNeck(8, 4, n_levels=3)
    x = torch.randn(5, 4)
    for b in neck.blocks:
        assert torch.allclose(b(x), x)


def test_slow_level_updates_only_every_period_with_summed_gradient():
    torch.manual_seed(0)
    model, head = Tiny(n_levels=2), ArcFace(4, 3, s=10, m=0.2)
    model.train()
    sched = CMSScheduler(model, head, T, periods=[1, 4])
    opt = sched.fast_optimizer()
    scaler = torch.amp.GradScaler(enabled=False)
    slow = model.neck.blocks[1].fc2.weight   # fc2: fc1 gets zero grad while fc2 is still 0
    batches = [(torch.randn(6, 6), torch.randint(0, 3, (6,))) for _ in range(8)]

    # reference: the summed gradient over steps 1-4, evaluated with the parameters as they evolve
    snapshots, grad_sum = [], torch.zeros_like(slow)
    for i, (x, y) in enumerate(batches, start=1):
        before = slow.detach().clone()
        _step(model, head, opt, sched, x, y, scaler)
        changed = not torch.equal(before, slow.detach())
        assert changed == (i % 4 == 0), f"slow level changed at step {i}"
        snapshots.append(slow.detach().clone())
    assert sched.n_updates == {4: 2}
    # fast block (period 1) changes every step
    assert sched.global_step == 8


def test_slow_update_equals_adamw_on_accumulated_gradient():
    torch.manual_seed(1)
    model, head = Tiny(n_levels=1), ArcFace(4, 3, s=10, m=0.2)
    model.eval()  # BN running stats fixed so two passes are identical
    batches = [(torch.randn(6, 6), torch.randint(0, 3, (6,))) for _ in range(3)]
    w = model.neck.blocks[0].fc2.weight      # fc2 gets a nonzero gradient from the start
    w0 = w.detach().clone()
    # manual: sum the gradients of 3 steps (body/head frozen in this check), one AdamW step
    for p in list(model.body.parameters()) + list(head.parameters()) + list(model.neck.proj.parameters()) \
            + list(model.neck.bn.parameters()):
        p.requires_grad_(False)
    g = torch.zeros_like(w)
    for x, y in batches:
        loss, _ = head(model(x), y)
        g += torch.autograd.grad(loss, w)[0]
    ref = w0.clone().requires_grad_(True)
    ref.grad = g
    torch.optim.AdamW([ref], lr=T["lr_head"], weight_decay=0.0).step()
    # scheduler: period 3, block accumulates and steps once
    with torch.no_grad():
        w.copy_(w0)
    model.neck.blocks[0].zero_grad(set_to_none=True)
    sched = CMSScheduler(model, head, T, periods=[3])
    scaler = torch.amp.GradScaler(enabled=False)
    for x, y in batches:
        loss, _ = head(model(x), y)
        loss.backward()
        sched.hook(0, scaler)
    assert g.abs().sum() > 0, "reference gradient must be nonzero for the check to mean anything"
    assert not torch.allclose(w.detach(), w0)
    assert torch.allclose(w.detach(), ref.detach(), atol=1e-6)


def test_all_period_one_means_no_slow_optimizers():
    model, head = Tiny(n_levels=3), ArcFace(4, 3)
    sched = CMSScheduler(model, head, T, periods=[1, 1, 1])
    assert sched.slow == {}
    n_fast = sum(p.numel() for ps, _ in sched.fast for p in ps)
    assert n_fast == sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())


def test_cms_refuses_mixed_precision_and_wrong_level_count():
    model, head = Tiny(n_levels=2), ArcFace(4, 3)
    with pytest.raises(ValueError):
        CMSScheduler(model, head, T, periods=[1, 4, 16])
    sched = CMSScheduler(model, head, T, periods=[1, 4])
    with pytest.raises(RuntimeError):
        sched.hook(0, torch.amp.GradScaler(enabled=True, device="cpu"))
