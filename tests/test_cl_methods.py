import torch
import torch.nn as nn

from nlear.cl_methods import EWC, LwF, Replay
from nlear.models import ArcFace


def _setup(n=40, n_classes=4):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 8))
    head = ArcFace(8, n_classes, s=10, m=0.2)
    images = torch.randint(0, 255, (n, 3, 12, 10), dtype=torch.uint8)
    rows = torch.arange(n)
    labels = torch.arange(n) % n_classes
    return model, head, images, rows, labels


def test_ewc_penalty_is_zero_until_a_task_ends_then_grows_with_drift():
    model, head, images, rows, labels = _setup()
    ewc = EWC(lam=100.0, fisher_batches=3)
    x = torch.randn(2, 3, 12, 10)
    assert ewc.extra_loss(model, x, model(x), None) == 0
    gen = torch.Generator().manual_seed(0)
    ewc.end_task(model, head, images, rows, labels, torch.device("cpu"), None, 8, gen)
    assert all((f >= 0).all() for f in ewc.fisher) and sum(f.sum() for f in ewc.fisher) > 0
    assert ewc.extra_loss(model, x, model(x), None).item() == 0     # at the anchor
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1)
    assert ewc.extra_loss(model, x, model(x), None).item() > 0
    old = [f.clone() for f in ewc.fisher]
    ewc.end_task(model, head, images, rows, labels, torch.device("cpu"), None, 8, gen)
    assert all((new >= o - 1e-12).all() for new, o in zip(ewc.fisher, old))  # online sum, gamma=1


def test_lwf_distills_toward_previous_model_only():
    model, head, images, rows, labels = _setup()
    lwf = LwF(lam=1.0)
    x = torch.randn(3, 3, 12, 10)
    assert lwf.extra_loss(model, x, model(x), None) == 0
    lwf.end_task(model)
    assert lwf.extra_loss(model, x, model(x), None).item() < 1e-10
    with torch.no_grad():
        model[-1].weight.add_(torch.randn_like(model[-1].weight))
    assert lwf.extra_loss(model, x, model(x), None).item() > 0
    assert not any(p.requires_grad for p in lwf.old.parameters())


def test_replay_buffer_keeps_k_per_identity_and_mixes_half_batches():
    model, head, images, rows, labels = _setup(n=40, n_classes=4)
    rep = Replay(per_id=3, seed=0)
    gen = torch.Generator().manual_seed(0)
    first = rep.batch_fn(rows, labels, 8, gen)
    r, lab = first(0)
    assert len(r) == 8                                     # empty buffer: all current
    rep.end_task(model, head, images, rows[:20], labels[:20])
    assert len(rep.rows) == 4 * 3
    assert (torch.bincount(rep.labels) == 3).all()
    assert set(rep.rows.tolist()) <= set(range(20))       # only the finished task's images
    new_rows, new_labels = rows[20:] + 100, labels[20:]
    r, lab = rep.batch_fn(new_rows, new_labels, 8, gen)(0)
    assert (r[:4] >= 100).all() and (r[4:] < 20).all()     # half new task, half buffer
