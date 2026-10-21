import copy

import torch

from nlear.cms import CMSScheduler
from nlear.models import ArcFace, CMSParallelMLP, EmbeddingNet

T = {"lr_body": 1e-3, "lr_head": 1e-3, "weight_decay": 0.0}
ARCH, SIZE = "vit_tiny_patch16_224", (32, 32)


def _plain_and_cms(levels=3, seed=0):
    torch.manual_seed(seed)
    plain = EmbeddingNet(ARCH, emb_dim=16, pretrained=False, img_size=SIZE)
    cms = copy.deepcopy(plain)
    for blk in cms.body.blocks:
        blk.mlp = CMSParallelMLP(blk.mlp, levels)
    return plain.eval(), cms.eval()


def test_cms_vit_starts_exactly_equal_to_the_plain_vit():
    plain, cms = _plain_and_cms()
    x = torch.randn(3, 3, *SIZE)
    assert torch.allclose(plain(x), cms(x), atol=1e-5)
    built = EmbeddingNet(ARCH, emb_dim=16, pretrained=False, img_size=SIZE, vit_cms_levels=3)
    assert all(isinstance(b.mlp, CMSParallelMLP) and len(b.mlp.copies) == 3 for b in built.body.blocks)


def _train(model, head, sched, steps, seed=1):
    g = torch.Generator().manual_seed(seed)
    opt = sched.fast_optimizer()
    scaler = torch.amp.GradScaler(enabled=False)
    model.train()
    for _ in range(steps):
        x = torch.randn(4, 3, *SIZE, generator=g)
        y = torch.randint(0, 5, (4,), generator=g)
        opt.zero_grad(set_to_none=True)
        loss, _ = head(model(x), y)
        loss.backward()
        sched.hook(0, scaler)
        opt.step()


def test_slow_copies_update_only_on_their_period():
    _, cms = _plain_and_cms()
    head = ArcFace(16, 5, s=10, m=0.2)
    sched = CMSScheduler.for_vit(cms, head, T, periods=[1, 3, 6])
    w = [cms.body.blocks[0].mlp.copies[i].fc2.weight for i in range(3)]
    history = []
    for _ in range(6):
        before = [x.detach().clone() for x in w]
        _train(cms, head, sched, 1)
        history.append([not torch.equal(b, x.detach()) for b, x in zip(before, w)])
    assert [h[0] for h in history] == [True] * 6
    assert [h[1] for h in history] == [False, False, True, False, False, True]
    assert [h[2] for h in history] == [False, False, False, False, False, True]
    assert sched.n_updates == {3: 2, 6: 1}


def test_all_period_one_keeps_copies_identical_i_e_plain_vit():
    _, cms = _plain_and_cms()
    head = ArcFace(16, 5, s=10, m=0.2)
    sched = CMSScheduler.for_vit(cms, head, T, periods=[1, 1, 1])
    assert sched.slow == {}
    _train(cms, head, sched, 5)
    for blk in cms.body.blocks:
        c0, c1, c2 = (list(c.parameters()) for c in blk.mlp.copies)
        for a, b, c in zip(c0, c1, c2):
            assert torch.allclose(a, b, atol=1e-6) and torch.allclose(a, c, atol=1e-6)


def test_multi_frequency_makes_copies_diverge():
    _, cms = _plain_and_cms()
    head = ArcFace(16, 5, s=10, m=0.2)
    sched = CMSScheduler.for_vit(cms, head, T, periods=[1, 4, 16])
    _train(cms, head, sched, 8)
    c = cms.body.blocks[0].mlp.copies
    assert not torch.allclose(c[0].fc1.weight, c[1].fc1.weight)
    assert torch.equal(c[2].fc1.weight, _plain_and_cms()[1].body.blocks[0].mlp.copies[2].fc1.weight)  # never updated yet
