"""Study v2 (plan.md R3): CMS inside the backbone of every model family.

For each family: the CMS model computes exactly the plain model at initialisation, the
right layers are replicated, slow copies update only on their period, and with all
periods = 1 the copies stay identical (so CMS-X with periods 1,1,1 is just X)."""
import copy

import pytest
import torch
import torch.nn as nn

from nlear.cms import CMSScheduler
from nlear.models import ArcFace, CMSParallel, EmbeddingNet, cms_targets

T = {"lr_body": 1e-3, "lr_head": 1e-3, "weight_decay": 0.0}
SIZE = (64, 48)
# family -> (timm tag, number of replicated layers in the real model)
FAMILIES = {
    "resnet50": ("resnet50.tv2_in1k", 32),                     # 16 bottlenecks x (conv1, conv3)
    "convnext": ("convnext_tiny.fb_in1k", 18),                 # 18 blocks x mlp
    "mobilenet": ("mobilenetv3_large_100.ra_in1k", 29),        # 1 dw-separable + 14 inverted residual x 2
    "deit": ("deit_small_patch16_224.fb_in1k", 12),            # 12 blocks x mlp
    "swin": ("swin_tiny_patch4_window7_224.ms_in1k", 12),      # 12 blocks x mlp
}


def _pair(tag, levels=3, seed=0):
    torch.manual_seed(seed)
    plain = EmbeddingNet(tag, emb_dim=16, pretrained=False, img_size=SIZE)
    torch.manual_seed(seed)
    cms = EmbeddingNet(tag, emb_dim=16, pretrained=False, img_size=SIZE, body_cms_levels=levels)
    # timm's random ResNet init zeroes each bottleneck's last BN scale ("zero_init_last"), which
    # blocks every gradient into the branch on the first step; pretrained weights never do this.
    for m in plain.modules():
        if isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
    cms.load_state_dict(_expand(plain.state_dict(), cms))
    return plain, cms


def _expand(sd, cms):
    """Plain weights -> CMS weights: every copy gets the original layer's weights."""
    out = {}
    for k in cms.state_dict():
        if ".copies." in k:
            pre, rest = k.split(".copies.", 1)
            out[k] = sd[f"{pre}.{rest.split('.', 1)[1]}"]
        elif k.endswith("agg_logits"):
            out[k] = cms.state_dict()[k]
        else:
            out[k] = sd[k]
    return out


@pytest.mark.parametrize("family", list(FAMILIES))
def test_cms_replicates_the_pointwise_layers(family):
    tag, n = FAMILIES[family]
    plain = EmbeddingNet(tag, emb_dim=16, pretrained=False, img_size=SIZE)
    targets = cms_targets(plain.body)
    assert len(targets) == n
    for parent, attr in targets:
        layer = getattr(parent, attr)
        assert not isinstance(layer, nn.Conv2d) or layer.kernel_size == (1, 1)
    cms = EmbeddingNet(tag, emb_dim=16, pretrained=False, img_size=SIZE, body_cms_levels=3)
    mods = [m for m in cms.body.modules() if isinstance(m, CMSParallel)]
    assert len(mods) == n and all(len(m.copies) == 3 for m in mods)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_cms_model_starts_exactly_equal_to_the_plain_model(family):
    plain, cms = _pair(FAMILIES[family][0])
    x = torch.randn(3, 3, *SIZE)
    with torch.no_grad():
        assert torch.allclose(plain.eval()(x), cms.eval()(x), atol=1e-4)


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


@pytest.mark.parametrize("family", ["resnet50", "convnext", "mobilenet", "swin"])
def test_slow_copies_update_only_on_their_period(family):
    _, cms = _pair(FAMILIES[family][0])
    head = ArcFace(16, 5, s=10, m=0.2)
    sched = CMSScheduler.for_body(cms, head, T, periods=[1, 2, 4])
    mod = next(m for m in cms.body.modules() if isinstance(m, CMSParallel))
    w = [next(c.parameters()) for c in mod.copies]
    changed = []
    for _ in range(4):
        before = [p.detach().clone() for p in w]
        _train(cms, head, sched, 1)
        changed.append([not torch.equal(b, p.detach()) for b, p in zip(before, w)])
    assert [c[0] for c in changed] == [True] * 4
    assert [c[1] for c in changed] == [False, True, False, True]
    assert [c[2] for c in changed] == [False, False, False, True]


@pytest.mark.parametrize("family", ["resnet50", "convnext", "mobilenet", "swin"])
def test_all_period_one_keeps_copies_identical(family):
    _, cms = _pair(FAMILIES[family][0])
    head = ArcFace(16, 5, s=10, m=0.2)
    sched = CMSScheduler.for_body(cms, head, T, periods=[1, 1, 1])
    assert sched.slow == {}
    _train(cms, head, sched, 3)
    for m in cms.body.modules():
        if isinstance(m, CMSParallel):
            c0, c1, c2 = (list(c.parameters()) for c in m.copies)
            for a, b, c in zip(c0, c1, c2):
                assert torch.allclose(a, b, atol=1e-5) and torch.allclose(a, c, atol=1e-5)


def test_v1_vit_flag_is_the_same_as_body_cms():
    tag = FAMILIES["deit"][0]
    a = EmbeddingNet(tag, emb_dim=16, pretrained=False, img_size=SIZE, vit_cms_levels=3)
    b = EmbeddingNet(tag, emb_dim=16, pretrained=False, img_size=SIZE, body_cms_levels=3)
    assert a.state_dict().keys() == b.state_dict().keys()
    assert all(isinstance(blk.mlp, CMSParallel) for blk in a.body.blocks)
