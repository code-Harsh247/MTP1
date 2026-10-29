"""Backbone (the slow NL level) and the ArcFace training head."""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _resnet(name, pretrained):
    weights = {"resnet18": "IMAGENET1K_V1", "resnet50": "IMAGENET1K_V2"}[name] if pretrained else None
    net = getattr(torchvision.models, name)(weights=weights)
    feat_dim = net.fc.in_features
    net.fc = nn.Identity()
    return net, feat_dim


def is_vit(arch):
    return arch.startswith(("deit", "vit"))


def is_timm(arch):
    """Study v2 names models by their timm weight tag (e.g. "convnext_tiny.fb_in1k"); v1's
    bare "resnet18"/"resnet50" still mean the torchvision models, and bare ViT names timm."""
    return "." in arch or is_vit(arch)


def _timm(name, pretrained, img_size):
    """Any timm backbone as a feature extractor at the ear input size. Models with position
    embeddings or attention windows (ViT, DeiT, Swin) are built for that size, and timm resamples
    their pretrained tables. Stochastic depth is off for every family (Swin's default is 0.1)."""
    import timm
    kw = {"img_size": tuple(img_size)} if name.startswith(("deit", "vit", "swin")) else {}
    net = timm.create_model(name, pretrained=pretrained, num_classes=0, drop_path_rate=0.0, **kw)
    # MobileNetV3 keeps its 1x1 head conv: its output is head_hidden_size (1280), not num_features (960).
    return net, getattr(net, "head_hidden_size", None) or net.num_features


class CMSParallel(nn.Module):
    """CMS for a pretrained layer ([NL] Sec. 7.3 + independent variant, Eq. 74).

    `levels` copies of the pretrained module (a Transformer/ConvNeXt MLP, or a CNN's 1x1 conv),
    combined by a learnable weighted sum (softmax of logits, initialised uniform). At
    initialisation every copy equals the pretrained module, so this computes exactly what it
    did. The copies differ only through their update frequencies (nlear.cms.CMSScheduler.for_body):
    if all copies updated every step they would receive identical (Adam-normalised) updates and
    stay identical, i.e. the plain model.
    """

    def __init__(self, mlp, levels=3):
        super().__init__()
        self.copies = nn.ModuleList([copy.deepcopy(mlp) for _ in range(levels)])
        self.agg_logits = nn.Parameter(torch.zeros(levels))

    def forward(self, x):
        w = torch.softmax(self.agg_logits, 0)
        return sum(w[i] * m(x) for i, m in enumerate(self.copies))


CMSParallelMLP = CMSParallel   # v1 name (Option B)


def cms_targets(body):
    """(parent, attribute) of every pointwise layer that CMS replicates, per family (plan.md R3):
    the MLP of each Transformer / Swin / ConvNeXt block, the 1x1 convs of each ResNet bottleneck,
    and the expand / project 1x1 convs of each MobileNet block."""
    attrs = {"Block": ("mlp",), "SwinTransformerBlock": ("mlp",), "ConvNeXtBlock": ("mlp",),
             "Bottleneck": ("conv1", "conv3"), "InvertedResidual": ("conv_pw", "conv_pwl"),
             "DepthwiseSeparableConv": ("conv_pw",)}
    out = []
    for m in body.modules():
        for a in attrs.get(type(m).__name__, ()):
            layer = getattr(m, a)
            if isinstance(layer, nn.Conv2d) and layer.kernel_size != (1, 1):
                raise ValueError(f"{type(m).__name__}.{a} is not a 1x1 conv")
            out.append((m, a))
    return out


def apply_body_cms(body, levels):
    targets = cms_targets(body)
    if not targets:
        raise ValueError(f"no CMS targets in {type(body).__name__}")
    for parent, attr in targets:
        setattr(parent, attr, CMSParallel(getattr(parent, attr), levels))
    return len(targets)


class CMSBlock(nn.Module):
    """One CMS memory level ([NL] Eq. 70): residual MLP x + W2 gelu(W1 LN(x)).
    W2 starts at zero, so a new block is the identity and CMS starts equal to the plain neck."""

    def __init__(self, d, mult=2):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, mult * d)
        self.fc2 = nn.Linear(mult * d, d)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.ln(x))))


class CMSNeck(nn.Module):
    """Continuum Memory System neck, sequential variant ([NL] Eq. 70, 73): projection, then a
    chain of CMS blocks (levels), then BN. Which level updates how often is decided by
    nlear.cms.CMSScheduler, not here."""

    def __init__(self, feat_dim, emb_dim, n_levels=3, mult=2, dropout=0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(feat_dim, emb_dim, bias=False)
        self.blocks = nn.ModuleList([CMSBlock(emb_dim, mult) for _ in range(n_levels)])
        self.bn = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        x = self.proj(self.drop(x))
        for b in self.blocks:
            x = b(x)
        return self.bn(x)


class EmbeddingNet(nn.Module):
    """ImageNet-pretrained ResNet -> neck -> embedding (not normalized here).
    neck="linear": dropout, linear, BN (the Phase 2 baseline); neck="cms": CMSNeck."""

    def __init__(self, arch="resnet18", emb_dim=512, pretrained=True, dropout=0.0,
                 neck="linear", cms_levels=3, cms_mult=2, img_size=(128, 96), vit_cms_levels=0,
                 body_cms_levels=0):
        super().__init__()
        body_cms_levels = body_cms_levels or vit_cms_levels     # v1 Option B flag = the same thing
        if is_timm(arch):
            self.body, feat_dim = _timm(arch, pretrained, img_size)
        else:
            self.body, feat_dim = _resnet(arch, pretrained)
        if body_cms_levels:
            apply_body_cms(self.body, body_cms_levels)
        if neck == "cms":
            self.neck = CMSNeck(feat_dim, emb_dim, cms_levels, cms_mult, dropout)
        else:
            self.neck = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, emb_dim, bias=False),
                                      nn.BatchNorm1d(emb_dim))
        self.emb_dim = emb_dim

    def forward(self, x):
        return self.neck(self.body(x))


class ArcFace(nn.Module):
    """Additive angular margin head (Deng et al., 2019): logits = s*cos(theta + m) for the true class."""

    def __init__(self, emb_dim, num_classes, s=30.0, m=0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m

    def forward(self, emb, labels, active=None):
        """active: optional bool mask [num_classes]; inactive classes are excluded from the
        softmax (class-incremental learning: only classes seen so far compete)."""
        cos = F.linear(F.normalize(emb), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        target = torch.cos(torch.acos(cos) + self.m)
        # Standard insightface guard: past theta = pi - m, cos(theta + m) stops being
        # monotonic, so fall back to a linear penalty there.
        th, mm = math.cos(math.pi - self.m), math.sin(math.pi - self.m) * self.m
        target = torch.where(cos > th, target, cos - mm)
        onehot = F.one_hot(labels, cos.shape[1]).bool()
        logits = torch.where(onehot, target, cos) * self.s
        if active is not None:
            logits = logits.masked_fill(~active.to(logits.device), float("-inf"))
            cos = cos.masked_fill(~active.to(cos.device), -2.0)
        return F.cross_entropy(logits, labels), cos
