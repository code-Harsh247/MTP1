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


def _vit(name, pretrained, img_size):
    """timm ViT/DeiT as a feature extractor at the ear input size (positional embeddings resampled)."""
    import timm
    net = timm.create_model(name, pretrained=pretrained, num_classes=0, img_size=tuple(img_size))
    return net, net.num_features


class CMSParallelMLP(nn.Module):
    """CMS for a pretrained Transformer MLP ([NL] Sec. 7.3 + independent variant, Eq. 74).

    `levels` copies of the pretrained MLP, combined by a learnable weighted sum (softmax of
    logits, initialised uniform). At initialisation every copy equals the pretrained MLP, so the
    module computes exactly what the pretrained block did. The copies differ only through their
    update frequencies (nlear.cms.CMSScheduler.for_vit): if all copies updated every step they
    would receive identical (Adam-normalised) updates and stay identical, i.e. a plain ViT.
    """

    def __init__(self, mlp, levels=3):
        super().__init__()
        self.copies = nn.ModuleList([copy.deepcopy(mlp) for _ in range(levels)])
        self.agg_logits = nn.Parameter(torch.zeros(levels))

    def forward(self, x):
        w = torch.softmax(self.agg_logits, 0)
        return sum(w[i] * m(x) for i, m in enumerate(self.copies))


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
                 neck="linear", cms_levels=3, cms_mult=2, img_size=(128, 96), vit_cms_levels=0):
        super().__init__()
        if is_vit(arch):
            self.body, feat_dim = _vit(arch, pretrained, img_size)
            if vit_cms_levels:
                for blk in self.body.blocks:
                    blk.mlp = CMSParallelMLP(blk.mlp, vit_cms_levels)
        else:
            self.body, feat_dim = _resnet(arch, pretrained)
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
