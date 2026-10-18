"""Backbone (the slow NL level) and the ArcFace training head."""
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


class EmbeddingNet(nn.Module):
    """ImageNet-pretrained ResNet -> linear -> BN embedding (not normalized here)."""

    def __init__(self, arch="resnet18", emb_dim=512, pretrained=True, dropout=0.0):
        super().__init__()
        self.body, feat_dim = _resnet(arch, pretrained)
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

    def forward(self, emb, labels):
        cos = F.linear(F.normalize(emb), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        target = torch.cos(torch.acos(cos) + self.m)
        # Standard insightface guard: past theta = pi - m, cos(theta + m) stops being
        # monotonic, so fall back to a linear penalty there.
        th, mm = math.cos(math.pi - self.m), math.sin(math.pi - self.m) * self.m
        target = torch.where(cos > th, target, cos - mm)
        onehot = F.one_hot(labels, cos.shape[1]).bool()
        logits = torch.where(onehot, target, cos) * self.s
        return F.cross_entropy(logits, labels), cos
