"""Extreme Value Machine (Rudd et al., TPAMI 2017) as an open-set rejection baseline.

For every gallery embedding x_i of identity c we look at its distances to *negatives*
(embeddings known to be other people), keep the `tail` smallest half-distances (the
margin), and fit a Weibull to them. The probability that a probe z belongs to x_i's class:

    Psi_i(z) = exp(-(d(x_i, z) / lambda_i) ** kappa_i)      (Weibull survival function)

An identity's score is the max of Psi over its gallery embeddings; a probe is accepted as
known if its best score exceeds tau. d is cosine distance on L2-normalized embeddings.
Simplification vs. the paper: no extreme-vector set-cover reduction (we keep every gallery
image as an extreme vector; galleries have only 2-5 images per identity).

Negatives: "gallery" uses only other enrolled identities' gallery images (what the original
EVM sees); "train" also adds the training-identity embeddings, which are legitimately
available at enrollment time and give a much better estimate of the margin.
"""
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import weibull_min


def fit_weibull_low(tail):
    """MLE Weibull (loc=0) on the smallest margins; returns (kappa, lambda)."""
    tail = np.asarray(tail, dtype=np.float64)
    tail = tail[tail > 1e-6]
    if len(tail) < 2 or np.ptp(tail) < 1e-9:
        # Degenerate tail: fall back to a steep Weibull around the median margin.
        return 10.0, float(np.median(tail)) if len(tail) else 1e-3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kappa, _, lam = weibull_min.fit(tail, floc=0)
    return float(kappa), float(lam)


def score_evm(G, gl, P, neg=None, tail=10):
    """G gallery [n,d], gl identities, P probes [m,d], neg optional extra negatives [k,d]."""
    classes = sorted(set(gl))
    idx = {c: i for i, c in enumerate(classes)}
    lab = np.array([idx[c] for c in gl])
    G, P = F.normalize(G.float(), dim=1), F.normalize(P.float(), dim=1)
    D_gg = (1 - G @ G.T).clamp_min(0).cpu().numpy()
    D_gn = (1 - G @ F.normalize(neg.float(), dim=1).T).clamp_min(0).cpu().numpy() if neg is not None else None
    params = []
    for i in range(len(G)):
        d = D_gg[i][lab != lab[i]]
        if D_gn is not None:
            d = np.concatenate([d, D_gn[i]])
        params.append(fit_weibull_low(np.sort(d)[:tail] / 2.0))
    kappa = torch.tensor([k for k, _ in params], dtype=torch.float32)
    lam = torch.tensor([l for _, l in params], dtype=torch.float32)
    D_pg = (1 - P @ G.T).clamp_min(1e-12).cpu().double()
    # Score s = -log(-log Psi) = kappa * log(lambda / d), a strictly increasing function of
    # Psi, so max/argmax, rejection decisions and AUROC are exactly those of Psi. Psi itself
    # underflows to 0 beyond a tight margin (argmax becomes an arbitrary tie), and even
    # log Psi = -(d/lambda)^kappa overflows when a tight tail gives a huge kappa.
    s = kappa.double() * torch.log(lam.double() / D_pg)              # [m, n]
    S = torch.empty(P.shape[0], len(classes), dtype=torch.float64)
    for c in range(len(classes)):
        S[:, c] = s[:, torch.as_tensor(lab == c)].max(1).values
    return S, classes


def make_evm(tail=10, negatives="gallery"):
    def fn(G, gl, P, neg=None):
        return score_evm(G, gl, P, neg=neg if negatives == "train" else None, tail=tail)
    fn.needs_train_negatives = negatives == "train"
    fn.__name__ = f"evm_{negatives}_t{tail}"
    return fn
