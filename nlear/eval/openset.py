"""Open-set identification with rejection: scorers and metrics.

A scorer turns gallery and probe embeddings into a score matrix S [n_probe, n_enrolled]
(higher = more similar). The probe is assigned argmax_c S and accepted as known if
max_c S > tau. Unknown probes are those whose identity was never enrolled.

Metrics:
  rank1      closed-set accuracy on known probes (no rejection)
  auroc      known-vs-unknown separation of the max score
  dir@far    Detection-and-Identification Rate: fraction of known probes that are accepted
             AND correctly identified, at the tau where the given fraction of unknown probes
             is (wrongly) accepted. "oracle": tau set on the test unknowns themselves (the
             usual curve in papers). "calibrated": tau set on validation unknowns, then applied
             to test. The calibrated numbers are the honest ones.
  eer, tar@far  1:1 verification over all (probe, enrolled identity) pairs.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve


def _classes(gallery_labels):
    classes = sorted(set(gallery_labels))
    return classes, {c: i for i, c in enumerate(classes)}


def score_cosine_proto(G, gl, P):
    """Cosine to each identity's mean (re-normalized) gallery embedding."""
    classes, idx = _classes(gl)
    G, P = F.normalize(G, dim=1), F.normalize(P, dim=1)
    lab = torch.tensor([idx[c] for c in gl], device=G.device)
    protos = torch.zeros(len(classes), G.shape[1], device=G.device).index_add_(0, lab, G)
    return P @ F.normalize(protos, dim=1).T, classes


def score_cosine_max(G, gl, P):
    """Cosine to the closest single gallery image of each identity."""
    classes, idx = _classes(gl)
    sims = F.normalize(P, dim=1) @ F.normalize(G, dim=1).T
    S = torch.full((P.shape[0], len(classes)), -2.0, device=P.device)
    for j, c in enumerate(gl):
        S[:, idx[c]] = torch.maximum(S[:, idx[c]], sims[:, j])
    return S, classes


def score_subspace(G, gl, P):
    """Norm of the probe's projection onto the span of each identity's gallery embeddings.

    For a unit probe p and orthonormal basis Q_c, ||Q_c^T p|| = cos(angle to the subspace),
    and the residual ||p - Q_c Q_c^T p|| = sqrt(1 - score^2). A linear associative memory
    written by the delta rule converges to this projection (notes/nl_equations.md 8.3), so
    this is the baseline any NL memory must beat.
    """
    classes, _ = _classes(gl)
    G, P = F.normalize(G, dim=1), F.normalize(P, dim=1)
    gl = np.asarray(gl)
    S = torch.empty(P.shape[0], len(classes), device=P.device)
    for k, c in enumerate(classes):
        Q, _ = torch.linalg.qr(G[torch.as_tensor(gl == c)].T)  # [d, n_c], orthonormal columns
        S[:, k] = (P @ Q).norm(dim=1)
    return S, classes


SCORERS = {"cosine_proto": score_cosine_proto, "cosine_max": score_cosine_max,
           "subspace": score_subspace}


def _register_evm():
    from .evm import make_evm  # local import: evm imports scipy
    # Tail sizes chosen on EarVN validation, 5 seeds (notes/phase2_baseline.md, EVM section;
    # log in runs/earvn/evm_tune.log). Rule: best val DIR@FAR=5%, tie-break val AUROC.
    SCORERS["evm"] = make_evm(tail=14, negatives="gallery")
    SCORERS["evm_trainneg"] = make_evm(tail=1000, negatives="train")


_register_evm()


def threshold_at_far(unknown_scores, far):
    """tau such that a fraction `far` of unknown probes has score > tau."""
    return float(np.quantile(unknown_scores, 1.0 - far)) if len(unknown_scores) else float("inf")


def evaluate(S, classes, probe_labels, probe_known, fars=(0.01, 0.05, 0.10), calib_taus=None):
    """S: [n_probe, n_classes] tensor. Returns a flat dict of metrics.

    calib_taus: {far: tau} chosen on validation; adds calibrated test numbers if given.
    """
    S = S.float().cpu().numpy()
    idx = {c: i for i, c in enumerate(classes)}
    own = np.array([idx.get(l, -1) for l in probe_labels])
    known = np.asarray(probe_known).astype(bool)
    assert (own[known] >= 0).all(), "a known probe's identity is not enrolled"
    assert (own[~known] == -1).all(), "an unknown probe's identity is enrolled"

    pred, score = S.argmax(1), S.max(1)
    correct = pred == own
    out = {"n_known": int(known.sum()), "n_unknown": int((~known).sum()),
           "n_classes": len(classes), "rank1": float(correct[known].mean())}
    if (~known).any():
        out["auroc"] = float(roc_auc_score(known, score))
        for far in fars:
            tau = threshold_at_far(score[~known], far)
            out[f"dir@far{far:g}"] = float((correct & (score > tau))[known].mean())
        for far, tau in (calib_taus or {}).items():
            out[f"cal_dir@far{far:g}"] = float((correct & (score > tau))[known].mean())
            out[f"cal_far@far{far:g}"] = float((score[~known] > tau).mean())

    # 1:1 verification: genuine = known probe vs its own identity, impostor = everything else.
    is_gen = np.zeros_like(S, dtype=bool)
    is_gen[np.where(known)[0], own[known]] = True
    fpr, tpr, _ = roc_curve(is_gen.ravel(), S.ravel())
    out["eer"] = float(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))])
    out["tar@far0.01"] = float(np.interp(0.01, fpr, tpr))
    return out
