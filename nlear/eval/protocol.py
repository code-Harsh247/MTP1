"""Run the full open-set protocol on saved embeddings: calibrate tau on val, report test."""
import numpy as np
import torch

from .openset import SCORERS, evaluate, threshold_at_far

FARS = (0.01, 0.05, 0.10)


def _subset(blob, split, role=None, known=None):
    m = blob["split"] == split
    if role is not None:
        m &= blob["role"] == role
    if known is not None:
        m &= blob["known"] == known
    return torch.as_tensor(np.where(m)[0])


def variant_blobs(blob, data_cfg, seed):
    """{'': blob, 'g5': blob with g5 roles, ...} for the evaluation-only split variants.

    Variants keep the same identities and splits and only change gallery/probe roles,
    so they reuse the same embeddings.
    """
    from pathlib import Path

    import pandas as pd

    out = {"": blob}
    template = Path(data_cfg["data"]["splits"].format(seed=seed))
    for name in (data_cfg["splits"].get("variants") or {}):
        vpath = template.with_name(template.name.replace("splits_", f"splits_{name}_", 1))
        v = pd.read_csv(vpath, dtype={"identity": str, "path": str}).set_index("path").loc[blob["path"]]
        assert (v["split"].to_numpy() == blob["split"]).all(), f"variant {name} changes splits"
        out[name] = dict(blob, role=v["role"].to_numpy(), known=v["known"].to_numpy())
    return out


def run_protocol(blob, scorer, fars=FARS, device="cpu"):
    """blob: dict with 'emb' [N,d] tensor and numpy arrays 'identity','split','role','known'.

    Returns {'val': metrics, 'test': metrics}; test includes cal_* numbers using val taus.
    """
    fn = SCORERS[scorer] if isinstance(scorer, str) else scorer
    emb = blob["emb"].to(device)
    # Training-identity embeddings: known non-matches, usable as negatives at enrollment.
    kw = {"neg": emb[_subset(blob, "train")]} if getattr(fn, "needs_train_negatives", False) else {}
    out, taus = {}, None
    for split in ("val", "test"):
        g = _subset(blob, split, "gallery", 1)
        p = _subset(blob, split, "probe")
        S, classes = fn(emb[g], blob["identity"][g.numpy()].tolist(), emb[p], **kw)
        labels = blob["identity"][p.numpy()].tolist()
        known = blob["known"][p.numpy()] == 1
        out[split] = evaluate(S, classes, labels, known, fars, calib_taus=taus)
        if split == "val":
            score = S.max(1).values.float().cpu().numpy()
            taus = {far: threshold_at_far(score[~known], far) for far in fars}
    return out
