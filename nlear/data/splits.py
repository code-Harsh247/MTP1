"""Identity-disjoint open-set splits.

Every image gets a `split` and a `role`:

    split  role     known  meaning
    train  train    -1     used to train the backbone / meta-train the memory
    val    gallery   1     enrollment images of a known validation identity
    val    probe     1     query images of a known validation identity
    val    probe     0     query images of an unknown validation identity (never enrolled)
    test   (same as val)

Train, val and test identities never overlap. Unknown identities never appear in
any gallery. The rejection threshold is calibrated on val only.
"""
import numpy as np
import pandas as pd


def _assign_identities(idents, rng, frac_train, frac_val, known_frac):
    idents = np.array(sorted(idents))
    rng.shuffle(idents)
    n = len(idents)
    n_train = int(round(frac_train * n))
    n_val = int(round(frac_val * n))
    out = {}
    for i in idents[:n_train]:
        out[i] = ("train", -1)
    for split, block in (("val", idents[n_train:n_train + n_val]),
                         ("test", idents[n_train + n_val:])):
        n_known = int(round(known_frac * len(block)))
        for j, i in enumerate(block):
            out[i] = (split, 1 if j < n_known else 0)
    return out


def make_splits(manifest, seed=0, min_images=4, frac_train=0.7, frac_val=0.1,
                known_frac=0.5, n_gallery=2, stratify_by="source"):
    """Return the manifest with split/role/known columns added (dropped identities removed)."""
    if n_gallery >= min_images:
        raise ValueError("n_gallery must be < min_images so every known identity has probes")
    rng = np.random.default_rng(seed)
    counts = manifest.groupby("identity").size()
    keep = counts[counts >= min_images].index
    df = manifest[manifest["identity"].isin(keep)].copy()

    # Shuffle identities separately within each source so every split has the same source mix.
    groups = df.groupby(stratify_by)["identity"].unique() if stratify_by in df.columns \
        else pd.Series({"all": df["identity"].unique()})
    assign = {}
    for _, idents in groups.items():
        assign.update(_assign_identities(idents, rng, frac_train, frac_val, known_frac))

    df["split"] = df["identity"].map(lambda i: assign[i][0])
    df["known"] = df["identity"].map(lambda i: assign[i][1])
    df["role"] = np.where(df["split"] == "train", "train", "probe")

    # First n_gallery images (after a seeded shuffle) of each known identity are the gallery.
    known = df[df["known"] == 1]
    order = known.sample(frac=1.0, random_state=seed).groupby("identity").cumcount()
    gallery_idx = order[order < n_gallery].index
    df.loc[gallery_idx, "role"] = "gallery"

    check_splits(df)
    return df.sort_values(["split", "identity", "path"]).reset_index(drop=True)


def check_splits(df):
    """Raise if any open-set protocol rule is broken."""
    per_ident = df.groupby("identity")["split"].nunique()
    assert (per_ident == 1).all(), "an identity appears in more than one split"
    assert not ((df["known"] == 0) & (df["role"] == "gallery")).any(), "unknown identity in a gallery"
    assert not ((df["split"] == "train") & (df["role"] != "train")).any(), "train rows must have role=train"
    for split in ("val", "test"):
        k = df[(df["split"] == split) & (df["known"] == 1)]
        if len(k):
            g = k[k["role"] == "gallery"].groupby("identity").size()
            p = k[k["role"] == "probe"].groupby("identity").size()
            assert set(g.index) == set(k["identity"]), f"{split}: known identity without gallery"
            assert set(p.index) == set(k["identity"]), f"{split}: known identity without probes"


def summarize(df):
    rows = []
    for split in ("train", "val", "test"):
        s = df[df["split"] == split]
        rows.append({
            "split": split,
            "identities": s["identity"].nunique(),
            "known_ids": s[s["known"] == 1]["identity"].nunique(),
            "unknown_ids": s[s["known"] == 0]["identity"].nunique(),
            "images": len(s),
            "gallery_imgs": int((s["role"] == "gallery").sum()),
            "probe_imgs": int((s["role"] == "probe").sum()),
        })
    return pd.DataFrame(rows)
