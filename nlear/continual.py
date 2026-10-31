"""Continual-learning protocol P1 (identity-incremental), see plan_continual.md section 3.

The train identities of an open-set split are dealt into n_tasks tasks. For every train
identity a fraction of its images is held out ("seen-person test images") so forgetting of
already-learned people can be measured; the rest are training images for its task.
Val/test identities are untouched: they stay the unseen-people open-set protocol.

Columns added to the split DataFrame:
    task        0..n_tasks-1 for train identities, -1 for val/test
    cl_role     train | held_gallery | held_probe   (train identities)
                eval                                (val/test rows; their own role column applies)
    cl_gallery  True for the rows that enroll a train identity when its forgetting is measured

Few-image datasets (IITD: median 3 images per person) cannot spare separate held-out gallery
images. With n_gallery=0 a person's own *training* images are its gallery and only probes are
held out, so A[t][j] still measures how well held-out probes match what the model learned.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .eval.openset import evaluate, score_cosine_proto


def make_task_splits(df, n_tasks=5, holdout_frac=0.2, n_gallery=2, seed=0):
    df = df.copy()
    rng = np.random.default_rng(2000 + seed)
    train_ids = np.array(sorted(df.loc[df["split"] == "train", "identity"].unique()))
    rng.shuffle(train_ids)
    task_of = {i: t for t, block in enumerate(np.array_split(train_ids, n_tasks)) for i in block}
    df["task"] = df["identity"].map(task_of).fillna(-1).astype(int)
    df["cl_role"] = np.where(df["split"] == "train", "train", "eval")

    for ident in train_ids:
        idx = df.index[df["identity"] == ident].to_numpy()
        idx = idx[rng.permutation(len(idx))]
        n_hold = max(n_gallery + 1, int(round(holdout_frac * len(idx))))
        if n_hold >= len(idx):
            raise ValueError(f"identity {ident} has too few images ({len(idx)}) to hold out {n_hold}")
        df.loc[idx[:n_gallery], "cl_role"] = "held_gallery"
        df.loc[idx[n_gallery:n_hold], "cl_role"] = "held_probe"
    gallery_role = "held_gallery" if n_gallery > 0 else "train"
    df["cl_gallery"] = (df["split"] == "train") & (df["cl_role"] == gallery_role)
    check_task_splits(df, n_tasks, n_gallery)
    return df


def check_task_splits(df, n_tasks, n_gallery=2):
    tr = df[df["split"] == "train"]
    assert set(tr["task"]) == set(range(n_tasks)), "every task needs identities"
    assert (df.loc[df["split"] != "train", "task"] == -1).all(), "val/test rows must not have a task"
    assert (tr.groupby("identity")["task"].nunique() == 1).all(), "an identity is in two tasks"
    roles = ("train", "held_gallery", "held_probe") if n_gallery > 0 else ("train", "held_probe")
    if n_gallery == 0:
        assert not (tr["cl_role"] == "held_gallery").any(), "n_gallery=0 must not hold out gallery images"
    for role in roles:
        has = set(tr.loc[tr["cl_role"] == role, "identity"])
        assert has == set(tr["identity"]), f"some train identity lacks {role} images"


def load_frozen_tasks(name, splits_template, seed, n_tasks, n_gallery, bench_dir="benchmark"):
    """Study v2: read the frozen task file of a dataset and seed, after checking that it and its
    split file have the md5 pinned in benchmark/hashes_<name>.txt. Any mismatch is an error."""
    split_file = Path(splits_template.format(seed=seed))
    task_file = split_file.with_name(split_file.name.replace("splits_", "tasks_", 1))
    pinned = {}
    for line in (Path(bench_dir) / f"hashes_{name}.txt").read_text().splitlines():
        digest, fname = line.split()
        pinned[fname] = digest
    for f in (split_file, task_file):
        if f.name not in pinned:
            raise ValueError(f"{f.name} is not in the frozen benchmark of {name}")
        if hashlib.md5(f.read_bytes()).hexdigest() != pinned[f.name]:
            raise ValueError(f"{f} differs from its frozen hash in hashes_{name}.txt")
    df = pd.read_csv(task_file, dtype={"identity": str, "path": str})
    check_task_splits(df, n_tasks, n_gallery)
    return df


def seen_task_metrics(emb, df, task, unknown_mask):
    """Open-set identification of ONE task's people from their held-out images.

    Gallery: the cl_gallery images of the task's identities (held-out gallery images, or the
    training images for few-image datasets). Known probes: their held_probe images. Unknown probes: a fixed set of never-trained people (unknown_mask rows), the same
    for every task and every stage, so A[t, j] is comparable across t.
    """
    t = df["task"].to_numpy() == task
    role = df["cl_role"].to_numpy()
    g = np.where(t & df["cl_gallery"].to_numpy())[0]
    pk = np.where(t & (role == "held_probe"))[0]
    pu = np.where(unknown_mask)[0]
    p = np.concatenate([pk, pu])
    ids = df["identity"].to_numpy()
    S, classes = score_cosine_proto(emb[g], ids[g].tolist(), emb[p])
    known = np.r_[np.ones(len(pk), bool), np.zeros(len(pu), bool)]
    return evaluate(S, classes, ids[p].tolist(), known)


def continual_summary(A, key="dir@far0.05"):
    """A: list over stages t of list over tasks j of metric dicts (A[t][j]).

    final_avg   mean_j A[T-1][j]                         (performance after learning everything)
    forgetting  mean_{j<T-1} max_{t>=j, t<T-1} A[t][j] - A[T-1][j]   (largest drop since learned)
    bwt         mean_{j<T-1} A[T-1][j] - A[j][j]         (backward transfer; negative = forgetting)
    """
    M = np.array([[A[t][j][key] for j in range(len(A[t]))] for t in range(len(A))])
    T = M.shape[0]
    out = {f"final_avg_{key}": float(M[-1].mean())}
    if T > 1:
        out[f"forgetting_{key}"] = float(np.mean([M[j:T - 1, j].max() - M[-1, j] for j in range(T - 1)]))
        out[f"bwt_{key}"] = float(np.mean([M[-1, j] - M[j, j] for j in range(T - 1)]))
    return out, M.tolist()
