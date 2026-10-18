"""Continual-learning protocol P1 (identity-incremental), see plan_continual.md section 3.

The train identities of an open-set split are dealt into n_tasks tasks. For every train
identity a fraction of its images is held out ("seen-person test images") so forgetting of
already-learned people can be measured; the rest are training images for its task.
Val/test identities are untouched: they stay the unseen-people open-set protocol.

Columns added to the split DataFrame:
    task     0..n_tasks-1 for train identities, -1 for val/test
    cl_role  train | held_gallery | held_probe   (train identities)
             eval                                (val/test rows; their own role column applies)
"""
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
    check_task_splits(df, n_tasks)
    return df


def check_task_splits(df, n_tasks):
    tr = df[df["split"] == "train"]
    assert set(tr["task"]) == set(range(n_tasks)), "every task needs identities"
    assert (df.loc[df["split"] != "train", "task"] == -1).all(), "val/test rows must not have a task"
    assert (tr.groupby("identity")["task"].nunique() == 1).all(), "an identity is in two tasks"
    for role in ("train", "held_gallery", "held_probe"):
        has = set(tr.loc[tr["cl_role"] == role, "identity"])
        assert has == set(tr["identity"]), f"some train identity lacks {role} images"


def seen_task_metrics(emb, df, task, unknown_mask):
    """Open-set identification of ONE task's people from their held-out images.

    Gallery: held_gallery images of the task's identities. Known probes: their held_probe
    images. Unknown probes: a fixed set of never-trained people (unknown_mask rows), the same
    for every task and every stage, so A[t, j] is comparable across t.
    """
    t = df["task"].to_numpy() == task
    role = df["cl_role"].to_numpy()
    g = np.where(t & (role == "held_gallery"))[0]
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
