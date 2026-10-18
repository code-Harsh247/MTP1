"""Continual ear recognition, protocol P1 (identity-incremental), plan_continual.md.

The train identities are dealt into --n-tasks tasks, learned one after another. After every
task t we record
    A[t][j]  open-set identification of task j's people from their held-out images
             (cosine_proto; fixed set of never-trained unknown people), for every task j
    U[t]     the Phase 2 open-set protocol on the unseen val/test people
and summarize final average, forgetting and backward transfer (nlear/continual.py).

Methods:
    joint   all tasks' data at once (upper bound; one stage)
    naive   plain sequential fine-tuning (lower bound)

    python scripts/train_continual.py --data configs/earvn.yaml --method naive --seed 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nlear.continual import continual_summary, make_task_splits, seen_task_metrics  # noqa: E402
from nlear.data.cache import GPUAugment, load_image_cache  # noqa: E402
from nlear.data.manifest import load_manifest  # noqa: E402
from nlear.engine import embed, train_steps  # noqa: E402
from nlear.eval.protocol import run_protocol  # noqa: E402
from nlear.models import ArcFace, EmbeddingNet  # noqa: E402

METHODS = ("joint", "naive")


def make_optimizer(model, head, t):
    return torch.optim.AdamW([
        {"params": model.body.parameters(), "lr": t["lr_body"]},
        {"params": list(model.neck.parameters()) + list(head.parameters()), "lr": t["lr_head"]},
    ], weight_decay=t["weight_decay"])


def evaluate_stage(model, images, df, device, n_tasks):
    """Embed held-out + val/test rows once; return (A_row, U) for this stage."""
    need = df["cl_role"].isin(["held_gallery", "held_probe"]) | df["split"].isin(["val", "test"])
    sub = df[need].reset_index(drop=True)
    emb = embed(model, images, torch.tensor(sub["row"].to_numpy()), device).cpu()
    unknown = ((sub["split"] == "test") & (sub["known"] == 0) & (sub["role"] == "probe")).to_numpy()
    A_row = [seen_task_metrics(emb, sub, j, unknown) for j in range(n_tasks)]
    blob = {"emb": emb, "identity": sub["identity"].to_numpy(), "split": sub["split"].to_numpy(),
            "role": sub["role"].to_numpy(), "known": sub["known"].to_numpy(), "path": sub["path"].to_numpy()}
    U = run_protocol(blob, "cosine_proto")
    return A_row, U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--train", default="configs/baseline.yaml")
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=5)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--steps-per-task", type=int, default=250)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    dcfg, tcfg = yaml.safe_load(open(args.data)), yaml.safe_load(open(args.train))
    d, t = dcfg["data"], tcfg["train"]
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(args.out) / dcfg["name"] / "continual" / (args.method + (f"_{args.tag}" if args.tag else "")) / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump({"data": dcfg, "train": tcfg, "args": vars(args)}, open(run_dir / "config.yaml", "w"))

    manifest = load_manifest(d["manifest"])
    h, w = d["img_size"]
    images, row_of = load_image_cache(manifest, d["root"], d["img_size"],
                                      Path(d["manifest"]).parent / f"cache_{h}x{w}.pt")
    df = pd.read_csv(d["splits"].format(seed=args.seed), dtype={"identity": str, "path": str})
    df["row"] = df["path"].map(row_of)
    df = make_task_splits(df, args.n_tasks, args.holdout, seed=args.seed)
    df.to_csv(run_dir / "task_splits.csv", index=False)

    classes = sorted(df.loc[df["split"] == "train", "identity"].unique())
    label_of = {c: i for i, c in enumerate(classes)}
    task_of_class = torch.tensor([int(df.loc[df["identity"] == c, "task"].iloc[0]) for c in classes])
    tr = df[df["cl_role"] == "train"]
    rows_of_task = [torch.tensor(tr.loc[tr["task"] == k, "row"].to_numpy()) for k in range(args.n_tasks)]
    labels_of_task = [torch.tensor(tr.loc[tr["task"] == k, "identity"].map(label_of).to_numpy())
                      for k in range(args.n_tasks)]
    print(f"[{dcfg['name']} seed {args.seed} {args.method}] {len(classes)} train ids in {args.n_tasks} tasks "
          f"({[len(r) for r in rows_of_task]} train imgs); held-out "
          f"{int(df['cl_role'].str.startswith('held').sum())} imgs")

    model = EmbeddingNet(t["arch"], t["emb_dim"], t["pretrained"], t.get("dropout", 0.0)).to(device)
    head = ArcFace(t["emb_dim"], len(classes), t["arcface_s"], t["arcface_m"]).to(device)
    scaler = torch.amp.GradScaler(enabled=t["amp"] and device.type == "cuda")
    augment = GPUAugment(**t.get("aug", {}))

    if args.method == "joint":
        stages = [list(range(args.n_tasks))]       # one stage with every task's data
    else:
        stages = [[k] for k in range(args.n_tasks)]

    A, U, log = [], [], []
    for s, tasks in enumerate(stages):
        tic = time.time()
        rows = torch.cat([rows_of_task[k] for k in tasks])
        labels = torch.cat([labels_of_task[k] for k in tasks])
        seen = max(tasks)
        active = task_of_class <= seen                     # classes seen so far compete
        steps = args.steps_per_task * len(tasks)

        def batch_fn(step):
            b = torch.randint(len(rows), (t["batch_size"],), generator=gen)
            return rows[b], labels[b]

        opt = make_optimizer(model, head, t)
        stats = train_steps(model, head, opt, images, batch_fn, steps, device, augment, scaler,
                            warm=max(1, steps // 10), active=active)
        A_row, U_s = evaluate_stage(model, images, df, device, args.n_tasks)
        A.append(A_row)
        U.append(U_s)
        rec = {"stage": s, "tasks": tasks, **stats, "sec": round(time.time() - tic, 1),
               "A_dir@5%": [round(a["dir@far0.05"], 3) for a in A_row],
               "U_test_dir@5%": round(U_s["test"]["dir@far0.05"], 4)}
        log.append(rec)
        print(" ".join(f"{k}={v}" for k, v in rec.items()), flush=True)

    summary = {}
    for key in ("dir@far0.05", "rank1", "auroc"):
        sm, M = continual_summary(A, key)
        summary.update(sm)
        summary[f"matrix_{key}"] = M
    summary["final_U_test"] = U[-1]["test"]
    summary["final_U_val"] = U[-1]["val"]
    json.dump({"log": log, "summary": summary, "A": A, "U": U}, open(run_dir / "metrics.json", "w"), indent=1)
    torch.save({"model": model.state_dict()}, run_dir / "model.pt")
    print("SUMMARY " + " ".join(f"{k}={v:.4f}" for k, v in summary.items() if isinstance(v, float)))
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
