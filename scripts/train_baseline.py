"""Phase 2 baseline: fine-tune a pretrained ResNet with ArcFace on the train identities,
then evaluate open-set identification on val/test and save all embeddings for Phase 3.

    python scripts/train_baseline.py --data configs/ami.yaml --train configs/baseline.yaml --seed 0
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nlear.data.cache import GPUAugment, load_image_cache, to_input  # noqa: E402
from nlear.data.manifest import load_manifest  # noqa: E402
from nlear.eval.protocol import run_protocol  # noqa: E402
from nlear.models import ArcFace, EmbeddingNet  # noqa: E402


@torch.no_grad()
def embed(model, images, rows, device, bs=256):
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        x = to_input(images[rows[i:i + bs]], device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(x).float())
    return torch.cat(out)


def _parse_value(s):
    """YAML, except '5e-5' becomes a float (YAML 1.1 reads it as a string without a '.')."""
    v = yaml.safe_load(s)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            pass
    return v


def make_blob(df, emb):
    return {"emb": emb.cpu(), "identity": df["identity"].to_numpy(), "split": df["split"].to_numpy(),
            "role": df["role"].to_numpy(), "known": df["known"].to_numpy(), "path": df["path"].to_numpy()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--train", default="configs/baseline.yaml")
    ap.add_argument("--seed", type=int, default=0, help="selects splits_seed{seed}.csv and the RNG seed")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int, help="override for quick tests")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override train config, e.g. arch=resnet50 aug.degrees=15")
    ap.add_argument("--tag", default="", help="suffix for the run folder")
    ap.add_argument("--val-only", action="store_true",
                    help="tuning mode: compute and save validation metrics only, never test")
    args = ap.parse_args()
    dcfg, tcfg = yaml.safe_load(open(args.data)), yaml.safe_load(open(args.train))
    d, t = dcfg["data"], tcfg["train"]
    if args.epochs:
        t["epochs"] = args.epochs
    for kv in args.set:
        key, val = kv.split("=", 1)
        node = t
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = _parse_value(val)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    method = f"baseline_{t['arch']}" + (f"_{args.tag}" if args.tag else "")
    run_dir = Path(args.out) / dcfg["name"] / ("tune" if args.val_only else "") / method / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump({"data": dcfg, "train": tcfg, "seed": args.seed}, open(run_dir / "config.yaml", "w"))

    manifest = load_manifest(d["manifest"])
    h, w = d["img_size"]
    images, row_of = load_image_cache(manifest, d["root"], d["img_size"],
                                      Path(d["manifest"]).parent / f"cache_{h}x{w}.pt")
    df = pd.read_csv(d["splits"].format(seed=args.seed), dtype={"identity": str, "path": str})
    df["row"] = df["path"].map(row_of)
    assert df["row"].notna().all(), "splits reference images missing from the manifest cache"

    tr = df[df["split"] == "train"].reset_index(drop=True)
    classes = sorted(tr["identity"].unique())
    tr_rows = torch.tensor(tr["row"].to_numpy())
    tr_lab = torch.tensor(tr["identity"].map({c: i for i, c in enumerate(classes)}).to_numpy())
    ev = df[df["split"] != "train"].reset_index(drop=True)
    ev_rows = torch.tensor(ev["row"].to_numpy())
    print(f"[{dcfg['name']} seed {args.seed}] train {len(tr)} imgs / {len(classes)} ids; "
          f"eval {len(ev)} imgs; device {device}")

    model = EmbeddingNet(t["arch"], t["emb_dim"], t["pretrained"], t.get("dropout", 0.0)).to(device)
    head = ArcFace(t["emb_dim"], len(classes), t["arcface_s"], t["arcface_m"]).to(device)
    opt = torch.optim.AdamW([
        {"params": model.body.parameters(), "lr": t["lr_body"]},
        {"params": list(model.neck.parameters()) + list(head.parameters()), "lr": t["lr_head"]},
    ], weight_decay=t["weight_decay"])
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = math.ceil(len(tr) / t["batch_size"])
    if t.get("steps"):
        # Length in optimizer steps: "5 epochs" is 1,550 steps on EarVN but only 40 on AMI.
        t["epochs"] = max(1, math.ceil(t["steps"] / steps_per_epoch))
        t["eval_every"] = max(1, t["epochs"] // 8)
        print(f"steps={t['steps']} -> {t['epochs']} epochs of {steps_per_epoch} steps")
    total, warm = t["epochs"] * steps_per_epoch, t["warmup_epochs"] * steps_per_epoch
    scaler = torch.amp.GradScaler(enabled=t["amp"] and device.type == "cuda")
    augment = GPUAugment(**t.get("aug", {}))

    log, step = [], 0
    for epoch in range(1, t["epochs"] + 1):
        model.train(); head.train()
        perm = torch.randperm(len(tr))
        tic, loss_sum, acc_sum = time.time(), 0.0, 0.0
        for i in range(steps_per_epoch):
            b = perm[i * t["batch_size"]:(i + 1) * t["batch_size"]]
            if len(b) < 2:
                continue  # BatchNorm needs >1 sample
            f = step / max(1, warm) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * f
            x = to_input(images[tr_rows[b]], device, augment)
            y = tr_lab[b].to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                loss, cos = head(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item() * len(b)
            acc_sum += (cos.argmax(1) == y).float().sum().item()
            step += 1
        rec = {"epoch": epoch, "loss": loss_sum / len(tr), "train_acc": acc_sum / len(tr),
               "sec": round(time.time() - tic, 1)}
        if epoch % t["eval_every"] == 0 or epoch == t["epochs"]:
            blob = make_blob(ev, embed(model, images, ev_rows, device))
            v = run_protocol(blob, "cosine_proto")["val"]
            rec.update({f"val_{k}": v[k] for k in ("rank1", "auroc", "dir@far0.05")})
        log.append(rec)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))

    # Final: embed every image (train embeddings feed the Phase 3 memory), evaluate all scorers.
    all_rows = torch.tensor(df["row"].to_numpy())
    blob = make_blob(df, embed(model, images, all_rows, device))
    if not args.val_only:
        torch.save(blob, run_dir / "embeddings.pt")
        torch.save({"model": model.state_dict(), "classes": classes}, run_dir / "model.pt")
    results = {s: run_protocol(blob, s) for s in tcfg["scorers"]}

    # Evaluation-only split variants (e.g. g5 = 5 gallery images): same embeddings, new roles.
    for name in (dcfg["splits"].get("variants") or {}):
        p = Path(d["splits"].format(seed=args.seed))
        vdf = pd.read_csv(p.with_name(p.name.replace("splits_", f"splits_{name}_", 1)),
                          dtype={"identity": str, "path": str}).set_index("path").loc[df["path"]]
        assert (vdf["split"].to_numpy() == blob["split"]).all(), f"variant {name} changes splits"
        vblob = dict(blob, role=vdf["role"].to_numpy(), known=vdf["known"].to_numpy())
        results.update({f"{s}@{name}": run_protocol(vblob, s) for s in tcfg["scorers"]})

    split = "val" if args.val_only else "test"
    if args.val_only:
        results = {s: {"val": r["val"]} for s, r in results.items()}
    json.dump({"log": log, "results": results}, open(run_dir / "metrics.json", "w"), indent=1)

    keys = ["rank1", "auroc", "dir@far0.01", "dir@far0.05", "cal_dir@far0.05", "cal_far@far0.05", "eer"]
    if args.val_only:
        keys = [k for k in keys if not k.startswith("cal_")]
    first = results[tcfg["scorers"][0]][split]
    print(f"\n{split.upper()} ({first['n_known']} known / {first['n_unknown']} unknown probes)")
    print(f"{'scorer':18s}" + "".join(f"{k:>16s}" for k in keys))
    for s, r in results.items():
        print(f"{s:18s}" + "".join(f"{r[split].get(k, float('nan')):16.4f}" for k in keys))
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
