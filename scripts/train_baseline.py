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
from nlear.eval.protocol import run_protocol, variant_blobs  # noqa: E402
from nlear.cms import CMSScheduler  # noqa: E402
from nlear.m3 import M3, split_params  # noqa: E402
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
    ap.add_argument("--meta-frac", type=float, default=0.0,
                    help="hold out this fraction of train identities from the backbone (split='meta')")
    ap.add_argument("--no-save", action="store_true", help="metrics only: skip embeddings.pt and model.pt")
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

    method = ("splitbb_" if args.meta_frac else "baseline_") + t["arch"] + (f"_{args.tag}" if args.tag else "")
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
    if args.meta_frac:
        # Split-backbone protocol (Phase 3): hold out a fraction of the TRAIN identities from
        # the backbone. Their embeddings are then "unseen people", usable to meta-train the
        # memory. They get split="meta" and never touch backbone training or evaluation.
        train_ids = np.array(sorted(df.loc[df["split"] == "train", "identity"].unique()))
        meta_ids = np.random.default_rng(1000 + args.seed).choice(
            train_ids, int(round(args.meta_frac * len(train_ids))), replace=False)
        df.loc[df["identity"].isin(meta_ids), "split"] = "meta"
        print(f"split-backbone: {len(meta_ids)} of {len(train_ids)} train identities held out as 'meta'")

    tr = df[df["split"] == "train"].reset_index(drop=True)
    classes = sorted(tr["identity"].unique())
    tr_rows = torch.tensor(tr["row"].to_numpy())
    tr_lab = torch.tensor(tr["identity"].map({c: i for i, c in enumerate(classes)}).to_numpy())
    ev = df[df["split"] != "train"].reset_index(drop=True)
    ev_rows = torch.tensor(ev["row"].to_numpy())
    print(f"[{dcfg['name']} seed {args.seed}] train {len(tr)} imgs / {len(classes)} ids; "
          f"eval {len(ev)} imgs; device {device}")

    vit_cms = t.get("vit_cms") or {}
    model = EmbeddingNet(t["arch"], t["emb_dim"], t["pretrained"], t.get("dropout", 0.0),
                         img_size=d["img_size"], vit_cms_levels=len(vit_cms.get("periods", []))).to(device)
    head = ArcFace(t["emb_dim"], len(classes), t["arcface_s"], t["arcface_m"]).to(device)
    cms = None
    if vit_cms:
        # Option B: CMS inside the ViT ([NL] Sec. 7.3); slow copies accumulate gradients -> fp32.
        t["amp"] = False
        cms = CMSScheduler.for_vit(model, head, t, list(vit_cms["periods"]), float(vit_cms.get("lr_pow", 0.0)))
        opts = [cms.fast_optimizer()]
        print(f"ViT-CMS: MLP copy periods {vit_cms['periods']}, lr_pow {vit_cms.get('lr_pow', 0.0)}; fp32")
    elif t.get("optimizer", "adamw") == "m3":
        # U4: NL's M3 optimizer (nlear/m3.py) for the weight matrices (body convs, neck projection);
        # AdamW keeps BN params, biases and the ArcFace head (Muon convention).
        m = t.get("m3", {})
        body_mats, body_rest = split_params(model.body)
        neck_mats, neck_rest = split_params(model.neck)
        opts = [M3(body_mats + neck_mats, lr=float(m.get("lr", 1e-4)), alpha=float(m.get("alpha", 1.0)),
                   freq=int(m.get("freq", 16)), betas=tuple(m.get("betas", (0.95, 0.999, 0.99))),
                   weight_decay=t["weight_decay"]),
                torch.optim.AdamW([{"params": body_rest, "lr": t["lr_body"]},
                                   {"params": neck_rest + list(head.parameters()), "lr": t["lr_head"]}],
                                  weight_decay=t["weight_decay"])]
        print(f"optimizer: M3 {m} on {sum(p.numel() for p in body_mats + neck_mats)/1e6:.1f}M matrix params; "
              f"AdamW on the rest")
    else:
        opts = [torch.optim.AdamW([
            {"params": model.body.parameters(), "lr": t["lr_body"]},
            {"params": list(model.neck.parameters()) + list(head.parameters()), "lr": t["lr_head"]},
        ], weight_decay=t["weight_decay"])]
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]
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
            for o, lrs in zip(opts, base_lrs):
                for g, lr in zip(o.param_groups, lrs):
                    g["lr"] = lr * f
            x = to_input(images[tr_rows[b]], device, augment)
            y = tr_lab[b].to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                loss, cos = head(model(x), y)
            for o in opts:
                o.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cms is not None:
                cms.hook(step, scaler)
            for o in opts:
                scaler.step(o)
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
    if not args.val_only and not args.no_save:
        torch.save(blob, run_dir / "embeddings.pt")
        torch.save({"model": model.state_dict(), "classes": classes}, run_dir / "model.pt")
    # Base split plus evaluation-only variants (e.g. g5 = 5 gallery images): same embeddings.
    results = {}
    for name, vblob in variant_blobs(blob, dcfg, args.seed).items():
        results.update({s + (f"@{name}" if name else ""): run_protocol(vblob, s) for s in tcfg["scorers"]})

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
