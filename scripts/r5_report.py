"""R5 report (exploratory): why CMS did not reduce forgetting (benchmark/r5_design.yaml).
Reads only the raw metrics.json and diag.json files.

    python scripts/r5_report.py --joint runs_kaggle/r1_final --r2 runs_kaggle/r2 --r3 runs_kaggle/r3 \
        --r5 runs_kaggle/r5 --diag runs_kaggle/diag
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r2_report import load  # noqa: E402
from r3_report import compare  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def task1_forgetting(m):
    """Rank-1 forgetting of task 1 alone: its best score before the last task minus its final score."""
    A = np.array([[c["rank1"] for c in row] for row in m["A"]])
    return float(A[:-1, 0].max() - A[-1, 0])


def mean(values):
    return f"{np.mean(values):.3f}" if values else ""


def main():
    ap = argparse.ArgumentParser()
    for arg in ("joint", "r2", "r3", "r5", "diag"):
        ap.add_argument(f"--{arg}", nargs="+", required=True)
    args = ap.parse_args()
    bm = ROOT / "benchmark"
    bench = yaml.safe_load(open(bm / "tasks.yaml"))["datasets"]
    rule = yaml.safe_load(open(bm / "r3_rule.yaml"))
    design = yaml.safe_load(open(bm / "r5_design.yaml"))
    recipes = yaml.safe_load(open(bm / "recipes.yaml"))["recipes"]
    pows = yaml.safe_load(open(bm / "r3_lr_pow.yaml"))["lr_pow"]
    seeds = [s for s in bench["earvn"]["seeds"] if s != 0]

    def tag(key):
        r = recipes["earvn"][key]
        return f"{key}_lr{r['lr_body']:g}_s{r['steps_per_task']}"

    def get(roots, folder, key, seed):
        return load(roots, Path("earvn") / "continual" / f"{folder}_{tag(key)}" / f"seed{seed}" / "metrics.json")

    print("## Experiment 1: full-size slow updates (EarVN, against naive, R3's test)\n")
    print("| Model | Variant | Naive forg. | R3 CMS forg. | R5 forg. | Seeds better | Naive final | R5 final | "
          "Slow updates | Verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    folders = {"pow1": "naive_bodycms1-16-256_p1", "perstep": "naive_bodycms1-16-256_p0_perstep"}
    for key in design["cells_exp1"]:
        r3_folder = f"naive_bodycms1-16-256_p{pows['earvn'][key]:g}"
        for vname in design["variants_exp1"]:
            rows, r3_f, upd = [], [], None
            for seed in seeds:
                naive, joint = get(args.r2, "naive", key, seed), get(args.joint, "joint", key, seed)
                new, r3 = get(args.r5, folders[vname], key, seed), get(args.r3, r3_folder, key, seed)
                if r3:
                    r3_f.append(r3["summary"]["forgetting_rank1"])
                if naive and joint and new:
                    upd = new["summary"].get("cms_updates_per_period")
                    rows.append({"base_f": naive["summary"]["forgetting_rank1"],
                                 "cms_f": new["summary"]["forgetting_rank1"],
                                 "base_final": naive["summary"]["final_avg_rank1"],
                                 "cms_final": new["summary"]["final_avg_rank1"],
                                 "joint": joint["summary"]["final_avg_rank1"]})
            v = compare(rows, len(seeds), rule)
            better = sum(r["cms_f"] < r["base_f"] for r in rows)
            print(f"| {key} | {vname} | {mean([r['base_f'] for r in rows])} | {mean(r3_f)} | "
                  f"{mean([r['cms_f'] for r in rows])} | {better} of {len(rows)} | "
                  f"{mean([r['base_final'] for r in rows])} | {mean([r['cms_final'] for r in rows])} | {upd} | **{v}** |")

    key = design["cell_exp3"]
    print(f"\n## Experiment 3: perfect-memory upper bound (EarVN {key}, copies 2-3 frozen after task "
          f"{design['snapshot_after_task']})\n")
    print("| Seed | Naive task-1 forg. | Snapshot task-1 forg. | Naive forg. | Snapshot forg. | Naive final | "
          "Snapshot final | Mixing weights at end |")
    print("|---|---|---|---|---|---|---|---|")
    acc = []
    for seed in seeds:
        naive = get(args.r2, "naive", key, seed)
        snap = get(args.r5, f"naive_snapshot{design['snapshot_after_task']}", key, seed)
        if not (naive and snap):
            print(f"| {seed} | missing | | | | | | |")
            continue
        row = [task1_forgetting(naive), task1_forgetting(snap), naive["summary"]["forgetting_rank1"],
               snap["summary"]["forgetting_rank1"], naive["summary"]["final_avg_rank1"],
               snap["summary"]["final_avg_rank1"]]
        acc.append(row)
        mix = "/".join(f"{x:.2f}" for x in snap["log"][-1].get("mix_weights") or [])
        print(f"| {seed} | " + " | ".join(f"{x:.3f}" for x in row) + f" | {mix} |")
    if acc:
        print("| mean | " + " | ".join(f"{x:.3f}" for x in np.mean(acc, 0)) + " | |")

    print("\n## Diagnostics (experiments 2 and 5): one run per row, seed "
          f"{design['diag_seed']}\n")
    print("| Run | Task 1: after task 1 → end | Part holding most of the loss (share) | "
          "Movement from pretrained, copies 1/2/3 | Each copy alone, task 1 rank-1 |")
    print("|---|---|---|---|---|")
    for root in args.diag:
        for f in sorted(Path(root).rglob("diag.json")):
            d = json.load(open(f))
            rel = f.parent.relative_to(root).as_posix()
            restore = {k: v["share_recovered"] for k, v in d["restore"].items()}
            top = max(restore, key=lambda k: -1 if np.isnan(restore[k]) else restore[k])
            copies = [k for k in d["moved_from_start"] if k.startswith("copy")]
            moved = "/".join(f"{d['moved_from_start'][c][-1]:.3f}" for c in copies) or "plain model"
            alone = "/".join(f"{v['task1']:.2f}" for k, v in (d.get("one_copy_only") or {}).items()
                             if k.startswith("copy"))
            print(f"| {rel} | {d['task1_after_task1']:.3f} → {d['task1_final']:.3f} | {top} ({restore[top]:.0%}) | "
                  f"{moved} | {alone} |")


if __name__ == "__main__":
    main()
