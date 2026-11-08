"""R5b report (exploratory): does the dilution result hold, and is the blend the bottleneck?
(benchmark/r5b_design.yaml). Reads only the raw metrics.json and diag.json files.

    python scripts/r5b_report.py --joint runs_kaggle/r1_final --r2 runs_kaggle/r2 \
        --snap runs_kaggle/r5 runs_kaggle/r5b --diag runs_kaggle/diag
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
from r5_report import task1_forgetting, mean  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def last_task(m):
    """Rank-1 on the last task at the end (how well the newest people are learned)."""
    return float(m["A"][-1][-1]["rank1"])


def main():
    ap = argparse.ArgumentParser()
    for arg in ("joint", "r2", "snap", "diag"):
        ap.add_argument(f"--{arg}", nargs="+", required=True)
    args = ap.parse_args()
    bm = ROOT / "benchmark"
    bench = yaml.safe_load(open(bm / "tasks.yaml"))["datasets"]
    rule = yaml.safe_load(open(bm / "r3_rule.yaml"))
    design = yaml.safe_load(open(bm / "r5b_design.yaml"))
    recipes = yaml.safe_load(open(bm / "recipes.yaml"))["recipes"]
    k = design["snapshot_after_task"]

    def seeds(name):
        return [s for s in bench[name]["seeds"] if s != 0]

    def get(roots, name, key, folder, seed):
        r = recipes[name][key]
        tag = f"{key}_lr{r['lr_body']:g}_s{r['steps_per_task']}"
        return load(roots, Path(name) / "continual" / f"{folder}_{tag}" / f"seed{seed}" / "metrics.json")

    print("## Experiment 1: the snapshot upper bound on every shown cell (mixing learned)\n")
    print("| Dataset | Model | Seeds | Naive task-1 forg. | Snapshot task-1 forg. | Seeds better (task 1) | "
          "Naive forg. | Snapshot forg. | Naive final | Snapshot final | R3 test (overall) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    cells = [{"dataset": "earvn", "model": "modern_cnn"}] + design["cells_exp1"]   # R5's cell first
    for c in cells:
        name, key = c["dataset"], c["model"]
        rows, t1 = [], []
        for s in seeds(name):
            naive, joint = get(args.r2, name, key, "naive", s), get(args.joint, name, key, "joint", s)
            snap = get(args.snap, name, key, f"naive_snapshot{k}", s)
            if naive and joint and snap:
                rows.append({"base_f": naive["summary"]["forgetting_rank1"],
                             "cms_f": snap["summary"]["forgetting_rank1"],
                             "base_final": naive["summary"]["final_avg_rank1"],
                             "cms_final": snap["summary"]["final_avg_rank1"],
                             "joint": joint["summary"]["final_avg_rank1"]})
                t1.append((task1_forgetting(naive), task1_forgetting(snap)))
        v = compare(rows, len(seeds(name)), rule) if rows else "missing"
        print(f"| {name} | {key} | {len(rows)} of {len(seeds(name))} | {mean([a for a, _ in t1])} | "
              f"{mean([b for _, b in t1])} | {sum(b < a for a, b in t1)} of {len(t1)} | "
              f"{mean([r['base_f'] for r in rows])} | {mean([r['cms_f'] for r in rows])} | "
              f"{mean([r['base_final'] for r in rows])} | {mean([r['cms_final'] for r in rows])} | **{v}** |")

    print("\n## Experiment 2: frozen copies' share of the output fixed after task 1\n")
    print("| Dataset | Model | Frozen share | Seeds | Task-1 forg. | Forgetting | Last-task rank-1 | "
          "Final rank-1 | Frozen share at end |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in design["cells_exp2"]:
        name, key = c["dataset"], c["model"]
        variants = [("learned", f"naive_snapshot{k}")] + [
            (f"{s:g}", f"naive_snapshot{k}_mix{round(s * 100)}") for s in design["mix_shares"]]
        naive = [get(args.r2, name, key, "naive", s) for s in seeds(name)]
        naive = [m for m in naive if m]
        print(f"| {name} | {key} | naive | {len(naive)} | {mean([task1_forgetting(m) for m in naive])} | "
              f"{mean([m['summary']['forgetting_rank1'] for m in naive])} | "
              f"{mean([last_task(m) for m in naive])} | "
              f"{mean([m['summary']['final_avg_rank1'] for m in naive])} | |")
        for label, folder in variants:
            ms = [m for m in (get(args.snap, name, key, folder, s) for s in seeds(name)) if m]
            share = [1 - m["log"][-1]["mix_weights"][0] for m in ms if m["log"][-1].get("mix_weights")]
            print(f"| {name} | {key} | {label} | {len(ms)} | {mean([task1_forgetting(m) for m in ms])} | "
                  f"{mean([m['summary']['forgetting_rank1'] for m in ms])} | {mean([last_task(m) for m in ms])} | "
                  f"{mean([m['summary']['final_avg_rank1'] for m in ms])} | {mean(share)} |")

    print("\n## Diagnostics: is the kept memory diluted? (snapshot runs and the second seed)\n")
    print("| Run | Task 1: after task 1 → end (whole model) | Frozen copy alone, task 1 | Gap | "
          "Part holding most of the loss (share) |")
    print("|---|---|---|---|---|")
    for root in args.diag:
        for f in sorted(Path(root).rglob("diag.json")):
            rel = f.parent.relative_to(root).as_posix()
            if "snapshot" not in rel and not rel.endswith("seed2"):
                continue
            d = json.load(open(f))
            restore = {n: v["share_recovered"] for n, v in d["restore"].items()}
            top = max(restore, key=lambda n: -1 if np.isnan(restore[n]) else restore[n])
            alone = {n: v["task1"] for n, v in (d.get("one_copy_only") or {}).items() if "frozen" in n}
            frozen = max(alone.values()) if alone else float("nan")
            gap = frozen - d["task1_final"]
            print(f"| {rel} | {d['task1_after_task1']:.3f} → {d['task1_final']:.3f} | "
                  f"{'' if np.isnan(frozen) else f'{frozen:.3f}'} | {'' if np.isnan(gap) else f'{gap:+.3f}'} | "
                  f"{top} ({restore[top]:.0%}) |")


if __name__ == "__main__":
    main()
