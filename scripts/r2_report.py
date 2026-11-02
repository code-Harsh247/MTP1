"""R2 report: apply the forgetting rule of benchmark/r2_grid.yaml to the naive arm of every
model x dataset, and print every arm's numbers. Reads only the raw metrics.json files.

    python scripts/r2_report.py --joint runs_kaggle/r1_final runs_dgx/r1_final --r2 runs_kaggle/r2 runs_dgx/r2
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def verdict(rows, n_seeds, forgetting_min, joint_gap):
    """rows: one dict per finished seed with forgetting, final, joint. Returns the cell's verdict."""
    if len(rows) < n_seeds or any(not math.isfinite(v) for r in rows for v in r.values()):
        return "INCOMPLETE"
    shown = all(r["forgetting"] >= forgetting_min and r["final"] <= r["joint"] - joint_gap for r in rows)
    return "SHOWN" if shown else "not shown"


def forgetting_of(A, key):
    """Forgetting of any stored metric, as nlear.continual.continual_summary defines it."""
    M = np.array([[A[t][j][key] for j in range(len(A[t]))] for t in range(len(A))])
    T = M.shape[0]
    return float(M[-1].mean()), float(np.mean([M[j:T - 1, j].max() - M[-1, j] for j in range(T - 1)]))


def find(roots, rel):
    return next((Path(p) / rel for p in roots if (Path(p) / rel).exists()), None)


def load(roots, rel):
    f = find(roots, rel)
    return json.load(open(f)) if f else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", nargs="+", required=True)
    ap.add_argument("--r2", nargs="+", required=True)
    args = ap.parse_args()
    bench = yaml.safe_load(open(ROOT / "benchmark" / "tasks.yaml"))["datasets"]
    grid = yaml.safe_load(open(ROOT / "benchmark" / "r2_grid.yaml"))
    recipes = yaml.safe_load(open(ROOT / "benchmark" / "recipes.yaml"))["recipes"]
    path = ROOT / "benchmark" / "r2_strengths.yaml"
    strengths = yaml.safe_load(open(path))["strengths"] if path.exists() else {}
    counts = {"SHOWN": 0, "not shown": 0, "INCOMPLETE": 0}
    for name, models in recipes.items():
        seeds = [s for s in bench[name]["seeds"] if s != 0]
        print(f"\n## {name} (seeds {seeds})\n")
        print("| Model | Arm | Rank-1 | Rank-1 forgetting | DIR@1% forg. | DIR@5% forg. | DIR@10% forg. | AUROC forg. | Runs |")
        print("|---|---|---|---|---|---|---|---|---|")
        for key, r in (models or {}).items():
            tag = f"{key}_lr{r['lr_body']:g}_s{r['steps_per_task']}"
            lam = strengths.get(name, {}).get(key, {})
            arms = [("naive", "naive"), ("replay", f"replay_k{grid['replay_k']}")]
            for method in grid["strengths"]:
                if method in lam:
                    arms.append((method, f"{method}_lam{lam[method]:g}"))
            rule_rows = []
            for arm, folder in arms:
                res = []
                for seed in seeds:
                    m = load(args.r2, Path(name) / "continual" / f"{folder}_{tag}" / f"seed{seed}" / "metrics.json")
                    if m is None:
                        continue
                    res.append(m)
                    if arm == "naive":
                        j = load(args.joint, Path(name) / "continual" / f"joint_{tag}" / f"seed{seed}" / "metrics.json")
                        if j is not None:
                            rule_rows.append({"forgetting": m["summary"]["forgetting_rank1"],
                                              "final": m["summary"]["final_avg_rank1"],
                                              "joint": j["summary"]["final_avg_rank1"]})
                if not res:
                    print(f"| {key} | {arm} | | | | | | | 0/{len(seeds)} |")
                    continue
                cols = []
                for k in ("rank1", "dir@far0.01", "dir@far0.05", "dir@far0.1", "auroc"):
                    vals = [forgetting_of(m["A"], k) for m in res]
                    cols.append((k, np.mean([v[0] for v in vals]), np.mean([v[1] for v in vals]),
                                 np.std([v[1] for v in vals], ddof=1) if len(vals) > 1 else 0.0))
                print(f"| {key} | {arm} | {cols[0][1]:.3f} | {cols[0][2]:.3f} ± {cols[0][3]:.3f} | {cols[1][2]:.3f} | "
                      f"{cols[2][2]:.3f} | {cols[3][2]:.3f} | {cols[4][2]:.3f} | {len(res)}/{len(seeds)} |")
            v = verdict(rule_rows, len(seeds), grid["forgetting_min"], grid["joint_gap"])
            counts[v] += 1
            detail = ", ".join(f"f {x['forgetting']:.3f} final {x['final']:.3f} joint {x['joint']:.3f}" for x in rule_rows)
            print(f"| {key} | **{v}** | {detail} | | | | | | |")
    print(f"\nCells: {counts}")


if __name__ == "__main__":
    main()
