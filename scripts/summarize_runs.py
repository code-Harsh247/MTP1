"""Mean +- std over seeds of test metrics for every run under runs/<dataset>/<method>/seed*/.

    python scripts/summarize_runs.py runs/ami/baseline_resnet18
"""
import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ["rank1", "auroc", "dir@far0.01", "dir@far0.05", "dir@far0.1",
        "cal_dir@far0.05", "cal_far@far0.05", "eer", "tar@far0.01"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method_dirs", nargs="+")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    for mdir in map(Path, args.method_dirs):
        runs = sorted(mdir.glob("seed*/metrics.json"))
        if not runs:
            print(f"no runs in {mdir}")
            continue
        res = [json.load(open(r))["results"] for r in runs]
        print(f"\n{mdir}  ({len(runs)} seeds: {', '.join(r.parent.name for r in runs)}), split={args.split}")
        print(f"{'scorer':14s}" + "".join(f"{k:>17s}" for k in KEYS))
        for scorer in res[0]:
            cells = []
            for k in KEYS:
                v = np.array([r[scorer][args.split].get(k, np.nan) for r in res])
                cells.append(f"{np.nanmean(v):8.3f} ±{np.nanstd(v):6.3f}")
            print(f"{scorer:14s}" + "".join(f"{c:>17s}" for c in cells))


if __name__ == "__main__":
    main()
