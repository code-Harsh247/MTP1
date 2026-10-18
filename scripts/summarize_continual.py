"""Mean +- std over seeds for continual-learning runs (runs/<ds>/continual/<method>/seed*/).

    python scripts/summarize_continual.py runs/earvn/continual
"""
import argparse
import json
from pathlib import Path

import numpy as np

COLS = [("final_avg_dir@far0.05", "seen DIR@5%"), ("final_avg_rank1", "seen rank1"),
        ("forgetting_dir@far0.05", "forget DIR@5%"), ("forgetting_rank1", "forget rank1"),
        ("bwt_dir@far0.05", "BWT DIR@5%")]
UCOLS = [("rank1", "unseen rank1"), ("auroc", "unseen AUROC"), ("dir@far0.05", "unseen DIR@5%")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--matrix", action="store_true", help="also print the mean A[t][j] DIR@5% matrix")
    args = ap.parse_args()
    print(f"{'method':22s}{'seeds':>6s}" + "".join(f"{n:>17s}" for _, n in COLS + UCOLS))
    for mdir in sorted(p for p in Path(args.root).iterdir() if p.is_dir()):
        runs = [json.load(open(f)) for f in sorted(mdir.glob("seed*/metrics.json"))]
        if not runs:
            continue
        cells = []
        for k, _ in COLS:
            v = np.array([r["summary"].get(k, np.nan) for r in runs], dtype=float)
            cells.append("—" if np.isnan(v).all() else f"{np.nanmean(v):.3f} ±{np.nanstd(v):.3f}")
        for k, _ in UCOLS:
            v = np.array([r["summary"]["final_U_test"][k] for r in runs])
            cells.append(f"{v.mean():.3f} ±{v.std():.3f}")
        print(f"{mdir.name:22s}{len(runs):>6d}" + "".join(f"{c:>17s}" for c in cells))
        if args.matrix and len(runs[0]["A"]) > 1:
            M = np.mean([r["summary"]["matrix_dir@far0.05"] for r in runs], axis=0)
            print("   mean A[stage][task] DIR@5% (rows: after task t; cols: task j)")
            for t, row in enumerate(M):
                print("   " + " ".join(f"{x:6.3f}" for x in row))


if __name__ == "__main__":
    main()
