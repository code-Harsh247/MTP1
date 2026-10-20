"""Apply the U3 decision rule (notes/u3_continual.md) mechanically.

    python scripts/u3_report.py runs_dgx/uerc/continual
"""
import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = [1, 2, 3]
ARMS = {"joint": "joint_fp32", "naive": "naive_fp32", "replay": "replay_k5_fp32",
        "ctrl+R": "replay_k5_cms1-1-1_b1_p0", "slowbb+R": "replay_k5_cms1-1-1_b16_p0.5",
        "D+R": "replay_k5_cms1-16-256_b16_p0.5"}
COLS = [("final_avg_rank1", "seen rank-1", +1), ("forgetting_rank1", "forget rank-1", -1),
        ("final_avg_dir@far0.05", "seen DIR@5%", +1), ("forgetting_dir@far0.05", "forget DIR@5%", -1),
        ("U:rank1", "unseen rank-1", +1), ("U:auroc", "unseen AUROC", +1), ("U:dir@far0.05", "unseen DIR@5%", +1)]


def series(root, folder, key):
    out = []
    for s in SEEDS:
        m = json.load(open(Path(root) / folder / f"seed{s}" / "metrics.json"))["summary"]
        out.append(m["final_U_test"][key[2:]] if key.startswith("U:") else m.get(key, np.nan))
    return np.array(out, dtype=float)


def paired(root, a, b):
    print(f"  {a} vs {b}:")
    for key, name, sign in COLS:
        d = (series(root, ARMS[a], key) - series(root, ARMS[b], key)) * sign
        if np.isnan(d).all():
            continue
        print(f"    {name:15s} better on {int((d > 0).sum())}/3   mean {d.mean():+.4f} ± {d.std():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    root = ap.parse_args().root
    print(f"TEST, UERC P1 (10 tasks), seeds {SEEDS}, all arms fp32")
    print(f"{'arm':10s}" + "".join(f"{n:>17s}" for _, n, _ in COLS))
    for arm, folder in ARMS.items():
        cells = []
        for key, _, _ in COLS:
            v = series(root, folder, key)
            cells.append("—" if np.isnan(v).all() else f"{np.nanmean(v):.3f} ±{np.nanstd(v):.3f}")
        print(f"{arm:10s}" + "".join(f"{c:>17s}" for c in cells))

    d = series(root, ARMS["slowbb+R"], "final_avg_rank1") - series(root, ARMS["ctrl+R"], "final_avg_rank1")
    wins = int((d > 0).sum())
    print(f"\nDECISION (primary: final seen rank-1, slowbb+R vs ctrl+R, paired)")
    print(f"  better on {wins}/3 seeds, mean {d.mean():+.4f} ± {d.std():.4f}  per seed {np.round(d, 4).tolist()}")
    print(f"  VERDICT: {'CONFIRMED' if wins == 3 else 'NOT CONFIRMED'}")
    print("\nREPORTED (not decision criteria)")
    paired(root, "slowbb+R", "ctrl+R")
    paired(root, "D+R", "slowbb+R")
    paired(root, "slowbb+R", "replay")
    paired(root, "ctrl+R", "replay")


if __name__ == "__main__":
    main()
