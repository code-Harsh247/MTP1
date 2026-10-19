"""Apply the C3b confirmatory rule (notes/c3b_confirm.md) mechanically.

    python scripts/c3b_report.py runs_dgx/earvn/continual
"""
import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = [6, 7, 8, 9, 10]
ARMS = {"R": "replay_k5",
        "ctrl+R": "replay_k5_cms1-1-1_b1_p0",
        "slowbb+R": "replay_k5_cms1-1-1_b16_p0.5",
        "D+R": "replay_k5_cms1-16-256_b16_p0.5"}
PRIMARY = [("forgetting_dir@far0.05", "forgetting", -1),   # lower is better
           ("final_avg_rank1", "seen rank-1", +1),
           ("U:auroc", "unseen AUROC", +1)]
EXTRA = [("final_avg_dir@far0.05", "seen DIR@5%", +1), ("U:dir@far0.05", "unseen DIR@5%", +1)]


def series(root, folder, key):
    out = []
    for s in SEEDS:
        m = json.load(open(Path(root) / folder / f"seed{s}" / "metrics.json"))["summary"]
        out.append(m["final_U_test"][key[2:]] if key.startswith("U:") else m[key])
    return np.array(out)


def beats(root, a, b, verbose=True):
    ok = True
    for key, name, sign in PRIMARY:
        d = (series(root, ARMS[a], key) - series(root, ARMS[b], key)) * sign
        w = int((d > 0).sum())
        ok &= w >= 4
        if verbose:
            print(f"    {name:13s} better on {w}/5   mean {d.mean():+.3f} ± {d.std():.3f}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    root = ap.parse_args().root
    print(f"TEST, fresh seeds {SEEDS}")
    print(f"{'arm':10s}" + "".join(f"{n:>17s}" for _, n, _ in PRIMARY + EXTRA))
    for arm, folder in ARMS.items():
        cells = [f"{series(root, folder, k).mean():.3f} ±{series(root, folder, k).std():.3f}"
                 for k, _, _ in PRIMARY + EXTRA]
        print(f"{arm:10s}" + "".join(f"{c:>17s}" for c in cells))
    print("\nDECISION (paired; X beats Y = better on all 3 primary metrics, each >= 4/5 seeds)")
    res = {}
    for a, b in (("D+R", "R"), ("slowbb+R", "R"), ("D+R", "slowbb+R")):
        print(f"  {a} vs {b}:")
        res[(a, b)] = beats(root, a, b)
        print(f"    -> {a} {'BEATS' if res[(a, b)] else 'does not beat'} {b}")
    if not res[("D+R", "R")]:
        verdict = "NOT CONFIRMED"
    elif res[("D+R", "slowbb+R")]:
        verdict = "CONFIRMED (CMS chain)"
    elif res[("slowbb+R", "R")]:
        verdict = "CONFIRMED (frequency split only)"
    else:
        verdict = "CONFIRMED vs R, but attribution unresolved (D+R beats neither control cleanly)"
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
