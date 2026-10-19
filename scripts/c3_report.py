"""Apply the C3 decision procedure (notes/c3_cms.md) to the DGX runs, mechanically.

    python scripts/c3_report.py runs_dgx/earvn/continual
"""
import argparse
import json
from pathlib import Path

import numpy as np

VARIANTS = {  # id -> run-folder suffix written by scripts/train_continual.py
    "ctrl": "cms1-1-1_b1_p0",
    "A": "cms1-16-256_b1_p0",
    "B": "cms1-16-256_b1_p0.5",
    "C": "cms1-16-256_b16_p0",
    "D": "cms1-16-256_b16_p0.5",
    "E": "cms1-4-16-64_b1_p0.5",
}
SEEDS_TEST = [1, 2, 3, 4, 5]
FINAL, FORGET = "final_avg_dir@far0.05", "forgetting_dir@far0.05"


def load(root, folder, seed):
    f = Path(root) / folder / f"seed{seed}" / "metrics.json"
    return json.load(open(f))["summary"] if f.exists() else None


def series(root, folder, key, seeds=SEEDS_TEST):
    out = []
    for s in seeds:
        m = load(root, folder, s)
        if m is None:
            raise FileNotFoundError(f"{folder}/seed{s} missing")
        out.append(m["final_U_test"][key[2:]] if key.startswith("U:") else m[key])
    return np.array(out)


def wins(a, b, higher=True):
    d = (a - b) if higher else (b - a)
    return int((d > 0).sum()), float(d.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    args = ap.parse_args()
    root = args.root

    # --- selection on seed 0 only ---
    chosen = {}
    for base in ("naive", "replay_k5"):
        scores = {v: load(root, f"{base}_{VARIANTS[v]}", 0)[FINAL] for v in "ABCDE"}
        chosen[base] = max(scores, key=scores.get)
        print(f"seed-0 selection ({base}): " + ", ".join(f"{v}={s:.3f}" for v, s in scores.items())
              + f"  -> {chosen[base]}")

    # --- test table, seeds 1-5 ---
    rows = [("naive", "naive"), ("EWC lam=1e2", "ewc_lam100"), ("ctrl (same arch)", f"naive_{VARIANTS['ctrl']}"),
            (f"CMS* = {chosen['naive']}", f"naive_{VARIANTS[chosen['naive']]}"),
            ("replay k=5", "replay_k5"), ("ctrl + replay", f"replay_k5_{VARIANTS['ctrl']}"),
            (f"CMS*+R = {chosen['replay_k5']}", f"replay_k5_{VARIANTS[chosen['replay_k5']]}"),
            ("joint", "joint")]
    keys = [(FINAL, "seen DIR@5%"), (FORGET, "forget DIR@5%"), ("final_avg_rank1", "seen rank1"),
            ("U:dir@far0.05", "unseen DIR@5%"), ("U:auroc", "unseen AUROC")]
    print(f"\nTEST, seeds 1-5\n{'method':26s}" + "".join(f"{n:>17s}" for _, n in keys))
    for label, folder in rows:
        cells = []
        for k, _ in keys:
            try:
                v = series(root, folder, k)
                cells.append(f"{v.mean():.3f} ±{v.std():.3f}")
            except (FileNotFoundError, KeyError):
                cells.append("—")
        print(f"{label:26s}" + "".join(f"{c:>17s}" for c in cells))

    # --- decision rule ---
    cms = f"naive_{VARIANTS[chosen['naive']]}"
    cmsr = f"replay_k5_{VARIANTS[chosen['replay_k5']]}"
    print("\nDECISION (paired, seeds 1-5; win = better on >= 4/5)")
    a_parts = []
    for ref in ("naive", "ewc_lam100"):
        wf, df = wins(series(root, cms, FORGET), series(root, ref, FORGET), higher=False)
        ws, ds = wins(series(root, cms, FINAL), series(root, ref, FINAL))
        print(f"  (a) CMS* vs {ref:11s}: forgetting lower {wf}/5 (mean {df:+.3f}); seen DIR@5% higher {ws}/5 (mean {ds:+.3f})")
        a_parts.append(wf >= 4 and ws >= 4)
    a = all(a_parts)
    wb, db = wins(series(root, cmsr, FINAL), series(root, "replay_k5", FINAL))
    b = wb >= 4
    print(f"  (b) CMS*+R vs replay k=5: seen DIR@5% higher {wb}/5 (mean {db:+.3f})")
    wa, da = wins(series(root, cms, FINAL), series(root, f"naive_{VARIANTS['ctrl']}", FINAL))
    wra, dra = wins(series(root, cmsr, FINAL), series(root, f"replay_k5_{VARIANTS['ctrl']}", FINAL))
    print(f"  attribution: CMS* vs ctrl {wa}/5 (mean {da:+.3f}); CMS*+R vs ctrl+R {wra}/5 (mean {dra:+.3f})")
    attributed = (a and wa >= 4) or (b and wra >= 4)
    verdict = "GO" if attributed else ("PARTIAL (gain not attributable to multi-frequency)" if (a or b) else "NO-GO")
    print(f"  (a) {'holds' if a else 'fails'}; (b) {'holds' if b else 'fails'}  =>  {verdict}")


if __name__ == "__main__":
    main()
