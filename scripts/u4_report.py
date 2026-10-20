"""Apply the U4 decision rule (notes/u4_m3.md) mechanically.

    python scripts/u4_report.py runs_dgx/uerc
"""
import argparse
import json
from pathlib import Path

import numpy as np

LRS = ["3e-6", "1e-5", "3e-5"]
SEEDS = [1, 2, 3]
PRIMARY = "dir@far0.05"


def tag(alpha, lr):
    return f"m3a{alpha}_lr{lr}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    root = Path(ap.parse_args().root)

    chosen = {}
    for alpha, fam in ((1, "M3"), (0, "control")):
        val = {}
        for lr in LRS:
            f = root / f"tune/baseline_resnet50_{tag(alpha, lr)}/seed0/metrics.json"
            r = json.load(open(f))["results"]["cosine_proto"]
            assert "test" not in r, "tuning output must not contain test metrics"
            val[lr] = r["val"]
        ranked = sorted(LRS, key=lambda lr: (val[lr][PRIMARY], val[lr]["auroc"]), reverse=True)
        top, nxt = ranked[0], ranked[1]
        pick = max((top, nxt), key=lambda lr: val[lr]["auroc"]) if val[top][PRIMARY] - val[nxt][PRIMARY] <= 0.005 else top
        chosen[fam] = tag(alpha, pick)
        print(f"seed-0 VAL {fam:8s}: " + "  ".join(f"lr {lr}: DIR@5% {val[lr][PRIMARY]:.4f} AUROC {val[lr]['auroc']:.4f}"
                                                   for lr in LRS) + f"  -> {chosen[fam]}")

    def test(folder, key="cosine_proto", metric=PRIMARY):
        return np.array([json.load(open(root / folder / f"seed{s}/metrics.json"))["results"][key]["test"][metric]
                         for s in SEEDS])

    arms = {"AdamW (U1)": "baseline_resnet50_e4", f"M3* ({chosen['M3']})": f"baseline_resnet50_{chosen['M3']}",
            f"control* ({chosen['control']})": f"baseline_resnet50_{chosen['control']}"}
    print(f"\nTEST, seeds {SEEDS}, cosine_proto")
    for var, title in (("", "2 gallery"), ("@g5", "5 gallery")):
        print(f"  [{title}]")
        print(f"  {'arm':28s}" + "".join(f"{m:>15s}" for m in ("rank1", "auroc", PRIMARY, "eer")))
        for label, folder in arms.items():
            print(f"  {label:28s}" + "".join(f"{test(folder, 'cosine_proto' + var, m).mean():9.3f}±{test(folder, 'cosine_proto' + var, m).std():.3f}"
                                             for m in ("rank1", "auroc", PRIMARY, "eer")))

    m3, ctrl, adamw = (test(f) for f in arms.values())
    w_a, w_c = int((m3 > adamw).sum()), int((m3 > ctrl).sum())
    print(f"\nDECISION (primary: test {PRIMARY}, 2 gallery, paired over seeds {SEEDS})")
    print(f"  M3* vs AdamW   : better on {w_a}/3, mean {np.mean(m3 - adamw):+.4f}  per seed {np.round(m3 - adamw, 4).tolist()}")
    print(f"  M3* vs control*: better on {w_c}/3, mean {np.mean(m3 - ctrl):+.4f}")
    print(f"  (reported) control* vs AdamW: better on {int((ctrl > adamw).sum())}/3, mean {np.mean(ctrl - adamw):+.4f}")
    verdict = "GO (NL)" if (w_a == 3 and w_c == 3) else ("PARTIAL" if w_a == 3 else "NO-GO")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
