"""Apply the U2 decision rule (notes/u2_killtest.md) mechanically.

    python scripts/u2_report.py runs_dgx/uerc
"""
import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = [1, 2, 3]
PRIMARY = "dir@far0.05"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="runs_dgx/uerc")
    root = Path(ap.parse_args().root)
    choice = json.load(open(root / "u2_arm3_choice.json"))
    arms = [("1 cosine", "cosine_proto"), ("2 linear mem", "subspace"), ("3 deep mem", choice["spec"]),
            ("4 meta mem (full)", "meta_memory_full"), ("4' meta rule", "meta_memory_rule"),
            ("4b control", "meta_proto_ctrl"), ("ref EVM", "evm")]
    R = {s: json.load(open(root / f"splitbb_resnet50_e4/seed{s}/metrics.json"))["results"] for s in SEEDS}
    print(f"arm 3 (seed-0 val choice): {choice['spec']}")
    print("seed-0 val table:", {k: round(v[PRIMARY], 4) for k, v in choice["val"].items()})

    def val(key, metric, s, suffix=""):
        return R[s][key + suffix]["test"][metric]

    for suffix, title in (("", "all sources, 2 gallery"), ("@g5", "all sources, 5 gallery"),
                          ("#jpg", "VGGFace2 (jpg), 2 gallery"), ("#png", "UERC19+new (png), 2 gallery")):
        print(f"\nTEST {title}, seeds {SEEDS}")
        print(f"{'arm':20s}{'rank-1':>14s}{'AUROC':>14s}{'DIR@1%':>14s}{'DIR@5%':>14s}{'DIR@10%':>14s}")
        for label, key in arms:
            try:
                cells = []
                for m in ("rank1", "auroc", "dir@far0.01", PRIMARY, "dir@far0.1"):
                    v = np.array([val(key, m, s, suffix) for s in SEEDS])
                    cells.append(f"{v.mean():.3f}±{v.std():.3f}")
                print(f"{label:20s}" + "".join(f"{c:>14s}" for c in cells))
            except KeyError:
                print(f"{label:20s}  (missing)")

    def wins(a, b):
        d = np.array([val(a, PRIMARY, s) - val(b, PRIMARY, s) for s in SEEDS])
        return int((d > 0).sum()), float(d.mean())

    print(f"\nDECISION (primary: test {PRIMARY}, 2 gallery, all sources; paired over seeds {SEEDS})")
    w1, d1 = wins("meta_memory_full", "cosine_proto")
    wc, dc = wins("meta_memory_full", "meta_proto_ctrl")
    print(f"  arm 4 vs arm 1 (cosine) : better on {w1}/3, mean {d1:+.4f}")
    print(f"  arm 4 vs arm 4b (control): better on {wc}/3, mean {dc:+.4f}")
    for label, key in arms[1:3] + arms[4:6]:
        w, d = wins(key, "cosine_proto")
        print(f"  (reported) {label:18s} vs cosine: better on {w}/3, mean {d:+.4f}")
    verdict = "GO" if (w1 == 3 and wc == 3) else ("PARTIAL" if w1 == 3 else "NO-GO")
    print(f"  VERDICT: {verdict}")

    print("\nselected checkpoint episode per seed (0 = untrained start won on val):")
    for key in ("meta_memory_full", "meta_memory_rule", "meta_proto_ctrl"):
        eps = []
        for s in SEEDS:
            log = json.load(open(root / f"splitbb_resnet50_e4/seed{s}/{key}/metrics.json"))["log"]
            eps.append(max(log, key=lambda e: e["val_dir@far0.05"])["episode"])
        print(f"  {key:18s} {eps}")


if __name__ == "__main__":
    main()
