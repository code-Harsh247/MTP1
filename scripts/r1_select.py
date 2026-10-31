"""R1: apply the selection rule of benchmark/r1_grid.yaml to the seed-0 tuning runs and write
benchmark/recipes.yaml. Refuses to write if a run is missing, unless --allow-missing is given
(the gaps are then listed in the output).

    python scripts/r1_select.py runs_dgx/r1_tune
"""
import argparse
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def pick(rows, tol=0.005):
    """rows: dicts with rank1, auroc, steps, lr. Returns the chosen row (rule in r1_grid.yaml)."""
    ok = [r for r in rows if math.isfinite(r["rank1"]) and math.isfinite(r["auroc"])]
    if not ok:
        return None
    best = max(r["rank1"] for r in ok)
    near = [r for r in ok if r["rank1"] >= best - tol]
    return sorted(near, key=lambda r: (-r["auroc"], r["steps"], r["lr"]))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()
    bench = yaml.safe_load(open(ROOT / "benchmark" / "tasks.yaml"))["datasets"]
    models = yaml.safe_load(open(ROOT / "benchmark" / "models.yaml"))["models"]
    grid = yaml.safe_load(open(ROOT / "benchmark" / "r1_grid.yaml"))
    recipes, missing = {}, []
    for name in bench:
        recipes[name] = {}
        print(f"\n{name}\n  {'model':26s}{'lr_body':>9s}{'steps':>7s}{'rank-1':>8s}{'val AUROC':>11s}")
        for key in models:
            rows = []
            for lr in grid["lr_body"]:
                for steps in grid["steps_per_task"][name]:
                    f = (Path(args.root) / name / "continual" / f"joint_{key}_lr{lr:g}_s{steps}"
                         / "seed0" / "metrics.json")
                    if not f.exists():
                        missing.append(str(f))
                        continue
                    s = json.load(open(f))["summary"]
                    rows.append({"lr": lr, "steps": steps, "rank1": s["final_avg_rank1"],
                                 "auroc": s["final_U_val"]["auroc"]})
            choice = pick(rows)
            for r in sorted(rows, key=lambda r: (r["lr"], r["steps"])):
                mark = " <-" if r is choice else ""
                print(f"  {key:26s}{r['lr']:9g}{r['steps']:7d}{r['rank1']:8.3f}{r['auroc']:11.3f}{mark}")
            if choice:
                recipes[name][key] = {"lr_body": choice["lr"], "steps_per_task": choice["steps"]}
    if missing:
        print(f"\n{len(missing)} runs missing, e.g. {missing[0]}")
        if not args.allow_missing:
            sys.exit("not writing recipes.yaml: runs are missing")
    out = ROOT / "benchmark" / "recipes.yaml"
    out.write_text("# Written by scripts/r1_select.py from the seed-0 tuning runs.\n"
                   + yaml.safe_dump({"recipes": recipes}, sort_keys=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
