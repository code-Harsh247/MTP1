"""R2: choose the EWC and LwF strength of each model x dataset from the seed-0 runs (rule in
benchmark/r2_grid.yaml) and write benchmark/r2_strengths.yaml. Refuses to write if a run is
missing, unless --allow-missing is given.

    python scripts/r2_select.py runs_kaggle/r2 runs_dgx/r2
"""
import argparse
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def pick(rows, tol=0.005):
    """rows: dicts with rank1, forgetting, lam. Returns the chosen row."""
    ok = [r for r in rows if math.isfinite(r["rank1"]) and math.isfinite(r["forgetting"])]
    if not ok:
        return None
    best = max(r["rank1"] for r in ok)
    near = [r for r in ok if r["rank1"] >= best - tol]
    return sorted(near, key=lambda r: (r["forgetting"], r["lam"]))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()
    grid = yaml.safe_load(open(ROOT / "benchmark" / "r2_grid.yaml"))
    recipes = yaml.safe_load(open(ROOT / "benchmark" / "recipes.yaml"))["recipes"]
    chosen, missing = {}, []
    for name, models in recipes.items():
        chosen[name] = {}
        print(f"\n{name}\n  {'model':26s}{'method':>7s}{'lam':>9s}{'rank-1':>8s}{'forgetting':>12s}")
        for key, r in (models or {}).items():
            chosen[name][key] = {}
            for method, values in grid["strengths"].items():
                rows = []
                for lam in values:
                    run = f"{method}_lam{lam:g}_{key}_lr{r['lr_body']:g}_s{r['steps_per_task']}"
                    rel = Path(name) / "continual" / run / "seed0" / "metrics.json"
                    f = next((Path(p) / rel for p in args.roots if (Path(p) / rel).exists()), None)
                    if f is None:
                        missing.append(str(rel))
                        continue
                    s = json.load(open(f))["summary"]
                    rows.append({"lam": lam, "rank1": s["final_avg_rank1"], "forgetting": s["forgetting_rank1"]})
                choice = pick(rows)
                for row in rows:
                    mark = " <-" if row is choice else ""
                    print(f"  {key:26s}{method:>7s}{row['lam']:9g}{row['rank1']:8.3f}{row['forgetting']:12.3f}{mark}")
                if choice:
                    chosen[name][key][method] = choice["lam"]
    if missing:
        print(f"\n{len(missing)} runs missing, e.g. {missing[0]}")
        if not args.allow_missing:
            sys.exit("not writing r2_strengths.yaml: runs are missing")
    out = ROOT / "benchmark" / "r2_strengths.yaml"
    out.write_text("# Written by scripts/r2_select.py from the seed-0 runs.\n"
                   + yaml.safe_dump({"strengths": chosen}, sort_keys=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
