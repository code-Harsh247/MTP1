"""R3: choose the slow-copy learning rate of each model x dataset from the seed-0 runs (rule in
benchmark/r3_rule.yaml) and write benchmark/r3_lr_pow.yaml. Refuses to write if a run is missing,
unless --allow-missing is given.

    python scripts/r3_select.py runs_kaggle/r3 runs_dgx/r3
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r2_select import pick  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()
    rule = yaml.safe_load(open(ROOT / "benchmark" / "r3_rule.yaml"))
    recipes = yaml.safe_load(open(ROOT / "benchmark" / "recipes.yaml"))["recipes"]
    periods = "-".join(str(p) for p in rule["periods"])
    chosen, missing = {}, []
    for name, models in recipes.items():
        chosen[name] = {}
        print(f"\n{name}\n  {'model':26s}{'lr_pow':>7s}{'rank-1':>8s}{'forgetting':>12s}")
        for key, r in (models or {}).items():
            rows = []
            for pow_ in rule["lr_pows"]:
                run = f"naive_bodycms{periods}_p{pow_:g}_{key}_lr{r['lr_body']:g}_s{r['steps_per_task']}"
                rel = Path(name) / "continual" / run / "seed0" / "metrics.json"
                f = next((Path(p) / rel for p in args.roots if (Path(p) / rel).exists()), None)
                if f is None:
                    missing.append(str(rel))
                    continue
                s = json.load(open(f))["summary"]
                rows.append({"lam": pow_, "rank1": s["final_avg_rank1"], "forgetting": s["forgetting_rank1"]})
            choice = pick(rows)
            for row in rows:
                mark = " <-" if row is choice else ""
                print(f"  {key:26s}{row['lam']:7g}{row['rank1']:8.3f}{row['forgetting']:12.3f}{mark}")
            if choice:
                chosen[name][key] = choice["lam"]
    if missing:
        print(f"\n{len(missing)} runs missing, e.g. {missing[0]}")
        if not args.allow_missing:
            sys.exit("not writing r3_lr_pow.yaml: runs are missing")
    out = ROOT / "benchmark" / "r3_lr_pow.yaml"
    out.write_text("# Written by scripts/r3_select.py from the seed-0 runs.\n"
                   + yaml.safe_dump({"lr_pow": chosen}, sort_keys=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
