"""R4 report (exploratory): task-boundary CMS against naive, by benchmark/r3_rule.yaml's test, and
against R3's CMS for comparison. Reads only the raw metrics.json files.

    python scripts/r4_report.py --joint runs_kaggle/r1_final --r2 runs_kaggle/r2 \
        --r3 runs_kaggle/r3 --r4 runs_kaggle/r4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "dgx"))
from make_r4_tasks import periods_of  # noqa: E402
from r2_report import load  # noqa: E402
from r3_report import compare  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    for arg in ("joint", "r2", "r3", "r4"):
        ap.add_argument(f"--{arg}", nargs="+", required=True)
    args = ap.parse_args()
    bench = yaml.safe_load(open(ROOT / "benchmark" / "tasks.yaml"))["datasets"]
    rule = yaml.safe_load(open(ROOT / "benchmark" / "r3_rule.yaml"))
    design = yaml.safe_load(open(ROOT / "benchmark" / "r4_design.yaml"))
    recipes = yaml.safe_load(open(ROOT / "benchmark" / "recipes.yaml"))["recipes"]
    pows = yaml.safe_load(open(ROOT / "benchmark" / "r3_lr_pow.yaml"))["lr_pow"]
    r3_periods = "-".join(str(p) for p in rule["periods"])
    verdicts = []
    print("| Dataset | Model | Periods | Slow updates | Naive forg. | R3 CMS forg. | R4 CMS forg. | "
          "Naive final | R4 final | Joint | R4 vs naive |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, keys in design["cells"].items():
        seeds = [s for s in bench[name]["seeds"] if s != 0]
        for key in keys:
            r = recipes[name][key]
            tag = f"{key}_lr{r['lr_body']:g}_s{r['steps_per_task']}"
            periods = periods_of(r["steps_per_task"], design["slow_tasks"])
            p = f"{pows[name][key]:g}"

            def get(roots, folder, seed):
                return load(roots, Path(name) / "continual" / f"{folder}_{tag}" / f"seed{seed}" / "metrics.json")

            rows, r3_f, updates = [], [], None
            for seed in seeds:
                naive, joint = get(args.r2, "naive", seed), get(args.joint, "joint", seed)
                r4 = get(args.r4, f"naive_bodycms{'-'.join(map(str, periods))}_p{p}", seed)
                r3 = get(args.r3, f"naive_bodycms{r3_periods}_p{p}", seed)
                if r3:
                    r3_f.append(r3["summary"]["forgetting_rank1"])
                if naive and joint and r4:
                    updates = r4["summary"].get("cms_updates_per_period")
                    rows.append({"base_f": naive["summary"]["forgetting_rank1"], "cms_f": r4["summary"]["forgetting_rank1"],
                                 "base_final": naive["summary"]["final_avg_rank1"], "cms_final": r4["summary"]["final_avg_rank1"],
                                 "joint": joint["summary"]["final_avg_rank1"]})
            v = compare(rows, len(seeds), rule)
            verdicts.append(v)

            def mean(values):
                return f"{np.mean(values):.3f}" if values else ""

            print(f"| {name} | {key} | {periods} | {updates or ''} | {mean([x['base_f'] for x in rows])} | "
                  f"{mean(r3_f)} | {mean([x['cms_f'] for x in rows])} | {mean([x['base_final'] for x in rows])} | "
                  f"{mean([x['cms_final'] for x in rows])} | {mean([x['joint'] for x in rows])} | **{v}** |")
    print(f"\nR4 verdicts (exploratory): { {v: verdicts.count(v) for v in sorted(set(verdicts))} }")


if __name__ == "__main__":
    main()
