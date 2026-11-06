"""R4b report (exploratory): CMS with task-boundary consolidation against naive, by
benchmark/r3_rule.yaml's test, with R3 and R4 CMS alongside. Reads only the raw metrics.json files.

    python scripts/r4b_report.py --joint runs_kaggle/r1_final --r2 runs_kaggle/r2 \
        --r3 runs_kaggle/r3 --r4 runs_kaggle/r4 --r4b runs_kaggle/r4b
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
    for arg in ("joint", "r2", "r3", "r4", "r4b"):
        ap.add_argument(f"--{arg}", nargs="+", required=True)
    args = ap.parse_args()
    bench = yaml.safe_load(open(ROOT / "benchmark" / "tasks.yaml"))["datasets"]
    rule = yaml.safe_load(open(ROOT / "benchmark" / "r3_rule.yaml"))
    design = yaml.safe_load(open(ROOT / "benchmark" / "r4b_design.yaml"))
    slow_tasks = yaml.safe_load(open(ROOT / "benchmark" / "r4_design.yaml"))["slow_tasks"]
    recipes = yaml.safe_load(open(ROOT / "benchmark" / "recipes.yaml"))["recipes"]
    pows = yaml.safe_load(open(ROOT / "benchmark" / "r3_lr_pow.yaml"))["lr_pow"]
    r3_periods = "-".join(str(p) for p in rule["periods"])
    folder_b = f"naive_consol{'-'.join(map(str, design['every']))}_b{design['beta']:g}"
    verdicts = []
    print("| Dataset | Model | Naive forg. | R3 CMS forg. | R4 CMS forg. | R4b forg. | Naive final | "
          "R4b final | Joint | Mixing weights at end | R4b vs naive |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, keys in design["cells"].items():
        seeds = [s for s in bench[name]["seeds"] if s != 0]
        for key in keys:
            r = recipes[name][key]
            tag = f"{key}_lr{r['lr_body']:g}_s{r['steps_per_task']}"
            p = f"{pows[name][key]:g}"
            r4_folder = f"naive_bodycms{'-'.join(map(str, periods_of(r['steps_per_task'], slow_tasks)))}_p{p}"

            def get(roots, folder, seed):
                return load(roots, Path(name) / "continual" / f"{folder}_{tag}" / f"seed{seed}" / "metrics.json")

            rows, r3_f, r4_f, mix = [], [], [], []
            for seed in seeds:
                naive, joint = get(args.r2, "naive", seed), get(args.joint, "joint", seed)
                new = get(args.r4b, folder_b, seed)
                for roots, folder, acc in ((args.r3, f"naive_bodycms{r3_periods}_p{p}", r3_f),
                                           (args.r4, r4_folder, r4_f)):
                    m = get(roots, folder, seed)
                    if m:
                        acc.append(m["summary"]["forgetting_rank1"])
                if naive and joint and new:
                    mix.append(new["log"][-1].get("mix_weights") or [])
                    rows.append({"base_f": naive["summary"]["forgetting_rank1"], "cms_f": new["summary"]["forgetting_rank1"],
                                 "base_final": naive["summary"]["final_avg_rank1"], "cms_final": new["summary"]["final_avg_rank1"],
                                 "joint": joint["summary"]["final_avg_rank1"]})
            v = compare(rows, len(seeds), rule)
            verdicts.append(v)

            def mean(values):
                return f"{np.mean(values):.3f}" if values else ""

            mix_s = "/".join(f"{x:.2f}" for x in np.mean(mix, 0)) if mix and mix[0] else ""
            print(f"| {name} | {key} | {mean([x['base_f'] for x in rows])} | {mean(r3_f)} | {mean(r4_f)} | "
                  f"{mean([x['cms_f'] for x in rows])} | {mean([x['base_final'] for x in rows])} | "
                  f"{mean([x['cms_final'] for x in rows])} | {mean([x['joint'] for x in rows])} | {mix_s} | **{v}** |")
    print(f"\nR4b verdicts (exploratory): { {v: verdicts.count(v) for v in sorted(set(verdicts))} }")


if __name__ == "__main__":
    main()
