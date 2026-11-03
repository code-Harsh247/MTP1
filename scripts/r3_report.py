"""R3 report: apply benchmark/r3_rule.yaml to every model x dataset and answer RQ2.
Reads only the raw metrics.json files.

    python scripts/r3_report.py --joint runs_kaggle/r1_final runs_dgx/r1_final \
        --r2 runs_kaggle/r2 runs_dgx/r2 --r3 runs_kaggle/r3 runs_dgx/r3
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r2_report import load, verdict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def compare(rows, n_seeds, rule):
    """rows: one dict per seed with base_f, cms_f, base_final, cms_final, joint (joint may be nan
    for the secondary comparison). Returns INCOMPLETE, not reduced, REDUCED or FIXED."""
    if len(rows) < n_seeds or any(not math.isfinite(r[k]) for r in rows for k in
                                  ("base_f", "cms_f", "base_final", "cms_final")):
        return "INCOMPLETE"
    drops = [r["base_f"] - r["cms_f"] for r in rows]
    need = n_seeds - 1 if n_seeds >= 5 else n_seeds
    reduced = (np.mean(drops) >= rule["min_mean_drop"]
               and sum(d > 0 for d in drops) >= need
               and np.mean([r["cms_final"] for r in rows]) >= np.mean([r["base_final"] for r in rows]) - rule["max_final_loss"])
    if not reduced:
        return "not reduced"
    fixed = (np.mean([r["cms_f"] for r in rows]) < rule["fixed_forgetting_below"]
             and np.mean([r["cms_final"] for r in rows]) > np.mean([r["joint"] for r in rows]) - rule["joint_gap"])
    return "FIXED" if fixed else "REDUCED"


def rq2(verdicts):
    """verdicts: the primary verdict of every cell where R2 showed forgetting."""
    good = sum(v in ("REDUCED", "FIXED") for v in verdicts)
    if good == 0:
        return "NO"
    return "YES" if good > len(verdicts) / 2 else "PARTIAL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", nargs="+", required=True)
    ap.add_argument("--r2", nargs="+", required=True)
    ap.add_argument("--r3", nargs="+", required=True)
    args = ap.parse_args()
    bench = yaml.safe_load(open(ROOT / "benchmark" / "tasks.yaml"))["datasets"]
    grid = yaml.safe_load(open(ROOT / "benchmark" / "r2_grid.yaml"))
    rule = yaml.safe_load(open(ROOT / "benchmark" / "r3_rule.yaml"))
    recipes = yaml.safe_load(open(ROOT / "benchmark" / "recipes.yaml"))["recipes"]
    path = ROOT / "benchmark" / "r3_lr_pow.yaml"
    chosen = yaml.safe_load(open(path))["lr_pow"] if path.exists() else {}
    periods = "-".join(str(p) for p in rule["periods"])
    k = rule["replay_k"]
    counted, incomplete = [], 0
    for name, models in recipes.items():
        seeds = [s for s in bench[name]["seeds"] if s != 0]
        print(f"\n## {name} (seeds {seeds})\n")
        print("| Model | R2 | lr_pow | Slow updates | Naive forg. | CMS forg. | Naive final | CMS final | Joint | Primary | Replay forg. | Replay+CMS forg. | Secondary |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for key, r in (models or {}).items():
            tag = f"{key}_lr{r['lr_body']:g}_s{r['steps_per_task']}"
            pow_ = chosen.get(name, {}).get(key)

            def get(roots, folder, seed):
                return load(roots, Path(name) / "continual" / f"{folder}_{tag}" / f"seed{seed}" / "metrics.json")

            r2_rows, primary, secondary, updates = [], [], [], None
            for seed in seeds:
                naive, joint = get(args.r2, "naive", seed), get(args.joint, "joint", seed)
                j = joint["summary"]["final_avg_rank1"] if joint else float("nan")
                if naive and joint:
                    r2_rows.append({"forgetting": naive["summary"]["forgetting_rank1"],
                                    "final": naive["summary"]["final_avg_rank1"], "joint": j})
                if pow_ is None:
                    continue
                cms = get(args.r3, f"naive_bodycms{periods}_p{pow_:g}", seed)
                if naive and cms and joint:
                    updates = cms["summary"].get("cms_updates_per_period")
                    primary.append({"base_f": naive["summary"]["forgetting_rank1"], "cms_f": cms["summary"]["forgetting_rank1"],
                                    "base_final": naive["summary"]["final_avg_rank1"], "cms_final": cms["summary"]["final_avg_rank1"],
                                    "joint": j})
                rep, repc = get(args.r2, f"replay_k{k}", seed), get(args.r3, f"replay_k{k}_bodycms{periods}_p{pow_:g}", seed)
                if rep and repc:
                    secondary.append({"base_f": rep["summary"]["forgetting_rank1"], "cms_f": repc["summary"]["forgetting_rank1"],
                                      "base_final": rep["summary"]["final_avg_rank1"], "cms_final": repc["summary"]["final_avg_rank1"],
                                      "joint": j})
            shown = verdict(r2_rows, len(seeds), grid["forgetting_min"], grid["joint_gap"])
            v1, v2 = compare(primary, len(seeds), rule), compare(secondary, len(seeds), rule)
            if shown == "SHOWN":
                if v1 == "INCOMPLETE":
                    incomplete += 1
                else:
                    counted.append(v1)

            def mean(rows, field):
                return f"{np.mean([x[field] for x in rows]):.3f}" if rows else ""

            print(f"| {key} | {shown} | {'' if pow_ is None else f'{pow_:g}'} | {updates or ''} | {mean(primary, 'base_f')} | "
                  f"{mean(primary, 'cms_f')} | {mean(primary, 'base_final')} | {mean(primary, 'cms_final')} | {mean(primary, 'joint')} | "
                  f"**{v1}** | {mean(secondary, 'base_f')} | {mean(secondary, 'cms_f')} | {v2} |")
    print(f"\nCells with shown forgetting and finished runs: {len(counted)}; shown but incomplete: {incomplete}")
    print(f"Their primary verdicts: { {v: counted.count(v) for v in sorted(set(counted))} }")
    print(f"RQ2: {rq2(counted) if counted else 'NOT ANSWERED YET'}")


if __name__ == "__main__":
    main()
