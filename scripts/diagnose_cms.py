"""Why did CMS not reduce forgetting? Diagnostics on runs trained with --save-stages.

For each run folder (the one holding config.yaml, metrics.json, init.pt and stage<k>.pt):
  1. where task 1 was forgotten: put one part of the final model back to its after-task-1 values
     and see how much of task 1's rank-1 comes back;
  2. how the copies were mixed: the mean softmax weight of each copy after every task;
  3. how far each part moved from its starting (pretrained) weights after every task;
  4. what each copy knows: score with one copy alone (its mixing weight set to 1).
Writes diag.md and diag.json into each run folder and prints the tables.

    python scripts/diagnose_cms.py runs/diag/earvn/continual/<run>/seed1 [more run folders]
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nlear.continual import load_frozen_tasks  # noqa: E402
from nlear.data.cache import load_image_cache  # noqa: E402
from nlear.data.manifest import load_manifest  # noqa: E402
from nlear.models import CMSParallel, EmbeddingNet, cms_targets  # noqa: E402

_spec = importlib.util.spec_from_file_location("train_continual", ROOT / "scripts" / "train_continual.py")
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)


def group_names(model, periods):
    """state_dict key -> part of the model, for the restore and drift tables."""
    name_of = {id(m): n for n, m in model.named_modules()}
    pointwise = [f"{name_of[id(parent)]}.{attr}." for parent, attr in cms_targets(model.body)]

    def group(k):
        if k.startswith("neck."):
            return "neck"
        if k.endswith("agg_logits"):
            return "mixing weights"
        m = re.search(r"\.copies\.(\d+)\.", k)
        if m:
            i = int(m.group(1))
            return f"copy {i + 1} (every {periods[i]} steps)"
        if any(k.startswith(p) for p in pointwise):
            return "pointwise layers"
        return "rest of backbone"

    return {k: group(k) for k in model.state_dict()}


def rel_change(a, b, keys):
    num = sum(float((b[k].float() - a[k].float()).pow(2).sum()) for k in keys if a[k].is_floating_point())
    den = sum(float(a[k].float().pow(2).sum()) for k in keys if a[k].is_floating_point())
    return (num / den) ** 0.5 if den else float("nan")


def diagnose(run_dir, device):
    run_dir = Path(run_dir)
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    d, t, a = cfg["data"]["data"], cfg["train"]["train"], cfg["args"]
    periods = [int(x) for x in a["body_cms_periods"].split(",")] if a.get("body_cms_periods") else []
    bench = yaml.safe_load(open(ROOT / "benchmark" / "tasks.yaml"))["datasets"][a["bench"]]
    T = bench["n_tasks"]
    h, w = d["img_size"]
    images, row_of = load_image_cache(load_manifest(d["manifest"]), d["root"], d["img_size"],
                                      Path(d["manifest"]).parent / f"cache_{h}x{w}.pt")
    df = load_frozen_tasks(a["bench"], d["splits"], a["seed"], T, bench["held_gallery"])
    df["row"] = df["path"].map(row_of)
    model = EmbeddingNet(t["arch"], t["emb_dim"], False, t.get("dropout", 0.0), img_size=d["img_size"],
                         body_cms_levels=len(periods)).to(device)
    init = torch.load(run_dir / "init.pt", map_location="cpu")
    stages = [torch.load(run_dir / f"stage{s}.pt", map_location="cpu") for s in range(T)]
    final, first = stages[-1], stages[0]
    groups = group_names(model, periods)
    parts = sorted(set(groups.values()))

    def rank1(state, mix=None):
        model.load_state_dict(state)
        if mix is not None:
            for m in model.modules():
                if isinstance(m, CMSParallel):
                    m.agg_logits.data = torch.full_like(m.agg_logits, -30.0)
                    m.agg_logits.data[mix] = 0.0
        A_row, _ = tc.evaluate_stage(model, images, df, device, T)
        return [r["rank1"] for r in A_row]

    out = {"run": str(run_dir), "periods": periods}
    r_first, r_final = rank1(first), rank1(final)
    logged = json.load(open(run_dir / "metrics.json"))["A"][-1]
    out["check_final_matches_metrics"] = bool(np.allclose(r_final, [x["rank1"] for x in logged], atol=1e-6))
    out["task1_after_task1"], out["task1_final"] = r_first[0], r_final[0]
    out["restore"] = {}
    for part in parts:
        state = {k: (first[k] if groups[k] == part else v) for k, v in final.items()}
        r = rank1(state)
        lost = r_first[0] - r_final[0]
        out["restore"][part] = {"task1": r[0], "last_task": r[-1],
                                "share_recovered": (r[0] - r_final[0]) / lost if lost > 0 else float("nan")}
    out["mix_weights"] = [x.get("mix_weights") for x in json.load(open(run_dir / "metrics.json"))["log"]]
    out["moved_from_start"] = {part: [rel_change(init, st, [k for k in groups if groups[k] == part]) for st in stages]
                               for part in parts}
    if periods:
        out["one_copy_only"] = {}
        for i, p in enumerate(periods):
            r = rank1(final, mix=i)
            out["one_copy_only"][f"copy {i + 1} (every {p} steps)"] = {
                "task1": r[0], "old_tasks": float(np.mean(r[:-1])), "last_task": r[-1]}
        out["one_copy_only"]["all copies, as trained"] = {
            "task1": r_final[0], "old_tasks": float(np.mean(r_final[:-1])), "last_task": r_final[-1]}
    return out


def to_markdown(o):
    lines = [f"## {o['run']}", "", f"Periods: {o['periods'] or 'plain model'}. "
             f"Final rank-1 matches metrics.json: {o['check_final_matches_metrics']}.", "",
             f"**1. Where task 1 was forgotten.** Task 1 rank-1: {o['task1_after_task1']:.3f} after task 1, "
             f"{o['task1_final']:.3f} at the end. Each row puts one part back to its after-task-1 values.", "",
             "| Part put back | Task 1 rank-1 | Share of the loss recovered | Last task rank-1 |", "|---|---|---|---|"]
    for part, r in o["restore"].items():
        lines.append(f"| {part} | {r['task1']:.3f} | {r['share_recovered']:.0%} | {r['last_task']:.3f} |")
    if any(o["mix_weights"]):
        lines += ["", "**2. Mixing weights** (mean over CMS modules, one value per copy):", "",
                  "| After task | " + " | ".join(f"copy {i + 1}" for i in range(len(o["periods"]))) + " |",
                  "|---|" + "---|" * len(o["periods"])]
        for s, wts in enumerate(o["mix_weights"]):
            lines.append(f"| {s + 1} | " + " | ".join(f"{x:.3f}" for x in wts) + " |")
    lines += ["", "**3. How far each part moved** from its starting weights (relative L2 change):", "",
              "| Part | " + " | ".join(f"after task {s + 1}" for s in range(len(next(iter(o["moved_from_start"].values()))))) + " |",
              "|---|" + "---|" * len(next(iter(o["moved_from_start"].values())))]
    for part, vals in o["moved_from_start"].items():
        if part == "mixing weights":
            continue   # they start at zero, so a relative change is undefined; table 2 shows them
        lines.append(f"| {part} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
    if o.get("one_copy_only"):
        lines += ["", "**4. One copy alone** (final model, that copy's mixing weight set to 1):", "",
                  "| Copy used | Task 1 rank-1 | Old tasks rank-1 | Last task rank-1 |", "|---|---|---|---|"]
        for c, r in o["one_copy_only"].items():
            lines.append(f"| {c} | {r['task1']:.3f} | {r['old_tasks']:.3f} | {r['last_task']:.3f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for run in args.runs:
        o = diagnose(run, device)
        md = to_markdown(o)
        json.dump(o, open(Path(run) / "diag.json", "w"), indent=1)
        (Path(run) / "diag.md").write_text(md, encoding="utf-8")
        print(md, flush=True)


if __name__ == "__main__":
    main()
