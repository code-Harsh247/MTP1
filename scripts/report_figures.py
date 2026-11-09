"""Figures for the MTP report (reports/mtp_report/figures/), read only from the raw metrics.json and
diag.json files.

    python scripts/report_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "dgx"))
from make_r4_tasks import periods_of  # noqa: E402

RUNS = ROOT / "runs_kaggle"
OUT = ROOT / "reports" / "mtp_report" / "figures"
# Categorical slots in fixed order (validated: light mode, adjacent-pair CVD and normal-vision pass).
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
HATCH = ["", "//", "..", "xx"]          # second encoding, for print and colour-blind readers
INK, MUTED = "#1f1f1e", "#6b6a64"
NAMES = {"classic_cnn": "ResNet-50", "modern_cnn": "ConvNeXt-T", "lightweight_cnn": "MobileNetV3",
         "plain_transformer": "DeiT-S", "hierarchical_transformer": "Swin-T"}
DS = {"ami": "AMI", "awe": "AWE", "iitd": "IITD", "earvn": "EarVN"}
SHOWN = [("ami", "plain_transformer"), ("awe", "modern_cnn"), ("awe", "plain_transformer"),
         ("awe", "hierarchical_transformer"), ("earvn", "classic_cnn"), ("earvn", "modern_cnn"),
         ("earvn", "lightweight_cnn"), ("earvn", "plain_transformer"), ("earvn", "hierarchical_transformer")]

bench = yaml.safe_load(open(ROOT / "benchmark" / "tasks.yaml"))["datasets"]
recipes = yaml.safe_load(open(ROOT / "benchmark" / "recipes.yaml"))["recipes"]
strengths = yaml.safe_load(open(ROOT / "benchmark" / "r2_strengths.yaml"))["strengths"]
pows = yaml.safe_load(open(ROOT / "benchmark" / "r3_lr_pow.yaml"))["lr_pow"]

plt.rcParams.update({"pdf.fonttype": 42, "font.family": "serif","font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def tag(name, key):
    r = recipes[name][key]
    return f"{key}_lr{r['lr_body']:g}_s{r['steps_per_task']}"


def forgetting(root, name, folder):
    vals = []
    for s in bench[name]["seeds"]:
        if s == 0:
            continue
        f = RUNS / root / name / "continual" / folder / f"seed{s}" / "metrics.json"
        vals.append(json.load(open(f))["summary"]["forgetting_rank1"])
    return float(np.mean(vals)), float(np.std(vals))


def bars(ax, groups, series, labels, ylabel):
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(groups))
    for i, (vals, lab) in enumerate(zip(series, labels)):
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w * 0.9, color=C[i], hatch=HATCH[i % 4],
               edgecolor="white", linewidth=0.5, label=lab)
    ax.set_xticks(x, groups, rotation=0)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color="#e4e3de", linewidth=0.6)
    ax.set_axisbelow(True)


def fig_r2():
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.6), sharey=True)
    arms = ["naive", "ewc", "lwf", "replay"]
    for ax, name in zip(axes.flat, ["ami", "awe", "iitd", "earvn"]):
        series = []
        for arm in arms:
            row = []
            for key in NAMES:
                st = strengths[name][key]
                folder = {"naive": "naive", "ewc": f"ewc_lam{st['ewc']:g}", "lwf": f"lwf_lam{st['lwf']:g}",
                          "replay": "replay_k5"}[arm]
                row.append(forgetting("r2", name, f"{folder}_{tag(name, key)}")[0])
            series.append(row)
        bars(ax, [NAMES[k].replace("MobileNetV3", "MobileNet") for k in NAMES], series,
             ["Naive", "EWC", "LwF", "Replay"], "Rank-1 forgetting")
        ax.axhline(0.10, color=MUTED, linestyle="--", linewidth=0.8)
        ax.set_title(DS[name], fontsize=10, color=INK)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        for lab in ax.get_xticklabels():
            lab.set_ha("right")
    axes[0, 0].text(4.45, 0.105, "0.10", color=MUTED, fontsize=7, ha="right", va="bottom")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "r2_forgetting.pdf")


def fig_matrix():
    m = json.load(open(RUNS / "r2" / "earvn" / "continual" / f"naive_{tag('earvn', 'modern_cnn')}" / "seed1" / "metrics.json"))
    A = np.array([[c["rank1"] for c in row] for row in m["A"]])
    T = len(A)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    shown = np.where(np.tril(np.ones_like(A)) > 0, A, np.nan)
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1)
    for t in range(T):
        for j in range(t + 1):
            ax.text(j, t, f"{A[t, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if A[t, j] > 0.6 else INK)
    ax.set_xticks(range(T), [f"T{j + 1}" for j in range(T)])
    ax.set_yticks(range(T), [f"T{t + 1}" for t in range(T)])
    ax.set_ylabel("After training on")
    ax.set_xlabel("Task evaluated")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Rank-1")
    fig.tight_layout()
    fig.savefig(OUT / "acc_matrix_earvn_convnext.pdf")


def fig_cms():
    rule_p = "-".join(str(p) for p in yaml.safe_load(open(ROOT / "benchmark" / "r3_rule.yaml"))["periods"])
    series = [[], [], [], []]
    groups = []
    for name, key in SHOWN:
        t = tag(name, key)
        p = f"{pows[name][key]:g}"
        r4p = "-".join(map(str, periods_of(recipes[name][key]["steps_per_task"], 2)))
        series[0].append(forgetting("r2", name, f"naive_{t}")[0])
        series[1].append(forgetting("r3", name, f"naive_bodycms{rule_p}_p{p}_{t}")[0])
        series[2].append(forgetting("r4", name, f"naive_bodycms{r4p}_p{p}_{t}")[0])
        series[3].append(forgetting("r4b", name, f"naive_consol1-2_b0.5_{t}")[0])
        groups.append(f"{DS[name]}\n{NAMES[key].replace('MobileNetV3', 'MobileNet')}")
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    bars(ax, groups, series, ["Naive", "R3: CMS", "R4: task-boundary CMS", "R4b: consolidation"],
         "Rank-1 forgetting")
    ax.tick_params(axis="x", labelsize=6.3)
    ax.legend(loc="upper left", ncol=4, fontsize=7.5, bbox_to_anchor=(0, 1.13))
    fig.tight_layout()
    fig.savefig(OUT / "cms_forgetting.pdf")


def fig_diag():
    base = RUNS / "diag" / "earvn" / "continual"
    r3 = json.load(open(base / "naive_bodycms1-16-256_p0_modern_cnn_lr0.0003_s500" / "seed1" / "diag.json"))
    r4 = json.load(open(base / "naive_bodycms1-500-1000_p0_modern_cnn_lr0.0003_s500" / "seed1" / "diag.json"))
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5))
    for i, (d, lab) in enumerate(((r3, "R3 (periods 1, 16, 256)"), (r4, "R4 (periods 1, 500, 1000)"))):
        copies = [k for k in d["moved_from_start"] if k.startswith("copy")]
        for c_i, c in enumerate(copies):
            axes[i].plot(range(1, 6), d["moved_from_start"][c], marker="o", markersize=4, linewidth=2,
                         color=C[c_i], label=f"Copy {c_i + 1}")
        axes[i].set_title(lab, fontsize=9, color=INK)
        axes[i].set_xlabel("After task")
        axes[i].set_xticks(range(1, 6))
        axes[i].yaxis.grid(True, color="#e4e3de", linewidth=0.6)
    axes[0].set_ylabel("Relative change from\npretrained weights")
    axes[1].legend(loc="center right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "cms_copy_movement.pdf")


def fig_r5():
    """R5 on EarVN ConvNeXt-T: task-1 rank-1 over the tasks (mean of seeds 1-3), and each copy alone (seed 1)."""
    t = tag("earvn", "modern_cnn")
    seeds = [s for s in bench["earvn"]["seeds"] if s != 0]
    arms = [("Naive", RUNS / "r2", "naive"), ("R3 CMS", RUNS / "r3", "naive_bodycms1-16-256_p0"),
            ("R5 per-step", RUNS / "r5", "naive_bodycms1-16-256_p0_perstep"),
            ("R5 snapshot", RUNS / "r5", "naive_snapshot1")]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), gridspec_kw={"width_ratios": [1, 1.25]})
    for i, (lab, root, folder) in enumerate(arms):
        cols = []
        for s in seeds:
            m = json.load(open(root / "earvn" / "continual" / f"{folder}_{t}" / f"seed{s}" / "metrics.json"))
            cols.append([row[0]["rank1"] for row in m["A"]])
        axes[0].plot(range(1, 6), np.mean(cols, 0), marker="o", markersize=4, linewidth=2, color=C[i], label=lab)
    axes[0].set_xlabel("After task")
    axes[0].set_ylabel("Task-1 rank-1")
    axes[0].set_xticks(range(1, 6))
    axes[0].set_ylim(0, 1)
    axes[0].yaxis.grid(True, color="#e4e3de", linewidth=0.6)
    axes[0].legend(loc="lower left", fontsize=7)
    axes[0].set_title("Task 1 over the tasks (3 seeds)", fontsize=9, color=INK)

    base = RUNS / "diag" / "earvn" / "continual"
    runs = [("R3", "naive_bodycms1-16-256_p0"), ("R5\nper-step", "naive_bodycms1-16-256_p0_perstep"),
            ("R4b", "naive_consol1-2_b0.5"), ("R5\nsnapshot", "naive_snapshot1")]
    labels = ["Copy 1 alone", "Copy 2 alone", "Copy 3 alone", "All copies"]
    x = np.arange(len(runs))
    w = 0.2
    for j, lab in enumerate(labels):
        vals = []
        for _, folder in runs:
            alone = json.load(open(base / f"{folder}_{t}" / "seed1" / "diag.json"))["one_copy_only"]
            vals.append(list(alone.values())[j]["task1"])
        colour = INK if j == 3 else C[j]
        axes[1].bar(x + (j - 1.5) * w, vals, w, color=colour, label=lab, edgecolor="white", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([r for r, _ in runs], fontsize=8)
    axes[1].set_ylim(0, 1)
    axes[1].yaxis.grid(True, color="#e4e3de", linewidth=0.6)
    axes[1].set_axisbelow(True)
    axes[1].set_ylabel("Task-1 rank-1 at the end")
    axes[1].legend(loc="upper left", fontsize=7, ncol=2)
    axes[1].set_title("Each copy used alone (seed 1)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "r5_memory.pdf")


def fig_r5b():
    """R5b: task-1 forgetting, naive vs. the snapshot oracle on all 9 shown cells (mean over seeds), and
    task-1 forgetting against the frozen copies' share of the output on EarVN ConvNeXt-T and DeiT-S."""
    def t1(m):
        A = np.array([[c["rank1"] for c in row] for row in m["A"]])
        return A[:-1, 0].max() - A[-1, 0]

    def runs(root, name, key, folder):
        out = []
        for s in [s for s in bench[name]["seeds"] if s != 0]:
            p = root / name / "continual" / f"{folder}_{tag(name, key)}" / f"seed{s}" / "metrics.json"
            out.append(json.load(open(p)))
        return out

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7), gridspec_kw={"width_ratios": [1.7, 1]})
    labels, naive, snap = [], [], []
    for name, key in SHOWN:
        labels.append(f"{DS[name]}\n{NAMES[key].replace('MobileNetV3', 'MobileNet')}")
        naive.append(np.mean([t1(m) for m in runs(RUNS / "r2", name, key, "naive")]))
        root = RUNS / ("r5" if (name, key) == ("earvn", "modern_cnn") else "r5b")
        snap.append(np.mean([t1(m) for m in runs(root, name, key, "naive_snapshot1")]))
    x = np.arange(len(labels))
    axes[0].bar(x - 0.2, naive, 0.4, color=C[0], label="Naive")
    axes[0].bar(x + 0.2, snap, 0.4, color=C[3], label="Snapshot (oracle)", hatch=HATCH[1], edgecolor="white",
                linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=6.5, rotation=90)
    axes[0].set_ylabel("Task-1 rank-1 forgetting")
    axes[0].yaxis.grid(True, color="#e4e3de", linewidth=0.6)
    axes[0].set_axisbelow(True)
    axes[0].set_ylim(0, 0.68)
    axes[0].legend(loc="upper left", fontsize=7, ncol=2)
    axes[0].set_title("A perfect memory of task 1, all 9 cells", fontsize=9, color=INK)

    for i, key in enumerate(("modern_cnn", "plain_transformer")):
        learned = runs(RUNS / ("r5" if key == "modern_cnn" else "r5b"), "earvn", key, "naive_snapshot1")
        xs = [np.mean([1 - m["log"][-1]["mix_weights"][0] for m in learned]), 0.8, 0.95]
        ys = [np.mean([t1(m) for m in learned])] + [
            np.mean([t1(m) for m in runs(RUNS / "r5b", "earvn", key, f"naive_snapshot1_mix{v}")]) for v in (80, 95)]
        base = np.mean([t1(m) for m in runs(RUNS / "r2", "earvn", key, "naive")])
        col = C[i * 2]
        axes[1].plot(xs, ys, marker="o", markersize=4, linewidth=2, color=col, label=NAMES[key])
        axes[1].axhline(base, color=col, linewidth=1, linestyle="--")
    axes[1].set_xlabel("Frozen copies' share of output")
    axes[1].set_ylim(0, 0.6)
    axes[1].yaxis.grid(True, color="#e4e3de", linewidth=0.6)
    axes[1].legend(loc="lower left", fontsize=7)
    axes[1].set_title("EarVN (dashed: naive)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "r5b_memory.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_r5b()
    fig_r2()
    fig_matrix()
    fig_cms()
    fig_diag()
    fig_r5()
    print("figures written to", OUT)
