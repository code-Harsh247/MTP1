"""U2 step 1 (seed 0): split-backbone training + arm-3 selection on VALIDATION (notes/u2_killtest.md).

Trains the UERC split backbone for seed 0 (half the train identities held out as 'meta'), then
scores the pre-registered grid of untrained deep memories on seed-0 val and writes the chosen
configuration to runs_dgx/uerc/u2_arm3_choice.json for scripts/u2_seed.py.

    python scripts/u2_tune.py
"""
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nlear.eval.protocol import run_protocol  # noqa: E402
from nlear.memory import AssocMemory, make_memory_scorer  # noqa: E402

RUN = ROOT / "runs_dgx/uerc/splitbb_resnet50_e4/seed0"
CHOICE = ROOT / "runs_dgx/uerc/u2_arm3_choice.json"
# Pre-registered grid (same family as Phase 3): MLP memory, 512 hidden, lr 0.5, momentum 0.9.
GRID = [dict(steps=st, forget=fg) for st in (3, 10) for fg in (0.0, 0.1, 0.3)]
HIDDEN, LR, MOM = 512, 0.5, 0.9


def spec(c):
    return f"mem:mlp:{HIDDEN}:{c['steps']}:{LR:g}:{MOM:g}:{c['forget']:g}"


def main():
    if not (RUN / "embeddings.pt").exists():
        cmd = [sys.executable, "scripts/train_baseline.py", "--data", "configs/uerc.yaml", "--seed", "0",
               "--meta-frac", "0.5", "--out", "runs_dgx", "--tag", "e4", "--set", "epochs=4"]
        print(">>> " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
    blob = torch.load(RUN / "embeddings.pt", weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    table = {"cosine_proto": run_protocol(blob, "cosine_proto")["val"], "subspace": run_protocol(blob, "subspace")["val"]}
    for c in GRID:
        mem = AssocMemory(blob["emb"].shape[1], arch="mlp", hidden=HIDDEN, steps=c["steps"], lr=LR,
                          momentum=MOM, forget=c["forget"])
        try:
            table[spec(c)] = run_protocol(blob, make_memory_scorer(mem, device))["val"]
        except ValueError as e:  # non-finite scores: the write diverged
            print(f"{spec(c)}: diverged ({e})")
    for k, v in table.items():
        print(f"VAL seed0 {k:32s} DIR@5% {v['dir@far0.05']:.4f}  AUROC {v['auroc']:.4f}  rank1 {v['rank1']:.4f}")
    mems = {k: v for k, v in table.items() if k.startswith("mem:")}
    best = max(mems, key=lambda k: (round(mems[k]["dir@far0.05"], 3), mems[k]["auroc"]))
    c = next(c for c in GRID if spec(c) == best)
    out = {"spec": best, "hidden": HIDDEN, "steps": c["steps"], "lr": LR, "momentum": MOM,
           "forget": c["forget"], "rule": "max seed-0 val DIR@5% (3 d.p.), tie -> AUROC",
           "val": {k: {m: v[m] for m in ("dir@far0.05", "auroc", "rank1")} for k, v in table.items()}}
    CHOICE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(CHOICE, "w"), indent=1)
    print(f"ARM 3 CHOICE: {best} -> {CHOICE}")


if __name__ == "__main__":
    main()
