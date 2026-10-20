"""U2 step 2 (one test seed): every kill-test arm on the same split-backbone embeddings.

    python scripts/u2_seed.py --seed 1

1. split-backbone training (seed s; half the train identities held out as 'meta')
2. arms 1, 2, 3 (+ EVM reference) via rescore.py, overall and per source
3. arm 4  : meta-trained memory (W_K, W_V, M_0, write rule)
   arm 4' : meta-trained write rule only
   arm 4b : control, learned projection + cosine (no memory)
Arm 3's configuration comes from runs_dgx/uerc/u2_arm3_choice.json (scripts/u2_tune.py, seed 0).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHOICE = ROOT / "runs_dgx/uerc/u2_arm3_choice.json"


def run(cmd):
    cmd = [sys.executable, *cmd]
    print(">>> " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--n-known", type=int, default=32)
    ap.add_argument("--n-unknown", type=int, default=32)
    args = ap.parse_args()
    run_dir = f"runs_dgx/uerc/splitbb_resnet50_e4/seed{args.seed}"

    if not (ROOT / run_dir / "embeddings.pt").exists():
        rc = run(["scripts/train_baseline.py", "--data", "configs/uerc.yaml", "--seed", str(args.seed),
                  "--meta-frac", "0.5", "--out", "runs_dgx", "--tag", "e4", "--set", "epochs=4"])
        if rc:
            sys.exit(rc)
    c = json.load(open(CHOICE))
    print(f"arm 3 = {c['spec']} (chosen on seed-0 val)")

    failures = 0
    failures += bool(run(["scripts/rescore.py", run_dir, "--scorer", "cosine_proto", "subspace", c["spec"], "evm",
                          "--by-source"]))
    mem = ["--hidden", str(c["hidden"]), "--steps", str(c["steps"]), "--mem-lr", str(c["lr"]),
           "--momentum", str(c["momentum"]), "--forget", str(c["forget"])]
    common = ["--run", run_dir, "--episodes", str(args.episodes), "--eval-every", str(args.eval_every),
              "--n-known", str(args.n_known), "--n-unknown", str(args.n_unknown), "--seed", str(args.seed),
              "--by-source"]
    failures += bool(run(["scripts/meta_train.py", *common, "--model", "memory", "--tag", "full", *mem]))
    failures += bool(run(["scripts/meta_train.py", *common, "--model", "memory", "--tag", "rule", *mem,
                          "--freeze", "proj", "init"]))
    failures += bool(run(["scripts/meta_train.py", *common, "--model", "proto", "--tag", "ctrl", "--lr", "3e-5"]))
    print(f"U2 seed {args.seed} done; {failures} step(s) failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
