"""Run several python commands in sequence, stopping at the first failure.

Lets one Slurm task line do e.g. "train, then rescore the saved embeddings":
    python scripts/chain.py scripts/train_baseline.py --seed 1 ... ';;' scripts/rescore.py runs/... --by-source
"""
import subprocess
import sys

cmds, cur = [], []
for tok in sys.argv[1:]:
    if tok == ";;":
        cmds.append(cur)
        cur = []
    else:
        cur.append(tok)
cmds.append(cur)
for c in cmds:
    if not c:
        continue
    print(">>> " + " ".join(c), flush=True)
    rc = subprocess.run([sys.executable, *c]).returncode
    if rc != 0:
        sys.exit(rc)
