#!/usr/bin/env bash
# Choose training length (optimizer steps) for AMI and AWE on their OWN validation sets.
# Everything else is the EarVN-tuned baseline (configs/baseline.yaml). Their val splits are
# tiny (5 unknown ids), so we average val DIR@FAR=5% over seeds 0-2.
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
for ds in ami awe; do
  for steps in 300 800 1550; do
    for s in 0 1 2; do
      echo "=== $ds steps=$steps seed $s"
      $PY scripts/train_baseline.py --data configs/$ds.yaml --seed $s --val-only --tag steps$steps \
          --set steps=$steps 2>&1 | grep -E '^cosine_proto |Error|Traceback'
    done
  done
done
echo "SWEEP DONE"
