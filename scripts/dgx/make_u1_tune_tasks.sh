#!/bin/bash
# U1 tuning (notes/u1_baseline.md): seed 0, VALIDATION ONLY. Training length (epochs of ~2,713
# steps at batch 64) and one larger batch; everything else is the Phase 2 recipe.
#   bash scripts/dgx/make_u1_tune_tasks.sh > tasks_u1_tune.txt
B="scripts/train_baseline.py --data configs/uerc.yaml --seed 0 --val-only --out runs_dgx"
echo "$B --tag e1 --set epochs=1"
echo "$B --tag e2 --set epochs=2"
echo "$B --tag e4 --set epochs=4"
echo "$B --tag e8 --set epochs=8"
echo "$B --tag e4_bs128 --set epochs=4 batch_size=128"
