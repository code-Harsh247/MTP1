#!/bin/bash
# U4 (notes/u4_m3.md): M3 (alpha=1) and its control (alpha=0) at 3 learning rates, on seeds 0-3.
# Seed 0 is validation-only (selection); seeds 1-3 are test runs (metrics only, --no-save). 24 lines.
#   bash scripts/dgx/make_u4_tasks.sh > tasks_u4.txt
B="scripts/train_baseline.py --data configs/uerc.yaml --out runs_dgx"
for s in 0 1 2 3; do
  if [ "$s" = 0 ]; then MODE="--val-only"; else MODE="--no-save"; fi
  for a in 1 0; do
    for lr in 3e-6 1e-5 3e-5; do
      echo "$B --seed $s $MODE --tag m3a${a}_lr${lr} --set epochs=4 optimizer=m3 m3.lr=$lr m3.alpha=$a m3.freq=16"
    done
  done
done
