#!/bin/bash
# B0 (notes/b_vit_cms.md): plain DeiT-S on UERC seed 0, validation only; lr_body x epochs (4 runs).
#   bash scripts/dgx/make_b0_tasks.sh > tasks_b0.txt
for lr in 3e-5 1e-4; do
  for e in 4 8; do
    echo "scripts/train_baseline.py --data configs/uerc.yaml --train configs/vit.yaml --seed 0 --val-only --out runs_dgx --tag vit_lr${lr}_e${e} --set lr_body=${lr} epochs=${e}"
  done
done
