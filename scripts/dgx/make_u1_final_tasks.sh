#!/bin/bash
# U1 final runs (notes/u1_baseline.md): selected config e4 (4 epochs, batch 64) on seeds 1-3,
# then all open-set scorers overall and per source (VGGFace2 = jpg vs others = png).
#   bash scripts/dgx/make_u1_final_tasks.sh > tasks_u1_final.txt
for s in 1 2 3; do
  echo "scripts/chain.py scripts/train_baseline.py --data configs/uerc.yaml --seed $s --out runs_dgx --tag e4 --set epochs=4 ;; scripts/rescore.py runs_dgx/uerc/baseline_resnet50_e4/seed$s --scorer cosine_proto cosine_max subspace evm evm_trainneg --by-source"
done
