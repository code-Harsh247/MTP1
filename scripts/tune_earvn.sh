#!/usr/bin/env bash
# Phase 2 validation sweep on EarVN seed 0 (--val-only: test is never computed).
# Selection metric: val DIR@FAR=5% of cosine_proto (2 gallery images); tie-break val AUROC.
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
run() { tag=$1; shift; echo "=== $tag"; $PY scripts/train_baseline.py --data configs/earvn.yaml --seed 0 --val-only --tag "$tag" "$@" 2>&1 | grep -E '^epoch=|^VAL|^cosine_proto|Error|Traceback'; }

STRONG_AUG='aug={degrees: 20, translate: 0.1, scale: [0.8, 1.2], jitter: 0.4}'

ROUND=${1:-2}
if [ "$ROUND" = 1 ]; then
  # Round 1 (2026-10-18). Finding: val DIR@FAR peaks ~epoch 5, then degrades.
  # (e15_lowlr / e15_reg crashed here: '5e-5' parsed as a string; fixed, rerun in round 2.)
  run e10            --set epochs=10
  run e15_frozen     --set epochs=15 lr_body=0
  run e15_lowlr      --set epochs=15 lr_body=5e-5 lr_head=1e-3
  run e15_reg        --set epochs=15 dropout=0.3 weight_decay=5e-3 "$STRONG_AUG"
  run e15_m05        --set epochs=15 arcface_m=0.5
  run e15            --set arch=resnet50 epochs=15
  run e15_reg        --set arch=resnet50 epochs=15 dropout=0.3 weight_decay=5e-3 "$STRONG_AUG"
else
  # Round 2: short runs, validation after every epoch, pick (config, epoch).
  E='epochs=8 eval_every=1'
  run e8_lowlr       --set $E lr_body=5e-5 lr_head=1e-3
  run e8_reg         --set $E dropout=0.3 weight_decay=5e-3 "$STRONG_AUG"
  run e8             --set arch=resnet50 $E
  run e8_lowlr       --set arch=resnet50 $E lr_body=5e-5 lr_head=1e-3
  run e8_reg         --set arch=resnet50 $E dropout=0.3 weight_decay=5e-3 "$STRONG_AUG"
fi
echo "SWEEP DONE"
