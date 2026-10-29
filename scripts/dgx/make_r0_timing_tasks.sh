#!/bin/bash
# R0 cost check: 1 epoch of the plain baseline for every model x dataset, seed 0, validation only,
# fp32, no saved models. 25 lines, model-major within each dataset, so with --array=0-4 element i
# runs model i on all five datasets (comparable timings on one GPU type).
#   bash scripts/dgx/make_r0_timing_tasks.sh > tasks_r0_timing.txt
#   sbatch --array=0-4 --export=ALL,TASKS=$PWD/tasks_r0_timing.txt scripts/dgx/py_array.sbatch
MODELS="resnet50.tv2_in1k convnext_tiny.fb_in1k mobilenetv3_large_100.ra_in1k deit_small_patch16_224.fb_in1k swin_tiny_patch4_window7_224.ms_in1k"
for data in configs/ami_dgx.yaml configs/awe_dgx.yaml configs/iitd_dgx.yaml configs/earvn_dgx.yaml configs/uerc.yaml; do
  for m in $MODELS; do
    echo "scripts/train_baseline.py --data $data --seed 0 --epochs 1 --val-only --no-save --out runs_dgx/r0_timing --set arch=$m amp=false"
  done
done
