#!/bin/bash
# B1 (static) + B2 (continual) on seeds 1-3 with the B0-selected recipe (configs/vit_selected.yaml).
#   bash scripts/dgx/make_b12_tasks.sh > tasks_b12.txt      (12 lines)
PY=/home/dsamantaai/miniconda3/envs/harsh_nlear/bin/python
[ -x "$PY" ] || PY=python
E=$("$PY" -c "import yaml; print(yaml.safe_load(open('configs/vit_selected.yaml'))['train']['epochs'])")
SPT=$(( 870 * E / 4 ))
CMS_STATIC="vit_cms.periods=[1,16,256] vit_cms.lr_pow=0.5"
for s in 1 2 3; do
  echo "scripts/train_baseline.py --data configs/uerc.yaml --train configs/vit_selected.yaml --seed $s --no-save --out runs_dgx --tag vit"
  echo "scripts/train_baseline.py --data configs/uerc.yaml --train configs/vit_selected.yaml --seed $s --no-save --out runs_dgx --tag vitcms --set $CMS_STATIC"
  echo "scripts/train_continual.py --data configs/uerc.yaml --train configs/vit_selected.yaml --n-tasks 10 --steps-per-task $SPT --out runs_dgx --seed $s --method replay --buffer-per-id 5 --fp32 --tag vit"
  echo "scripts/train_continual.py --data configs/uerc.yaml --train configs/vit_selected.yaml --n-tasks 10 --steps-per-task $SPT --out runs_dgx --seed $s --method replay --buffer-per-id 5 --tag vit --vit-cms-periods 1,16,256 --vit-cms-lr-pow 0.5"
done
