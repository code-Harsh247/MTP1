#!/bin/bash
# U3 (notes/u3_continual.md): 6 arms x seeds 1-3 = 18 continual runs on UERC, all fp32.
#   bash scripts/dgx/make_u3_tasks.sh > tasks_u3.txt
B="scripts/train_continual.py --data configs/uerc.yaml --n-tasks 10 --steps-per-task 870 --out runs_dgx"
for s in 1 2 3; do
  echo "$B --seed $s --method joint --fp32"
  echo "$B --seed $s --method naive --fp32"
  echo "$B --seed $s --method replay --buffer-per-id 5 --fp32"
  echo "$B --seed $s --method replay --buffer-per-id 5 --cms-periods 1,1,1"
  echo "$B --seed $s --method replay --buffer-per-id 5 --cms-periods 1,1,1 --cms-body-period 16 --cms-lr-pow 0.5"
  echo "$B --seed $s --method replay --buffer-per-id 5 --cms-periods 1,16,256 --cms-body-period 16 --cms-lr-pow 0.5"
done
