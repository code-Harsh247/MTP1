#!/bin/bash
# Write the C2 task list for seeds 1-5 (seed 0 was used for tuning on the laptop).
#   bash scripts/dgx/make_c2_tasks.sh <ewc_lambda> <lwf_lambda> > tasks_c2.txt
EWC_LAM=${1:?ewc lambda}; LWF_LAM=${2:?lwf lambda}
for s in 1 2 3 4 5; do
  echo "joint 0 0 $s"
  echo "naive 0 0 $s"
  echo "ewc $EWC_LAM 0 $s"
  echo "lwf $LWF_LAM 0 $s"
  echo "replay 0 2 $s"
  echo "replay 0 5 $s"
done
