#!/bin/bash
# C2 task list for seeds 1-5 (seed 0 was the laptop tuning run, runs/earvn/c2_tune.log).
# Per method, both lambdas are run: the rule's pick (best seed-0 final seen DIR@5%) and the
# stability setting that nearly removes forgetting (2026-10-20 tuning):
#   EWC: 1e2 (rule) and 1e4 (forgetting 0.207 -> 0.015);  LwF: 1 (rule) and 10 (0.166 -> 0.035)
#   bash scripts/dgx/make_c2_tasks.sh > tasks_c2.txt          # 40 lines
for s in 1 2 3 4 5; do
  for t in "joint 0 0" "naive 0 0" "ewc 1e2 0" "ewc 1e4 0" "lwf 1 0" "lwf 10 0" "replay 0 2" "replay 0 5"; do
    echo "$t $s"
  done
done
