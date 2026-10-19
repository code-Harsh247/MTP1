#!/bin/bash
# C3b confirmatory task list (notes/c3b_confirm.md): 4 replay arms x fresh seeds 6-10 = 20 lines.
#   bash scripts/dgx/make_c3b_tasks.sh > tasks_c3b.txt
ARMS=(
  ""                                                                 # R: plain replay k=5
  "--cms-periods 1,1,1"                                              # ctrl+R
  "--cms-periods 1,1,1 --cms-body-period 16 --cms-lr-pow 0.5"        # slowbb+R
  "--cms-periods 1,16,256 --cms-body-period 16 --cms-lr-pow 0.5"     # D+R
)
for s in 6 7 8 9 10; do
  for a in "${ARMS[@]}"; do
    echo "replay 0 5 $s${a:+ $a}"
  done
done
