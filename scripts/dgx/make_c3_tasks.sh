#!/bin/bash
# C3 task list: CMS neck variants, each without and with replay k=5, on seeds 0-5 (72 lines).
# Seed 0 selects the variant (rule in notes/c3_cms.md); seeds 1-5 are the reported test runs.
#   bash scripts/dgx/make_c3_tasks.sh > tasks_c3.txt
VARIANTS=(
  "--cms-periods 1,1,1"                                              # control: same arch, no multi-frequency
  "--cms-periods 1,16,256"                                           # A
  "--cms-periods 1,16,256 --cms-lr-pow 0.5"                          # B: slow levels get a larger lr
  "--cms-periods 1,16,256 --cms-body-period 16"                      # C: backbone is a slow level too
  "--cms-periods 1,16,256 --cms-body-period 16 --cms-lr-pow 0.5"     # D
  "--cms-periods 1,4,16,64 --cms-lr-pow 0.5"                         # E: 4 levels, faster spectrum
)
for s in 0 1 2 3 4 5; do
  for v in "${VARIANTS[@]}"; do
    echo "naive 0 0 $s $v"
    echo "replay 0 5 $s $v"
  done
done
