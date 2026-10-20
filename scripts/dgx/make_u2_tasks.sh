#!/bin/bash
# U2 (notes/u2_killtest.md): one tuning task (seed 0) and three test-seed tasks.
#   bash scripts/dgx/make_u2_tasks.sh   -> writes tasks_u2_tune.txt and tasks_u2_seeds.txt
echo "scripts/u2_tune.py" > tasks_u2_tune.txt
printf 'scripts/u2_seed.py --seed %s\n' 1 2 3 > tasks_u2_seeds.txt
wc -l tasks_u2_tune.txt tasks_u2_seeds.txt
