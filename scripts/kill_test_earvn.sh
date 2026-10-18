#!/usr/bin/env bash
# Phase 3 kill test on EarVN split-backbone embeddings (runs/earvn/splitbb_resnet50/seed0..4).
# Arms fixed in advance (notes/phase3_killtest.md):
#   1  cosine_proto                       baseline
#   2  subspace (= linear memory)         NL memory, linear (exactly equal, tests/test_memory.py)
#   3  mem:mlp:512:10:0.5:0.9:0.3         NL deep memory, untrained (config chosen on val, arm-3 sweep)
#   4  meta_memory_full                   NL deep memory, meta-trained W_K/W_V, M_0, write rule
#   4' meta_memory_rule                   meta-trained write rule only (lr, momentum, forget)
#   4b meta_proto_ctrl                    control: same episodes, learned projection + cosine
#   +  evm, evm_trainneg                  reference
# Meta-trained arms select their checkpoint on validation (episode 0 = untrained start competes).
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
M=runs/earvn/splitbb_resnet50
for s in 0 1 2 3 4; do
  echo "=== seed $s"
  $PY scripts/rescore.py $M/seed$s --scorer cosine_proto subspace mem:mlp:512:10:0.5:0.9:0.3 evm evm_trainneg 2>&1 | grep -E 'Error|Traceback|diverged' || true
  $PY scripts/meta_train.py --run $M/seed$s --model memory --episodes 1000 --eval-every 100 --tag full 2>&1 | grep -E '^BEST|Error|Traceback'
  $PY scripts/meta_train.py --run $M/seed$s --model memory --freeze proj init --episodes 1000 --eval-every 100 --tag rule 2>&1 | grep -E '^BEST|Error|Traceback'
  $PY scripts/meta_train.py --run $M/seed$s --model proto --lr 3e-5 --episodes 1000 --eval-every 100 --tag ctrl 2>&1 | grep -E '^BEST|Error|Traceback'
done
echo "KILL TEST DONE"
