# R4b (exploratory): CMS with task-boundary consolidation

*Results 2026-11-06 (Kaggle, 2 × Tesla T4).* Design: `benchmark/r4b_design.yaml` (commit `f914ebd`), designed after R3, R4 and the CMS diagnostics were seen and committed before any R4b run. Full table: [r4b_tables.md](r4b_tables.md), produced by `scripts/r4b_report.py`. All 35 runs finished. Every summary was recomputed from the raw accuracy matrices (0 mismatches), and the per-seed comparisons against naive were redone separately.

**Because it was designed after seeing R3 and R4, this is a follow-up, not a test of RQ2.**

## What changed from R4

The diagnostics ([cms_diagnostics.md](cms_diagnostics.md)) showed the slow copies barely moved, because each slow AdamW step is normalised to the size of one ordinary step. In R4b the slow copies take no gradient at all. Copy 1 trains every step. At the end of every task, copy 2 moves half-way toward copy 1. At the end of every second task, copy 3 moves half-way toward copy 2 first (β = 0.5, `nlear.cms.TaskConsolidation`). Every run logged 5 consolidations of copy 2 and 2 of copy 3, as designed.

## Result

**Reduced in 0 of the 9 cells with shown forgetting, fixed in none.** The test is R3's: mean drop ≥ 0.05, better on every seed (all but one with 5), and final rank-1 at most 0.02 lower.

| Cell | Naive forg. | R3 CMS | R4 CMS | R4b | Seeds better | Final rank-1 change | Verdict |
|---|---|---|---|---|---|---|---|
| AMI DeiT-S | 0.161 | 0.129 | 0.100 | 0.132 | 3 of 5 | +0.023 | not reduced |
| AWE ConvNeXt-T | 0.232 | 0.243 | 0.239 | 0.246 | 3 of 5 | −0.029 | not reduced |
| AWE DeiT-S | 0.271 | 0.250 | 0.311 | 0.268 | 3 of 5 | −0.006 | not reduced |
| AWE Swin-T | 0.286 | 0.296 | 0.264 | 0.325 | 2 of 5 | −0.040 | not reduced |
| EarVN ResNet-50 | 0.411 | 0.409 | 0.353 | 0.400 | 2 of 3 | −0.007 | not reduced |
| EarVN ConvNeXt-T | 0.407 | 0.371 | 0.361 | 0.403 | 2 of 3 | −0.007 | not reduced |
| EarVN MobileNetV3 | 0.329 | 0.312 | 0.300 | 0.317 | 2 of 3 | −0.014 | not reduced |
| EarVN DeiT-S | 0.476 | 0.479 | 0.473 | 0.501 | 0 of 3 | −0.027 | not reduced |
| EarVN Swin-T | 0.463 | 0.452 | 0.434 | 0.470 | 1 of 3 | −0.011 | not reduced |

## Reading

- **Consolidation did worse than R4's task-boundary CMS:** R4 had lower forgetting in 8 of 9 cells (the exception is AWE DeiT-S). The largest R4b drop is 0.029 (AMI DeiT-S); on EarVN the drops are at most 0.012, and DeiT-S and Swin-T forget more than naive.
- **The model learned to rely on the fast copy.** At the end, the fast copy's mixing weight is 0.37–0.45 on the EarVN CNNs (0.33 at the start), so the copies that hold old knowledge are partly switched off. Copies that never take a gradient cannot push back on this; the mixing logits do take a gradient, and they follow the current task.
- **Halving the gap each task is still fast forgetting.** With β = 0.5, copy 2 keeps 0.5⁴ ≈ 6% of what it held after task 1 by the end of 5 tasks. Copy 3 is consolidated only twice, but it is one of three summed copies, so its old signal is diluted too. A smaller β or a fixed mixing weight would be the next things to try; neither was run.

## What this means for the study

Four ways of running CMS have now been tried on the 9 cells: R3 fixed periods (0 reduced), R3 replay + CMS (0 of 20), R4 task-boundary periods (2 reduced) and R4b consolidation (0 reduced). In this setting, a CMS layer in place of the backbone's pointwise layers does not reduce forgetting by the pre-registered margin. R4 is the only version that came close, and it is exploratory.
