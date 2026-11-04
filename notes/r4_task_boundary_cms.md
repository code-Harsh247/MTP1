# R4 (exploratory): task-boundary CMS

*Results 2026-11-04 (Kaggle, 2 × Tesla T4).* Design: `benchmark/r4_design.yaml` (commit `4c25f68`), designed after R3's result was seen and committed before any R4 run. Full table: [r4_tables.md](r4_tables.md), produced by `scripts/r4_report.py`. The verdicts were cross-checked by a separate calculation from the raw accuracy matrices.

**Because it was designed after seeing R3, this is a follow-up, not a test of RQ2.**

## What changed from R3

Only the update periods. They now follow each cell's task length S: copy 1 every step, copy 2 every S steps (once per task), copy 3 every 2S steps. So the slow copies update 5 and 2 times on every dataset. In R3 the slowest copy updated only 1–3 times on AMI and AWE.

## Result

**Reduced in 2 of the 9 cells with shown forgetting, fixed in none.** The test is R3's: mean drop ≥ 0.05, better on every seed (all but one with 5), and final rank-1 at most 0.02 lower.

| Cell | Naive forg. | R3 CMS forg. | R4 CMS forg. | Seeds better | Final rank-1 change | Verdict |
|---|---|---|---|---|---|---|
| AMI DeiT-S | 0.161 | 0.129 | 0.100 | 4 of 5 | +0.046 | **REDUCED** |
| AWE ConvNeXt-T | 0.232 | 0.243 | 0.239 | 3 of 5 | −0.063 | not reduced |
| AWE DeiT-S | 0.271 | 0.250 | 0.311 | 1 of 5 | +0.023 | not reduced |
| AWE Swin-T | 0.286 | 0.296 | 0.264 | 3 of 5 | +0.009 | not reduced |
| EarVN ResNet-50 | 0.411 | 0.409 | 0.353 | 3 of 3 | −0.001 | **REDUCED** |
| EarVN ConvNeXt-T | 0.407 | 0.371 | 0.361 | 3 of 3 | +0.018 | not reduced |
| EarVN MobileNetV3 | 0.329 | 0.312 | 0.300 | 3 of 3 | +0.000 | not reduced |
| EarVN DeiT-S | 0.476 | 0.479 | 0.473 | 2 of 3 | +0.004 | not reduced |
| EarVN Swin-T | 0.463 | 0.452 | 0.434 | 3 of 3 | +0.020 | not reduced |

## Reading

- **Task-boundary updates help a little more than R3's fixed periods.** Forgetting is lower than R3's CMS in 8 of 9 cells, and on EarVN it falls below naive on every seed in 4 of 5 families. Drops there are 0.03–0.06 in 4 of 5 families, close to the threshold.
- **AWE gets no benefit,** and ConvNeXt-T loses 0.063 final rank-1 there.
- **The effect is still far smaller than LwF or replay,** which lower EarVN forgetting by 0.17–0.29 (R2).
- **Two REDUCED cells out of 9 is a weak signal,** since a few cells can pass a threshold by chance. It suggests that how CMS consolidates matters, and that tying it to task boundaries is the direction to test properly, with fresh seeds and a pre-registered rule.

## Limits

- Exploratory: no correction for having looked at R3 first.
- Slow-lr values were reused from R3, not re-tuned for the new periods.
- UERC is not included.
