# R2: is there catastrophic forgetting? (RQ1)

*Results 2026-11-03, for AMI, AWE, IITD and EarVN (Kaggle, 2 × Tesla T4). UERC is not in yet.*

Design and rule: `benchmark/r2_grid.yaml`, committed before any R2 run. Full tables, all arms and metrics: [r2_tables.md](r2_tables.md), produced by `scripts/r2_report.py`. The verdicts were cross-checked by a separate calculation from the raw accuracy matrices, which gave the same 9 cells.

## Verdict

Forgetting counts as **shown** for a model × dataset if, on every reported seed, naive rank-1 forgetting ≥ 0.10 and final rank-1 ≤ that seed's joint − 0.10.

| Dataset | Shown | Models | Naive rank-1 forgetting (mean, per model) |
|---|---|---|---|
| EarVN | 5 of 5 | all | 0.33–0.48 |
| AWE | 3 of 5 | ConvNeXt-T, DeiT-S, Swin-T | 0.15–0.29 |
| AMI | 1 of 5 | DeiT-S | 0.03–0.16 |
| IITD | 0 of 5 | — | 0.01–0.04 |

**9 of 20 cells.** Every model family forgets heavily on EarVN, the largest dataset here.

## What else the runs show

- **Near misses fail on single seeds:** AWE ResNet-50 (mean 0.150) and MobileNetV3 (0.182), and AMI ResNet-50 (0.100), each have a seed below 0.10. The rule needs every seed, so they stay "not shown".
- **IITD forgets on strict thresholds, not rank-1.** Naive DIR@1% forgetting is 0.18–0.22 for ResNet-50, DeiT-S and Swin-T. The rule uses rank-1, so IITD stays "not shown".
- **Forgetting can be reduced.** On EarVN, replay (5 images per person) roughly halves forgetting in every family. LwF does as well or better without stored images: MobileNetV3 falls from 0.329 to 0.042, ResNet-50 from 0.411 to 0.151. EWC helps less (0.27–0.36).
- **AMI and IITD are near the ceiling.** With joint rank-1 at 0.97–1.00, there is little to forget on rank-1.

## Limits

- **EarVN recipes sit at the grid's corner** (R1). Longer training may change how much is forgotten.
- **AWE is noisy:** one probe moves rank-1 by about 0.014.
- **EWC and LwF strengths** were chosen from two values each on seed 0 (`benchmark/r2_strengths.yaml`).
