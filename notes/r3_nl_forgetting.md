# R3: does Nested Learning reduce forgetting? (RQ2)

*Results 2026-11-03, for AMI, AWE, IITD and EarVN (Kaggle, 2 × Tesla T4). UERC is not in yet.*

Rule: `benchmark/r3_rule.yaml` (commit `ec56ffc`). That commit came a few minutes after the Kaggle session started; the session ran the same files, and no R3 result had been seen. Full table: [r3_tables.md](r3_tables.md), produced by `scripts/r3_report.py`. The verdicts were cross-checked by a separate calculation from the raw accuracy matrices, and the slow-lr choices made on Kaggle were reproduced locally.

## Answer

**RQ2: NO.** In none of the 9 cells where R2 showed forgetting did naive + CMS reduce it by the pre-registered amount.

| Cell (R2: shown) | Forgetting drop, naive − CMS (mean) | Seeds where CMS forgot less | Final rank-1 change |
|---|---|---|---|
| AMI DeiT-S | +0.032 | 3 of 5 | +0.037 |
| AWE ConvNeXt-T | −0.011 | 2 of 5 | −0.026 |
| AWE DeiT-S | +0.021 | 2 of 5 | +0.057 |
| AWE Swin-T | −0.011 | 2 of 5 | −0.017 |
| EarVN ResNet-50 | +0.002 | 2 of 3 | −0.007 |
| EarVN ConvNeXt-T | +0.036 | 3 of 3 | +0.014 |
| EarVN MobileNetV3 | +0.017 | 3 of 3 | +0.013 |
| EarVN DeiT-S | −0.003 | 1 of 3 | −0.005 |
| EarVN Swin-T | +0.012 | 2 of 3 | +0.010 |

REDUCED needs a mean drop ≥ 0.05, CMS better on every seed (all but one with 5 seeds), and final rank-1 no more than 0.02 lower. The 11 cells without shown forgetting are also "not reduced".

## Reading

- **CMS neither helps nor hurts much.** Drops are within ±0.04, and final rank-1 moves by less than 0.06 either way.
- **The small effects lean positive on EarVN:** ConvNeXt-T and MobileNetV3 forget less on all 3 seeds, by 0.036 and 0.017. That is consistent but well short of the 0.05 threshold.
- **Other methods do far better on the same cells.** On EarVN, LwF lowers forgetting by 0.17–0.29 and replay by 0.17–0.24 (R2).

## Limits

- **Slow copies update rarely.** A period-256 copy updates 1–3 times on AMI, AWE and IITD, and 9 times on EarVN. The NL arm's slow memory therefore barely acts on the small datasets. The periods were fixed in advance for every cell.
- **One NL arm only:** CMS on each block's pointwise MLP, with periods 1, 16, 256 and two seed-0 slow-lr values. Other CMS placements, periods or consolidation could behave differently (R4).
- **Secondary comparison (replay + CMS against replay), run 2026-11-04: not reduced in any of the 20 cells.** Among the 9 shown cells, the largest gain is AWE DeiT-S (forgetting 0.114 → 0.064, 4 of 5 seeds), just under the 0.05 mean drop. AWE ConvNeXt-T gets worse (0.036 → 0.104). On EarVN, CMS lowers replay's forgetting by about 0.02 in 4 of 5 families. It does not change the RQ2 answer.
- **UERC is not included.** If it finishes, its cells are added by re-running the report.
