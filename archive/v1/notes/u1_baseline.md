# U1: UERC static baselines

*Selection rule committed before the tuning runs (2026-10-20). Results are appended below.*

## Recipe

This is the Phase 2 recipe (`configs/baseline.yaml`): ImageNet-pretrained ResNet-50 + ArcFace (s=30, m=0.3), 512-d embedding, dropout 0.3, weight decay 5e-3, strong GPU augmentation, AdamW with warmup and cosine schedule, AMP, 128×96 input.

The data is UERC `public/` (`configs/uerc.yaml`): 917 training identities per seed, about 174k training images. One epoch is about 2,713 steps at batch 64.

## Tuning (seed 0, validation only; `scripts/dgx/make_u1_tune_tasks.sh`)

EarVN's "5 epochs" does not transfer to 8× more data, so only the training length is tuned, plus one larger batch:

| Tag | Epochs | Batch | Approx. steps |
|---|---|---|---|
| e1 | 1 | 64 | 2.7k |
| e2 | 2 | 64 | 5.4k |
| e4 | 4 | 64 | 10.9k |
| e8 | 8 | 64 | 21.7k |
| e4_bs128 | 4 | 128 | 5.4k |

**Selection rule:** pick the configuration with the highest **validation DIR@FAR=5% (cosine_proto, 2 gallery images)** at its final epoch, breaking ties (within 0.005) by validation AUROC. This is the same rule as on EarVN. Test splits are never computed into any output during tuning (`--val-only`).

### Tuning result (2026-10-20; DGX array 34942; `runs_dgx/u1_tune_selection.txt`)

| Config (seed 0, VAL) | DIR@5% | AUROC | rank-1 | EER | g5 DIR@5% |
|---|---|---|---|---|---|
| **e4** (4 epochs, batch 64) | **0.427** | 0.820 | **0.721** | **0.072** | **0.591** |
| e2 | 0.413 | **0.822** | 0.700 | 0.076 | 0.562 |
| e8 | 0.408 | 0.820 | 0.694 | 0.081 | 0.588 |
| e4, batch 128 | 0.384 | 0.799 | 0.709 | 0.076 | 0.556 |
| e1 | 0.267 | 0.750 | 0.569 | 0.112 | 0.371 |

**Selected: e4** (margin 0.014 over e2, so no tie-break). No test metric was computed into any tuning output; the script checks this.

**Observations:**

- **Training longer than about 4 epochs does not help.** The 8-epoch run's validation curve dips and recovers (0.38 → 0.32 → 0.41). This is the same "open-set quality peaks early" pattern as on EarVN, but at about 11k steps instead of 1.5k.
- **On UERC validation, the subspace scorer slightly exceeds cosine_proto** (0.430 vs 0.427), unlike on EarVN, where it was clearly worse. The test runs will show whether this holds.

## Final runs

- The selected configuration runs on **seeds 1–3** (full test protocol), with embeddings saved.
- **Scorers:** cosine_proto, cosine_max, subspace, EVM, EVM + train negatives. The EVM tail sizes are re-chosen on seed-0 validation embeddings if needed.
- **Reporting:** results **overall** and **per source** (`rescore.py --by-source`: VGGFace2 = jpg, the other two sources = png), plus the 5-gallery variant.
- These numbers are the reference for U2 (memory kill test at scale) and U3 (continual).

## Results (2026-10-20; test, seeds 1–3; `runs_dgx/u1_final_tables.txt`)

Each seed has 131 enrolled identities and about 25.5k known + 24.7k unknown probes.

**All sources, 2 gallery images:**

| Scorer | Rank-1 | AUROC | DIR@1% | DIR@5% | DIR@10% | cal DIR@5% (actual FAR) | EER | TAR@1% |
|---|---|---|---|---|---|---|---|---|
| **cosine_proto** | **0.662 ± 0.010** | **0.784 ± 0.004** | **0.226** | **0.370 ± 0.007** | **0.445** | 0.425 (0.085) | **0.069** | **0.752** |
| cosine_max | 0.614 | 0.755 | 0.188 | 0.316 | 0.386 | 0.384 (0.098) | 0.078 | 0.702 |
| subspace | 0.622 | 0.764 | 0.216 | 0.347 | 0.415 | 0.412 (0.097) | 0.096 | 0.704 |
| EVM | 0.535 | 0.725 | 0.193 | 0.311 | 0.368 | 0.287 (**0.037**) | 0.143 | 0.600 |
| EVM + train negatives | 0.148 | 0.550 | 0.115 | 0.128 | 0.133 | — (0.737) | 0.188 | 0.169 |

**All sources, 5 gallery images (cosine_proto):** rank-1 **0.809 ± 0.005**, AUROC **0.860 ± 0.004**, DIR@5% **0.536 ± 0.006**, EER 0.045, TAR@1% 0.876.

**Per source** (each source scored with its own gallery and probes, cosine_proto):

| Source | Enrolled ids | Rank-1 | AUROC | DIR@5% | Rank-1 (g5) | AUROC (g5) | DIR@5% (g5) |
|---|---|---|---|---|---|---|---|
| VGGFace2 (jpg) | 66 | 0.699 ± 0.009 | 0.803 ± 0.008 | 0.398 ± 0.010 | 0.833 | 0.876 | 0.569 |
| UERC 2019 + new (png) | 65 | 0.803 ± 0.013 | 0.834 ± 0.012 | 0.462 ± 0.030 | 0.889 | 0.887 | 0.602 |

## Findings

1. **More training identities help most.** With 917 training identities, unknown rejection is much better than on EarVN with 115: AUROC 0.784 vs 0.716 and DIR@5% 0.370 vs 0.209 (2 gallery images; different datasets, so indicative only). The seed-to-seed std is tiny (≤ 0.010), so these UERC numbers are reliable with 3 seeds.
2. **cosine_proto stays the best scorer** on every metric and every subgroup. The validation hint that subspace might match it did not hold on test (0.347 vs 0.370 DIR@5%).
3. **VGGFace2 ears are the harder part,** even though they dominate training (94% of images): rank-1 0.70 vs 0.80 for the PNG sources at similar gallery size. Per-source numbers are not comparable to the "all sources" row, which has twice the enrolled identities and therefore more confusers.
4. **EVM:**
   - Plain EVM remains the best-calibrated scorer (actual FAR 0.037 for a 0.05 target, vs 0.085 for cosine) at a clear accuracy cost.
   - **EVM with training negatives collapses on VGGFace2** (rank-1 0.25) but works on the PNG sources (0.74). The likely cause is VGGFace2's many near-duplicate frames, which make the 1,000 nearest negatives degenerate margins.
5. **Calibration** is better than on EarVN (actual FAR 0.085 vs about 0.10 at a 0.05 target), but still off. UERC's validation set has 66 unknown identities instead of 8, so the remaining gap is not just validation size.

**The U1 reference for U2 and U3 is cosine_proto:** 2 gallery images, rank-1 0.662, AUROC 0.784, DIR@5% 0.370. Embeddings for seeds 1–3 are on the DGX (`runs_dgx/uerc/baseline_resnet50_e4/seed*/embeddings.pt`).
