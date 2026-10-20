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

## Results

*(appended after the runs)*
