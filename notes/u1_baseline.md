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

## Final runs

- The selected configuration runs on **seeds 1–3** (full test protocol), with embeddings saved.
- **Scorers:** cosine_proto, cosine_max, subspace, EVM, EVM + train negatives. The EVM tail sizes are re-chosen on seed-0 validation embeddings if needed.
- **Reporting:** results **overall** and **per source** (`rescore.py --by-source`: VGGFace2 = jpg, the other two sources = png), plus the 5-gallery variant.
- These numbers are the reference for U2 (memory kill test at scale) and U3 (continual).

## Results

*(appended after the runs)*
