# Option B: CMS inside a ViT (closest test of the NL paper's own design)

*This file and its scripts were committed **before** any Option-B run on UERC (2026-10-21), including the B1/B2 decision rules. Results are appended below.*

## Why

On ResNet-50, CMS only touched a small neck on top of a CNN without MLP blocks (C3, C3b, U3). The NL paper puts multi-frequency memory **inside every Transformer block** (Hope, Hope-Attention) and recommends **initializing CMS from a pretrained model** ([NL] §7.3). Option B tests that design, so an examiner cannot say "CMS was not tested as intended".

## Design (`nlear/models.py` CMSParallelMLP, `nlear/cms.py` CMSScheduler.for_vit)

- **Backbone:** ImageNet-pretrained **DeiT-S** (`deit_small_patch16_224`, 22M parameters) at 128×96 (48 patches, positional embeddings resampled), with the same embedding neck and ArcFace head as before (`configs/vit.yaml`).
- **CMS-ViT:**
  - Each of the 12 blocks' MLPs becomes **3 parallel copies of the pretrained MLP**, combined by learnable softmax weights initialized uniform ([NL] Eq. 74, independent variant, plus §7.3 pretrained initialization).
  - **Copy ℓ updates every Cℓ steps** on its accumulated gradient: C = (1, 16, 256), slow lr × C^0.5 (as variant D in C3).
  - Everything else (attention, embeddings, norms, aggregation weights, neck, head) updates every step.
- **Why plain ViT is the exact control:**
  - At initialization CMS-ViT computes exactly the pretrained ViT (tested).
  - With all periods = 1, the copies get identical updates and stay identical (tested), so they are a plain ViT.
  - CMS-ViT vs ViT therefore differs **only** in the multi-frequency updates, i.e. the NL mechanism.
- **Precision:** all Option-B arms run in fp32.

## B0: tuning the plain ViT (seed 0, validation only)

- **Grid:** `lr_body` ∈ {3e-5, 1e-4} × epochs ∈ {4, 8}, plain ViT only (4 runs).
- **Selection:** best val DIR@5% (cosine_proto, 2 gallery images); a tie within 0.005 is broken by AUROC (`scripts/b0_select.py`, which writes `configs/vit_selected.yaml`).
- **CMS-ViT gets no extra tuning:** it uses the selected recipe. The CMS settings above are fixed in advance.

## B1: static open-set (seeds 1–3, test)

- **Arms:** ViT vs CMS-ViT, both with the selected recipe.
- **Primary:** test DIR@5% (cosine_proto, 2 gallery images, all sources), paired.
- **GO** if CMS-ViT is higher on **3/3** seeds; otherwise **NO-GO**.
- **Reported:** rank-1, AUROC, EER, the 5-gallery variant, and ViT vs the ResNet-50 U1 baseline.

## B2: continual learning (seeds 1–3, test)

- **Protocol:** UERC P1, 10 tasks, replay k = 5 in both arms. Steps per task = 870 × (E\*/4), where E\* is B0's selected epochs (the same budget rule as U3).
- **Arms:** ViT + replay vs CMS-ViT + replay.
- **Primary:** final seen rank-1, paired.
- **GO** if CMS-ViT + replay is higher on **3/3** seeds; otherwise **NO-GO**.
- **Reported:** forgetting (rank-1 and DIR@5%), seen DIR@5%, and the unseen-people metrics.

## Results

### B0 tuning (2026-10-21; `runs_dgx/b0_selection.txt`)

| Plain ViT, seed-0 VAL | DIR@5% | AUROC | Rank-1 |
|---|---|---|---|
| lr_body 3e-5, 4 epochs | 0.435 | 0.825 | 0.725 |
| **lr_body 3e-5, 8 epochs** | **0.470** | **0.839** | **0.750** |
| lr_body 1e-4, 4 epochs | 0.452 | 0.826 | 0.730 |
| lr_body 1e-4, 8 epochs | 0.385 | 0.798 | 0.660 (overfits) |

**Selected: lr_body 3e-5, 8 epochs**, a clear winner with no tie-break (`configs/vit_selected.yaml`). For B2 this means 870 × 8/4 = **1,740 steps per task**.

On the same seed-0 validation split, the plain ViT is already stronger than the ResNet-50 U1 recipe (val DIR@5% 0.47 vs 0.43). This is validation only; B1 reports test.

### B1 / B2

*(appended after round 2)*
