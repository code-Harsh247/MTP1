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

### B1 / B2 (2026-10-23; DGX job 35048, 12 runs, 0 failures; `runs_dgx/b12_report.txt`)

The report script, task list and model/scheduler code were unchanged since the pre-registration commit `1797822`. An **independent cross-check** computed straight from the raw `metrics.json` files agrees with every decision number.

**B1, static open-set (test, seeds 1–3, cosine_proto):**

| Model | Rank-1 | AUROC | **DIR@5%** | EER | Rank-1 (g5) | DIR@5% (g5) |
|---|---|---|---|---|---|---|
| ResNet-50 (U1, AMP, 4 epochs) | 0.662 ± 0.010 | 0.784 | 0.370 ± 0.007 | 0.069 | 0.809 | 0.536 |
| **ViT / DeiT-S** (fp32, 8 epochs) | **0.721 ± 0.008** | **0.808** | **0.433 ± 0.014** | **0.064** | **0.843** | **0.605** |
| CMS-ViT (fp32, 8 epochs) | 0.687 ± 0.011 | 0.798 | 0.408 ± 0.026 | 0.069 | 0.821 | 0.578 |

**Decision B1:** CMS-ViT vs ViT, DIR@5% better on **0/3** seeds (−0.021, −0.012, −0.042; mean −0.025). **VERDICT B1: NO-GO.** CMS makes the static open-set model worse on every metric.

**B2, continual learning (UERC P1, 10 tasks, replay k=5, test, seeds 1–3):**

| Arm | **Seen rank-1** | Forget rank-1 ↓ | Seen DIR@5% | Forget DIR@5% ↓ | Unseen AUROC | Unseen DIR@5% |
|---|---|---|---|---|---|---|
| ViT + R | 0.732 ± 0.008 | **0.107** | 0.427 | **0.168** | **0.716** | **0.240** |
| CMS-ViT + R | **0.743 ± 0.005** | 0.120 | **0.438** | 0.194 | 0.698 | 0.209 |

**Decision B2:** final seen rank-1 better on **3/3** seeds (+0.008, +0.009, +0.018; mean **+0.012**). **VERDICT B2: GO** on the pre-registered metric.

### Reading B2 honestly

1. **The gain is small**: +0.012 rank-1, about 1.5 seed-stds. With 3 seeds, "3 of 3" would happen by chance 1 time in 8 (sign test). It passed the rule we set, but it is not strong evidence on its own.
2. **It does not work the way NL claims.** NL proposes CMS to **reduce forgetting**. Here forgetting is **worse** in 3/3 seeds (rank-1 forgetting 0.120 vs 0.107; DIR@5% 0.194 vs 0.168). The gain comes from **learning each new group better** (`runs_dgx/b2_mechanism.txt`):

   | | Rank-1 right after learning a task | Final, old tasks 1–9 |
   |---|---|---|
   | ViT + R | 0.828 | 0.715 |
   | CMS-ViT + R | 0.851 (+0.023, every seed) | 0.725 |

   The multi-frequency copies act as **extra plasticity / effective capacity**, not as a better long-term memory.
3. **It costs unseen-people performance:** unseen AUROC is lower on 3/3 seeds (−0.018), as is unseen DIR@5% (−0.031). For a biometric system that must also handle new people, that is a real downside.

### Conclusions from Option B

- **CMS inside a pretrained ViT, the paper's own recommended design, does not improve open-set ear recognition** (B1: worse).
- **In continual learning it gives a small, consistent gain in final accuracy on enrolled people** (B2: +0.012 rank-1, 3/3). But it gets there through more plasticity rather than less forgetting, and it makes unseen-people rejection worse. This is the **only pre-registered positive NL result in the project**, and it should be reported with exactly these caveats.
- **A useful side result: the plain DeiT-S ViT is the strongest ear model in the project** (UERC test rank-1 0.721, AUROC 0.808, DIR@5% 0.433, vs 0.662 / 0.784 / 0.370 for ResNet-50). This comparison is not perfectly controlled: the ViT ran in fp32 for 8 epochs, the ResNet in AMP for 4, each with its own tuned recipe.
