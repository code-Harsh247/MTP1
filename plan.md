# Plan v2: catastrophic forgetting in ear recognition, and whether Nested Learning fixes it

*Started 2026-10-29 on branch `restart-forgetting`, after the professor's review. Study v1 is archived in [archive/v1/](archive/v1/) and tagged `v1-final`.*

## The professor's brief

1. First **prove catastrophic forgetting** in baseline models, using **different types of models** (CNN-based, transformer-based, …).
2. Then **apply Nested Learning** and test whether it fixes the forgetting.
3. Then design a **way to use Nested Learning for continual learning**.
4. **Always use the same datasets and the same splits.** Use **many different ear datasets**.

## Research questions

- **RQ1.** Do standard ear models from different families forget catastrophically when they learn new people one group after another?
- **RQ2.** Does Nested Learning (multi-frequency CMS memory) reduce that forgetting, on exactly the same models, datasets and splits?
- **RQ3.** Can Nested Learning be turned into a practical continual-enrollment method for ears?

## Design: the same experiment, repeated on every dataset

Every dataset is **its own, separate experiment**. For each dataset:

1. Start from a fresh ImageNet-pretrained model.
2. Split that dataset's training people into tasks (groups of new people).
3. Train on the tasks one after another, measuring how much the model forgets the earlier groups.

Models never move from one dataset to another. A result counts as general when it holds across the six datasets and the five model families.

Because datasets never touch, a person who appears in two datasets causes no leak, so no cross-dataset overlap check is needed. Duplicate images *within* a dataset are already removed when its manifest is built.

## Rules for the whole study

- **One frozen benchmark.** R0 fixes the splits, task lists, held-out images, input size, metrics and training budget for every dataset, and commits them before any training. Every later experiment uses them unchanged.
- **Same conditions for every arm:** fp32 precision, input size, augmentation, embedding head (512-d + ArcFace), and the same budget rule.
- **Seeds:**
  - Seed 0 is for tuning (validation only). Seeds 1–3 are reported (test) on every dataset.
  - The small datasets (AMI, AWE, IITD) also report seeds 4–5, because they are cheap to run and their numbers are noisy.
- **Pre-registration:** every decision rule, report script and task list is committed before the runs it judges. Each verdict is cross-checked against the raw metric files.
- **Negative results are reported as negative results.**

Lessons carried over from v1 (details in [archive/v1/](archive/v1/)):

- Replay is a very strong baseline. It removes most forgetting and leaves little for NL to fix.
- Precision alone (fp32 vs AMP) changes forgetting, so all arms must use the same precision.
- Each arm needs a fair recipe.
- A report script once printed a wrong verdict, so verdicts are always cross-checked.

## Datasets

| Dataset | Ids / images (known) | Notes |
|---|---|---|
| AMI | 100 / 700 | 7 images per id, uniform size |
| AWE | 100 / 1,000 | 10 per id, very variable size |
| EarVN1.0 | 164 / 28,412 | many images per id, tiny images |
| IITD-I | 221 / 793 | only 3–6 per id (median 3), grayscale |
| UERC 2023 (public) | 1,310 / ≈250k | 3 sources; VGGFace2-derived ears are 94% of images; DGX only |
| KinEar | *to inspect in R0* | on the DGX, never opened yet |

VGGFace-Ear stays excluded, as in v1.

## Model families

All are ImageNet-pretrained through `timm`, with the same embedding neck and ArcFace head.

| Family | Model | Params |
|---|---|---|
| Classic CNN | ResNet-50 | 25M |
| Modern CNN | ConvNeXt-Tiny | 28M |
| Lightweight CNN | MobileNetV3-Large (or EfficientNet-B0) | 5M |
| Plain transformer | DeiT-S | 22M |
| Hierarchical transformer | Swin-T | 28M |

**Input size.** v1 used 128×96 (ears are about 1.4× taller than wide). Swin needs token grids that its windows divide. R0 picks the smallest size that all five models support and uses it for all of them.

## Protocol (identity-incremental, per dataset)

This is the v1 protocol P1, applied to every dataset.

- **Splits.** Each dataset keeps its identity-disjoint open-set split: train / val / test people, with val and test half known and half unknown.
- **Tasks.** The training people are dealt into T tasks. T is fixed per dataset in R0, for example 10 for UERC and 5 for the smaller sets. On AMI and AWE that gives about 14 people per task.
- **Held-out images.** Some images of every training person are held out, never trained on, to measure forgetting of people the model has learned.
  - v1 held out 20% (2 gallery images, the rest probes).
  - That does not work for IITD (median 3 images per person). R0 sets a per-dataset rule, e.g. 1 held-out probe per person with the gallery taken from that person's training images.
- **Class-incremental head.** At task t, only the people seen so far take part in the ArcFace softmax.

**Metrics.**

- **Accuracy matrix A[t][j]:** rank-1 on task j's held-out images after training up to task t. All the other metrics come from it.
- **Final average accuracy** over all tasks.
- **Forgetting:** the drop from each task's best accuracy to its final accuracy.
- **Backward transfer (BWT).**
- Plots of each task's accuracy over time.
- **Secondary:** open-set AUROC and DIR@FAR at **1%, 5% and 10%** on learned people, and the same metrics on unseen val/test people. All three DIR values come from the same evaluation (`nlear/eval/protocol.py`, `FARS`).
- **Caution on DIR@1% for small datasets.** On AMI and AWE the unknown probes number only about 70–100. A 1% FAR threshold is then set by the top one or two unknown probes, so DIR@1% will be very noisy there. It is reported with its seed spread and never used alone for a decision.

## Phases

| Phase | Work | Output |
|---|---|---|
| **R0. Freeze the benchmark** | Inspect KinEar on the DGX. Prepare IITD and KinEar caches. Build the split set for each of the six datasets (seeds 0–5). Fix the task count and held-out-image rule per dataset. Check that all five models train at the common input size (memory, speed per step). Measure the cost of one run per dataset. Commit everything under `benchmark/`, with split-file hashes. | `notes/r0_benchmark.md`, `benchmark/` |
| **R1. Joint baselines** (upper bound) | Train each model on all of a dataset's tasks at once. Tune each model's recipe on seed-0 validation with the **same small grid** (learning rate × epochs). | One tuned recipe per model and dataset; joint numbers |
| **R2. Prove forgetting** (RQ1) | Naive sequential fine-tuning for every model × dataset. Reference continual-learning methods with the same budget: EWC, LwF, and replay (k = 5 images per person, or fewer where a dataset has fewer). **Pre-registered rule:** forgetting is shown for a (model, dataset) cell if, on every reported seed, rank-1 forgetting ≥ 0.10 **and** final average rank-1 ≤ joint − 0.10. | `notes/r2_forgetting.md`: forgetting curves for every model and dataset |
| **R3. Does NL fix it?** (RQ2) | **NL arm, applied the same way to every family:** each block's pointwise MLP becomes 3 parallel copies of the pretrained weights, updated every 1, 16 and 256 steps (NL Eq. 74 + §7.3). At initialization this equals the plain model, and with all periods = 1 it stays equal (unit-tested). **Primary comparison:** X vs CMS-X, both with *naive* training (no replay). This is NL's own claim, and it is the realistic biometric case, since storing old ear images may not be allowed. **Secondary:** both with replay. **The decision rule is finalized and committed at the end of R2, before any R3 run.** | `notes/r3_nl_forgetting.md` |
| **R4. NL for continual enrollment** (RQ3) | Design depends on R3. Candidates: (a) CMS with consolidation from fast to slow copies; (b) update periods tied to task boundaries; (c) per-person associative memory for enrolling without touching the backbone. Pre-registered before running. | Method + results |
| **R5. Write-up** | Thesis chapters and the final report. | |

**What "pointwise MLP" means in each model:**

| Model | Its pointwise MLP |
|---|---|
| DeiT-S, Swin-T | the FFN |
| ConvNeXt | the inverted-bottleneck MLP |
| ResNet-50 | the bottleneck's 1×1 convolutions |
| MobileNetV3 | the expand and project 1×1 convolutions |

## Code changes needed (before R1)

- `nlear/models.py`: a generic timm backbone for all five families, plus a generic CMS wrapper for pointwise MLPs or 1×1 convs.
- `nlear/data/`: manifests, caches and splits for IITD and KinEar.
- `nlear/continual.py`: the per-dataset task count and held-out-image rule (including the few-images case for IITD).
- Tests: exactness of CMS at initialization for every family, plus the task-list and split invariants.

## Compute and priorities

Rough size: 6 datasets × 5 models × 7 arms (joint, naive, EWC, LwF, replay, CMS, CMS + replay) × 3–5 seeds is about 700 runs, plus tuning. Most of them are on the small datasets and take minutes. UERC dominates the cost. R0 measures the real cost. If it is too much, scope is cut in this order:

1. run EWC and LwF only on UERC and EarVN;
2. run CMS + replay (the secondary NL comparison) only on UERC and EarVN.

Never cut: the five model families, the six datasets, and the joint, naive and CMS arms. These are what the professor asked for.

## Questions for the professor

1. Is running the same experiment separately on each dataset (rather than one model learning dataset after dataset) what you meant by "use many different ear datasets"?
2. Is a clear negative answer to RQ2 ("NL does not reduce forgetting") acceptable as a thesis outcome?
3. Should replay count as allowed? It stores old people's ear images, which may be restricted in a real biometric system.
