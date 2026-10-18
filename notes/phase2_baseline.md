# Phase 2: Baseline results

*Date: 2026-10-19. Code: `scripts/train_baseline.py`, `nlear/eval/`. Runs: `runs/<dataset>/baseline_resnet50/seed0..4`.*

## Setup

- **Model:** ImageNet-pretrained ResNet-50 → 512-d embedding, trained with ArcFace (s=30, m=0.3) on the **train identities only**.
- **Input:** 128×96 images with GPU augmentation (rotation ±20°, scale 0.8–1.2, translation 10%, colour jitter 0.4), dropout 0.3, weight decay 5e-3, AdamW with cosine schedule.
- **Open-set protocol:** test identities are disjoint from training and are half enrolled (known) and half never enrolled (unknown).
  - Gallery: 2 images per known identity, or 5 in the "g5" variant (same people, same model).
  - Probes: every other image of known *and* unknown test identities.
- **Scorers on the embeddings:**
  - `cosine_proto`: cosine to each identity's mean gallery embedding
  - `cosine_max`: cosine to the closest gallery image
  - `subspace`: projection onto the span of the identity's gallery embeddings, which is the linear-memory equivalent (`notes/nl_equations.md` §8.3)
- **Metrics:**
  - rank-1 on known probes
  - AUROC for known vs unknown, using the max score
  - **DIR@FAR**: fraction of known probes accepted *and* correctly identified when a given fraction of unknowns is wrongly accepted
    - "oracle": τ is set on the test unknowns
    - "cal": τ is set on validation unknowns and then applied to test
  - EER and TAR@FAR=1% for 1:1 verification
- **Reporting:** mean ± std over 5 random identity splits (seeds 0–4).

## Tuning (EarVN seed 0, validation only)

Test was never computed during tuning (`--val-only`). Selection metric: val DIR@FAR=5% (cosine_proto, 2 gallery images). Two rounds, 12 configs (`scripts/tune_earvn.sh`, logs in `runs/earvn/tune_sweep*.log`).

| Config (val, seed 0) | DIR@5% | AUROC | Rank-1 |
|---|---|---|---|
| ResNet-18, 30 epochs (first run) | 0.222 | 0.719 | 0.647 |
| ResNet-18, frozen ImageNet body | 0.020 | 0.473 | 0.294 |
| ResNet-18, ArcFace m=0.5 | 0.181 | 0.672 | 0.616 |
| ResNet-18, 10 epochs | 0.255 | 0.715 | 0.691 |
| ResNet-50, 15 epochs | 0.282 | 0.736 | 0.685 |
| **ResNet-50 + reg, epoch 5 (chosen)** | **0.377** | **0.781** | **0.747** |

Findings:

1. **Open-set quality peaks early and then degrades** (about epoch 5 of 8 on EarVN), while training accuracy goes to 100%. The embedding over-specializes to the 115 training identities, and unknown people stop looking unfamiliar. *Relevance to the thesis:* this is exactly the gap a better rejection mechanism has to close.
2. **Fine-tuning is essential.** Frozen ImageNet features are at chance for rejection (AUROC 0.47).
3. **Regularization plus a bigger backbone helps.** A larger ArcFace margin hurts.

**Training length is set in optimizer steps** (`steps:` in the config). The EarVN choice of 5 epochs is ≈1,550 steps; the same "5 epochs" is only 40 steps on AMI. See the AMI/AWE section.

## EarVN1.0 test results (5 seeds)

2 gallery images per known identity; about 2,800 known and 2,850 unknown probes per seed.

| Scorer | Rank-1 | AUROC | DIR@1% | DIR@5% | DIR@10% | cal DIR@5% (actual FAR) | EER | TAR@1% |
|---|---|---|---|---|---|---|---|---|
| cosine_proto | 0.591 ± 0.084 | 0.716 ± 0.044 | 0.091 ± 0.027 | 0.209 ± 0.049 | 0.284 ± 0.062 | 0.280 ± 0.107 (0.103) | 0.159 | 0.343 |
| cosine_max | 0.539 ± 0.074 | 0.691 ± 0.038 | 0.076 ± 0.029 | 0.168 ± 0.048 | 0.234 ± 0.057 | 0.214 ± 0.079 (0.093) | 0.174 | 0.285 |
| subspace | 0.518 ± 0.083 | 0.688 ± 0.039 | 0.090 ± 0.031 | 0.189 ± 0.051 | 0.257 ± 0.063 | 0.241 ± 0.087 (0.095) | 0.208 | 0.306 |

5 gallery images (g5 variant, same models):

| Scorer | Rank-1 | AUROC | DIR@1% | DIR@5% | DIR@10% | cal DIR@5% (actual FAR) | EER | TAR@1% |
|---|---|---|---|---|---|---|---|---|
| cosine_proto | 0.720 ± 0.032 | 0.767 ± 0.033 | 0.173 ± 0.026 | 0.316 ± 0.036 | 0.401 ± 0.036 | 0.402 ± 0.061 (0.109) | 0.125 | 0.462 |
| cosine_max | 0.633 ± 0.030 | 0.725 ± 0.031 | 0.107 ± 0.022 | 0.232 ± 0.033 | 0.307 ± 0.034 | 0.295 ± 0.051 (0.099) | 0.147 | 0.370 |
| subspace | 0.625 ± 0.026 | 0.727 ± 0.029 | 0.143 ± 0.016 | 0.267 ± 0.029 | 0.344 ± 0.016 | 0.330 ± 0.050 (0.094) | 0.174 | 0.404 |

Reading:

- **Unknown rejection is the weak point.** Even with a reasonable rank-1, at FAR=5% only about 21% (2 gallery) to 32% (5 gallery) of known probes are both accepted and correct. There is a lot of room for an NL memory to improve on this.
- **cosine_proto is the strongest baseline.** The subspace scorer (≈ a linear memory) is *worse* than the prototype, so linear-memory surprise alone will not beat it. That matches the expectation in `nl_equations.md` §8.3.
- **Calibration is off by about 2×.** τ set on validation gives an actual test FAR of ~10% when 5% is targeted. Validation has only 8 unknown identities, which is too few to estimate the threshold.
- Seed-0 test numbers are below seed-0 validation (0.377). That is expected after selecting on validation, and the gap shows the split-to-split variance (std 0.05).

## AMI and AWE

**A mistake, caught and fixed.** First I reused EarVN's "5 epochs". On AMI and AWE that is only 40–55 optimizer steps, which badly undertrained them (AWE AUROC 0.50, i.e. chance). Those runs are kept in `runs/<ds>/baseline_resnet50_ep5_undertrained/` for the record and are **not** results.

**Fix.** Training length is now in steps. The step count was chosen on each dataset's **own validation set**, averaged over seeds 0–2 because each split has only 5 unknown validation ids (`scripts/tune_steps_small.sh`). Everything else is the EarVN-tuned config.

| Val, mean of seeds 0–2 (cosine_proto) | AMI DIR@5% | AMI AUROC | AWE DIR@5% | AWE AUROC |
|---|---|---|---|---|
| 300 steps (**chosen for both**) | **0.893 ± 0.050** | 0.947 | **0.275 ± 0.089** | 0.710 |
| 800 steps | 0.853 ± 0.082 | 0.953 | 0.183 ± 0.012 | 0.700 |
| 1,550 steps | 0.787 ± 0.132 | 0.939 | 0.208 ± 0.031 | 0.763 |

For AWE the choice is noisy: AUROC favours 1,550 steps. We kept the rule fixed in advance (DIR@5%).

### Test results, 300 steps, 5 seeds

AMI (10 known / 10 unknown test ids; 50 known and 70 unknown probes):

| Scorer | Rank-1 | AUROC | DIR@1% | DIR@5% | DIR@10% | cal DIR@5% (actual FAR) | EER | TAR@1% |
|---|---|---|---|---|---|---|---|---|
| cosine_proto | 0.948 ± 0.027 | 0.937 ± 0.022 | 0.704 ± 0.050 | 0.760 ± 0.052 | 0.824 ± 0.048 | 0.852 ± 0.065 (0.194) | 0.053 | 0.840 |
| cosine_max | 0.944 ± 0.023 | 0.933 ± 0.029 | 0.656 ± 0.106 | 0.756 ± 0.087 | 0.800 ± 0.054 | 0.848 ± 0.066 (0.209) | 0.056 | 0.816 |
| subspace | 0.948 ± 0.027 | 0.937 ± 0.023 | 0.704 ± 0.089 | 0.780 ± 0.064 | 0.832 ± 0.048 | 0.860 ± 0.062 (0.229) | 0.053 | 0.840 |

For reference, the earlier ResNet-18 run (30 epochs ≈ 240 steps, default regularization) reached 0.968 / 0.946 / 0.820 (rank-1 / AUROC / DIR@5%). That is within about one std, so AMI is near its ceiling either way.

AWE (10 known / 10 unknown test ids; 80 known and 100 unknown probes):

| Scorer | Rank-1 | AUROC | DIR@1% | DIR@5% | DIR@10% | cal DIR@5% (actual FAR) | EER | TAR@1% |
|---|---|---|---|---|---|---|---|---|
| cosine_proto | 0.410 ± 0.065 | 0.626 ± 0.052 | 0.040 ± 0.029 | 0.105 ± 0.048 | 0.150 ± 0.058 | 0.230 ± 0.087 (0.284) | 0.271 | 0.148 |
| cosine_max | 0.390 ± 0.066 | 0.628 ± 0.068 | 0.035 ± 0.025 | 0.068 ± 0.032 | 0.140 ± 0.049 | 0.205 ± 0.038 (0.236) | 0.277 | 0.107 |
| subspace | 0.343 ± 0.049 | 0.601 ± 0.067 | 0.035 ± 0.022 | 0.085 ± 0.043 | 0.158 ± 0.064 | 0.203 ± 0.057 (0.226) | 0.346 | 0.128 |

## Summary across datasets (cosine_proto, 2 gallery images, test, 5 seeds)

| Dataset | Rank-1 | AUROC | DIR@5% | Calibrated FAR (target 0.05) |
|---|---|---|---|---|
| AMI (controlled) | 0.948 ± 0.027 | 0.937 ± 0.022 | 0.760 ± 0.052 | 0.194 |
| EarVN1.0 (in the wild) | 0.591 ± 0.084 | 0.716 ± 0.044 | 0.209 ± 0.049 | 0.103 |
| AWE (in the wild, tiny) | 0.410 ± 0.065 | 0.626 ± 0.052 | 0.105 ± 0.048 | 0.284 |

## Conclusions for Phase 3

1. **The baseline to beat is `cosine_proto`.** It is best or tied on all three datasets. The `subspace` scorer (≈ linear-memory surprise) never clearly beats it and is worse on EarVN and AWE. So the NL memory must do something a linear subspace can't (deep memory, meta-learned write rule), exactly as anticipated in the notes.
2. **Rejection is the weak point on in-the-wild data.** AUROC is 0.63–0.72 even where closed-set rank-1 is reasonable. The headroom is real.
3. **Threshold calibration fails on small validation sets.** The actual FAR is 2–6× the 5% target. Recommendation: report oracle DIR@FAR as the main open-set metric (the literature standard) and calibrated numbers as a secondary "deployment" check. Calibration can only be judged properly on UERC, which has many more validation unknowns.
4. **Phase 3 kill test on EarVN** uses the saved embeddings `runs/earvn/baseline_resnet50/seed*/embeddings.pt` (all train/val/test images, 512-d). Memory experiments need no further backbone training.

## Not done yet

- **EVM** (Extreme Value Machine) as an extra rejection baseline. It is listed in plan.md Phase 2 but not implemented. Add it before the final thesis tables; it isn't needed for the Phase 3 go/no-go decision.
- **Cross-dataset tests** (EarVN → AWE/IITD) are Phase 5.
