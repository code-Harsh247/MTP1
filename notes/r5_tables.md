## Experiment 1: full-size slow updates (EarVN, against naive, R3's test)

| Model | Variant | Naive forg. | R3 CMS forg. | R5 forg. | Seeds better | Naive final | R5 final | Slow updates | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| classic_cnn | pow1 | 0.411 | 0.409 | 0.420 | 1 of 3 | 0.553 | 0.501 | {'16': 156, '256': 9} | **not reduced** |
| classic_cnn | perstep | 0.411 | 0.409 | 0.445 | 0 of 3 | 0.553 | 0.535 | {'16': 156, '256': 9} | **not reduced** |
| modern_cnn | pow1 | 0.407 | 0.371 | 0.233 | 3 of 3 | 0.597 | 0.130 | {'16': 156, '256': 9} | **not reduced** |
| modern_cnn | perstep | 0.407 | 0.371 | 0.435 | 0 of 3 | 0.597 | 0.582 | {'16': 156, '256': 9} | **not reduced** |

## Experiment 3: perfect-memory upper bound (EarVN modern_cnn, copies 2-3 frozen after task 1)

| Seed | Naive task-1 forg. | Snapshot task-1 forg. | Naive forg. | Snapshot forg. | Naive final | Snapshot final | Mixing weights at end |
|---|---|---|---|---|---|---|---|
| 1 | 0.421 | 0.228 | 0.414 | 0.333 | 0.595 | 0.631 | 0.41/0.30/0.30 |
| 2 | 0.496 | 0.301 | 0.402 | 0.299 | 0.601 | 0.642 | 0.41/0.30/0.30 |
| 3 | 0.419 | 0.288 | 0.404 | 0.318 | 0.596 | 0.636 | 0.41/0.30/0.30 |
| mean | 0.445 | 0.272 | 0.407 | 0.317 | 0.597 | 0.636 | |

## Diagnostics (experiments 2 and 5): one run per row, seed 1

| Run | Task 1: after task 1 → end | Part holding most of the loss (share) | Movement from pretrained, copies 1/2/3 | Each copy alone, task 1 rank-1 |
|---|---|---|---|---|
| awe/continual/naive_bodycms1-16-256_p0_hierarchical_transformer_lr0.0003_s160/seed1 | 0.857 → 0.429 | rest of backbone (117%) | 0.078/0.041/0.011 | 0.29/0.43/0.36 |
| awe/continual/naive_hierarchical_transformer_lr0.0003_s160/seed1 | 0.786 → 0.500 | pointwise layers (50%) | plain model |  |
| earvn/continual/naive_bodycms1-16-256_p0_modern_cnn_lr0.0003_s500/seed1 | 0.877 → 0.460 | copy 1 (every 1 steps) (47%) | 0.097/0.041/0.010 | 0.42/0.28/0.39 |
| earvn/continual/naive_bodycms1-16-256_p0_perstep_modern_cnn_lr0.0003_s500/seed1 | 0.891 → 0.490 | copy 2 (every 16 steps) (50%) | 0.091/0.154/0.148 | 0.54/0.36/0.33 |
| earvn/continual/naive_bodycms1-16-256_p0_plain_transformer_lr0.0003_s500/seed1 | 0.853 → 0.341 | rest of backbone (79%) | 0.162/0.065/0.019 | 0.22/0.30/0.39 |
| earvn/continual/naive_bodycms1-16-256_p1_modern_cnn_lr0.0003_s500/seed1 | 0.837 → 0.115 | neck (0%) | 0.096/0.330/2.565 | 0.09/0.09/0.10 |
| earvn/continual/naive_bodycms1-500-1000_p0_modern_cnn_lr0.0003_s500/seed1 | 0.871 → 0.510 | copy 1 (every 1 steps) (69%) | 0.100/0.007/0.004 | 0.39/0.39/0.36 |
| earvn/continual/naive_consol1-2_b0.5_modern_cnn_lr0.0003_s500/seed1 | 0.879 → 0.503 | copy 1 (every step) (40%) | 0.099/0.084/0.036 | 0.37/0.52/0.64 |
| earvn/continual/naive_modern_cnn_lr0.0003_s500/seed1 | 0.885 → 0.493 | pointwise layers (93%) | plain model |  |
| earvn/continual/naive_plain_transformer_lr0.0003_s500/seed1 | 0.874 → 0.368 | rest of backbone (62%) | plain model |  |
| earvn/continual/naive_snapshot1_modern_cnn_lr0.0003_s500/seed1 | 0.801 → 0.577 | copy 1 (every step) (94%) | 0.100/0.054/0.054 | 0.34/0.79/0.79 |
