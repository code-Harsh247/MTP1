## Experiment 1: the snapshot upper bound on every shown cell (mixing learned)

| Dataset | Model | Seeds | Naive task-1 forg. | Snapshot task-1 forg. | Seeds better (task 1) | Naive forg. | Snapshot forg. | Naive final | Snapshot final | R3 test (overall) |
|---|---|---|---|---|---|---|---|---|---|---|
| earvn | modern_cnn | 3 of 3 | 0.445 | 0.272 | 3 of 3 | 0.407 | 0.317 | 0.597 | 0.636 | **REDUCED** |
| earvn | classic_cnn | 3 of 3 | 0.432 | 0.268 | 3 of 3 | 0.411 | 0.338 | 0.553 | 0.578 | **REDUCED** |
| earvn | lightweight_cnn | 3 of 3 | 0.287 | 0.264 | 2 of 3 | 0.329 | 0.306 | 0.536 | 0.525 | **not reduced** |
| earvn | plain_transformer | 3 of 3 | 0.525 | 0.438 | 3 of 3 | 0.476 | 0.444 | 0.524 | 0.538 | **not reduced** |
| earvn | hierarchical_transformer | 3 of 3 | 0.502 | 0.436 | 3 of 3 | 0.463 | 0.430 | 0.546 | 0.563 | **not reduced** |
| awe | modern_cnn | 5 of 5 | 0.329 | 0.243 | 4 of 5 | 0.232 | 0.221 | 0.529 | 0.520 | **not reduced** |
| awe | plain_transformer | 5 of 5 | 0.386 | 0.314 | 3 of 5 | 0.271 | 0.243 | 0.340 | 0.369 | **not reduced** |
| awe | hierarchical_transformer | 5 of 5 | 0.443 | 0.271 | 4 of 5 | 0.286 | 0.246 | 0.426 | 0.417 | **not reduced** |
| ami | plain_transformer | 5 of 5 | 0.157 | 0.086 | 4 of 5 | 0.161 | 0.079 | 0.843 | 0.903 | **FIXED** |

## Experiment 2: frozen copies' share of the output fixed after task 1

| Dataset | Model | Frozen share | Seeds | Task-1 forg. | Forgetting | Last-task rank-1 | Final rank-1 | Frozen share at end |
|---|---|---|---|---|---|---|---|---|
| earvn | modern_cnn | naive | 3 | 0.445 | 0.407 | 0.945 | 0.597 | |
| earvn | modern_cnn | learned | 3 | 0.272 | 0.317 | 0.929 | 0.636 | 0.595 |
| earvn | modern_cnn | 0.8 | 3 | 0.228 | 0.289 | 0.913 | 0.639 | 0.800 |
| earvn | modern_cnn | 0.95 | 3 | 0.171 | 0.258 | 0.872 | 0.632 | 0.950 |
| earvn | plain_transformer | naive | 3 | 0.525 | 0.476 | 0.928 | 0.524 | |
| earvn | plain_transformer | learned | 3 | 0.438 | 0.444 | 0.928 | 0.538 | 0.644 |
| earvn | plain_transformer | 0.8 | 3 | 0.445 | 0.443 | 0.919 | 0.532 | 0.800 |
| earvn | plain_transformer | 0.95 | 3 | 0.434 | 0.445 | 0.905 | 0.520 | 0.950 |

## Diagnostics: is the kept memory diluted? (snapshot runs and the second seed)

| Run | Task 1: after task 1 → end (whole model) | Frozen copy alone, task 1 | Gap | Part holding most of the loss (share) |
|---|---|---|---|---|
| awe/continual/naive_snapshot1_hierarchical_transformer_lr0.0003_s160/seed1 | 0.714 → 0.429 | 0.429 | +0.000 | rest of backbone (125%) |
| earvn/continual/naive_bodycms1-16-256_p0_modern_cnn_lr0.0003_s500/seed2 | 0.878 → 0.389 |  |  | copy 1 (every 1 steps) (47%) |
| earvn/continual/naive_modern_cnn_lr0.0003_s500/seed2 | 0.900 → 0.378 |  |  | pointwise layers (86%) |
| earvn/continual/naive_snapshot1_classic_cnn_lr0.0003_s500/seed1 | 0.719 → 0.441 | 0.573 | +0.132 | rest of backbone (64%) |
| earvn/continual/naive_snapshot1_mix95_modern_cnn_lr0.0003_s500/seed2 | 0.802 → 0.576 | 0.694 | +0.118 | neck (93%) |
| earvn/continual/naive_snapshot1_modern_cnn_lr0.0003_s500/seed1 | 0.801 → 0.577 | 0.787 | +0.210 | copy 1 (every step) (94%) |
| earvn/continual/naive_snapshot1_modern_cnn_lr0.0003_s500/seed2 | 0.802 → 0.500 | 0.825 | +0.325 | copy 1 (every step) (108%) |
| earvn/continual/naive_snapshot1_plain_transformer_lr0.0003_s500/seed1 | 0.806 → 0.365 | 0.469 | +0.104 | rest of backbone (104%) |
