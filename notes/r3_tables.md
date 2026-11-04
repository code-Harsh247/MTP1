
## ami (seeds [1, 2, 3, 4, 5])

| Model | R2 | lr_pow | Slow updates | Naive forg. | CMS forg. | Naive final | CMS final | Joint | Primary | Replay forg. | Replay+CMS forg. | Secondary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| classic_cnn | not shown | 0.5 | {'16': 50, '256': 3} | 0.100 | 0.054 | 0.871 | 0.871 | 0.980 | **not reduced** | -0.004 | 0.000 | not reduced |
| modern_cnn | not shown | 0.5 | {'16': 18, '256': 1} | 0.039 | 0.036 | 0.937 | 0.940 | 0.994 | **not reduced** | -0.007 | -0.011 | not reduced |
| lightweight_cnn | not shown | 0 | {'16': 50, '256': 3} | 0.032 | 0.025 | 0.957 | 0.966 | 0.997 | **not reduced** | 0.004 | -0.011 | not reduced |
| plain_transformer | SHOWN | 0 | {'16': 18, '256': 1} | 0.161 | 0.129 | 0.843 | 0.880 | 1.000 | **not reduced** | 0.004 | 0.004 | not reduced |
| hierarchical_transformer | not shown | 0 | {'16': 18, '256': 1} | 0.029 | 0.025 | 0.974 | 0.977 | 0.997 | **not reduced** | -0.004 | 0.004 | not reduced |

## awe (seeds [1, 2, 3, 4, 5])

| Model | R2 | lr_pow | Slow updates | Naive forg. | CMS forg. | Naive final | CMS final | Joint | Primary | Replay forg. | Replay+CMS forg. | Secondary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| classic_cnn | not shown | 0.5 | {'16': 18, '256': 1} | 0.150 | 0.179 | 0.366 | 0.360 | 0.557 | **not reduced** | 0.086 | 0.064 | not reduced |
| modern_cnn | SHOWN | 0.5 | {'16': 50, '256': 3} | 0.232 | 0.243 | 0.529 | 0.503 | 0.694 | **not reduced** | 0.036 | 0.104 | not reduced |
| lightweight_cnn | not shown | 0.5 | {'16': 50, '256': 3} | 0.182 | 0.157 | 0.414 | 0.420 | 0.580 | **not reduced** | 0.050 | 0.079 | not reduced |
| plain_transformer | SHOWN | 0 | {'16': 18, '256': 1} | 0.271 | 0.250 | 0.340 | 0.397 | 0.689 | **not reduced** | 0.114 | 0.064 | not reduced |
| hierarchical_transformer | SHOWN | 0 | {'16': 50, '256': 3} | 0.286 | 0.296 | 0.426 | 0.409 | 0.700 | **not reduced** | 0.057 | 0.046 | not reduced |

## iitd (seeds [1, 2, 3, 4, 5])

| Model | R2 | lr_pow | Slow updates | Naive forg. | CMS forg. | Naive final | CMS final | Joint | Primary | Replay forg. | Replay+CMS forg. | Secondary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| classic_cnn | not shown | 0 | {'16': 50, '256': 3} | 0.010 | 0.011 | 0.979 | 0.973 | 0.978 | **not reduced** | 0.000 | -0.003 | not reduced |
| modern_cnn | not shown | 0.5 | {'16': 18, '256': 1} | 0.008 | 0.003 | 0.970 | 0.972 | 0.979 | **not reduced** | -0.003 | -0.005 | not reduced |
| lightweight_cnn | not shown | 0.5 | {'16': 50, '256': 3} | 0.015 | 0.015 | 0.966 | 0.968 | 0.979 | **not reduced** | -0.003 | -0.002 | not reduced |
| plain_transformer | not shown | 0 | {'16': 50, '256': 3} | 0.027 | 0.026 | 0.970 | 0.970 | 0.990 | **not reduced** | -0.003 | 0.002 | not reduced |
| hierarchical_transformer | not shown | 0 | {'16': 50, '256': 3} | 0.040 | 0.023 | 0.957 | 0.973 | 0.990 | **not reduced** | 0.005 | 0.000 | not reduced |

## earvn (seeds [1, 2, 3])

| Model | R2 | lr_pow | Slow updates | Naive forg. | CMS forg. | Naive final | CMS final | Joint | Primary | Replay forg. | Replay+CMS forg. | Secondary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| classic_cnn | SHOWN | 0.5 | {'16': 156, '256': 9} | 0.411 | 0.409 | 0.553 | 0.546 | 0.932 | **not reduced** | 0.229 | 0.228 | not reduced |
| modern_cnn | SHOWN | 0 | {'16': 156, '256': 9} | 0.407 | 0.371 | 0.597 | 0.612 | 0.946 | **not reduced** | 0.171 | 0.151 | not reduced |
| lightweight_cnn | SHOWN | 0.5 | {'16': 156, '256': 9} | 0.329 | 0.312 | 0.536 | 0.549 | 0.840 | **not reduced** | 0.162 | 0.146 | not reduced |
| plain_transformer | SHOWN | 0 | {'16': 156, '256': 9} | 0.476 | 0.479 | 0.524 | 0.519 | 0.937 | **not reduced** | 0.285 | 0.265 | not reduced |
| hierarchical_transformer | SHOWN | 0 | {'16': 156, '256': 9} | 0.463 | 0.452 | 0.546 | 0.556 | 0.952 | **not reduced** | 0.242 | 0.223 | not reduced |

## uerc (seeds [1, 2, 3])

| Model | R2 | lr_pow | Slow updates | Naive forg. | CMS forg. | Naive final | CMS final | Joint | Primary | Replay forg. | Replay+CMS forg. | Secondary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

Cells with shown forgetting and finished runs: 9; shown but incomplete: 0
Their primary verdicts: {'not reduced': 9}
RQ2: NO
