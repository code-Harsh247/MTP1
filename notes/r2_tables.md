
## ami (seeds [1, 2, 3, 4, 5])

| Model | Arm | Rank-1 | Rank-1 forgetting | DIR@1% forg. | DIR@5% forg. | DIR@10% forg. | AUROC forg. | Runs |
|---|---|---|---|---|---|---|---|---|
| classic_cnn | naive | 0.871 | 0.100 ± 0.053 | 0.146 | 0.136 | 0.143 | 0.046 | 5/5 |
| classic_cnn | replay | 0.991 | -0.004 ± 0.020 | 0.036 | 0.000 | -0.029 | -0.009 | 5/5 |
| classic_cnn | ewc | 0.911 | 0.050 ± 0.039 | 0.086 | 0.093 | 0.075 | 0.026 | 5/5 |
| classic_cnn | lwf | 0.837 | 0.118 ± 0.053 | 0.193 | 0.221 | 0.182 | 0.104 | 5/5 |
| classic_cnn | **not shown** | f 0.125 final 0.871 joint 0.986, f 0.018 final 0.957 joint 0.986, f 0.161 final 0.771 joint 0.986, f 0.107 final 0.871 joint 0.957, f 0.089 final 0.886 joint 0.986 | | | | | | |
| modern_cnn | naive | 0.937 | 0.039 ± 0.023 | 0.068 | 0.086 | 0.086 | 0.019 | 5/5 |
| modern_cnn | replay | 0.986 | -0.007 ± 0.027 | -0.025 | -0.054 | -0.032 | -0.007 | 5/5 |
| modern_cnn | ewc | 0.931 | 0.057 ± 0.034 | 0.107 | 0.089 | 0.082 | 0.029 | 5/5 |
| modern_cnn | lwf | 0.963 | 0.004 ± 0.023 | -0.029 | 0.011 | 0.025 | 0.001 | 5/5 |
| modern_cnn | **not shown** | f 0.036 final 0.943 joint 0.986, f 0.071 final 0.914 joint 1.000, f 0.054 final 0.929 joint 1.000, f 0.018 final 0.957 joint 0.986, f 0.018 final 0.943 joint 1.000 | | | | | | |
| lightweight_cnn | naive | 0.957 | 0.032 ± 0.023 | 0.157 | 0.100 | 0.075 | 0.041 | 5/5 |
| lightweight_cnn | replay | 0.991 | 0.004 ± 0.015 | -0.007 | 0.014 | -0.007 | -0.002 | 5/5 |
| lightweight_cnn | ewc | 0.946 | 0.039 ± 0.039 | 0.143 | 0.086 | 0.075 | 0.026 | 5/5 |
| lightweight_cnn | lwf | 0.966 | 0.021 ± 0.026 | 0.189 | 0.164 | 0.182 | 0.087 | 5/5 |
| lightweight_cnn | **not shown** | f 0.000 final 1.000 joint 1.000, f 0.036 final 0.957 joint 0.986, f 0.018 final 0.986 joint 1.000, f 0.054 final 0.943 joint 1.000, f 0.054 final 0.900 joint 1.000 | | | | | | |
| plain_transformer | naive | 0.843 | 0.161 ± 0.056 | 0.254 | 0.243 | 0.271 | 0.107 | 5/5 |
| plain_transformer | replay | 0.991 | 0.004 ± 0.015 | 0.029 | 0.029 | 0.018 | 0.002 | 5/5 |
| plain_transformer | ewc | 0.854 | 0.143 ± 0.052 | 0.225 | 0.179 | 0.214 | 0.073 | 5/5 |
| plain_transformer | lwf | 0.920 | 0.068 ± 0.008 | 0.246 | 0.200 | 0.136 | 0.069 | 5/5 |
| plain_transformer | **SHOWN** | f 0.125 final 0.871 joint 1.000, f 0.143 final 0.871 joint 1.000, f 0.250 final 0.757 joint 1.000, f 0.179 final 0.843 joint 1.000, f 0.107 final 0.871 joint 1.000 | | | | | | |
| hierarchical_transformer | naive | 0.974 | 0.029 ± 0.024 | 0.082 | 0.054 | 0.064 | 0.016 | 5/5 |
| hierarchical_transformer | replay | 0.994 | -0.004 ± 0.008 | 0.000 | -0.004 | -0.007 | -0.004 | 5/5 |
| hierarchical_transformer | ewc | 0.969 | 0.032 ± 0.015 | -0.007 | 0.014 | 0.043 | 0.005 | 5/5 |
| hierarchical_transformer | lwf | 0.977 | 0.025 ± 0.010 | 0.036 | 0.036 | 0.011 | 0.009 | 5/5 |
| hierarchical_transformer | **not shown** | f 0.054 final 0.957 joint 1.000, f 0.000 final 1.000 joint 1.000, f 0.054 final 0.943 joint 1.000, f 0.018 final 0.986 joint 1.000, f 0.018 final 0.986 joint 0.986 | | | | | | |

## awe (seeds [1, 2, 3, 4, 5])

| Model | Arm | Rank-1 | Rank-1 forgetting | DIR@1% forg. | DIR@5% forg. | DIR@10% forg. | AUROC forg. | Runs |
|---|---|---|---|---|---|---|---|---|
| classic_cnn | naive | 0.366 | 0.150 ± 0.054 | 0.046 | 0.079 | 0.086 | 0.036 | 5/5 |
| classic_cnn | replay | 0.454 | 0.086 ± 0.092 | 0.000 | 0.039 | 0.032 | 0.032 | 5/5 |
| classic_cnn | ewc | 0.340 | 0.150 ± 0.041 | 0.054 | 0.064 | 0.100 | 0.051 | 5/5 |
| classic_cnn | lwf | 0.394 | 0.082 ± 0.072 | 0.025 | 0.071 | 0.089 | 0.061 | 5/5 |
| classic_cnn | **not shown** | f 0.143 final 0.357 joint 0.614, f 0.089 final 0.414 joint 0.514, f 0.214 final 0.329 joint 0.500, f 0.107 final 0.400 joint 0.586, f 0.196 final 0.329 joint 0.571 | | | | | | |
| modern_cnn | naive | 0.529 | 0.232 ± 0.051 | 0.079 | 0.136 | 0.182 | 0.075 | 5/5 |
| modern_cnn | replay | 0.697 | 0.036 ± 0.063 | 0.007 | -0.021 | -0.021 | -0.004 | 5/5 |
| modern_cnn | ewc | 0.500 | 0.218 ± 0.023 | 0.064 | 0.107 | 0.154 | 0.072 | 5/5 |
| modern_cnn | lwf | 0.514 | 0.118 ± 0.065 | 0.032 | 0.046 | 0.089 | 0.032 | 5/5 |
| modern_cnn | **SHOWN** | f 0.286 final 0.600 joint 0.771, f 0.179 final 0.443 joint 0.657, f 0.268 final 0.514 joint 0.657, f 0.250 final 0.543 joint 0.643, f 0.179 final 0.543 joint 0.743 | | | | | | |
| lightweight_cnn | naive | 0.414 | 0.182 ± 0.101 | 0.093 | 0.143 | 0.129 | 0.077 | 5/5 |
| lightweight_cnn | replay | 0.546 | 0.050 ± 0.029 | 0.068 | 0.061 | 0.054 | 0.027 | 5/5 |
| lightweight_cnn | ewc | 0.411 | 0.196 ± 0.087 | 0.121 | 0.150 | 0.161 | 0.061 | 5/5 |
| lightweight_cnn | lwf | 0.469 | 0.132 ± 0.063 | 0.079 | 0.089 | 0.125 | 0.050 | 5/5 |
| lightweight_cnn | **not shown** | f 0.250 final 0.386 joint 0.629, f 0.071 final 0.471 joint 0.586, f 0.196 final 0.414 joint 0.529, f 0.089 final 0.429 joint 0.543, f 0.304 final 0.371 joint 0.614 | | | | | | |
| plain_transformer | naive | 0.340 | 0.271 ± 0.060 | 0.079 | 0.132 | 0.154 | 0.095 | 5/5 |
| plain_transformer | replay | 0.589 | 0.114 ± 0.060 | 0.046 | 0.029 | 0.054 | 0.040 | 5/5 |
| plain_transformer | ewc | 0.354 | 0.243 ± 0.114 | 0.071 | 0.139 | 0.150 | 0.038 | 5/5 |
| plain_transformer | lwf | 0.400 | 0.211 ± 0.051 | 0.089 | 0.111 | 0.100 | 0.045 | 5/5 |
| plain_transformer | **SHOWN** | f 0.250 final 0.457 joint 0.714, f 0.232 final 0.286 joint 0.671, f 0.268 final 0.271 joint 0.714, f 0.232 final 0.371 joint 0.671, f 0.375 final 0.314 joint 0.671 | | | | | | |
| hierarchical_transformer | naive | 0.426 | 0.286 ± 0.087 | 0.096 | 0.121 | 0.175 | 0.091 | 5/5 |
| hierarchical_transformer | replay | 0.657 | 0.057 ± 0.064 | 0.021 | 0.018 | -0.011 | 0.011 | 5/5 |
| hierarchical_transformer | ewc | 0.403 | 0.307 ± 0.079 | 0.107 | 0.171 | 0.186 | 0.082 | 5/5 |
| hierarchical_transformer | lwf | 0.434 | 0.239 ± 0.076 | 0.089 | 0.139 | 0.182 | 0.061 | 5/5 |
| hierarchical_transformer | **SHOWN** | f 0.339 final 0.457 joint 0.714, f 0.321 final 0.243 joint 0.686, f 0.143 final 0.514 joint 0.671, f 0.357 final 0.514 joint 0.757, f 0.268 final 0.400 joint 0.671 | | | | | | |

## iitd (seeds [1, 2, 3, 4, 5])

| Model | Arm | Rank-1 | Rank-1 forgetting | DIR@1% forg. | DIR@5% forg. | DIR@10% forg. | AUROC forg. | Runs |
|---|---|---|---|---|---|---|---|---|
| classic_cnn | naive | 0.979 | 0.010 ± 0.016 | 0.181 | 0.116 | 0.085 | 0.023 | 5/5 |
| classic_cnn | replay | 0.988 | 0.000 ± 0.008 | 0.024 | 0.003 | -0.000 | -0.002 | 5/5 |
| classic_cnn | ewc | 0.972 | 0.019 ± 0.015 | 0.108 | 0.087 | 0.073 | 0.029 | 5/5 |
| classic_cnn | lwf | 0.937 | 0.044 ± 0.012 | 0.189 | 0.123 | 0.084 | 0.039 | 5/5 |
| classic_cnn | **not shown** | f 0.016 final 0.968 joint 0.955, f 0.000 final 0.987 joint 0.974, f 0.008 final 0.981 joint 0.987, f -0.008 final 1.000 joint 0.994, f 0.032 final 0.961 joint 0.981 | | | | | | |
| modern_cnn | naive | 0.970 | 0.008 ± 0.013 | 0.050 | 0.035 | 0.024 | 0.008 | 5/5 |
| modern_cnn | replay | 0.981 | -0.003 ± 0.012 | -0.006 | -0.010 | -0.010 | -0.002 | 5/5 |
| modern_cnn | ewc | 0.970 | 0.005 ± 0.011 | 0.035 | 0.015 | 0.010 | 0.007 | 5/5 |
| modern_cnn | lwf | 0.974 | 0.002 ± 0.009 | 0.024 | 0.034 | 0.031 | 0.007 | 5/5 |
| modern_cnn | **not shown** | f -0.000 final 0.961 joint 0.974, f 0.024 final 0.968 joint 0.981, f 0.008 final 0.968 joint 0.987, f -0.008 final 0.987 joint 0.987, f 0.016 final 0.968 joint 0.968 | | | | | | |
| lightweight_cnn | naive | 0.966 | 0.015 ± 0.009 | 0.065 | 0.035 | 0.040 | 0.009 | 5/5 |
| lightweight_cnn | replay | 0.985 | -0.003 ± 0.009 | -0.018 | -0.016 | -0.008 | -0.005 | 5/5 |
| lightweight_cnn | ewc | 0.972 | 0.008 ± 0.015 | 0.068 | 0.026 | 0.011 | 0.010 | 5/5 |
| lightweight_cnn | lwf | 0.966 | 0.016 ± 0.014 | 0.081 | 0.035 | 0.040 | 0.008 | 5/5 |
| lightweight_cnn | **not shown** | f 0.008 final 0.961 joint 0.968, f 0.008 final 0.955 joint 0.968, f 0.008 final 0.987 joint 0.981, f 0.024 final 0.968 joint 0.987, f 0.024 final 0.961 joint 0.994 | | | | | | |
| plain_transformer | naive | 0.970 | 0.027 ± 0.019 | 0.179 | 0.126 | 0.081 | 0.029 | 5/5 |
| plain_transformer | replay | 0.994 | -0.003 ± 0.009 | -0.010 | -0.011 | -0.005 | -0.002 | 5/5 |
| plain_transformer | ewc | 0.969 | 0.024 ± 0.015 | 0.166 | 0.124 | 0.084 | 0.029 | 5/5 |
| plain_transformer | lwf | 0.970 | 0.021 ± 0.012 | 0.181 | 0.144 | 0.098 | 0.037 | 5/5 |
| plain_transformer | **not shown** | f 0.016 final 0.981 joint 0.987, f 0.040 final 0.955 joint 0.994, f 0.048 final 0.955 joint 0.987, f 0.000 final 0.994 joint 0.994, f 0.032 final 0.968 joint 0.987 | | | | | | |
| hierarchical_transformer | naive | 0.957 | 0.040 ± 0.024 | 0.223 | 0.168 | 0.135 | 0.041 | 5/5 |
| hierarchical_transformer | replay | 0.994 | 0.005 ± 0.007 | -0.005 | 0.002 | 0.008 | 0.002 | 5/5 |
| hierarchical_transformer | ewc | 0.974 | 0.018 ± 0.009 | 0.252 | 0.156 | 0.111 | 0.037 | 5/5 |
| hierarchical_transformer | lwf | 0.963 | 0.027 ± 0.020 | 0.227 | 0.190 | 0.132 | 0.053 | 5/5 |
| hierarchical_transformer | **not shown** | f 0.040 final 0.961 joint 0.981, f 0.048 final 0.948 joint 0.987, f 0.065 final 0.929 joint 0.987, f 0.000 final 0.987 joint 0.994, f 0.048 final 0.961 joint 1.000 | | | | | | |

## earvn (seeds [1, 2, 3])

| Model | Arm | Rank-1 | Rank-1 forgetting | DIR@1% forg. | DIR@5% forg. | DIR@10% forg. | AUROC forg. | Runs |
|---|---|---|---|---|---|---|---|---|
| classic_cnn | naive | 0.553 | 0.411 ± 0.013 | 0.257 | 0.370 | 0.409 | 0.185 | 3/3 |
| classic_cnn | replay | 0.679 | 0.229 ± 0.017 | 0.149 | 0.236 | 0.269 | 0.137 | 3/3 |
| classic_cnn | ewc | 0.657 | 0.272 ± 0.012 | 0.183 | 0.265 | 0.288 | 0.131 | 3/3 |
| classic_cnn | lwf | 0.609 | 0.151 ± 0.018 | 0.079 | 0.134 | 0.152 | 0.053 | 3/3 |
| classic_cnn | **SHOWN** | f 0.426 final 0.552 joint 0.936, f 0.403 final 0.562 joint 0.930, f 0.403 final 0.546 joint 0.929 | | | | | | |
| modern_cnn | naive | 0.597 | 0.407 ± 0.007 | 0.310 | 0.429 | 0.460 | 0.197 | 3/3 |
| modern_cnn | replay | 0.774 | 0.171 ± 0.012 | 0.166 | 0.216 | 0.220 | 0.107 | 3/3 |
| modern_cnn | ewc | 0.646 | 0.346 ± 0.008 | 0.297 | 0.405 | 0.427 | 0.181 | 3/3 |
| modern_cnn | lwf | 0.723 | 0.235 ± 0.014 | 0.214 | 0.315 | 0.329 | 0.139 | 3/3 |
| modern_cnn | **SHOWN** | f 0.414 final 0.595 joint 0.951, f 0.402 final 0.601 joint 0.951, f 0.404 final 0.596 joint 0.938 | | | | | | |
| lightweight_cnn | naive | 0.536 | 0.329 ± 0.024 | 0.141 | 0.246 | 0.295 | 0.140 | 3/3 |
| lightweight_cnn | replay | 0.625 | 0.162 ± 0.030 | 0.074 | 0.134 | 0.164 | 0.091 | 3/3 |
| lightweight_cnn | ewc | 0.545 | 0.320 ± 0.023 | 0.139 | 0.234 | 0.276 | 0.119 | 3/3 |
| lightweight_cnn | lwf | 0.626 | 0.042 ± 0.012 | 0.023 | 0.043 | 0.052 | 0.019 | 3/3 |
| lightweight_cnn | **SHOWN** | f 0.355 final 0.534 joint 0.861, f 0.307 final 0.548 joint 0.837, f 0.324 final 0.527 joint 0.822 | | | | | | |
| plain_transformer | naive | 0.524 | 0.476 ± 0.015 | 0.277 | 0.431 | 0.488 | 0.230 | 3/3 |
| plain_transformer | replay | 0.649 | 0.285 ± 0.012 | 0.227 | 0.317 | 0.347 | 0.182 | 3/3 |
| plain_transformer | ewc | 0.614 | 0.362 ± 0.014 | 0.289 | 0.408 | 0.429 | 0.199 | 3/3 |
| plain_transformer | lwf | 0.629 | 0.187 ± 0.009 | 0.104 | 0.169 | 0.206 | 0.083 | 3/3 |
| plain_transformer | **SHOWN** | f 0.459 final 0.540 joint 0.942, f 0.483 final 0.521 joint 0.939, f 0.486 final 0.511 joint 0.930 | | | | | | |
| hierarchical_transformer | naive | 0.546 | 0.463 ± 0.015 | 0.275 | 0.431 | 0.476 | 0.211 | 3/3 |
| hierarchical_transformer | replay | 0.703 | 0.242 ± 0.008 | 0.170 | 0.286 | 0.317 | 0.154 | 3/3 |
| hierarchical_transformer | ewc | 0.649 | 0.338 ± 0.016 | 0.259 | 0.389 | 0.412 | 0.177 | 3/3 |
| hierarchical_transformer | lwf | 0.638 | 0.172 ± 0.019 | 0.090 | 0.154 | 0.182 | 0.071 | 3/3 |
| hierarchical_transformer | **SHOWN** | f 0.468 final 0.552 joint 0.952, f 0.446 final 0.558 joint 0.957, f 0.476 final 0.528 joint 0.947 | | | | | | |

## uerc (seeds [1, 2, 3])

| Model | Arm | Rank-1 | Rank-1 forgetting | DIR@1% forg. | DIR@5% forg. | DIR@10% forg. | AUROC forg. | Runs |
|---|---|---|---|---|---|---|---|---|

Cells: {'SHOWN': 9, 'not shown': 11, 'INCOMPLETE': 0}
