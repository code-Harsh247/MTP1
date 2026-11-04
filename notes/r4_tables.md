| Dataset | Model | Periods | Slow updates | Naive forg. | R3 CMS forg. | R4 CMS forg. | Naive final | R4 final | Joint | R4 vs naive |
|---|---|---|---|---|---|---|---|---|---|---|
| ami | plain_transformer | [1, 60, 120] | {'60': 5, '120': 2} | 0.161 | 0.129 | 0.100 | 0.843 | 0.889 | 1.000 | **REDUCED** |
| awe | modern_cnn | [1, 160, 320] | {'160': 5, '320': 2} | 0.232 | 0.243 | 0.239 | 0.529 | 0.466 | 0.694 | **not reduced** |
| awe | plain_transformer | [1, 60, 120] | {'60': 5, '120': 2} | 0.271 | 0.250 | 0.311 | 0.340 | 0.363 | 0.689 | **not reduced** |
| awe | hierarchical_transformer | [1, 160, 320] | {'160': 5, '320': 2} | 0.286 | 0.296 | 0.264 | 0.426 | 0.434 | 0.700 | **not reduced** |
| earvn | classic_cnn | [1, 500, 1000] | {'500': 5, '1000': 2} | 0.411 | 0.409 | 0.353 | 0.553 | 0.552 | 0.932 | **REDUCED** |
| earvn | modern_cnn | [1, 500, 1000] | {'500': 5, '1000': 2} | 0.407 | 0.371 | 0.361 | 0.597 | 0.615 | 0.946 | **not reduced** |
| earvn | lightweight_cnn | [1, 500, 1000] | {'500': 5, '1000': 2} | 0.329 | 0.312 | 0.300 | 0.536 | 0.537 | 0.840 | **not reduced** |
| earvn | plain_transformer | [1, 500, 1000] | {'500': 5, '1000': 2} | 0.476 | 0.479 | 0.473 | 0.524 | 0.528 | 0.937 | **not reduced** |
| earvn | hierarchical_transformer | [1, 500, 1000] | {'500': 5, '1000': 2} | 0.463 | 0.452 | 0.434 | 0.546 | 0.566 | 0.952 | **not reduced** |

R4 verdicts (exploratory): {'REDUCED': 2, 'not reduced': 7}
