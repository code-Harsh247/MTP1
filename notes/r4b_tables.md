| Dataset | Model | Naive forg. | R3 CMS forg. | R4 CMS forg. | R4b forg. | Naive final | R4b final | Joint | Mixing weights at end | R4b vs naive |
|---|---|---|---|---|---|---|---|---|---|---|
| earvn | classic_cnn | 0.411 | 0.409 | 0.353 | 0.400 | 0.553 | 0.546 | 0.932 | 0.37/0.31/0.32 | **not reduced** |
| earvn | modern_cnn | 0.407 | 0.371 | 0.361 | 0.403 | 0.597 | 0.590 | 0.946 | 0.40/0.29/0.31 | **not reduced** |
| earvn | lightweight_cnn | 0.329 | 0.312 | 0.300 | 0.317 | 0.536 | 0.523 | 0.840 | 0.45/0.27/0.28 | **not reduced** |
| earvn | plain_transformer | 0.476 | 0.479 | 0.473 | 0.501 | 0.524 | 0.497 | 0.937 | 0.35/0.32/0.33 | **not reduced** |
| earvn | hierarchical_transformer | 0.463 | 0.452 | 0.434 | 0.470 | 0.546 | 0.535 | 0.952 | 0.31/0.35/0.34 | **not reduced** |
| awe | modern_cnn | 0.232 | 0.243 | 0.239 | 0.246 | 0.529 | 0.500 | 0.694 | 0.34/0.33/0.33 | **not reduced** |
| awe | plain_transformer | 0.271 | 0.250 | 0.311 | 0.268 | 0.340 | 0.334 | 0.689 | 0.34/0.33/0.33 | **not reduced** |
| awe | hierarchical_transformer | 0.286 | 0.296 | 0.264 | 0.325 | 0.426 | 0.386 | 0.700 | 0.33/0.33/0.34 | **not reduced** |
| ami | plain_transformer | 0.161 | 0.129 | 0.100 | 0.132 | 0.843 | 0.866 | 1.000 | 0.34/0.33/0.33 | **not reduced** |

R4b verdicts (exploratory): {'not reduced': 9}
