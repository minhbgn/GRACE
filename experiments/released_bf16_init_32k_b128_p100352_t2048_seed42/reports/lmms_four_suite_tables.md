## scores

| Model | SQA-IMG | AI2D | SEED-IMG | MMStar | Mean-4 |
| --- | --- | --- | --- | --- | --- |
| vanilla_2b | 87.26 | 76.13 | 76.09 | 55.08 | 73.64 |
| vanilla_8b_teacher | 94.70 | 84.00 | 78.56 | 64.55 | 80.45 |
| sft_2b | 86.76 | 77.53 | 76.68 | 56.10 | 74.27 |
| released_bf16 | 88.30 | 81.80 | 76.27 | 69.39 | 78.94 |
| released_grace_w4 | 87.11 | 80.41 | 75.70 | 67.49 | 77.68 |
| released_grace_w8 | 86.66 | 79.57 | 75.45 | 65.20 | 76.72 |
| qat_w4 | 86.12 | 80.63 | 76.39 | 68.29 | 77.86 |
| grace_w4 | 86.71 | 79.57 | 77.09 | 63.98 | 76.84 |
| qat_w8 | 87.70 | 82.32 | 76.73 | 70.66 | 79.35 |
| grace_w8 | 87.95 | 81.02 | 77.14 | 64.99 | 77.78 |

## paper

| Model | SQA paper → ours | AI2D paper → ours | SEED paper → ours | MMStar paper → ours | Paper Mean-4 | Our Mean-4 | Δ Mean-4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vanilla_8b_teacher | 85.0 → 94.70 | 85.7 → 84.00 | 77.5 → 78.56 | 70.9 → 64.55 | 79.775 | 80.45 | +0.68 |
| vanilla_2b | 81.4 → 87.26 | 76.9 → 76.13 | 71.2 → 76.09 | 58.3 → 55.08 | 71.950 | 73.64 | +1.69 |
| released_bf16 | 86.2 → 88.30 | 81.3 → 81.80 | 76.7 → 76.27 | 67.3 → 69.39 | 77.875 | 78.94 | +1.07 |
| released_grace_w8 | 85.3 → 86.66 | 80.4 → 79.57 | 75.9 → 75.45 | 66.5 → 65.20 | 77.025 | 76.72 | -0.31 |
| released_grace_w4 | 84.3 → 87.11 | 79.5 → 80.41 | 75.1 → 75.70 | 65.8 → 67.49 | 76.175 | 77.68 | +1.50 |

## old

| Model | Old SQA-IMG | New SQA-IMG | Old Mean-4 | New Mean-4 | Δ Mean-4 |
| --- | --- | --- | --- | --- | --- |
| vanilla_2b | 85.62 | 87.26 | 69.39 | 73.64 | +4.25 |
| vanilla_8b_teacher | 93.36 | 94.70 | 77.68 | 80.45 | +2.77 |
| sft_2b | 84.78 | 86.76 | 71.31 | 74.27 | +2.96 |
| released_bf16 | 86.07 | 88.30 | 76.10 | 78.94 | +2.84 |
| released_grace_w4 | 85.52 | 87.11 | 75.31 | 77.68 | +2.36 |
| released_grace_w8 | 84.88 | 86.66 | 74.65 | 76.72 | +2.07 |
| qat_w4 | 84.28 | 86.12 | 74.99 | 77.86 | +2.87 |
| grace_w4 | 85.47 | 86.71 | 73.24 | 76.84 | +3.60 |
| qat_w8 | 86.22 | 87.70 | 76.34 | 79.35 | +3.02 |
| grace_w8 | 86.61 | 87.95 | 73.01 | 77.78 | +4.77 |

## pairs

| Contrast (left − right) | scienceqa | ai2d | seedbench | mmstar | Δ Mean-4 |
| --- | --- | --- | --- | --- | --- |
| grace_w4_minus_qat_w4 | +0.59 | -1.07 | +0.70 | -4.31 | -1.02 |
| grace_w8_minus_qat_w8 | +0.25 | -1.30 | +0.41 | -5.67 | -1.58 |
| released_grace_w4_minus_released_grace_w8 | +0.45 | +0.84 | +0.25 | +2.30 | +0.96 |
| grace_w4_minus_released_grace_w4 | -0.40 | -0.84 | +1.39 | -3.51 | -0.84 |
| grace_w8_minus_released_grace_w8 | +1.29 | +1.46 | +1.69 | -0.21 | +1.06 |

## ci

| Bits | Benchmark | GRACE-only correct / QAT-only correct | Δ pp | Paired 95% interval |
| --- | --- | --- | --- | --- |
| w4 | scienceqa | 69 / 57 | +0.59 | [-0.50, +1.69] |
| w4 | ai2d | 114 / 147 | -1.07 | [-2.07, -0.03] |
| w4 | seedbench | 495 / 395 | +0.70 | [+0.29, +1.12] |
| w4 | mmstar | 60 / 139 | -4.31 | [-6.22, -2.39] |
| w8 | scienceqa | 68 / 63 | +0.25 | [-0.84, +1.39] |
| w8 | ai2d | 120 / 160 | -1.30 | [-2.36, -0.23] |
| w8 | seedbench | 582 / 524 | +0.41 | [-0.05, +0.86] |
| w8 | mmstar | 64 / 155 | -5.67 | [-7.72, -3.59] |

## categories

| MMStar category | W4 GRACE − QAT | W8 GRACE − QAT | Teacher − BF16 init |
| --- | --- | --- | --- |
| coarse perception | -6.63 | -4.69 | -4.42 |
| fine-grained perception | -2.31 | -4.77 | -0.67 |
| instance reasoning | -3.68 | -8.58 | -7.64 |
| logical reasoning | -3.31 | -2.78 | +1.03 |
| math | -4.24 | -5.30 | -6.72 |
| science & technology | -5.68 | -7.91 | -10.63 |
