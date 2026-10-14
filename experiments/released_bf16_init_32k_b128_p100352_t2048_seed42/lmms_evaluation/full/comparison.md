# Pinned lmms-eval comparison

Harness protocol; not verified paper reproduction. Means require all seven suites.

Completed 40/70 model/benchmark pairs.

| Model | scienceqa | ai2d | mmmu | mmstar | seedbench | hallusionbench | mmbench | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vanilla_2b | 87.26 | 76.13 | — | 55.08 | 76.09 | — | — | — |
| vanilla_8b_teacher | 94.70 | 84.00 | — | 64.55 | 78.56 | — | — | — |
| sft_2b | 86.76 | 77.53 | — | 56.10 | 76.68 | — | — | — |
| released_bf16 | 88.30 | 81.80 | — | 69.39 | 76.27 | — | — | — |
| released_grace_w4 | 87.11 | 80.41 | — | 67.49 | 75.70 | — | — | — |
| released_grace_w8 | 86.66 | 79.57 | — | 65.20 | 75.45 | — | — | — |
| qat_w4 | 86.12 | 80.63 | — | 68.29 | 76.39 | — | — | — |
| grace_w4 | 86.71 | 79.57 | — | 63.98 | 77.09 | — | — | — |
| qat_w8 | 87.70 | 82.32 | — | 70.66 | 76.73 | — | — | — |
| grace_w8 | 87.95 | 81.02 | — | 64.99 | 77.14 | — | — | — |
