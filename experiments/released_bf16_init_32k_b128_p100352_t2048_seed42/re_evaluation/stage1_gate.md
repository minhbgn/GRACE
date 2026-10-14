# Reference-stage acceptance

Gate: BLOCKED. Complete: 35/35; tolerance: ±3 pp per pair.

All seven local scorers are unjudged; no 8-point judge allowance. Numerical agreement alone does not prove paper-protocol identity.

| Model | Suite | Paper % | Local % | Delta pp | Status |
| --- | --- | ---: | ---: | ---: | --- |
| vanilla_2b | hallusionbench | 51.40 | 62.44 | 11.04 | fail |
| vanilla_2b | mmbench | 78.40 | 83.81 | 5.41 | fail |
| vanilla_2b | scienceqa | 81.40 | 83.19 | 1.79 | pass |
| vanilla_2b | ai2d | 76.90 | 76.62 | -0.28 | pass |
| vanilla_2b | mmmu | 53.40 | 41.22 | -12.18 | fail |
| vanilla_2b | seedbench_img | 71.20 | 76.79 | 5.59 | fail |
| vanilla_2b | mmstar | 58.30 | 46.87 | -11.43 | fail |
| vanilla_8b_teacher | hallusionbench | 61.10 | 72.90 | 11.80 | fail |
| vanilla_8b_teacher | mmbench | 84.50 | 89.79 | 5.29 | fail |
| vanilla_8b_teacher | scienceqa | 85.00 | 92.95 | 7.95 | fail |
| vanilla_8b_teacher | ai2d | 85.70 | 83.97 | -1.73 | pass |
| vanilla_8b_teacher | mmmu | 69.60 | 52.78 | -16.82 | fail |
| vanilla_8b_teacher | seedbench_img | 77.50 | 79.39 | 1.89 | pass |
| vanilla_8b_teacher | mmstar | 70.90 | 62.73 | -8.17 | fail |
| released_bf16 | hallusionbench | 66.90 | 67.05 | 0.15 | pass |
| released_bf16 | mmbench | 86.40 | 94.66 | 8.26 | fail |
| released_bf16 | scienceqa | 86.20 | 86.18 | -0.02 | pass |
| released_bf16 | ai2d | 81.30 | 81.12 | -0.18 | pass |
| released_bf16 | mmmu | 72.10 | 72.00 | -0.10 | pass |
| released_bf16 | seedbench_img | 76.70 | 76.28 | -0.42 | pass |
| released_bf16 | mmstar | 67.30 | 68.00 | 0.70 | pass |
| released_grace_w4 | hallusionbench | 65.40 | 64.57 | -0.83 | pass |
| released_grace_w4 | mmbench | 84.60 | 92.63 | 8.03 | fail |
| released_grace_w4 | scienceqa | 84.30 | 84.08 | -0.22 | pass |
| released_grace_w4 | ai2d | 79.50 | 79.40 | -0.10 | pass |
| released_grace_w4 | mmmu | 70.50 | 61.56 | -8.94 | fail |
| released_grace_w4 | seedbench_img | 75.10 | 73.86 | -1.24 | pass |
| released_grace_w4 | mmstar | 65.80 | 66.20 | 0.40 | pass |
| released_grace_w8 | hallusionbench | 66.10 | 59.17 | -6.93 | fail |
| released_grace_w8 | mmbench | 85.50 | 92.49 | 6.99 | fail |
| released_grace_w8 | scienceqa | 85.30 | 82.48 | -2.82 | pass |
| released_grace_w8 | ai2d | 80.40 | 75.94 | -4.46 | barely_acceptable |
| released_grace_w8 | mmmu | 71.30 | 58.11 | -13.19 | fail |
| released_grace_w8 | seedbench_img | 75.90 | 70.08 | -5.82 | fail |
| released_grace_w8 | mmstar | 66.50 | 63.67 | -2.83 | pass |
