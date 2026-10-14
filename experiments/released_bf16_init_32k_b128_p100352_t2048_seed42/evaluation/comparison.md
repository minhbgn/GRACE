# Released-BF16 comparison

Validated all 70 merged results, expected counts, unique identities, matching gold labels, model/processor paths and evaluation policy.

| Model | hallusionbench | mmbench | scienceqa | ai2d | mmmu | seedbench_img | mmstar | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grace_w4 | 64.92 | 92.28 | 83.73 | 79.05 | 64.78 | 73.04 | 55.40 | 73.32 |
| grace_w8 | 66.43 | 92.47 | 85.71 | 79.44 | 67.44 | 72.77 | 53.20 | 73.92 |
| qat_w4 | 63.86 | 92.31 | 82.88 | 79.37 | 66.33 | 72.44 | 63.87 | 74.44 |
| qat_w8 | 64.92 | 93.32 | 85.00 | 80.76 | 70.67 | 72.63 | 65.73 | 76.15 |
| released_bf16 | 66.34 | 92.95 | 85.03 | 80.31 | 70.00 | 72.29 | 65.73 | 76.09 |
| released_grace_w4 | 63.86 | 92.47 | 83.68 | 78.82 | 61.22 | 72.25 | 64.67 | 73.85 |
| released_grace_w8 | 60.85 | 91.55 | 82.95 | 78.11 | 56.89 | 71.99 | 63.60 | 72.28 |
| sft_2b | 64.13 | 83.85 | 82.72 | 74.94 | 43.00 | 73.06 | 52.47 | 67.74 |
| vanilla_2b | 59.88 | 83.37 | 82.95 | 74.81 | 40.00 | 72.46 | 44.67 | 65.45 |
| vanilla_8b_teacher | 70.33 | 89.10 | 92.36 | 82.09 | 51.33 | 76.13 | 59.13 | 74.35 |

Scores are percentages. The unweighted seven-suite mean is descriptive, not an official leaderboard metric. sft_2b is the previous vanilla-initialized local SFT export; the four new low-bit runs start independently from released_bf16.

## Differences (percentage points)

| Comparison | hallusionbench | mmbench | scienceqa | ai2d | mmmu | seedbench_img | mmstar | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W4: distillation gain (grace_w4 minus qat_w4) | +1.06 | -0.02 | +0.85 | -0.32 | -1.56 | +0.60 | -8.47 | -1.12 |
| W4: gap to full method (released_grace_w4 minus grace_w4) | -1.06 | +0.18 | -0.05 | -0.23 | -3.56 | -0.79 | +9.27 | +0.54 |
| W4: gap to BF16 initialization (released_bf16 minus grace_w4) | +1.42 | +0.67 | +1.30 | +1.26 | +5.22 | -0.75 | +10.33 | +2.78 |
| W4: gain over vanilla 2B (grace_w4 minus vanilla_2b) | +5.05 | +8.92 | +0.78 | +4.24 | +24.78 | +0.58 | +10.73 | +7.87 |
| W4: difference from teacher (grace_w4 minus vanilla_8b_teacher) | -5.40 | +3.19 | -8.63 | -3.04 | +13.44 | -3.08 | -3.73 | -1.04 |
| W4: difference from previous SFT (grace_w4 minus sft_2b) | +0.80 | +8.43 | +1.01 | +4.11 | +21.78 | -0.02 | +2.93 | +5.58 |
| W8: distillation gain (grace_w8 minus qat_w8) | +1.51 | -0.85 | +0.71 | -1.33 | -3.22 | +0.15 | -12.53 | -2.23 |
| W8: gap to full method (released_grace_w8 minus grace_w8) | -5.58 | -0.92 | -2.76 | -1.33 | -10.56 | -0.78 | +10.40 | -1.65 |
| W8: gap to BF16 initialization (released_bf16 minus grace_w8) | -0.09 | +0.49 | -0.68 | +0.87 | +2.56 | -0.48 | +12.53 | +2.17 |
| W8: gain over vanilla 2B (grace_w8 minus vanilla_2b) | +6.55 | +9.10 | +2.76 | +4.63 | +27.44 | +0.32 | +8.53 | +8.48 |
| W8: difference from teacher (grace_w8 minus vanilla_8b_teacher) | -3.90 | +3.37 | -6.65 | -2.66 | +16.11 | -3.35 | -5.93 | -0.43 |
| W8: difference from previous SFT (grace_w8 minus sft_2b) | +2.30 | +8.62 | +2.99 | +4.50 | +24.44 | -0.29 | +0.73 | +6.19 |

## Errors and parse failures

| Model | Suite | Errors | Parse failures |
| --- | --- | --- | --- |
| grace_w4 | hallusionbench | 0 | 0 |
| grace_w8 | hallusionbench | 0 | 0 |
| qat_w4 | hallusionbench | 0 | 0 |
| qat_w8 | hallusionbench | 0 | 0 |
| released_bf16 | hallusionbench | 0 | 0 |
| released_grace_w4 | hallusionbench | 0 | 0 |
| released_grace_w8 | hallusionbench | 0 | 9 |
| sft_2b | hallusionbench | 0 | 0 |
| vanilla_2b | hallusionbench | 0 | 0 |
| vanilla_8b_teacher | hallusionbench | 0 | 0 |
| grace_w4 | mmbench | 0 | 0 |
| grace_w8 | mmbench | 0 | 0 |
| qat_w4 | mmbench | 0 | 0 |
| qat_w8 | mmbench | 0 | 0 |
| released_bf16 | mmbench | 0 | 0 |
| released_grace_w4 | mmbench | 0 | 0 |
| released_grace_w8 | mmbench | 0 | 0 |
| sft_2b | mmbench | 0 | 0 |
| vanilla_2b | mmbench | 0 | 0 |
| vanilla_8b_teacher | mmbench | 0 | 0 |
| grace_w4 | scienceqa | 0 | 0 |
| grace_w8 | scienceqa | 0 | 0 |
| qat_w4 | scienceqa | 0 | 0 |
| qat_w8 | scienceqa | 0 | 0 |
| released_bf16 | scienceqa | 0 | 0 |
| released_grace_w4 | scienceqa | 0 | 0 |
| released_grace_w8 | scienceqa | 0 | 0 |
| sft_2b | scienceqa | 0 | 0 |
| vanilla_2b | scienceqa | 0 | 1 |
| vanilla_8b_teacher | scienceqa | 0 | 0 |
| grace_w4 | ai2d | 0 | 0 |
| grace_w8 | ai2d | 0 | 0 |
| qat_w4 | ai2d | 0 | 0 |
| qat_w8 | ai2d | 0 | 0 |
| released_bf16 | ai2d | 0 | 0 |
| released_grace_w4 | ai2d | 0 | 0 |
| released_grace_w8 | ai2d | 0 | 0 |
| sft_2b | ai2d | 0 | 0 |
| vanilla_2b | ai2d | 0 | 2 |
| vanilla_8b_teacher | ai2d | 0 | 0 |
| grace_w4 | mmmu | 0 | not reported |
| grace_w8 | mmmu | 0 | not reported |
| qat_w4 | mmmu | 0 | not reported |
| qat_w8 | mmmu | 0 | not reported |
| released_bf16 | mmmu | 0 | not reported |
| released_grace_w4 | mmmu | 0 | not reported |
| released_grace_w8 | mmmu | 0 | not reported |
| sft_2b | mmmu | 0 | not reported |
| vanilla_2b | mmmu | 0 | not reported |
| vanilla_8b_teacher | mmmu | 0 | not reported |
| grace_w4 | seedbench_img | 0 | 0 |
| grace_w8 | seedbench_img | 0 | 0 |
| qat_w4 | seedbench_img | 0 | 0 |
| qat_w8 | seedbench_img | 0 | 0 |
| released_bf16 | seedbench_img | 0 | 0 |
| released_grace_w4 | seedbench_img | 0 | 0 |
| released_grace_w8 | seedbench_img | 0 | 0 |
| sft_2b | seedbench_img | 0 | 0 |
| vanilla_2b | seedbench_img | 0 | 0 |
| vanilla_8b_teacher | seedbench_img | 0 | 0 |
| grace_w4 | mmstar | 0 | 0 |
| grace_w8 | mmstar | 0 | 0 |
| qat_w4 | mmstar | 0 | 0 |
| qat_w8 | mmstar | 0 | 0 |
| released_bf16 | mmstar | 0 | 0 |
| released_grace_w4 | mmstar | 0 | 0 |
| released_grace_w8 | mmstar | 0 | 0 |
| sft_2b | mmstar | 0 | 0 |
| vanilla_2b | mmstar | 0 | 1 |
| vanilla_8b_teacher | mmstar | 0 | 2 |

Failed parses remain in the denominator. MMMU uses the existing local mixed-answer scorer.

## Additional metrics and categories

| Model | Suite | Category/metric | N | Score (%) |
| --- | --- | --- | --- | --- |
| grace_w4 | hallusionbench | qAcc | 455 | 31.87 |
| grace_w4 | hallusionbench | fAcc | 346 | 36.13 |
| grace_w4 | hallusionbench | by_category/VD | 591 | 64.47 |
| grace_w4 | hallusionbench | by_category/VS | 538 | 65.43 |
| grace_w4 | hallusionbench | by_modality/image | 951 | 66.04 |
| grace_w4 | hallusionbench | by_modality/text | 178 | 58.99 |
| grace_w4 | hallusionbench | by_subcategory/VD/figure | 80 | 71.25 |
| grace_w4 | hallusionbench | by_subcategory/VD/illusion | 144 | 61.11 |
| grace_w4 | hallusionbench | by_subcategory/VD/math | 108 | 67.59 |
| grace_w4 | hallusionbench | by_subcategory/VD/ocr | 89 | 79.78 |
| grace_w4 | hallusionbench | by_subcategory/VD/video | 170 | 54.12 |
| grace_w4 | hallusionbench | by_subcategory/VS/chart | 206 | 65.05 |
| grace_w4 | hallusionbench | by_subcategory/VS/map | 96 | 62.50 |
| grace_w4 | hallusionbench | by_subcategory/VS/ocr | 81 | 65.43 |
| grace_w4 | hallusionbench | by_subcategory/VS/table | 155 | 67.74 |
| grace_w8 | hallusionbench | qAcc | 455 | 33.63 |
| grace_w8 | hallusionbench | fAcc | 346 | 37.57 |
| grace_w8 | hallusionbench | by_category/VD | 591 | 65.48 |
| grace_w8 | hallusionbench | by_category/VS | 538 | 67.47 |
| grace_w8 | hallusionbench | by_modality/image | 951 | 66.77 |
| grace_w8 | hallusionbench | by_modality/text | 178 | 64.61 |
| grace_w8 | hallusionbench | by_subcategory/VD/figure | 80 | 76.25 |
| grace_w8 | hallusionbench | by_subcategory/VD/illusion | 144 | 61.81 |
| grace_w8 | hallusionbench | by_subcategory/VD/math | 108 | 66.67 |
| grace_w8 | hallusionbench | by_subcategory/VD/ocr | 89 | 82.02 |
| grace_w8 | hallusionbench | by_subcategory/VD/video | 170 | 54.12 |
| grace_w8 | hallusionbench | by_subcategory/VS/chart | 206 | 68.45 |
| grace_w8 | hallusionbench | by_subcategory/VS/map | 96 | 63.54 |
| grace_w8 | hallusionbench | by_subcategory/VS/ocr | 81 | 64.20 |
| grace_w8 | hallusionbench | by_subcategory/VS/table | 155 | 70.32 |
| qat_w4 | hallusionbench | qAcc | 455 | 29.89 |
| qat_w4 | hallusionbench | fAcc | 346 | 31.21 |
| qat_w4 | hallusionbench | by_category/VD | 591 | 64.30 |
| qat_w4 | hallusionbench | by_category/VS | 538 | 63.38 |
| qat_w4 | hallusionbench | by_modality/image | 951 | 64.46 |
| qat_w4 | hallusionbench | by_modality/text | 178 | 60.67 |
| qat_w4 | hallusionbench | by_subcategory/VD/figure | 80 | 67.50 |
| qat_w4 | hallusionbench | by_subcategory/VD/illusion | 144 | 59.72 |
| qat_w4 | hallusionbench | by_subcategory/VD/math | 108 | 70.37 |
| qat_w4 | hallusionbench | by_subcategory/VD/ocr | 89 | 78.65 |
| qat_w4 | hallusionbench | by_subcategory/VD/video | 170 | 55.29 |
| qat_w4 | hallusionbench | by_subcategory/VS/chart | 206 | 66.02 |
| qat_w4 | hallusionbench | by_subcategory/VS/map | 96 | 60.42 |
| qat_w4 | hallusionbench | by_subcategory/VS/ocr | 81 | 56.79 |
| qat_w4 | hallusionbench | by_subcategory/VS/table | 155 | 65.16 |
| qat_w8 | hallusionbench | qAcc | 455 | 31.21 |
| qat_w8 | hallusionbench | fAcc | 346 | 32.08 |
| qat_w8 | hallusionbench | by_category/VD | 591 | 65.48 |
| qat_w8 | hallusionbench | by_category/VS | 538 | 64.31 |
| qat_w8 | hallusionbench | by_modality/image | 951 | 65.83 |
| qat_w8 | hallusionbench | by_modality/text | 178 | 60.11 |
| qat_w8 | hallusionbench | by_subcategory/VD/figure | 80 | 70.00 |
| qat_w8 | hallusionbench | by_subcategory/VD/illusion | 144 | 60.42 |
| qat_w8 | hallusionbench | by_subcategory/VD/math | 108 | 72.22 |
| qat_w8 | hallusionbench | by_subcategory/VD/ocr | 89 | 79.78 |
| qat_w8 | hallusionbench | by_subcategory/VD/video | 170 | 55.88 |
| qat_w8 | hallusionbench | by_subcategory/VS/chart | 206 | 66.99 |
| qat_w8 | hallusionbench | by_subcategory/VS/map | 96 | 59.38 |
| qat_w8 | hallusionbench | by_subcategory/VS/ocr | 81 | 60.49 |
| qat_w8 | hallusionbench | by_subcategory/VS/table | 155 | 65.81 |
| released_bf16 | hallusionbench | qAcc | 455 | 33.19 |
| released_bf16 | hallusionbench | fAcc | 346 | 36.99 |
| released_bf16 | hallusionbench | by_category/VD | 591 | 66.84 |
| released_bf16 | hallusionbench | by_category/VS | 538 | 65.80 |
| released_bf16 | hallusionbench | by_modality/image | 951 | 66.88 |
| released_bf16 | hallusionbench | by_modality/text | 178 | 63.48 |
| released_bf16 | hallusionbench | by_subcategory/VD/figure | 80 | 72.50 |
| released_bf16 | hallusionbench | by_subcategory/VD/illusion | 144 | 63.89 |
| released_bf16 | hallusionbench | by_subcategory/VD/math | 108 | 71.30 |
| released_bf16 | hallusionbench | by_subcategory/VD/ocr | 89 | 79.78 |
| released_bf16 | hallusionbench | by_subcategory/VD/video | 170 | 57.06 |
| released_bf16 | hallusionbench | by_subcategory/VS/chart | 206 | 68.45 |
| released_bf16 | hallusionbench | by_subcategory/VS/map | 96 | 59.38 |
| released_bf16 | hallusionbench | by_subcategory/VS/ocr | 81 | 60.49 |
| released_bf16 | hallusionbench | by_subcategory/VS/table | 155 | 69.03 |
| released_grace_w4 | hallusionbench | qAcc | 455 | 29.23 |
| released_grace_w4 | hallusionbench | fAcc | 346 | 35.55 |
| released_grace_w4 | hallusionbench | by_category/VD | 591 | 64.97 |
| released_grace_w4 | hallusionbench | by_category/VS | 538 | 62.64 |
| released_grace_w4 | hallusionbench | by_modality/image | 951 | 65.93 |
| released_grace_w4 | hallusionbench | by_modality/text | 178 | 52.81 |
| released_grace_w4 | hallusionbench | by_subcategory/VD/figure | 80 | 76.25 |
| released_grace_w4 | hallusionbench | by_subcategory/VD/illusion | 144 | 65.97 |
| released_grace_w4 | hallusionbench | by_subcategory/VD/math | 108 | 61.11 |
| released_grace_w4 | hallusionbench | by_subcategory/VD/ocr | 89 | 75.28 |
| released_grace_w4 | hallusionbench | by_subcategory/VD/video | 170 | 55.88 |
| released_grace_w4 | hallusionbench | by_subcategory/VS/chart | 206 | 58.74 |
| released_grace_w4 | hallusionbench | by_subcategory/VS/map | 96 | 57.29 |
| released_grace_w4 | hallusionbench | by_subcategory/VS/ocr | 81 | 58.02 |
| released_grace_w4 | hallusionbench | by_subcategory/VS/table | 155 | 73.55 |
| released_grace_w8 | hallusionbench | qAcc | 455 | 26.37 |
| released_grace_w8 | hallusionbench | fAcc | 346 | 33.53 |
| released_grace_w8 | hallusionbench | by_category/VD | 591 | 61.76 |
| released_grace_w8 | hallusionbench | by_category/VS | 538 | 59.85 |
| released_grace_w8 | hallusionbench | by_modality/image | 951 | 63.09 |
| released_grace_w8 | hallusionbench | by_modality/text | 178 | 48.88 |
| released_grace_w8 | hallusionbench | by_subcategory/VD/figure | 80 | 66.25 |
| released_grace_w8 | hallusionbench | by_subcategory/VD/illusion | 144 | 65.97 |
| released_grace_w8 | hallusionbench | by_subcategory/VD/math | 108 | 55.56 |
| released_grace_w8 | hallusionbench | by_subcategory/VD/ocr | 89 | 73.03 |
| released_grace_w8 | hallusionbench | by_subcategory/VD/video | 170 | 54.12 |
| released_grace_w8 | hallusionbench | by_subcategory/VS/chart | 206 | 56.31 |
| released_grace_w8 | hallusionbench | by_subcategory/VS/map | 96 | 59.38 |
| released_grace_w8 | hallusionbench | by_subcategory/VS/ocr | 81 | 60.49 |
| released_grace_w8 | hallusionbench | by_subcategory/VS/table | 155 | 64.52 |
| sft_2b | hallusionbench | qAcc | 455 | 31.87 |
| sft_2b | hallusionbench | fAcc | 346 | 35.55 |
| sft_2b | hallusionbench | by_category/VD | 591 | 62.94 |
| sft_2b | hallusionbench | by_category/VS | 538 | 65.43 |
| sft_2b | hallusionbench | by_modality/image | 951 | 65.40 |
| sft_2b | hallusionbench | by_modality/text | 178 | 57.30 |
| sft_2b | hallusionbench | by_subcategory/VD/figure | 80 | 77.50 |
| sft_2b | hallusionbench | by_subcategory/VD/illusion | 144 | 62.50 |
| sft_2b | hallusionbench | by_subcategory/VD/math | 108 | 53.70 |
| sft_2b | hallusionbench | by_subcategory/VD/ocr | 89 | 78.65 |
| sft_2b | hallusionbench | by_subcategory/VD/video | 170 | 54.12 |
| sft_2b | hallusionbench | by_subcategory/VS/chart | 206 | 61.65 |
| sft_2b | hallusionbench | by_subcategory/VS/map | 96 | 66.67 |
| sft_2b | hallusionbench | by_subcategory/VS/ocr | 81 | 62.96 |
| sft_2b | hallusionbench | by_subcategory/VS/table | 155 | 70.97 |
| vanilla_2b | hallusionbench | qAcc | 455 | 26.37 |
| vanilla_2b | hallusionbench | fAcc | 346 | 31.79 |
| vanilla_2b | hallusionbench | by_category/VD | 591 | 59.39 |
| vanilla_2b | hallusionbench | by_category/VS | 538 | 60.41 |
| vanilla_2b | hallusionbench | by_modality/image | 951 | 61.20 |
| vanilla_2b | hallusionbench | by_modality/text | 178 | 52.81 |
| vanilla_2b | hallusionbench | by_subcategory/VD/figure | 80 | 71.25 |
| vanilla_2b | hallusionbench | by_subcategory/VD/illusion | 144 | 60.42 |
| vanilla_2b | hallusionbench | by_subcategory/VD/math | 108 | 47.22 |
| vanilla_2b | hallusionbench | by_subcategory/VD/ocr | 89 | 78.65 |
| vanilla_2b | hallusionbench | by_subcategory/VD/video | 170 | 50.59 |
| vanilla_2b | hallusionbench | by_subcategory/VS/chart | 206 | 51.94 |
| vanilla_2b | hallusionbench | by_subcategory/VS/map | 96 | 57.29 |
| vanilla_2b | hallusionbench | by_subcategory/VS/ocr | 81 | 58.02 |
| vanilla_2b | hallusionbench | by_subcategory/VS/table | 155 | 74.84 |
| vanilla_8b_teacher | hallusionbench | qAcc | 455 | 38.90 |
| vanilla_8b_teacher | hallusionbench | fAcc | 346 | 47.40 |
| vanilla_8b_teacher | hallusionbench | by_category/VD | 591 | 67.34 |
| vanilla_8b_teacher | hallusionbench | by_category/VS | 538 | 73.61 |
| vanilla_8b_teacher | hallusionbench | by_modality/image | 951 | 70.77 |
| vanilla_8b_teacher | hallusionbench | by_modality/text | 178 | 67.98 |
| vanilla_8b_teacher | hallusionbench | by_subcategory/VD/figure | 80 | 83.75 |
| vanilla_8b_teacher | hallusionbench | by_subcategory/VD/illusion | 144 | 69.44 |
| vanilla_8b_teacher | hallusionbench | by_subcategory/VD/math | 108 | 62.96 |
| vanilla_8b_teacher | hallusionbench | by_subcategory/VD/ocr | 89 | 79.78 |
| vanilla_8b_teacher | hallusionbench | by_subcategory/VD/video | 170 | 54.12 |
| vanilla_8b_teacher | hallusionbench | by_subcategory/VS/chart | 206 | 75.24 |
| vanilla_8b_teacher | hallusionbench | by_subcategory/VS/map | 96 | 66.67 |
| vanilla_8b_teacher | hallusionbench | by_subcategory/VS/ocr | 81 | 74.07 |
| vanilla_8b_teacher | hallusionbench | by_subcategory/VS/table | 155 | 75.48 |
| grace_w4 | mmbench | by_category/action_recognition | 215 | 98.14 |
| grace_w4 | mmbench | by_category/attribute_comparison | 141 | 83.69 |
| grace_w4 | mmbench | by_category/attribute_recognition | 264 | 94.70 |
| grace_w4 | mmbench | by_category/celebrity_recognition | 396 | 93.43 |
| grace_w4 | mmbench | by_category/function_reasoning | 304 | 98.68 |
| grace_w4 | mmbench | by_category/future_prediction | 130 | 93.08 |
| grace_w4 | mmbench | by_category/identity_reasoning | 176 | 100.00 |
| grace_w4 | mmbench | by_category/image_emotion | 200 | 94.00 |
| grace_w4 | mmbench | by_category/image_quality | 150 | 73.33 |
| grace_w4 | mmbench | by_category/image_scene | 407 | 99.26 |
| grace_w4 | mmbench | by_category/image_style | 212 | 97.17 |
| grace_w4 | mmbench | by_category/image_topic | 140 | 94.29 |
| grace_w4 | mmbench | by_category/nature_relation | 179 | 92.18 |
| grace_w4 | mmbench | by_category/object_localization | 315 | 86.03 |
| grace_w4 | mmbench | by_category/ocr | 156 | 92.95 |
| grace_w4 | mmbench | by_category/physical_property_reasoning | 219 | 94.98 |
| grace_w4 | mmbench | by_category/physical_relation | 94 | 86.17 |
| grace_w4 | mmbench | by_category/social_relation | 172 | 95.93 |
| grace_w4 | mmbench | by_category/spatial_relationship | 177 | 74.58 |
| grace_w4 | mmbench | by_category/structuralized_imagetext_understanding | 282 | 85.82 |
| grace_w4 | mmbench | by_l2_category/attribute_reasoning | 699 | 97.85 |
| grace_w4 | mmbench | by_l2_category/coarse_perception | 1109 | 93.78 |
| grace_w4 | mmbench | by_l2_category/finegrained_perception (cross-instance) | 533 | 86.49 |
| grace_w4 | mmbench | by_l2_category/finegrained_perception (instance-level) | 1131 | 91.60 |
| grace_w4 | mmbench | by_l2_category/logic_reasoning | 412 | 88.11 |
| grace_w4 | mmbench | by_l2_category/relation_reasoning | 445 | 92.36 |
| grace_w8 | mmbench | by_category/action_recognition | 215 | 98.14 |
| grace_w8 | mmbench | by_category/attribute_comparison | 141 | 88.65 |
| grace_w8 | mmbench | by_category/attribute_recognition | 264 | 95.83 |
| grace_w8 | mmbench | by_category/celebrity_recognition | 396 | 95.96 |
| grace_w8 | mmbench | by_category/function_reasoning | 304 | 97.70 |
| grace_w8 | mmbench | by_category/future_prediction | 130 | 92.31 |
| grace_w8 | mmbench | by_category/identity_reasoning | 176 | 100.00 |
| grace_w8 | mmbench | by_category/image_emotion | 200 | 93.00 |
| grace_w8 | mmbench | by_category/image_quality | 150 | 71.33 |
| grace_w8 | mmbench | by_category/image_scene | 407 | 99.26 |
| grace_w8 | mmbench | by_category/image_style | 212 | 97.64 |
| grace_w8 | mmbench | by_category/image_topic | 140 | 95.00 |
| grace_w8 | mmbench | by_category/nature_relation | 179 | 92.74 |
| grace_w8 | mmbench | by_category/object_localization | 315 | 82.86 |
| grace_w8 | mmbench | by_category/ocr | 156 | 95.51 |
| grace_w8 | mmbench | by_category/physical_property_reasoning | 219 | 92.24 |
| grace_w8 | mmbench | by_category/physical_relation | 94 | 88.30 |
| grace_w8 | mmbench | by_category/social_relation | 172 | 95.93 |
| grace_w8 | mmbench | by_category/spatial_relationship | 177 | 77.97 |
| grace_w8 | mmbench | by_category/structuralized_imagetext_understanding | 282 | 85.11 |
| grace_w8 | mmbench | by_l2_category/attribute_reasoning | 699 | 96.57 |
| grace_w8 | mmbench | by_l2_category/coarse_perception | 1109 | 93.51 |
| grace_w8 | mmbench | by_l2_category/finegrained_perception (cross-instance) | 533 | 88.93 |
| grace_w8 | mmbench | by_l2_category/finegrained_perception (instance-level) | 1131 | 92.22 |
| grace_w8 | mmbench | by_l2_category/logic_reasoning | 412 | 87.38 |
| grace_w8 | mmbench | by_l2_category/relation_reasoning | 445 | 93.03 |
| qat_w4 | mmbench | by_category/action_recognition | 215 | 98.14 |
| qat_w4 | mmbench | by_category/attribute_comparison | 141 | 82.27 |
| qat_w4 | mmbench | by_category/attribute_recognition | 264 | 95.83 |
| qat_w4 | mmbench | by_category/celebrity_recognition | 396 | 94.95 |
| qat_w4 | mmbench | by_category/function_reasoning | 304 | 98.36 |
| qat_w4 | mmbench | by_category/future_prediction | 130 | 91.54 |
| qat_w4 | mmbench | by_category/identity_reasoning | 176 | 100.00 |
| qat_w4 | mmbench | by_category/image_emotion | 200 | 95.00 |
| qat_w4 | mmbench | by_category/image_quality | 150 | 75.33 |
| qat_w4 | mmbench | by_category/image_scene | 407 | 98.77 |
| qat_w4 | mmbench | by_category/image_style | 212 | 98.58 |
| qat_w4 | mmbench | by_category/image_topic | 140 | 92.86 |
| qat_w4 | mmbench | by_category/nature_relation | 179 | 88.83 |
| qat_w4 | mmbench | by_category/object_localization | 315 | 83.17 |
| qat_w4 | mmbench | by_category/ocr | 156 | 95.51 |
| qat_w4 | mmbench | by_category/physical_property_reasoning | 219 | 96.80 |
| qat_w4 | mmbench | by_category/physical_relation | 94 | 89.36 |
| qat_w4 | mmbench | by_category/social_relation | 172 | 95.93 |
| qat_w4 | mmbench | by_category/spatial_relationship | 177 | 79.66 |
| qat_w4 | mmbench | by_category/structuralized_imagetext_understanding | 282 | 81.56 |
| qat_w4 | mmbench | by_l2_category/attribute_reasoning | 699 | 98.28 |
| qat_w4 | mmbench | by_l2_category/coarse_perception | 1109 | 94.14 |
| qat_w4 | mmbench | by_l2_category/finegrained_perception (cross-instance) | 533 | 87.80 |
| qat_w4 | mmbench | by_l2_category/finegrained_perception (instance-level) | 1131 | 91.95 |
| qat_w4 | mmbench | by_l2_category/logic_reasoning | 412 | 84.71 |
| qat_w4 | mmbench | by_l2_category/relation_reasoning | 445 | 91.69 |
| qat_w8 | mmbench | by_category/action_recognition | 215 | 100.00 |
| qat_w8 | mmbench | by_category/attribute_comparison | 141 | 83.69 |
| qat_w8 | mmbench | by_category/attribute_recognition | 264 | 96.21 |
| qat_w8 | mmbench | by_category/celebrity_recognition | 396 | 97.73 |
| qat_w8 | mmbench | by_category/function_reasoning | 304 | 98.68 |
| qat_w8 | mmbench | by_category/future_prediction | 130 | 90.00 |
| qat_w8 | mmbench | by_category/identity_reasoning | 176 | 99.43 |
| qat_w8 | mmbench | by_category/image_emotion | 200 | 95.00 |
| qat_w8 | mmbench | by_category/image_quality | 150 | 81.33 |
| qat_w8 | mmbench | by_category/image_scene | 407 | 99.75 |
| qat_w8 | mmbench | by_category/image_style | 212 | 98.11 |
| qat_w8 | mmbench | by_category/image_topic | 140 | 95.00 |
| qat_w8 | mmbench | by_category/nature_relation | 179 | 92.18 |
| qat_w8 | mmbench | by_category/object_localization | 315 | 82.54 |
| qat_w8 | mmbench | by_category/ocr | 156 | 93.59 |
| qat_w8 | mmbench | by_category/physical_property_reasoning | 219 | 94.06 |
| qat_w8 | mmbench | by_category/physical_relation | 94 | 86.17 |
| qat_w8 | mmbench | by_category/social_relation | 172 | 95.35 |
| qat_w8 | mmbench | by_category/spatial_relationship | 177 | 83.05 |
| qat_w8 | mmbench | by_category/structuralized_imagetext_understanding | 282 | 87.23 |
| qat_w8 | mmbench | by_l2_category/attribute_reasoning | 699 | 97.42 |
| qat_w8 | mmbench | by_l2_category/coarse_perception | 1109 | 95.49 |
| qat_w8 | mmbench | by_l2_category/finegrained_perception (cross-instance) | 533 | 90.06 |
| qat_w8 | mmbench | by_l2_category/finegrained_perception (instance-level) | 1131 | 92.57 |
| qat_w8 | mmbench | by_l2_category/logic_reasoning | 412 | 88.11 |
| qat_w8 | mmbench | by_l2_category/relation_reasoning | 445 | 92.13 |
| released_bf16 | mmbench | by_category/action_recognition | 215 | 99.07 |
| released_bf16 | mmbench | by_category/attribute_comparison | 141 | 87.23 |
| released_bf16 | mmbench | by_category/attribute_recognition | 264 | 97.73 |
| released_bf16 | mmbench | by_category/celebrity_recognition | 396 | 97.47 |
| released_bf16 | mmbench | by_category/function_reasoning | 304 | 98.36 |
| released_bf16 | mmbench | by_category/future_prediction | 130 | 90.77 |
| released_bf16 | mmbench | by_category/identity_reasoning | 176 | 100.00 |
| released_bf16 | mmbench | by_category/image_emotion | 200 | 94.00 |
| released_bf16 | mmbench | by_category/image_quality | 150 | 72.67 |
| released_bf16 | mmbench | by_category/image_scene | 407 | 99.51 |
| released_bf16 | mmbench | by_category/image_style | 212 | 98.58 |
| released_bf16 | mmbench | by_category/image_topic | 140 | 94.29 |
| released_bf16 | mmbench | by_category/nature_relation | 179 | 91.06 |
| released_bf16 | mmbench | by_category/object_localization | 315 | 82.54 |
| released_bf16 | mmbench | by_category/ocr | 156 | 94.23 |
| released_bf16 | mmbench | by_category/physical_property_reasoning | 219 | 94.98 |
| released_bf16 | mmbench | by_category/physical_relation | 94 | 89.36 |
| released_bf16 | mmbench | by_category/social_relation | 172 | 93.02 |
| released_bf16 | mmbench | by_category/spatial_relationship | 177 | 79.66 |
| released_bf16 | mmbench | by_category/structuralized_imagetext_understanding | 282 | 86.88 |
| released_bf16 | mmbench | by_l2_category/attribute_reasoning | 699 | 97.71 |
| released_bf16 | mmbench | by_l2_category/coarse_perception | 1109 | 94.05 |
| released_bf16 | mmbench | by_l2_category/finegrained_perception (cross-instance) | 533 | 89.49 |
| released_bf16 | mmbench | by_l2_category/finegrained_perception (instance-level) | 1131 | 92.93 |
| released_bf16 | mmbench | by_l2_category/logic_reasoning | 412 | 88.11 |
| released_bf16 | mmbench | by_l2_category/relation_reasoning | 445 | 91.46 |
| released_grace_w4 | mmbench | by_category/action_recognition | 215 | 96.28 |
| released_grace_w4 | mmbench | by_category/attribute_comparison | 141 | 87.94 |
| released_grace_w4 | mmbench | by_category/attribute_recognition | 264 | 97.35 |
| released_grace_w4 | mmbench | by_category/celebrity_recognition | 396 | 95.71 |
| released_grace_w4 | mmbench | by_category/function_reasoning | 304 | 96.38 |
| released_grace_w4 | mmbench | by_category/future_prediction | 130 | 93.08 |
| released_grace_w4 | mmbench | by_category/identity_reasoning | 176 | 100.00 |
| released_grace_w4 | mmbench | by_category/image_emotion | 200 | 98.50 |
| released_grace_w4 | mmbench | by_category/image_quality | 150 | 76.00 |
| released_grace_w4 | mmbench | by_category/image_scene | 407 | 100.00 |
| released_grace_w4 | mmbench | by_category/image_style | 212 | 96.70 |
| released_grace_w4 | mmbench | by_category/image_topic | 140 | 97.14 |
| released_grace_w4 | mmbench | by_category/nature_relation | 179 | 94.97 |
| released_grace_w4 | mmbench | by_category/object_localization | 315 | 81.90 |
| released_grace_w4 | mmbench | by_category/ocr | 156 | 95.51 |
| released_grace_w4 | mmbench | by_category/physical_property_reasoning | 219 | 94.52 |
| released_grace_w4 | mmbench | by_category/physical_relation | 94 | 88.30 |
| released_grace_w4 | mmbench | by_category/social_relation | 172 | 93.02 |
| released_grace_w4 | mmbench | by_category/spatial_relationship | 177 | 72.32 |
| released_grace_w4 | mmbench | by_category/structuralized_imagetext_understanding | 282 | 82.27 |
| released_grace_w4 | mmbench | by_l2_category/attribute_reasoning | 699 | 96.71 |
| released_grace_w4 | mmbench | by_l2_category/coarse_perception | 1109 | 95.49 |
| released_grace_w4 | mmbench | by_l2_category/finegrained_perception (cross-instance) | 533 | 86.12 |
| released_grace_w4 | mmbench | by_l2_category/finegrained_perception (instance-level) | 1131 | 92.22 |
| released_grace_w4 | mmbench | by_l2_category/logic_reasoning | 412 | 85.68 |
| released_grace_w4 | mmbench | by_l2_category/relation_reasoning | 445 | 92.81 |
| released_grace_w8 | mmbench | by_category/action_recognition | 215 | 96.74 |
| released_grace_w8 | mmbench | by_category/attribute_comparison | 141 | 88.65 |
| released_grace_w8 | mmbench | by_category/attribute_recognition | 264 | 97.35 |
| released_grace_w8 | mmbench | by_category/celebrity_recognition | 396 | 95.45 |
| released_grace_w8 | mmbench | by_category/function_reasoning | 304 | 95.39 |
| released_grace_w8 | mmbench | by_category/future_prediction | 130 | 91.54 |
| released_grace_w8 | mmbench | by_category/identity_reasoning | 176 | 100.00 |
| released_grace_w8 | mmbench | by_category/image_emotion | 200 | 98.00 |
| released_grace_w8 | mmbench | by_category/image_quality | 150 | 73.33 |
| released_grace_w8 | mmbench | by_category/image_scene | 407 | 100.00 |
| released_grace_w8 | mmbench | by_category/image_style | 212 | 95.28 |
| released_grace_w8 | mmbench | by_category/image_topic | 140 | 95.00 |
| released_grace_w8 | mmbench | by_category/nature_relation | 179 | 94.41 |
| released_grace_w8 | mmbench | by_category/object_localization | 315 | 81.59 |
| released_grace_w8 | mmbench | by_category/ocr | 156 | 96.15 |
| released_grace_w8 | mmbench | by_category/physical_property_reasoning | 219 | 93.15 |
| released_grace_w8 | mmbench | by_category/physical_relation | 94 | 87.23 |
| released_grace_w8 | mmbench | by_category/social_relation | 172 | 92.44 |
| released_grace_w8 | mmbench | by_category/spatial_relationship | 177 | 57.63 |
| released_grace_w8 | mmbench | by_category/structuralized_imagetext_understanding | 282 | 84.75 |
| released_grace_w8 | mmbench | by_l2_category/attribute_reasoning | 699 | 95.85 |
| released_grace_w8 | mmbench | by_l2_category/coarse_perception | 1109 | 94.50 |
| released_grace_w8 | mmbench | by_l2_category/finegrained_perception (cross-instance) | 533 | 81.61 |
| released_grace_w8 | mmbench | by_l2_category/finegrained_perception (instance-level) | 1131 | 92.13 |
| released_grace_w8 | mmbench | by_l2_category/logic_reasoning | 412 | 86.89 |
| released_grace_w8 | mmbench | by_l2_category/relation_reasoning | 445 | 92.13 |
| sft_2b | mmbench | by_category/action_recognition | 215 | 94.88 |
| sft_2b | mmbench | by_category/attribute_comparison | 141 | 71.63 |
| sft_2b | mmbench | by_category/attribute_recognition | 264 | 94.70 |
| sft_2b | mmbench | by_category/celebrity_recognition | 396 | 95.20 |
| sft_2b | mmbench | by_category/function_reasoning | 304 | 92.11 |
| sft_2b | mmbench | by_category/future_prediction | 130 | 41.54 |
| sft_2b | mmbench | by_category/identity_reasoning | 176 | 97.16 |
| sft_2b | mmbench | by_category/image_emotion | 200 | 85.00 |
| sft_2b | mmbench | by_category/image_quality | 150 | 64.67 |
| sft_2b | mmbench | by_category/image_scene | 407 | 97.79 |
| sft_2b | mmbench | by_category/image_style | 212 | 93.87 |
| sft_2b | mmbench | by_category/image_topic | 140 | 90.71 |
| sft_2b | mmbench | by_category/nature_relation | 179 | 75.42 |
| sft_2b | mmbench | by_category/object_localization | 315 | 68.57 |
| sft_2b | mmbench | by_category/ocr | 156 | 90.38 |
| sft_2b | mmbench | by_category/physical_property_reasoning | 219 | 80.82 |
| sft_2b | mmbench | by_category/physical_relation | 94 | 79.79 |
| sft_2b | mmbench | by_category/social_relation | 172 | 92.44 |
| sft_2b | mmbench | by_category/spatial_relationship | 177 | 43.50 |
| sft_2b | mmbench | by_category/structuralized_imagetext_understanding | 282 | 78.72 |
| sft_2b | mmbench | by_l2_category/attribute_reasoning | 699 | 89.84 |
| sft_2b | mmbench | by_l2_category/coarse_perception | 1109 | 89.36 |
| sft_2b | mmbench | by_l2_category/finegrained_perception (cross-instance) | 533 | 71.67 |
| sft_2b | mmbench | by_l2_category/finegrained_perception (instance-level) | 1131 | 87.00 |
| sft_2b | mmbench | by_l2_category/logic_reasoning | 412 | 66.99 |
| sft_2b | mmbench | by_l2_category/relation_reasoning | 445 | 82.92 |
| vanilla_2b | mmbench | by_category/action_recognition | 215 | 96.28 |
| vanilla_2b | mmbench | by_category/attribute_comparison | 141 | 77.30 |
| vanilla_2b | mmbench | by_category/attribute_recognition | 264 | 92.80 |
| vanilla_2b | mmbench | by_category/celebrity_recognition | 396 | 93.43 |
| vanilla_2b | mmbench | by_category/function_reasoning | 304 | 92.11 |
| vanilla_2b | mmbench | by_category/future_prediction | 130 | 44.62 |
| vanilla_2b | mmbench | by_category/identity_reasoning | 176 | 97.16 |
| vanilla_2b | mmbench | by_category/image_emotion | 200 | 83.00 |
| vanilla_2b | mmbench | by_category/image_quality | 150 | 65.33 |
| vanilla_2b | mmbench | by_category/image_scene | 407 | 97.54 |
| vanilla_2b | mmbench | by_category/image_style | 212 | 92.92 |
| vanilla_2b | mmbench | by_category/image_topic | 140 | 94.29 |
| vanilla_2b | mmbench | by_category/nature_relation | 179 | 74.86 |
| vanilla_2b | mmbench | by_category/object_localization | 315 | 70.16 |
| vanilla_2b | mmbench | by_category/ocr | 156 | 91.67 |
| vanilla_2b | mmbench | by_category/physical_property_reasoning | 219 | 79.00 |
| vanilla_2b | mmbench | by_category/physical_relation | 94 | 68.09 |
| vanilla_2b | mmbench | by_category/social_relation | 172 | 90.12 |
| vanilla_2b | mmbench | by_category/spatial_relationship | 177 | 41.81 |
| vanilla_2b | mmbench | by_category/structuralized_imagetext_understanding | 282 | 76.24 |
| vanilla_2b | mmbench | by_l2_category/attribute_reasoning | 699 | 89.27 |
| vanilla_2b | mmbench | by_l2_category/coarse_perception | 1109 | 89.27 |
| vanilla_2b | mmbench | by_l2_category/finegrained_perception (cross-instance) | 533 | 73.17 |
| vanilla_2b | mmbench | by_l2_category/finegrained_perception (instance-level) | 1131 | 86.56 |
| vanilla_2b | mmbench | by_l2_category/logic_reasoning | 412 | 66.26 |
| vanilla_2b | mmbench | by_l2_category/relation_reasoning | 445 | 79.33 |
| vanilla_8b_teacher | mmbench | by_category/action_recognition | 215 | 95.81 |
| vanilla_8b_teacher | mmbench | by_category/attribute_comparison | 141 | 87.23 |
| vanilla_8b_teacher | mmbench | by_category/attribute_recognition | 264 | 96.21 |
| vanilla_8b_teacher | mmbench | by_category/celebrity_recognition | 396 | 96.97 |
| vanilla_8b_teacher | mmbench | by_category/function_reasoning | 304 | 94.08 |
| vanilla_8b_teacher | mmbench | by_category/future_prediction | 130 | 74.62 |
| vanilla_8b_teacher | mmbench | by_category/identity_reasoning | 176 | 97.73 |
| vanilla_8b_teacher | mmbench | by_category/image_emotion | 200 | 86.00 |
| vanilla_8b_teacher | mmbench | by_category/image_quality | 150 | 67.33 |
| vanilla_8b_teacher | mmbench | by_category/image_scene | 407 | 98.03 |
| vanilla_8b_teacher | mmbench | by_category/image_style | 212 | 92.45 |
| vanilla_8b_teacher | mmbench | by_category/image_topic | 140 | 95.71 |
| vanilla_8b_teacher | mmbench | by_category/nature_relation | 179 | 92.18 |
| vanilla_8b_teacher | mmbench | by_category/object_localization | 315 | 75.87 |
| vanilla_8b_teacher | mmbench | by_category/ocr | 156 | 99.36 |
| vanilla_8b_teacher | mmbench | by_category/physical_property_reasoning | 219 | 81.28 |
| vanilla_8b_teacher | mmbench | by_category/physical_relation | 94 | 77.66 |
| vanilla_8b_teacher | mmbench | by_category/social_relation | 172 | 93.02 |
| vanilla_8b_teacher | mmbench | by_category/spatial_relationship | 177 | 70.62 |
| vanilla_8b_teacher | mmbench | by_category/structuralized_imagetext_understanding | 282 | 84.40 |
| vanilla_8b_teacher | mmbench | by_l2_category/attribute_reasoning | 699 | 90.99 |
| vanilla_8b_teacher | mmbench | by_l2_category/coarse_perception | 1109 | 90.35 |
| vanilla_8b_teacher | mmbench | by_l2_category/finegrained_perception (cross-instance) | 533 | 85.18 |
| vanilla_8b_teacher | mmbench | by_l2_category/finegrained_perception (instance-level) | 1131 | 91.25 |
| vanilla_8b_teacher | mmbench | by_l2_category/logic_reasoning | 412 | 81.31 |
| vanilla_8b_teacher | mmbench | by_l2_category/relation_reasoning | 445 | 89.44 |
| grace_w4 | scienceqa | by_category/20th century American history | 4 | 100.00 |
| grace_w4 | scienceqa | by_category/Adaptations | 98 | 94.90 |
| grace_w4 | scienceqa | by_category/Age of Exploration | 1 | 0.00 |
| grace_w4 | scienceqa | by_category/Analyzing literature | 4 | 100.00 |
| grace_w4 | scienceqa | by_category/Anatomy and physiology | 2 | 100.00 |
| grace_w4 | scienceqa | by_category/Ancient Egypt and Kush | 1 | 100.00 |
| grace_w4 | scienceqa | by_category/Ancient Mesopotamia | 2 | 100.00 |
| grace_w4 | scienceqa | by_category/Animals | 8 | 100.00 |
| grace_w4 | scienceqa | by_category/Astronomy | 37 | 56.76 |
| grace_w4 | scienceqa | by_category/Atoms and molecules | 64 | 93.75 |
| grace_w4 | scienceqa | by_category/Audience, purpose, and tone | 17 | 100.00 |
| grace_w4 | scienceqa | by_category/Author's purpose and tone | 27 | 81.48 |
| grace_w4 | scienceqa | by_category/Banking and finance | 1 | 100.00 |
| grace_w4 | scienceqa | by_category/Basic economic principles | 116 | 93.97 |
| grace_w4 | scienceqa | by_category/Capitalization | 18 | 72.22 |
| grace_w4 | scienceqa | by_category/Categories | 11 | 72.73 |
| grace_w4 | scienceqa | by_category/Cells | 17 | 82.35 |
| grace_w4 | scienceqa | by_category/Chemical reactions | 43 | 86.05 |
| grace_w4 | scienceqa | by_category/Cities | 12 | 66.67 |
| grace_w4 | scienceqa | by_category/Classification | 161 | 96.27 |
| grace_w4 | scienceqa | by_category/Classification and scientific names | 46 | 95.65 |
| grace_w4 | scienceqa | by_category/Climate change | 1 | 100.00 |
| grace_w4 | scienceqa | by_category/Colonial America | 38 | 84.21 |
| grace_w4 | scienceqa | by_category/Comprehension strategies | 5 | 100.00 |
| grace_w4 | scienceqa | by_category/Conservation and natural resources | 1 | 100.00 |
| grace_w4 | scienceqa | by_category/Context clues | 5 | 60.00 |
| grace_w4 | scienceqa | by_category/Creative techniques | 19 | 68.42 |
| grace_w4 | scienceqa | by_category/Cultural celebrations | 2 | 100.00 |
| grace_w4 | scienceqa | by_category/Descriptive details | 3 | 100.00 |
| grace_w4 | scienceqa | by_category/Designing experiments | 134 | 97.01 |
| grace_w4 | scienceqa | by_category/Developing and supporting arguments | 101 | 83.17 |
| grace_w4 | scienceqa | by_category/Domain-specific vocabulary | 4 | 100.00 |
| grace_w4 | scienceqa | by_category/Early 19th century American history | 2 | 100.00 |
| grace_w4 | scienceqa | by_category/Earth events | 18 | 100.00 |
| grace_w4 | scienceqa | by_category/Ecological interactions | 24 | 75.00 |
| grace_w4 | scienceqa | by_category/Economics | 11 | 81.82 |
| grace_w4 | scienceqa | by_category/Ecosystems | 78 | 87.18 |
| grace_w4 | scienceqa | by_category/Editing and revising | 9 | 44.44 |
| grace_w4 | scienceqa | by_category/Electricity | 1 | 100.00 |
| grace_w4 | scienceqa | by_category/Engineering practices | 55 | 100.00 |
| grace_w4 | scienceqa | by_category/English colonies in North America | 39 | 87.18 |
| grace_w4 | scienceqa | by_category/Force and motion | 58 | 81.03 |
| grace_w4 | scienceqa | by_category/Formatting | 73 | 73.97 |
| grace_w4 | scienceqa | by_category/Fossils | 22 | 63.64 |
| grace_w4 | scienceqa | by_category/Genes to traits | 195 | 86.15 |
| grace_w4 | scienceqa | by_category/Geography | 46 | 97.83 |
| grace_w4 | scienceqa | by_category/Government | 16 | 81.25 |
| grace_w4 | scienceqa | by_category/Greece | 1 | 100.00 |
| grace_w4 | scienceqa | by_category/Heat and thermal energy | 42 | 90.48 |
| grace_w4 | scienceqa | by_category/Heredity | 41 | 100.00 |
| grace_w4 | scienceqa | by_category/Historical figures | 2 | 100.00 |
| grace_w4 | scienceqa | by_category/Independent reading comprehension | 5 | 100.00 |
| grace_w4 | scienceqa | by_category/Informational texts: level 1 | 5 | 100.00 |
| grace_w4 | scienceqa | by_category/Kinetic and potential energy | 19 | 89.47 |
| grace_w4 | scienceqa | by_category/Literary devices | 267 | 81.65 |
| grace_w4 | scienceqa | by_category/Magnets | 110 | 50.91 |
| grace_w4 | scienceqa | by_category/Maps | 46 | 91.30 |
| grace_w4 | scienceqa | by_category/Materials | 167 | 94.01 |
| grace_w4 | scienceqa | by_category/Medieval Asia | 2 | 50.00 |
| grace_w4 | scienceqa | by_category/Mixtures | 6 | 83.33 |
| grace_w4 | scienceqa | by_category/Oceania: geography | 51 | 82.35 |
| grace_w4 | scienceqa | by_category/Oceans and continents | 19 | 100.00 |
| grace_w4 | scienceqa | by_category/Oceans and continents	 | 4 | 100.00 |
| grace_w4 | scienceqa | by_category/Opinion writing | 4 | 100.00 |
| grace_w4 | scienceqa | by_category/Particle motion and energy | 41 | 92.68 |
| grace_w4 | scienceqa | by_category/Persuasive strategies | 11 | 72.73 |
| grace_w4 | scienceqa | by_category/Photosynthesis | 4 | 100.00 |
| grace_w4 | scienceqa | by_category/Phrases and clauses | 32 | 43.75 |
| grace_w4 | scienceqa | by_category/Physical Geography | 51 | 80.39 |
| grace_w4 | scienceqa | by_category/Physical and chemical change | 52 | 88.46 |
| grace_w4 | scienceqa | by_category/Plant reproduction | 6 | 66.67 |
| grace_w4 | scienceqa | by_category/Plants | 12 | 100.00 |
| grace_w4 | scienceqa | by_category/Plate tectonics | 3 | 100.00 |
| grace_w4 | scienceqa | by_category/Poetry elements | 4 | 25.00 |
| grace_w4 | scienceqa | by_category/Pronouns | 37 | 70.27 |
| grace_w4 | scienceqa | by_category/Pronouns and antecedents | 18 | 61.11 |
| grace_w4 | scienceqa | by_category/Read-alone texts | 2 | 100.00 |
| grace_w4 | scienceqa | by_category/Reference skills | 140 | 51.43 |
| grace_w4 | scienceqa | by_category/Research skills | 4 | 75.00 |
| grace_w4 | scienceqa | by_category/Rhyming | 17 | 47.06 |
| grace_w4 | scienceqa | by_category/Rocks and minerals | 46 | 93.48 |
| grace_w4 | scienceqa | by_category/Scientific names | 32 | 100.00 |
| grace_w4 | scienceqa | by_category/Sentences, fragments, and run-ons | 135 | 77.78 |
| grace_w4 | scienceqa | by_category/Shades of meaning | 10 | 80.00 |
| grace_w4 | scienceqa | by_category/Short and long vowels | 5 | 80.00 |
| grace_w4 | scienceqa | by_category/Social studies skills | 12 | 100.00 |
| grace_w4 | scienceqa | by_category/Solutions | 54 | 68.52 |
| grace_w4 | scienceqa | by_category/State capitals | 312 | 96.79 |
| grace_w4 | scienceqa | by_category/States | 18 | 100.00 |
| grace_w4 | scienceqa | by_category/States of matter | 47 | 89.36 |
| grace_w4 | scienceqa | by_category/Supply and demand | 27 | 66.67 |
| grace_w4 | scienceqa | by_category/Text structure | 2 | 100.00 |
| grace_w4 | scienceqa | by_category/The American Revolution | 4 | 75.00 |
| grace_w4 | scienceqa | by_category/The Americas: geography | 45 | 73.33 |
| grace_w4 | scienceqa | by_category/The Antebellum period | 1 | 100.00 |
| grace_w4 | scienceqa | by_category/The Civil War and Reconstruction | 2 | 100.00 |
| grace_w4 | scienceqa | by_category/Thermal energy | 40 | 72.50 |
| grace_w4 | scienceqa | by_category/Topographic maps | 1 | 100.00 |
| grace_w4 | scienceqa | by_category/Traits and heredity | 81 | 98.77 |
| grace_w4 | scienceqa | by_category/Units and measurement | 200 | 88.00 |
| grace_w4 | scienceqa | by_category/Velocity, acceleration, and forces | 86 | 54.65 |
| grace_w4 | scienceqa | by_category/Verb tense | 52 | 84.62 |
| grace_w4 | scienceqa | by_category/Visual elements | 9 | 77.78 |
| grace_w4 | scienceqa | by_category/Water cycle | 2 | 100.00 |
| grace_w4 | scienceqa | by_category/Weather and climate | 97 | 69.07 |
| grace_w4 | scienceqa | by_category/Word usage and nuance | 45 | 75.56 |
| grace_w4 | scienceqa | by_category/World religions | 1 | 100.00 |
| grace_w4 | scienceqa | by_modality/image | 2017 | 85.47 |
| grace_w4 | scienceqa | by_modality/text | 2224 | 82.15 |
| grace_w4 | scienceqa | by_subject/language science | 1100 | 74.09 |
| grace_w4 | scienceqa | by_subject/natural science | 2252 | 85.79 |
| grace_w4 | scienceqa | by_subject/social science | 889 | 90.44 |
| grace_w8 | scienceqa | by_category/20th century American history | 4 | 100.00 |
| grace_w8 | scienceqa | by_category/Adaptations | 98 | 96.94 |
| grace_w8 | scienceqa | by_category/Age of Exploration | 1 | 100.00 |
| grace_w8 | scienceqa | by_category/Analyzing literature | 4 | 100.00 |
| grace_w8 | scienceqa | by_category/Anatomy and physiology | 2 | 100.00 |
| grace_w8 | scienceqa | by_category/Ancient Egypt and Kush | 1 | 100.00 |
| grace_w8 | scienceqa | by_category/Ancient Mesopotamia | 2 | 100.00 |
| grace_w8 | scienceqa | by_category/Animals | 8 | 100.00 |
| grace_w8 | scienceqa | by_category/Astronomy | 37 | 62.16 |
| grace_w8 | scienceqa | by_category/Atoms and molecules | 64 | 96.88 |
| grace_w8 | scienceqa | by_category/Audience, purpose, and tone | 17 | 100.00 |
| grace_w8 | scienceqa | by_category/Author's purpose and tone | 27 | 77.78 |
| grace_w8 | scienceqa | by_category/Banking and finance | 1 | 100.00 |
| grace_w8 | scienceqa | by_category/Basic economic principles | 116 | 97.41 |
| grace_w8 | scienceqa | by_category/Capitalization | 18 | 77.78 |
| grace_w8 | scienceqa | by_category/Categories | 11 | 100.00 |
| grace_w8 | scienceqa | by_category/Cells | 17 | 88.24 |
| grace_w8 | scienceqa | by_category/Chemical reactions | 43 | 83.72 |
| grace_w8 | scienceqa | by_category/Cities | 12 | 50.00 |
| grace_w8 | scienceqa | by_category/Classification | 161 | 96.27 |
| grace_w8 | scienceqa | by_category/Classification and scientific names | 46 | 100.00 |
| grace_w8 | scienceqa | by_category/Climate change | 1 | 100.00 |
| grace_w8 | scienceqa | by_category/Colonial America | 38 | 89.47 |
| grace_w8 | scienceqa | by_category/Comprehension strategies | 5 | 100.00 |
| grace_w8 | scienceqa | by_category/Conservation and natural resources | 1 | 100.00 |
| grace_w8 | scienceqa | by_category/Context clues | 5 | 60.00 |
| grace_w8 | scienceqa | by_category/Creative techniques | 19 | 73.68 |
| grace_w8 | scienceqa | by_category/Cultural celebrations | 2 | 100.00 |
| grace_w8 | scienceqa | by_category/Descriptive details | 3 | 100.00 |
| grace_w8 | scienceqa | by_category/Designing experiments | 134 | 97.76 |
| grace_w8 | scienceqa | by_category/Developing and supporting arguments | 101 | 86.14 |
| grace_w8 | scienceqa | by_category/Domain-specific vocabulary | 4 | 75.00 |
| grace_w8 | scienceqa | by_category/Early 19th century American history | 2 | 100.00 |
| grace_w8 | scienceqa | by_category/Earth events | 18 | 94.44 |
| grace_w8 | scienceqa | by_category/Ecological interactions | 24 | 79.17 |
| grace_w8 | scienceqa | by_category/Economics | 11 | 81.82 |
| grace_w8 | scienceqa | by_category/Ecosystems | 78 | 87.18 |
| grace_w8 | scienceqa | by_category/Editing and revising | 9 | 44.44 |
| grace_w8 | scienceqa | by_category/Electricity | 1 | 100.00 |
| grace_w8 | scienceqa | by_category/Engineering practices | 55 | 100.00 |
| grace_w8 | scienceqa | by_category/English colonies in North America | 39 | 97.44 |
| grace_w8 | scienceqa | by_category/Force and motion | 58 | 94.83 |
| grace_w8 | scienceqa | by_category/Formatting | 73 | 93.15 |
| grace_w8 | scienceqa | by_category/Fossils | 22 | 72.73 |
| grace_w8 | scienceqa | by_category/Genes to traits | 195 | 84.10 |
| grace_w8 | scienceqa | by_category/Geography | 46 | 97.83 |
| grace_w8 | scienceqa | by_category/Government | 16 | 87.50 |
| grace_w8 | scienceqa | by_category/Greece | 1 | 100.00 |
| grace_w8 | scienceqa | by_category/Heat and thermal energy | 42 | 95.24 |
| grace_w8 | scienceqa | by_category/Heredity | 41 | 100.00 |
| grace_w8 | scienceqa | by_category/Historical figures | 2 | 100.00 |
| grace_w8 | scienceqa | by_category/Independent reading comprehension | 5 | 100.00 |
| grace_w8 | scienceqa | by_category/Informational texts: level 1 | 5 | 100.00 |
| grace_w8 | scienceqa | by_category/Kinetic and potential energy | 19 | 89.47 |
| grace_w8 | scienceqa | by_category/Literary devices | 267 | 84.27 |
| grace_w8 | scienceqa | by_category/Magnets | 110 | 60.91 |
| grace_w8 | scienceqa | by_category/Maps | 46 | 93.48 |
| grace_w8 | scienceqa | by_category/Materials | 167 | 95.81 |
| grace_w8 | scienceqa | by_category/Medieval Asia | 2 | 50.00 |
| grace_w8 | scienceqa | by_category/Mixtures | 6 | 83.33 |
| grace_w8 | scienceqa | by_category/Oceania: geography | 51 | 64.71 |
| grace_w8 | scienceqa | by_category/Oceans and continents | 19 | 100.00 |
| grace_w8 | scienceqa | by_category/Oceans and continents	 | 4 | 100.00 |
| grace_w8 | scienceqa | by_category/Opinion writing | 4 | 100.00 |
| grace_w8 | scienceqa | by_category/Particle motion and energy | 41 | 92.68 |
| grace_w8 | scienceqa | by_category/Persuasive strategies | 11 | 72.73 |
| grace_w8 | scienceqa | by_category/Photosynthesis | 4 | 100.00 |
| grace_w8 | scienceqa | by_category/Phrases and clauses | 32 | 59.38 |
| grace_w8 | scienceqa | by_category/Physical Geography | 51 | 78.43 |
| grace_w8 | scienceqa | by_category/Physical and chemical change | 52 | 92.31 |
| grace_w8 | scienceqa | by_category/Plant reproduction | 6 | 66.67 |
| grace_w8 | scienceqa | by_category/Plants | 12 | 100.00 |
| grace_w8 | scienceqa | by_category/Plate tectonics | 3 | 100.00 |
| grace_w8 | scienceqa | by_category/Poetry elements | 4 | 50.00 |
| grace_w8 | scienceqa | by_category/Pronouns | 37 | 75.68 |
| grace_w8 | scienceqa | by_category/Pronouns and antecedents | 18 | 66.67 |
| grace_w8 | scienceqa | by_category/Read-alone texts | 2 | 100.00 |
| grace_w8 | scienceqa | by_category/Reference skills | 140 | 50.71 |
| grace_w8 | scienceqa | by_category/Research skills | 4 | 75.00 |
| grace_w8 | scienceqa | by_category/Rhyming | 17 | 47.06 |
| grace_w8 | scienceqa | by_category/Rocks and minerals | 46 | 93.48 |
| grace_w8 | scienceqa | by_category/Scientific names | 32 | 100.00 |
| grace_w8 | scienceqa | by_category/Sentences, fragments, and run-ons | 135 | 80.74 |
| grace_w8 | scienceqa | by_category/Shades of meaning | 10 | 90.00 |
| grace_w8 | scienceqa | by_category/Short and long vowels | 5 | 100.00 |
| grace_w8 | scienceqa | by_category/Social studies skills | 12 | 100.00 |
| grace_w8 | scienceqa | by_category/Solutions | 54 | 66.67 |
| grace_w8 | scienceqa | by_category/State capitals | 312 | 97.44 |
| grace_w8 | scienceqa | by_category/States | 18 | 100.00 |
| grace_w8 | scienceqa | by_category/States of matter | 47 | 91.49 |
| grace_w8 | scienceqa | by_category/Supply and demand | 27 | 66.67 |
| grace_w8 | scienceqa | by_category/Text structure | 2 | 100.00 |
| grace_w8 | scienceqa | by_category/The American Revolution | 4 | 100.00 |
| grace_w8 | scienceqa | by_category/The Americas: geography | 45 | 73.33 |
| grace_w8 | scienceqa | by_category/The Antebellum period | 1 | 100.00 |
| grace_w8 | scienceqa | by_category/The Civil War and Reconstruction | 2 | 100.00 |
| grace_w8 | scienceqa | by_category/Thermal energy | 40 | 75.00 |
| grace_w8 | scienceqa | by_category/Topographic maps | 1 | 100.00 |
| grace_w8 | scienceqa | by_category/Traits and heredity | 81 | 100.00 |
| grace_w8 | scienceqa | by_category/Units and measurement | 200 | 86.00 |
| grace_w8 | scienceqa | by_category/Velocity, acceleration, and forces | 86 | 56.98 |
| grace_w8 | scienceqa | by_category/Verb tense | 52 | 92.31 |
| grace_w8 | scienceqa | by_category/Visual elements | 9 | 88.89 |
| grace_w8 | scienceqa | by_category/Water cycle | 2 | 100.00 |
| grace_w8 | scienceqa | by_category/Weather and climate | 97 | 69.07 |
| grace_w8 | scienceqa | by_category/Word usage and nuance | 45 | 77.78 |
| grace_w8 | scienceqa | by_category/World religions | 1 | 100.00 |
| grace_w8 | scienceqa | by_modality/image | 2017 | 86.61 |
| grace_w8 | scienceqa | by_modality/text | 2224 | 84.89 |
| grace_w8 | scienceqa | by_subject/language science | 1100 | 78.36 |
| grace_w8 | scienceqa | by_subject/natural science | 2252 | 87.26 |
| grace_w8 | scienceqa | by_subject/social science | 889 | 90.89 |
| qat_w4 | scienceqa | by_category/20th century American history | 4 | 100.00 |
| qat_w4 | scienceqa | by_category/Adaptations | 98 | 93.88 |
| qat_w4 | scienceqa | by_category/Age of Exploration | 1 | 0.00 |
| qat_w4 | scienceqa | by_category/Analyzing literature | 4 | 100.00 |
| qat_w4 | scienceqa | by_category/Anatomy and physiology | 2 | 100.00 |
| qat_w4 | scienceqa | by_category/Ancient Egypt and Kush | 1 | 100.00 |
| qat_w4 | scienceqa | by_category/Ancient Mesopotamia | 2 | 100.00 |
| qat_w4 | scienceqa | by_category/Animals | 8 | 87.50 |
| qat_w4 | scienceqa | by_category/Astronomy | 37 | 48.65 |
| qat_w4 | scienceqa | by_category/Atoms and molecules | 64 | 93.75 |
| qat_w4 | scienceqa | by_category/Audience, purpose, and tone | 17 | 100.00 |
| qat_w4 | scienceqa | by_category/Author's purpose and tone | 27 | 77.78 |
| qat_w4 | scienceqa | by_category/Banking and finance | 1 | 100.00 |
| qat_w4 | scienceqa | by_category/Basic economic principles | 116 | 93.97 |
| qat_w4 | scienceqa | by_category/Capitalization | 18 | 83.33 |
| qat_w4 | scienceqa | by_category/Categories | 11 | 72.73 |
| qat_w4 | scienceqa | by_category/Cells | 17 | 82.35 |
| qat_w4 | scienceqa | by_category/Chemical reactions | 43 | 86.05 |
| qat_w4 | scienceqa | by_category/Cities | 12 | 66.67 |
| qat_w4 | scienceqa | by_category/Classification | 161 | 90.68 |
| qat_w4 | scienceqa | by_category/Classification and scientific names | 46 | 89.13 |
| qat_w4 | scienceqa | by_category/Climate change | 1 | 100.00 |
| qat_w4 | scienceqa | by_category/Colonial America | 38 | 81.58 |
| qat_w4 | scienceqa | by_category/Comprehension strategies | 5 | 100.00 |
| qat_w4 | scienceqa | by_category/Conservation and natural resources | 1 | 100.00 |
| qat_w4 | scienceqa | by_category/Context clues | 5 | 80.00 |
| qat_w4 | scienceqa | by_category/Creative techniques | 19 | 68.42 |
| qat_w4 | scienceqa | by_category/Cultural celebrations | 2 | 100.00 |
| qat_w4 | scienceqa | by_category/Descriptive details | 3 | 100.00 |
| qat_w4 | scienceqa | by_category/Designing experiments | 134 | 95.52 |
| qat_w4 | scienceqa | by_category/Developing and supporting arguments | 101 | 81.19 |
| qat_w4 | scienceqa | by_category/Domain-specific vocabulary | 4 | 50.00 |
| qat_w4 | scienceqa | by_category/Early 19th century American history | 2 | 50.00 |
| qat_w4 | scienceqa | by_category/Earth events | 18 | 94.44 |
| qat_w4 | scienceqa | by_category/Ecological interactions | 24 | 70.83 |
| qat_w4 | scienceqa | by_category/Economics | 11 | 81.82 |
| qat_w4 | scienceqa | by_category/Ecosystems | 78 | 89.74 |
| qat_w4 | scienceqa | by_category/Editing and revising | 9 | 33.33 |
| qat_w4 | scienceqa | by_category/Electricity | 1 | 100.00 |
| qat_w4 | scienceqa | by_category/Engineering practices | 55 | 100.00 |
| qat_w4 | scienceqa | by_category/English colonies in North America | 39 | 87.18 |
| qat_w4 | scienceqa | by_category/Force and motion | 58 | 79.31 |
| qat_w4 | scienceqa | by_category/Formatting | 73 | 79.45 |
| qat_w4 | scienceqa | by_category/Fossils | 22 | 59.09 |
| qat_w4 | scienceqa | by_category/Genes to traits | 195 | 84.10 |
| qat_w4 | scienceqa | by_category/Geography | 46 | 95.65 |
| qat_w4 | scienceqa | by_category/Government | 16 | 81.25 |
| qat_w4 | scienceqa | by_category/Greece | 1 | 100.00 |
| qat_w4 | scienceqa | by_category/Heat and thermal energy | 42 | 88.10 |
| qat_w4 | scienceqa | by_category/Heredity | 41 | 100.00 |
| qat_w4 | scienceqa | by_category/Historical figures | 2 | 100.00 |
| qat_w4 | scienceqa | by_category/Independent reading comprehension | 5 | 100.00 |
| qat_w4 | scienceqa | by_category/Informational texts: level 1 | 5 | 100.00 |
| qat_w4 | scienceqa | by_category/Kinetic and potential energy | 19 | 89.47 |
| qat_w4 | scienceqa | by_category/Literary devices | 267 | 80.52 |
| qat_w4 | scienceqa | by_category/Magnets | 110 | 58.18 |
| qat_w4 | scienceqa | by_category/Maps | 46 | 91.30 |
| qat_w4 | scienceqa | by_category/Materials | 167 | 95.21 |
| qat_w4 | scienceqa | by_category/Medieval Asia | 2 | 50.00 |
| qat_w4 | scienceqa | by_category/Mixtures | 6 | 83.33 |
| qat_w4 | scienceqa | by_category/Oceania: geography | 51 | 78.43 |
| qat_w4 | scienceqa | by_category/Oceans and continents | 19 | 100.00 |
| qat_w4 | scienceqa | by_category/Oceans and continents	 | 4 | 100.00 |
| qat_w4 | scienceqa | by_category/Opinion writing | 4 | 100.00 |
| qat_w4 | scienceqa | by_category/Particle motion and energy | 41 | 85.37 |
| qat_w4 | scienceqa | by_category/Persuasive strategies | 11 | 90.91 |
| qat_w4 | scienceqa | by_category/Photosynthesis | 4 | 100.00 |
| qat_w4 | scienceqa | by_category/Phrases and clauses | 32 | 43.75 |
| qat_w4 | scienceqa | by_category/Physical Geography | 51 | 84.31 |
| qat_w4 | scienceqa | by_category/Physical and chemical change | 52 | 86.54 |
| qat_w4 | scienceqa | by_category/Plant reproduction | 6 | 66.67 |
| qat_w4 | scienceqa | by_category/Plants | 12 | 91.67 |
| qat_w4 | scienceqa | by_category/Plate tectonics | 3 | 100.00 |
| qat_w4 | scienceqa | by_category/Poetry elements | 4 | 50.00 |
| qat_w4 | scienceqa | by_category/Pronouns | 37 | 64.86 |
| qat_w4 | scienceqa | by_category/Pronouns and antecedents | 18 | 61.11 |
| qat_w4 | scienceqa | by_category/Read-alone texts | 2 | 100.00 |
| qat_w4 | scienceqa | by_category/Reference skills | 140 | 49.29 |
| qat_w4 | scienceqa | by_category/Research skills | 4 | 75.00 |
| qat_w4 | scienceqa | by_category/Rhyming | 17 | 47.06 |
| qat_w4 | scienceqa | by_category/Rocks and minerals | 46 | 93.48 |
| qat_w4 | scienceqa | by_category/Scientific names | 32 | 100.00 |
| qat_w4 | scienceqa | by_category/Sentences, fragments, and run-ons | 135 | 80.74 |
| qat_w4 | scienceqa | by_category/Shades of meaning | 10 | 90.00 |
| qat_w4 | scienceqa | by_category/Short and long vowels | 5 | 80.00 |
| qat_w4 | scienceqa | by_category/Social studies skills | 12 | 100.00 |
| qat_w4 | scienceqa | by_category/Solutions | 54 | 68.52 |
| qat_w4 | scienceqa | by_category/State capitals | 312 | 93.27 |
| qat_w4 | scienceqa | by_category/States | 18 | 100.00 |
| qat_w4 | scienceqa | by_category/States of matter | 47 | 89.36 |
| qat_w4 | scienceqa | by_category/Supply and demand | 27 | 66.67 |
| qat_w4 | scienceqa | by_category/Text structure | 2 | 100.00 |
| qat_w4 | scienceqa | by_category/The American Revolution | 4 | 75.00 |
| qat_w4 | scienceqa | by_category/The Americas: geography | 45 | 73.33 |
| qat_w4 | scienceqa | by_category/The Antebellum period | 1 | 100.00 |
| qat_w4 | scienceqa | by_category/The Civil War and Reconstruction | 2 | 100.00 |
| qat_w4 | scienceqa | by_category/Thermal energy | 40 | 72.50 |
| qat_w4 | scienceqa | by_category/Topographic maps | 1 | 100.00 |
| qat_w4 | scienceqa | by_category/Traits and heredity | 81 | 98.77 |
| qat_w4 | scienceqa | by_category/Units and measurement | 200 | 86.50 |
| qat_w4 | scienceqa | by_category/Velocity, acceleration, and forces | 86 | 51.16 |
| qat_w4 | scienceqa | by_category/Verb tense | 52 | 84.62 |
| qat_w4 | scienceqa | by_category/Visual elements | 9 | 77.78 |
| qat_w4 | scienceqa | by_category/Water cycle | 2 | 100.00 |
| qat_w4 | scienceqa | by_category/Weather and climate | 97 | 70.10 |
| qat_w4 | scienceqa | by_category/Word usage and nuance | 45 | 82.22 |
| qat_w4 | scienceqa | by_category/World religions | 1 | 100.00 |
| qat_w4 | scienceqa | by_modality/image | 2017 | 84.28 |
| qat_w4 | scienceqa | by_modality/text | 2224 | 81.61 |
| qat_w4 | scienceqa | by_subject/language science | 1100 | 74.45 |
| qat_w4 | scienceqa | by_subject/natural science | 2252 | 84.64 |
| qat_w4 | scienceqa | by_subject/social science | 889 | 88.86 |
| qat_w8 | scienceqa | by_category/20th century American history | 4 | 75.00 |
| qat_w8 | scienceqa | by_category/Adaptations | 98 | 93.88 |
| qat_w8 | scienceqa | by_category/Age of Exploration | 1 | 100.00 |
| qat_w8 | scienceqa | by_category/Analyzing literature | 4 | 100.00 |
| qat_w8 | scienceqa | by_category/Anatomy and physiology | 2 | 100.00 |
| qat_w8 | scienceqa | by_category/Ancient Egypt and Kush | 1 | 0.00 |
| qat_w8 | scienceqa | by_category/Ancient Mesopotamia | 2 | 100.00 |
| qat_w8 | scienceqa | by_category/Animals | 8 | 100.00 |
| qat_w8 | scienceqa | by_category/Astronomy | 37 | 64.86 |
| qat_w8 | scienceqa | by_category/Atoms and molecules | 64 | 100.00 |
| qat_w8 | scienceqa | by_category/Audience, purpose, and tone | 17 | 100.00 |
| qat_w8 | scienceqa | by_category/Author's purpose and tone | 27 | 74.07 |
| qat_w8 | scienceqa | by_category/Banking and finance | 1 | 100.00 |
| qat_w8 | scienceqa | by_category/Basic economic principles | 116 | 97.41 |
| qat_w8 | scienceqa | by_category/Capitalization | 18 | 83.33 |
| qat_w8 | scienceqa | by_category/Categories | 11 | 81.82 |
| qat_w8 | scienceqa | by_category/Cells | 17 | 88.24 |
| qat_w8 | scienceqa | by_category/Chemical reactions | 43 | 81.40 |
| qat_w8 | scienceqa | by_category/Cities | 12 | 66.67 |
| qat_w8 | scienceqa | by_category/Classification | 161 | 94.41 |
| qat_w8 | scienceqa | by_category/Classification and scientific names | 46 | 95.65 |
| qat_w8 | scienceqa | by_category/Climate change | 1 | 100.00 |
| qat_w8 | scienceqa | by_category/Colonial America | 38 | 89.47 |
| qat_w8 | scienceqa | by_category/Comprehension strategies | 5 | 100.00 |
| qat_w8 | scienceqa | by_category/Conservation and natural resources | 1 | 100.00 |
| qat_w8 | scienceqa | by_category/Context clues | 5 | 60.00 |
| qat_w8 | scienceqa | by_category/Creative techniques | 19 | 73.68 |
| qat_w8 | scienceqa | by_category/Cultural celebrations | 2 | 100.00 |
| qat_w8 | scienceqa | by_category/Descriptive details | 3 | 100.00 |
| qat_w8 | scienceqa | by_category/Designing experiments | 134 | 99.25 |
| qat_w8 | scienceqa | by_category/Developing and supporting arguments | 101 | 83.17 |
| qat_w8 | scienceqa | by_category/Domain-specific vocabulary | 4 | 75.00 |
| qat_w8 | scienceqa | by_category/Early 19th century American history | 2 | 0.00 |
| qat_w8 | scienceqa | by_category/Earth events | 18 | 100.00 |
| qat_w8 | scienceqa | by_category/Ecological interactions | 24 | 66.67 |
| qat_w8 | scienceqa | by_category/Economics | 11 | 81.82 |
| qat_w8 | scienceqa | by_category/Ecosystems | 78 | 88.46 |
| qat_w8 | scienceqa | by_category/Editing and revising | 9 | 44.44 |
| qat_w8 | scienceqa | by_category/Electricity | 1 | 100.00 |
| qat_w8 | scienceqa | by_category/Engineering practices | 55 | 100.00 |
| qat_w8 | scienceqa | by_category/English colonies in North America | 39 | 92.31 |
| qat_w8 | scienceqa | by_category/Force and motion | 58 | 89.66 |
| qat_w8 | scienceqa | by_category/Formatting | 73 | 89.04 |
| qat_w8 | scienceqa | by_category/Fossils | 22 | 77.27 |
| qat_w8 | scienceqa | by_category/Genes to traits | 195 | 86.15 |
| qat_w8 | scienceqa | by_category/Geography | 46 | 95.65 |
| qat_w8 | scienceqa | by_category/Government | 16 | 87.50 |
| qat_w8 | scienceqa | by_category/Greece | 1 | 100.00 |
| qat_w8 | scienceqa | by_category/Heat and thermal energy | 42 | 90.48 |
| qat_w8 | scienceqa | by_category/Heredity | 41 | 100.00 |
| qat_w8 | scienceqa | by_category/Historical figures | 2 | 100.00 |
| qat_w8 | scienceqa | by_category/Independent reading comprehension | 5 | 100.00 |
| qat_w8 | scienceqa | by_category/Informational texts: level 1 | 5 | 100.00 |
| qat_w8 | scienceqa | by_category/Kinetic and potential energy | 19 | 89.47 |
| qat_w8 | scienceqa | by_category/Literary devices | 267 | 83.52 |
| qat_w8 | scienceqa | by_category/Magnets | 110 | 55.45 |
| qat_w8 | scienceqa | by_category/Maps | 46 | 91.30 |
| qat_w8 | scienceqa | by_category/Materials | 167 | 93.41 |
| qat_w8 | scienceqa | by_category/Medieval Asia | 2 | 50.00 |
| qat_w8 | scienceqa | by_category/Mixtures | 6 | 83.33 |
| qat_w8 | scienceqa | by_category/Oceania: geography | 51 | 74.51 |
| qat_w8 | scienceqa | by_category/Oceans and continents | 19 | 100.00 |
| qat_w8 | scienceqa | by_category/Oceans and continents	 | 4 | 100.00 |
| qat_w8 | scienceqa | by_category/Opinion writing | 4 | 100.00 |
| qat_w8 | scienceqa | by_category/Particle motion and energy | 41 | 90.24 |
| qat_w8 | scienceqa | by_category/Persuasive strategies | 11 | 81.82 |
| qat_w8 | scienceqa | by_category/Photosynthesis | 4 | 100.00 |
| qat_w8 | scienceqa | by_category/Phrases and clauses | 32 | 59.38 |
| qat_w8 | scienceqa | by_category/Physical Geography | 51 | 92.16 |
| qat_w8 | scienceqa | by_category/Physical and chemical change | 52 | 92.31 |
| qat_w8 | scienceqa | by_category/Plant reproduction | 6 | 66.67 |
| qat_w8 | scienceqa | by_category/Plants | 12 | 91.67 |
| qat_w8 | scienceqa | by_category/Plate tectonics | 3 | 100.00 |
| qat_w8 | scienceqa | by_category/Poetry elements | 4 | 75.00 |
| qat_w8 | scienceqa | by_category/Pronouns | 37 | 70.27 |
| qat_w8 | scienceqa | by_category/Pronouns and antecedents | 18 | 55.56 |
| qat_w8 | scienceqa | by_category/Read-alone texts | 2 | 100.00 |
| qat_w8 | scienceqa | by_category/Reference skills | 140 | 52.14 |
| qat_w8 | scienceqa | by_category/Research skills | 4 | 75.00 |
| qat_w8 | scienceqa | by_category/Rhyming | 17 | 47.06 |
| qat_w8 | scienceqa | by_category/Rocks and minerals | 46 | 84.78 |
| qat_w8 | scienceqa | by_category/Scientific names | 32 | 100.00 |
| qat_w8 | scienceqa | by_category/Sentences, fragments, and run-ons | 135 | 82.22 |
| qat_w8 | scienceqa | by_category/Shades of meaning | 10 | 90.00 |
| qat_w8 | scienceqa | by_category/Short and long vowels | 5 | 80.00 |
| qat_w8 | scienceqa | by_category/Social studies skills | 12 | 100.00 |
| qat_w8 | scienceqa | by_category/Solutions | 54 | 68.52 |
| qat_w8 | scienceqa | by_category/State capitals | 312 | 94.55 |
| qat_w8 | scienceqa | by_category/States | 18 | 100.00 |
| qat_w8 | scienceqa | by_category/States of matter | 47 | 93.62 |
| qat_w8 | scienceqa | by_category/Supply and demand | 27 | 77.78 |
| qat_w8 | scienceqa | by_category/Text structure | 2 | 100.00 |
| qat_w8 | scienceqa | by_category/The American Revolution | 4 | 75.00 |
| qat_w8 | scienceqa | by_category/The Americas: geography | 45 | 73.33 |
| qat_w8 | scienceqa | by_category/The Antebellum period | 1 | 100.00 |
| qat_w8 | scienceqa | by_category/The Civil War and Reconstruction | 2 | 100.00 |
| qat_w8 | scienceqa | by_category/Thermal energy | 40 | 70.00 |
| qat_w8 | scienceqa | by_category/Topographic maps | 1 | 100.00 |
| qat_w8 | scienceqa | by_category/Traits and heredity | 81 | 98.77 |
| qat_w8 | scienceqa | by_category/Units and measurement | 200 | 83.50 |
| qat_w8 | scienceqa | by_category/Velocity, acceleration, and forces | 86 | 56.98 |
| qat_w8 | scienceqa | by_category/Verb tense | 52 | 88.46 |
| qat_w8 | scienceqa | by_category/Visual elements | 9 | 88.89 |
| qat_w8 | scienceqa | by_category/Water cycle | 2 | 100.00 |
| qat_w8 | scienceqa | by_category/Weather and climate | 97 | 78.35 |
| qat_w8 | scienceqa | by_category/Word usage and nuance | 45 | 77.78 |
| qat_w8 | scienceqa | by_category/World religions | 1 | 100.00 |
| qat_w8 | scienceqa | by_modality/image | 2017 | 86.22 |
| qat_w8 | scienceqa | by_modality/text | 2224 | 83.90 |
| qat_w8 | scienceqa | by_subject/language science | 1100 | 77.36 |
| qat_w8 | scienceqa | by_subject/natural science | 2252 | 86.46 |
| qat_w8 | scienceqa | by_subject/social science | 889 | 90.78 |
| released_bf16 | scienceqa | by_category/20th century American history | 4 | 75.00 |
| released_bf16 | scienceqa | by_category/Adaptations | 98 | 96.94 |
| released_bf16 | scienceqa | by_category/Age of Exploration | 1 | 100.00 |
| released_bf16 | scienceqa | by_category/Analyzing literature | 4 | 100.00 |
| released_bf16 | scienceqa | by_category/Anatomy and physiology | 2 | 100.00 |
| released_bf16 | scienceqa | by_category/Ancient Egypt and Kush | 1 | 0.00 |
| released_bf16 | scienceqa | by_category/Ancient Mesopotamia | 2 | 100.00 |
| released_bf16 | scienceqa | by_category/Animals | 8 | 100.00 |
| released_bf16 | scienceqa | by_category/Astronomy | 37 | 67.57 |
| released_bf16 | scienceqa | by_category/Atoms and molecules | 64 | 98.44 |
| released_bf16 | scienceqa | by_category/Audience, purpose, and tone | 17 | 100.00 |
| released_bf16 | scienceqa | by_category/Author's purpose and tone | 27 | 74.07 |
| released_bf16 | scienceqa | by_category/Banking and finance | 1 | 100.00 |
| released_bf16 | scienceqa | by_category/Basic economic principles | 116 | 97.41 |
| released_bf16 | scienceqa | by_category/Capitalization | 18 | 83.33 |
| released_bf16 | scienceqa | by_category/Categories | 11 | 81.82 |
| released_bf16 | scienceqa | by_category/Cells | 17 | 88.24 |
| released_bf16 | scienceqa | by_category/Chemical reactions | 43 | 83.72 |
| released_bf16 | scienceqa | by_category/Cities | 12 | 75.00 |
| released_bf16 | scienceqa | by_category/Classification | 161 | 93.17 |
| released_bf16 | scienceqa | by_category/Classification and scientific names | 46 | 100.00 |
| released_bf16 | scienceqa | by_category/Climate change | 1 | 100.00 |
| released_bf16 | scienceqa | by_category/Colonial America | 38 | 92.11 |
| released_bf16 | scienceqa | by_category/Comprehension strategies | 5 | 100.00 |
| released_bf16 | scienceqa | by_category/Conservation and natural resources | 1 | 100.00 |
| released_bf16 | scienceqa | by_category/Context clues | 5 | 80.00 |
| released_bf16 | scienceqa | by_category/Creative techniques | 19 | 78.95 |
| released_bf16 | scienceqa | by_category/Cultural celebrations | 2 | 50.00 |
| released_bf16 | scienceqa | by_category/Descriptive details | 3 | 100.00 |
| released_bf16 | scienceqa | by_category/Designing experiments | 134 | 100.00 |
| released_bf16 | scienceqa | by_category/Developing and supporting arguments | 101 | 83.17 |
| released_bf16 | scienceqa | by_category/Domain-specific vocabulary | 4 | 100.00 |
| released_bf16 | scienceqa | by_category/Early 19th century American history | 2 | 0.00 |
| released_bf16 | scienceqa | by_category/Earth events | 18 | 94.44 |
| released_bf16 | scienceqa | by_category/Ecological interactions | 24 | 66.67 |
| released_bf16 | scienceqa | by_category/Economics | 11 | 72.73 |
| released_bf16 | scienceqa | by_category/Ecosystems | 78 | 85.90 |
| released_bf16 | scienceqa | by_category/Editing and revising | 9 | 55.56 |
| released_bf16 | scienceqa | by_category/Electricity | 1 | 100.00 |
| released_bf16 | scienceqa | by_category/Engineering practices | 55 | 100.00 |
| released_bf16 | scienceqa | by_category/English colonies in North America | 39 | 94.87 |
| released_bf16 | scienceqa | by_category/Force and motion | 58 | 91.38 |
| released_bf16 | scienceqa | by_category/Formatting | 73 | 84.93 |
| released_bf16 | scienceqa | by_category/Fossils | 22 | 72.73 |
| released_bf16 | scienceqa | by_category/Genes to traits | 195 | 88.72 |
| released_bf16 | scienceqa | by_category/Geography | 46 | 95.65 |
| released_bf16 | scienceqa | by_category/Government | 16 | 87.50 |
| released_bf16 | scienceqa | by_category/Greece | 1 | 100.00 |
| released_bf16 | scienceqa | by_category/Heat and thermal energy | 42 | 95.24 |
| released_bf16 | scienceqa | by_category/Heredity | 41 | 100.00 |
| released_bf16 | scienceqa | by_category/Historical figures | 2 | 100.00 |
| released_bf16 | scienceqa | by_category/Independent reading comprehension | 5 | 100.00 |
| released_bf16 | scienceqa | by_category/Informational texts: level 1 | 5 | 100.00 |
| released_bf16 | scienceqa | by_category/Kinetic and potential energy | 19 | 89.47 |
| released_bf16 | scienceqa | by_category/Literary devices | 267 | 83.52 |
| released_bf16 | scienceqa | by_category/Magnets | 110 | 53.64 |
| released_bf16 | scienceqa | by_category/Maps | 46 | 95.65 |
| released_bf16 | scienceqa | by_category/Materials | 167 | 91.62 |
| released_bf16 | scienceqa | by_category/Medieval Asia | 2 | 50.00 |
| released_bf16 | scienceqa | by_category/Mixtures | 6 | 100.00 |
| released_bf16 | scienceqa | by_category/Oceania: geography | 51 | 72.55 |
| released_bf16 | scienceqa | by_category/Oceans and continents | 19 | 100.00 |
| released_bf16 | scienceqa | by_category/Oceans and continents	 | 4 | 100.00 |
| released_bf16 | scienceqa | by_category/Opinion writing | 4 | 100.00 |
| released_bf16 | scienceqa | by_category/Particle motion and energy | 41 | 90.24 |
| released_bf16 | scienceqa | by_category/Persuasive strategies | 11 | 90.91 |
| released_bf16 | scienceqa | by_category/Photosynthesis | 4 | 100.00 |
| released_bf16 | scienceqa | by_category/Phrases and clauses | 32 | 56.25 |
| released_bf16 | scienceqa | by_category/Physical Geography | 51 | 92.16 |
| released_bf16 | scienceqa | by_category/Physical and chemical change | 52 | 94.23 |
| released_bf16 | scienceqa | by_category/Plant reproduction | 6 | 66.67 |
| released_bf16 | scienceqa | by_category/Plants | 12 | 100.00 |
| released_bf16 | scienceqa | by_category/Plate tectonics | 3 | 100.00 |
| released_bf16 | scienceqa | by_category/Poetry elements | 4 | 0.00 |
| released_bf16 | scienceqa | by_category/Pronouns | 37 | 64.86 |
| released_bf16 | scienceqa | by_category/Pronouns and antecedents | 18 | 66.67 |
| released_bf16 | scienceqa | by_category/Read-alone texts | 2 | 100.00 |
| released_bf16 | scienceqa | by_category/Reference skills | 140 | 52.86 |
| released_bf16 | scienceqa | by_category/Research skills | 4 | 75.00 |
| released_bf16 | scienceqa | by_category/Rhyming | 17 | 52.94 |
| released_bf16 | scienceqa | by_category/Rocks and minerals | 46 | 91.30 |
| released_bf16 | scienceqa | by_category/Scientific names | 32 | 100.00 |
| released_bf16 | scienceqa | by_category/Sentences, fragments, and run-ons | 135 | 78.52 |
| released_bf16 | scienceqa | by_category/Shades of meaning | 10 | 90.00 |
| released_bf16 | scienceqa | by_category/Short and long vowels | 5 | 80.00 |
| released_bf16 | scienceqa | by_category/Social studies skills | 12 | 100.00 |
| released_bf16 | scienceqa | by_category/Solutions | 54 | 70.37 |
| released_bf16 | scienceqa | by_category/State capitals | 312 | 95.19 |
| released_bf16 | scienceqa | by_category/States | 18 | 100.00 |
| released_bf16 | scienceqa | by_category/States of matter | 47 | 89.36 |
| released_bf16 | scienceqa | by_category/Supply and demand | 27 | 74.07 |
| released_bf16 | scienceqa | by_category/Text structure | 2 | 100.00 |
| released_bf16 | scienceqa | by_category/The American Revolution | 4 | 100.00 |
| released_bf16 | scienceqa | by_category/The Americas: geography | 45 | 68.89 |
| released_bf16 | scienceqa | by_category/The Antebellum period | 1 | 100.00 |
| released_bf16 | scienceqa | by_category/The Civil War and Reconstruction | 2 | 100.00 |
| released_bf16 | scienceqa | by_category/Thermal energy | 40 | 62.50 |
| released_bf16 | scienceqa | by_category/Topographic maps | 1 | 100.00 |
| released_bf16 | scienceqa | by_category/Traits and heredity | 81 | 100.00 |
| released_bf16 | scienceqa | by_category/Units and measurement | 200 | 86.50 |
| released_bf16 | scienceqa | by_category/Velocity, acceleration, and forces | 86 | 50.00 |
| released_bf16 | scienceqa | by_category/Verb tense | 52 | 88.46 |
| released_bf16 | scienceqa | by_category/Visual elements | 9 | 88.89 |
| released_bf16 | scienceqa | by_category/Water cycle | 2 | 100.00 |
| released_bf16 | scienceqa | by_category/Weather and climate | 97 | 78.35 |
| released_bf16 | scienceqa | by_category/Word usage and nuance | 45 | 71.11 |
| released_bf16 | scienceqa | by_category/World religions | 1 | 100.00 |
| released_bf16 | scienceqa | by_modality/image | 2017 | 86.07 |
| released_bf16 | scienceqa | by_modality/text | 2224 | 84.08 |
| released_bf16 | scienceqa | by_subject/language science | 1100 | 76.64 |
| released_bf16 | scienceqa | by_subject/natural science | 2252 | 86.77 |
| released_bf16 | scienceqa | by_subject/social science | 889 | 91.00 |
| released_grace_w4 | scienceqa | by_category/20th century American history | 4 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Adaptations | 98 | 94.90 |
| released_grace_w4 | scienceqa | by_category/Age of Exploration | 1 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Analyzing literature | 4 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Anatomy and physiology | 2 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Ancient Egypt and Kush | 1 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Ancient Mesopotamia | 2 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Animals | 8 | 87.50 |
| released_grace_w4 | scienceqa | by_category/Astronomy | 37 | 64.86 |
| released_grace_w4 | scienceqa | by_category/Atoms and molecules | 64 | 96.88 |
| released_grace_w4 | scienceqa | by_category/Audience, purpose, and tone | 17 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Author's purpose and tone | 27 | 77.78 |
| released_grace_w4 | scienceqa | by_category/Banking and finance | 1 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Basic economic principles | 116 | 89.66 |
| released_grace_w4 | scienceqa | by_category/Capitalization | 18 | 77.78 |
| released_grace_w4 | scienceqa | by_category/Categories | 11 | 54.55 |
| released_grace_w4 | scienceqa | by_category/Cells | 17 | 88.24 |
| released_grace_w4 | scienceqa | by_category/Chemical reactions | 43 | 69.77 |
| released_grace_w4 | scienceqa | by_category/Cities | 12 | 58.33 |
| released_grace_w4 | scienceqa | by_category/Classification | 161 | 93.79 |
| released_grace_w4 | scienceqa | by_category/Classification and scientific names | 46 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Climate change | 1 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Colonial America | 38 | 86.84 |
| released_grace_w4 | scienceqa | by_category/Comprehension strategies | 5 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Conservation and natural resources | 1 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Context clues | 5 | 80.00 |
| released_grace_w4 | scienceqa | by_category/Creative techniques | 19 | 63.16 |
| released_grace_w4 | scienceqa | by_category/Cultural celebrations | 2 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Descriptive details | 3 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Designing experiments | 134 | 99.25 |
| released_grace_w4 | scienceqa | by_category/Developing and supporting arguments | 101 | 88.12 |
| released_grace_w4 | scienceqa | by_category/Domain-specific vocabulary | 4 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Early 19th century American history | 2 | 50.00 |
| released_grace_w4 | scienceqa | by_category/Earth events | 18 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Ecological interactions | 24 | 70.83 |
| released_grace_w4 | scienceqa | by_category/Economics | 11 | 63.64 |
| released_grace_w4 | scienceqa | by_category/Ecosystems | 78 | 88.46 |
| released_grace_w4 | scienceqa | by_category/Editing and revising | 9 | 55.56 |
| released_grace_w4 | scienceqa | by_category/Electricity | 1 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Engineering practices | 55 | 100.00 |
| released_grace_w4 | scienceqa | by_category/English colonies in North America | 39 | 94.87 |
| released_grace_w4 | scienceqa | by_category/Force and motion | 58 | 82.76 |
| released_grace_w4 | scienceqa | by_category/Formatting | 73 | 84.93 |
| released_grace_w4 | scienceqa | by_category/Fossils | 22 | 68.18 |
| released_grace_w4 | scienceqa | by_category/Genes to traits | 195 | 83.59 |
| released_grace_w4 | scienceqa | by_category/Geography | 46 | 89.13 |
| released_grace_w4 | scienceqa | by_category/Government | 16 | 87.50 |
| released_grace_w4 | scienceqa | by_category/Greece | 1 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Heat and thermal energy | 42 | 90.48 |
| released_grace_w4 | scienceqa | by_category/Heredity | 41 | 80.49 |
| released_grace_w4 | scienceqa | by_category/Historical figures | 2 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Independent reading comprehension | 5 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Informational texts: level 1 | 5 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Kinetic and potential energy | 19 | 84.21 |
| released_grace_w4 | scienceqa | by_category/Literary devices | 267 | 82.77 |
| released_grace_w4 | scienceqa | by_category/Magnets | 110 | 61.82 |
| released_grace_w4 | scienceqa | by_category/Maps | 46 | 97.83 |
| released_grace_w4 | scienceqa | by_category/Materials | 167 | 94.61 |
| released_grace_w4 | scienceqa | by_category/Medieval Asia | 2 | 50.00 |
| released_grace_w4 | scienceqa | by_category/Mixtures | 6 | 83.33 |
| released_grace_w4 | scienceqa | by_category/Oceania: geography | 51 | 72.55 |
| released_grace_w4 | scienceqa | by_category/Oceans and continents | 19 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Oceans and continents	 | 4 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Opinion writing | 4 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Particle motion and energy | 41 | 87.80 |
| released_grace_w4 | scienceqa | by_category/Persuasive strategies | 11 | 90.91 |
| released_grace_w4 | scienceqa | by_category/Photosynthesis | 4 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Phrases and clauses | 32 | 81.25 |
| released_grace_w4 | scienceqa | by_category/Physical Geography | 51 | 86.27 |
| released_grace_w4 | scienceqa | by_category/Physical and chemical change | 52 | 86.54 |
| released_grace_w4 | scienceqa | by_category/Plant reproduction | 6 | 83.33 |
| released_grace_w4 | scienceqa | by_category/Plants | 12 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Plate tectonics | 3 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Poetry elements | 4 | 25.00 |
| released_grace_w4 | scienceqa | by_category/Pronouns | 37 | 59.46 |
| released_grace_w4 | scienceqa | by_category/Pronouns and antecedents | 18 | 66.67 |
| released_grace_w4 | scienceqa | by_category/Read-alone texts | 2 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Reference skills | 140 | 52.14 |
| released_grace_w4 | scienceqa | by_category/Research skills | 4 | 75.00 |
| released_grace_w4 | scienceqa | by_category/Rhyming | 17 | 41.18 |
| released_grace_w4 | scienceqa | by_category/Rocks and minerals | 46 | 95.65 |
| released_grace_w4 | scienceqa | by_category/Scientific names | 32 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Sentences, fragments, and run-ons | 135 | 84.44 |
| released_grace_w4 | scienceqa | by_category/Shades of meaning | 10 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Short and long vowels | 5 | 80.00 |
| released_grace_w4 | scienceqa | by_category/Social studies skills | 12 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Solutions | 54 | 74.07 |
| released_grace_w4 | scienceqa | by_category/State capitals | 312 | 91.03 |
| released_grace_w4 | scienceqa | by_category/States | 18 | 100.00 |
| released_grace_w4 | scienceqa | by_category/States of matter | 47 | 93.62 |
| released_grace_w4 | scienceqa | by_category/Supply and demand | 27 | 74.07 |
| released_grace_w4 | scienceqa | by_category/Text structure | 2 | 100.00 |
| released_grace_w4 | scienceqa | by_category/The American Revolution | 4 | 100.00 |
| released_grace_w4 | scienceqa | by_category/The Americas: geography | 45 | 68.89 |
| released_grace_w4 | scienceqa | by_category/The Antebellum period | 1 | 100.00 |
| released_grace_w4 | scienceqa | by_category/The Civil War and Reconstruction | 2 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Thermal energy | 40 | 62.50 |
| released_grace_w4 | scienceqa | by_category/Topographic maps | 1 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Traits and heredity | 81 | 83.95 |
| released_grace_w4 | scienceqa | by_category/Units and measurement | 200 | 88.00 |
| released_grace_w4 | scienceqa | by_category/Velocity, acceleration, and forces | 86 | 50.00 |
| released_grace_w4 | scienceqa | by_category/Verb tense | 52 | 76.92 |
| released_grace_w4 | scienceqa | by_category/Visual elements | 9 | 88.89 |
| released_grace_w4 | scienceqa | by_category/Water cycle | 2 | 100.00 |
| released_grace_w4 | scienceqa | by_category/Weather and climate | 97 | 71.13 |
| released_grace_w4 | scienceqa | by_category/Word usage and nuance | 45 | 77.78 |
| released_grace_w4 | scienceqa | by_category/World religions | 1 | 100.00 |
| released_grace_w4 | scienceqa | by_modality/image | 2017 | 85.52 |
| released_grace_w4 | scienceqa | by_modality/text | 2224 | 82.01 |
| released_grace_w4 | scienceqa | by_subject/language science | 1100 | 77.27 |
| released_grace_w4 | scienceqa | by_subject/natural science | 2252 | 85.17 |
| released_grace_w4 | scienceqa | by_subject/social science | 889 | 87.85 |
| released_grace_w8 | scienceqa | by_category/20th century American history | 4 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Adaptations | 98 | 94.90 |
| released_grace_w8 | scienceqa | by_category/Age of Exploration | 1 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Analyzing literature | 4 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Anatomy and physiology | 2 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Ancient Egypt and Kush | 1 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Ancient Mesopotamia | 2 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Animals | 8 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Astronomy | 37 | 54.05 |
| released_grace_w8 | scienceqa | by_category/Atoms and molecules | 64 | 95.31 |
| released_grace_w8 | scienceqa | by_category/Audience, purpose, and tone | 17 | 94.12 |
| released_grace_w8 | scienceqa | by_category/Author's purpose and tone | 27 | 77.78 |
| released_grace_w8 | scienceqa | by_category/Banking and finance | 1 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Basic economic principles | 116 | 89.66 |
| released_grace_w8 | scienceqa | by_category/Capitalization | 18 | 72.22 |
| released_grace_w8 | scienceqa | by_category/Categories | 11 | 72.73 |
| released_grace_w8 | scienceqa | by_category/Cells | 17 | 82.35 |
| released_grace_w8 | scienceqa | by_category/Chemical reactions | 43 | 74.42 |
| released_grace_w8 | scienceqa | by_category/Cities | 12 | 58.33 |
| released_grace_w8 | scienceqa | by_category/Classification | 161 | 94.41 |
| released_grace_w8 | scienceqa | by_category/Classification and scientific names | 46 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Climate change | 1 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Colonial America | 38 | 89.47 |
| released_grace_w8 | scienceqa | by_category/Comprehension strategies | 5 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Conservation and natural resources | 1 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Context clues | 5 | 80.00 |
| released_grace_w8 | scienceqa | by_category/Creative techniques | 19 | 68.42 |
| released_grace_w8 | scienceqa | by_category/Cultural celebrations | 2 | 50.00 |
| released_grace_w8 | scienceqa | by_category/Descriptive details | 3 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Designing experiments | 134 | 97.01 |
| released_grace_w8 | scienceqa | by_category/Developing and supporting arguments | 101 | 88.12 |
| released_grace_w8 | scienceqa | by_category/Domain-specific vocabulary | 4 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Early 19th century American history | 2 | 50.00 |
| released_grace_w8 | scienceqa | by_category/Earth events | 18 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Ecological interactions | 24 | 62.50 |
| released_grace_w8 | scienceqa | by_category/Economics | 11 | 72.73 |
| released_grace_w8 | scienceqa | by_category/Ecosystems | 78 | 88.46 |
| released_grace_w8 | scienceqa | by_category/Editing and revising | 9 | 55.56 |
| released_grace_w8 | scienceqa | by_category/Electricity | 1 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Engineering practices | 55 | 100.00 |
| released_grace_w8 | scienceqa | by_category/English colonies in North America | 39 | 94.87 |
| released_grace_w8 | scienceqa | by_category/Force and motion | 58 | 84.48 |
| released_grace_w8 | scienceqa | by_category/Formatting | 73 | 80.82 |
| released_grace_w8 | scienceqa | by_category/Fossils | 22 | 68.18 |
| released_grace_w8 | scienceqa | by_category/Genes to traits | 195 | 85.13 |
| released_grace_w8 | scienceqa | by_category/Geography | 46 | 91.30 |
| released_grace_w8 | scienceqa | by_category/Government | 16 | 87.50 |
| released_grace_w8 | scienceqa | by_category/Greece | 1 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Heat and thermal energy | 42 | 90.48 |
| released_grace_w8 | scienceqa | by_category/Heredity | 41 | 87.80 |
| released_grace_w8 | scienceqa | by_category/Historical figures | 2 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Independent reading comprehension | 5 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Informational texts: level 1 | 5 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Kinetic and potential energy | 19 | 84.21 |
| released_grace_w8 | scienceqa | by_category/Literary devices | 267 | 80.52 |
| released_grace_w8 | scienceqa | by_category/Magnets | 110 | 61.82 |
| released_grace_w8 | scienceqa | by_category/Maps | 46 | 80.43 |
| released_grace_w8 | scienceqa | by_category/Materials | 167 | 95.81 |
| released_grace_w8 | scienceqa | by_category/Medieval Asia | 2 | 50.00 |
| released_grace_w8 | scienceqa | by_category/Mixtures | 6 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Oceania: geography | 51 | 72.55 |
| released_grace_w8 | scienceqa | by_category/Oceans and continents | 19 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Oceans and continents	 | 4 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Opinion writing | 4 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Particle motion and energy | 41 | 90.24 |
| released_grace_w8 | scienceqa | by_category/Persuasive strategies | 11 | 81.82 |
| released_grace_w8 | scienceqa | by_category/Photosynthesis | 4 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Phrases and clauses | 32 | 53.12 |
| released_grace_w8 | scienceqa | by_category/Physical Geography | 51 | 76.47 |
| released_grace_w8 | scienceqa | by_category/Physical and chemical change | 52 | 82.69 |
| released_grace_w8 | scienceqa | by_category/Plant reproduction | 6 | 66.67 |
| released_grace_w8 | scienceqa | by_category/Plants | 12 | 91.67 |
| released_grace_w8 | scienceqa | by_category/Plate tectonics | 3 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Poetry elements | 4 | 25.00 |
| released_grace_w8 | scienceqa | by_category/Pronouns | 37 | 59.46 |
| released_grace_w8 | scienceqa | by_category/Pronouns and antecedents | 18 | 66.67 |
| released_grace_w8 | scienceqa | by_category/Read-alone texts | 2 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Reference skills | 140 | 50.71 |
| released_grace_w8 | scienceqa | by_category/Research skills | 4 | 75.00 |
| released_grace_w8 | scienceqa | by_category/Rhyming | 17 | 35.29 |
| released_grace_w8 | scienceqa | by_category/Rocks and minerals | 46 | 97.83 |
| released_grace_w8 | scienceqa | by_category/Scientific names | 32 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Sentences, fragments, and run-ons | 135 | 82.96 |
| released_grace_w8 | scienceqa | by_category/Shades of meaning | 10 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Short and long vowels | 5 | 80.00 |
| released_grace_w8 | scienceqa | by_category/Social studies skills | 12 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Solutions | 54 | 70.37 |
| released_grace_w8 | scienceqa | by_category/State capitals | 312 | 89.74 |
| released_grace_w8 | scienceqa | by_category/States | 18 | 100.00 |
| released_grace_w8 | scienceqa | by_category/States of matter | 47 | 87.23 |
| released_grace_w8 | scienceqa | by_category/Supply and demand | 27 | 74.07 |
| released_grace_w8 | scienceqa | by_category/Text structure | 2 | 100.00 |
| released_grace_w8 | scienceqa | by_category/The American Revolution | 4 | 100.00 |
| released_grace_w8 | scienceqa | by_category/The Americas: geography | 45 | 73.33 |
| released_grace_w8 | scienceqa | by_category/The Antebellum period | 1 | 100.00 |
| released_grace_w8 | scienceqa | by_category/The Civil War and Reconstruction | 2 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Thermal energy | 40 | 62.50 |
| released_grace_w8 | scienceqa | by_category/Topographic maps | 1 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Traits and heredity | 81 | 88.89 |
| released_grace_w8 | scienceqa | by_category/Units and measurement | 200 | 87.00 |
| released_grace_w8 | scienceqa | by_category/Velocity, acceleration, and forces | 86 | 59.30 |
| released_grace_w8 | scienceqa | by_category/Verb tense | 52 | 73.08 |
| released_grace_w8 | scienceqa | by_category/Visual elements | 9 | 88.89 |
| released_grace_w8 | scienceqa | by_category/Water cycle | 2 | 100.00 |
| released_grace_w8 | scienceqa | by_category/Weather and climate | 97 | 72.16 |
| released_grace_w8 | scienceqa | by_category/Word usage and nuance | 45 | 77.78 |
| released_grace_w8 | scienceqa | by_category/World religions | 1 | 100.00 |
| released_grace_w8 | scienceqa | by_modality/image | 2017 | 84.88 |
| released_grace_w8 | scienceqa | by_modality/text | 2224 | 81.21 |
| released_grace_w8 | scienceqa | by_subject/language science | 1100 | 75.00 |
| released_grace_w8 | scienceqa | by_subject/natural science | 2252 | 85.48 |
| released_grace_w8 | scienceqa | by_subject/social science | 889 | 86.39 |
| sft_2b | scienceqa | by_category/20th century American history | 4 | 100.00 |
| sft_2b | scienceqa | by_category/Adaptations | 98 | 89.80 |
| sft_2b | scienceqa | by_category/Age of Exploration | 1 | 100.00 |
| sft_2b | scienceqa | by_category/Analyzing literature | 4 | 100.00 |
| sft_2b | scienceqa | by_category/Anatomy and physiology | 2 | 100.00 |
| sft_2b | scienceqa | by_category/Ancient Egypt and Kush | 1 | 0.00 |
| sft_2b | scienceqa | by_category/Ancient Mesopotamia | 2 | 100.00 |
| sft_2b | scienceqa | by_category/Animals | 8 | 100.00 |
| sft_2b | scienceqa | by_category/Astronomy | 37 | 62.16 |
| sft_2b | scienceqa | by_category/Atoms and molecules | 64 | 98.44 |
| sft_2b | scienceqa | by_category/Audience, purpose, and tone | 17 | 94.12 |
| sft_2b | scienceqa | by_category/Author's purpose and tone | 27 | 74.07 |
| sft_2b | scienceqa | by_category/Banking and finance | 1 | 100.00 |
| sft_2b | scienceqa | by_category/Basic economic principles | 116 | 89.66 |
| sft_2b | scienceqa | by_category/Capitalization | 18 | 66.67 |
| sft_2b | scienceqa | by_category/Categories | 11 | 63.64 |
| sft_2b | scienceqa | by_category/Cells | 17 | 76.47 |
| sft_2b | scienceqa | by_category/Chemical reactions | 43 | 76.74 |
| sft_2b | scienceqa | by_category/Cities | 12 | 58.33 |
| sft_2b | scienceqa | by_category/Classification | 161 | 97.52 |
| sft_2b | scienceqa | by_category/Classification and scientific names | 46 | 100.00 |
| sft_2b | scienceqa | by_category/Climate change | 1 | 100.00 |
| sft_2b | scienceqa | by_category/Colonial America | 38 | 86.84 |
| sft_2b | scienceqa | by_category/Comprehension strategies | 5 | 100.00 |
| sft_2b | scienceqa | by_category/Conservation and natural resources | 1 | 100.00 |
| sft_2b | scienceqa | by_category/Context clues | 5 | 80.00 |
| sft_2b | scienceqa | by_category/Creative techniques | 19 | 68.42 |
| sft_2b | scienceqa | by_category/Cultural celebrations | 2 | 100.00 |
| sft_2b | scienceqa | by_category/Descriptive details | 3 | 100.00 |
| sft_2b | scienceqa | by_category/Designing experiments | 134 | 96.27 |
| sft_2b | scienceqa | by_category/Developing and supporting arguments | 101 | 88.12 |
| sft_2b | scienceqa | by_category/Domain-specific vocabulary | 4 | 75.00 |
| sft_2b | scienceqa | by_category/Early 19th century American history | 2 | 100.00 |
| sft_2b | scienceqa | by_category/Earth events | 18 | 100.00 |
| sft_2b | scienceqa | by_category/Ecological interactions | 24 | 58.33 |
| sft_2b | scienceqa | by_category/Economics | 11 | 72.73 |
| sft_2b | scienceqa | by_category/Ecosystems | 78 | 89.74 |
| sft_2b | scienceqa | by_category/Editing and revising | 9 | 55.56 |
| sft_2b | scienceqa | by_category/Electricity | 1 | 100.00 |
| sft_2b | scienceqa | by_category/Engineering practices | 55 | 98.18 |
| sft_2b | scienceqa | by_category/English colonies in North America | 39 | 89.74 |
| sft_2b | scienceqa | by_category/Force and motion | 58 | 81.03 |
| sft_2b | scienceqa | by_category/Formatting | 73 | 75.34 |
| sft_2b | scienceqa | by_category/Fossils | 22 | 63.64 |
| sft_2b | scienceqa | by_category/Genes to traits | 195 | 81.54 |
| sft_2b | scienceqa | by_category/Geography | 46 | 93.48 |
| sft_2b | scienceqa | by_category/Government | 16 | 81.25 |
| sft_2b | scienceqa | by_category/Greece | 1 | 100.00 |
| sft_2b | scienceqa | by_category/Heat and thermal energy | 42 | 100.00 |
| sft_2b | scienceqa | by_category/Heredity | 41 | 82.93 |
| sft_2b | scienceqa | by_category/Historical figures | 2 | 100.00 |
| sft_2b | scienceqa | by_category/Independent reading comprehension | 5 | 100.00 |
| sft_2b | scienceqa | by_category/Informational texts: level 1 | 5 | 100.00 |
| sft_2b | scienceqa | by_category/Kinetic and potential energy | 19 | 78.95 |
| sft_2b | scienceqa | by_category/Literary devices | 267 | 79.03 |
| sft_2b | scienceqa | by_category/Magnets | 110 | 44.55 |
| sft_2b | scienceqa | by_category/Maps | 46 | 84.78 |
| sft_2b | scienceqa | by_category/Materials | 167 | 94.01 |
| sft_2b | scienceqa | by_category/Medieval Asia | 2 | 50.00 |
| sft_2b | scienceqa | by_category/Mixtures | 6 | 83.33 |
| sft_2b | scienceqa | by_category/Oceania: geography | 51 | 68.63 |
| sft_2b | scienceqa | by_category/Oceans and continents | 19 | 100.00 |
| sft_2b | scienceqa | by_category/Oceans and continents	 | 4 | 100.00 |
| sft_2b | scienceqa | by_category/Opinion writing | 4 | 100.00 |
| sft_2b | scienceqa | by_category/Particle motion and energy | 41 | 80.49 |
| sft_2b | scienceqa | by_category/Persuasive strategies | 11 | 81.82 |
| sft_2b | scienceqa | by_category/Photosynthesis | 4 | 100.00 |
| sft_2b | scienceqa | by_category/Phrases and clauses | 32 | 56.25 |
| sft_2b | scienceqa | by_category/Physical Geography | 51 | 96.08 |
| sft_2b | scienceqa | by_category/Physical and chemical change | 52 | 92.31 |
| sft_2b | scienceqa | by_category/Plant reproduction | 6 | 83.33 |
| sft_2b | scienceqa | by_category/Plants | 12 | 91.67 |
| sft_2b | scienceqa | by_category/Plate tectonics | 3 | 100.00 |
| sft_2b | scienceqa | by_category/Poetry elements | 4 | 50.00 |
| sft_2b | scienceqa | by_category/Pronouns | 37 | 64.86 |
| sft_2b | scienceqa | by_category/Pronouns and antecedents | 18 | 55.56 |
| sft_2b | scienceqa | by_category/Read-alone texts | 2 | 100.00 |
| sft_2b | scienceqa | by_category/Reference skills | 140 | 50.71 |
| sft_2b | scienceqa | by_category/Research skills | 4 | 75.00 |
| sft_2b | scienceqa | by_category/Rhyming | 17 | 41.18 |
| sft_2b | scienceqa | by_category/Rocks and minerals | 46 | 95.65 |
| sft_2b | scienceqa | by_category/Scientific names | 32 | 100.00 |
| sft_2b | scienceqa | by_category/Sentences, fragments, and run-ons | 135 | 88.15 |
| sft_2b | scienceqa | by_category/Shades of meaning | 10 | 90.00 |
| sft_2b | scienceqa | by_category/Short and long vowels | 5 | 60.00 |
| sft_2b | scienceqa | by_category/Social studies skills | 12 | 100.00 |
| sft_2b | scienceqa | by_category/Solutions | 54 | 61.11 |
| sft_2b | scienceqa | by_category/State capitals | 312 | 96.79 |
| sft_2b | scienceqa | by_category/States | 18 | 100.00 |
| sft_2b | scienceqa | by_category/States of matter | 47 | 87.23 |
| sft_2b | scienceqa | by_category/Supply and demand | 27 | 70.37 |
| sft_2b | scienceqa | by_category/Text structure | 2 | 100.00 |
| sft_2b | scienceqa | by_category/The American Revolution | 4 | 100.00 |
| sft_2b | scienceqa | by_category/The Americas: geography | 45 | 66.67 |
| sft_2b | scienceqa | by_category/The Antebellum period | 1 | 100.00 |
| sft_2b | scienceqa | by_category/The Civil War and Reconstruction | 2 | 100.00 |
| sft_2b | scienceqa | by_category/Thermal energy | 40 | 70.00 |
| sft_2b | scienceqa | by_category/Topographic maps | 1 | 100.00 |
| sft_2b | scienceqa | by_category/Traits and heredity | 81 | 83.95 |
| sft_2b | scienceqa | by_category/Units and measurement | 200 | 87.00 |
| sft_2b | scienceqa | by_category/Velocity, acceleration, and forces | 86 | 54.65 |
| sft_2b | scienceqa | by_category/Verb tense | 52 | 76.92 |
| sft_2b | scienceqa | by_category/Visual elements | 9 | 100.00 |
| sft_2b | scienceqa | by_category/Water cycle | 2 | 100.00 |
| sft_2b | scienceqa | by_category/Weather and climate | 97 | 80.41 |
| sft_2b | scienceqa | by_category/Word usage and nuance | 45 | 66.67 |
| sft_2b | scienceqa | by_category/World religions | 1 | 100.00 |
| sft_2b | scienceqa | by_modality/image | 2017 | 84.78 |
| sft_2b | scienceqa | by_modality/text | 2224 | 80.85 |
| sft_2b | scienceqa | by_subject/language science | 1100 | 74.45 |
| sft_2b | scienceqa | by_subject/natural science | 2252 | 84.15 |
| sft_2b | scienceqa | by_subject/social science | 889 | 89.31 |
| vanilla_2b | scienceqa | by_category/20th century American history | 4 | 100.00 |
| vanilla_2b | scienceqa | by_category/Adaptations | 98 | 84.69 |
| vanilla_2b | scienceqa | by_category/Age of Exploration | 1 | 100.00 |
| vanilla_2b | scienceqa | by_category/Analyzing literature | 4 | 100.00 |
| vanilla_2b | scienceqa | by_category/Anatomy and physiology | 2 | 100.00 |
| vanilla_2b | scienceqa | by_category/Ancient Egypt and Kush | 1 | 100.00 |
| vanilla_2b | scienceqa | by_category/Ancient Mesopotamia | 2 | 100.00 |
| vanilla_2b | scienceqa | by_category/Animals | 8 | 100.00 |
| vanilla_2b | scienceqa | by_category/Astronomy | 37 | 54.05 |
| vanilla_2b | scienceqa | by_category/Atoms and molecules | 64 | 98.44 |
| vanilla_2b | scienceqa | by_category/Audience, purpose, and tone | 17 | 100.00 |
| vanilla_2b | scienceqa | by_category/Author's purpose and tone | 27 | 85.19 |
| vanilla_2b | scienceqa | by_category/Banking and finance | 1 | 100.00 |
| vanilla_2b | scienceqa | by_category/Basic economic principles | 116 | 86.21 |
| vanilla_2b | scienceqa | by_category/Capitalization | 18 | 61.11 |
| vanilla_2b | scienceqa | by_category/Categories | 11 | 36.36 |
| vanilla_2b | scienceqa | by_category/Cells | 17 | 76.47 |
| vanilla_2b | scienceqa | by_category/Chemical reactions | 43 | 93.02 |
| vanilla_2b | scienceqa | by_category/Cities | 12 | 58.33 |
| vanilla_2b | scienceqa | by_category/Classification | 161 | 96.89 |
| vanilla_2b | scienceqa | by_category/Classification and scientific names | 46 | 100.00 |
| vanilla_2b | scienceqa | by_category/Climate change | 1 | 100.00 |
| vanilla_2b | scienceqa | by_category/Colonial America | 38 | 92.11 |
| vanilla_2b | scienceqa | by_category/Comprehension strategies | 5 | 100.00 |
| vanilla_2b | scienceqa | by_category/Conservation and natural resources | 1 | 100.00 |
| vanilla_2b | scienceqa | by_category/Context clues | 5 | 60.00 |
| vanilla_2b | scienceqa | by_category/Creative techniques | 19 | 68.42 |
| vanilla_2b | scienceqa | by_category/Cultural celebrations | 2 | 100.00 |
| vanilla_2b | scienceqa | by_category/Descriptive details | 3 | 100.00 |
| vanilla_2b | scienceqa | by_category/Designing experiments | 134 | 94.78 |
| vanilla_2b | scienceqa | by_category/Developing and supporting arguments | 101 | 87.13 |
| vanilla_2b | scienceqa | by_category/Domain-specific vocabulary | 4 | 100.00 |
| vanilla_2b | scienceqa | by_category/Early 19th century American history | 2 | 100.00 |
| vanilla_2b | scienceqa | by_category/Earth events | 18 | 100.00 |
| vanilla_2b | scienceqa | by_category/Ecological interactions | 24 | 58.33 |
| vanilla_2b | scienceqa | by_category/Economics | 11 | 81.82 |
| vanilla_2b | scienceqa | by_category/Ecosystems | 78 | 87.18 |
| vanilla_2b | scienceqa | by_category/Editing and revising | 9 | 55.56 |
| vanilla_2b | scienceqa | by_category/Electricity | 1 | 100.00 |
| vanilla_2b | scienceqa | by_category/Engineering practices | 55 | 100.00 |
| vanilla_2b | scienceqa | by_category/English colonies in North America | 39 | 94.87 |
| vanilla_2b | scienceqa | by_category/Force and motion | 58 | 86.21 |
| vanilla_2b | scienceqa | by_category/Formatting | 73 | 64.38 |
| vanilla_2b | scienceqa | by_category/Fossils | 22 | 59.09 |
| vanilla_2b | scienceqa | by_category/Genes to traits | 195 | 83.59 |
| vanilla_2b | scienceqa | by_category/Geography | 46 | 95.65 |
| vanilla_2b | scienceqa | by_category/Government | 16 | 81.25 |
| vanilla_2b | scienceqa | by_category/Greece | 1 | 100.00 |
| vanilla_2b | scienceqa | by_category/Heat and thermal energy | 42 | 92.86 |
| vanilla_2b | scienceqa | by_category/Heredity | 41 | 100.00 |
| vanilla_2b | scienceqa | by_category/Historical figures | 2 | 100.00 |
| vanilla_2b | scienceqa | by_category/Independent reading comprehension | 5 | 100.00 |
| vanilla_2b | scienceqa | by_category/Informational texts: level 1 | 5 | 100.00 |
| vanilla_2b | scienceqa | by_category/Kinetic and potential energy | 19 | 84.21 |
| vanilla_2b | scienceqa | by_category/Literary devices | 267 | 77.90 |
| vanilla_2b | scienceqa | by_category/Magnets | 110 | 41.82 |
| vanilla_2b | scienceqa | by_category/Maps | 46 | 86.96 |
| vanilla_2b | scienceqa | by_category/Materials | 167 | 92.81 |
| vanilla_2b | scienceqa | by_category/Medieval Asia | 2 | 50.00 |
| vanilla_2b | scienceqa | by_category/Mixtures | 6 | 83.33 |
| vanilla_2b | scienceqa | by_category/Oceania: geography | 51 | 82.35 |
| vanilla_2b | scienceqa | by_category/Oceans and continents | 19 | 100.00 |
| vanilla_2b | scienceqa | by_category/Oceans and continents	 | 4 | 100.00 |
| vanilla_2b | scienceqa | by_category/Opinion writing | 4 | 100.00 |
| vanilla_2b | scienceqa | by_category/Particle motion and energy | 41 | 95.12 |
| vanilla_2b | scienceqa | by_category/Persuasive strategies | 11 | 100.00 |
| vanilla_2b | scienceqa | by_category/Photosynthesis | 4 | 100.00 |
| vanilla_2b | scienceqa | by_category/Phrases and clauses | 32 | 59.38 |
| vanilla_2b | scienceqa | by_category/Physical Geography | 51 | 98.04 |
| vanilla_2b | scienceqa | by_category/Physical and chemical change | 52 | 94.23 |
| vanilla_2b | scienceqa | by_category/Plant reproduction | 6 | 83.33 |
| vanilla_2b | scienceqa | by_category/Plants | 12 | 91.67 |
| vanilla_2b | scienceqa | by_category/Plate tectonics | 3 | 100.00 |
| vanilla_2b | scienceqa | by_category/Poetry elements | 4 | 75.00 |
| vanilla_2b | scienceqa | by_category/Pronouns | 37 | 59.46 |
| vanilla_2b | scienceqa | by_category/Pronouns and antecedents | 18 | 44.44 |
| vanilla_2b | scienceqa | by_category/Read-alone texts | 2 | 100.00 |
| vanilla_2b | scienceqa | by_category/Reference skills | 140 | 46.43 |
| vanilla_2b | scienceqa | by_category/Research skills | 4 | 75.00 |
| vanilla_2b | scienceqa | by_category/Rhyming | 17 | 58.82 |
| vanilla_2b | scienceqa | by_category/Rocks and minerals | 46 | 97.83 |
| vanilla_2b | scienceqa | by_category/Scientific names | 32 | 100.00 |
| vanilla_2b | scienceqa | by_category/Sentences, fragments, and run-ons | 135 | 85.93 |
| vanilla_2b | scienceqa | by_category/Shades of meaning | 10 | 100.00 |
| vanilla_2b | scienceqa | by_category/Short and long vowels | 5 | 40.00 |
| vanilla_2b | scienceqa | by_category/Social studies skills | 12 | 100.00 |
| vanilla_2b | scienceqa | by_category/Solutions | 54 | 68.52 |
| vanilla_2b | scienceqa | by_category/State capitals | 312 | 98.08 |
| vanilla_2b | scienceqa | by_category/States | 18 | 100.00 |
| vanilla_2b | scienceqa | by_category/States of matter | 47 | 89.36 |
| vanilla_2b | scienceqa | by_category/Supply and demand | 27 | 74.07 |
| vanilla_2b | scienceqa | by_category/Text structure | 2 | 100.00 |
| vanilla_2b | scienceqa | by_category/The American Revolution | 4 | 100.00 |
| vanilla_2b | scienceqa | by_category/The Americas: geography | 45 | 66.67 |
| vanilla_2b | scienceqa | by_category/The Antebellum period | 1 | 100.00 |
| vanilla_2b | scienceqa | by_category/The Civil War and Reconstruction | 2 | 100.00 |
| vanilla_2b | scienceqa | by_category/Thermal energy | 40 | 70.00 |
| vanilla_2b | scienceqa | by_category/Topographic maps | 1 | 100.00 |
| vanilla_2b | scienceqa | by_category/Traits and heredity | 81 | 91.36 |
| vanilla_2b | scienceqa | by_category/Units and measurement | 200 | 86.00 |
| vanilla_2b | scienceqa | by_category/Velocity, acceleration, and forces | 86 | 52.33 |
| vanilla_2b | scienceqa | by_category/Verb tense | 52 | 78.85 |
| vanilla_2b | scienceqa | by_category/Visual elements | 9 | 100.00 |
| vanilla_2b | scienceqa | by_category/Water cycle | 2 | 100.00 |
| vanilla_2b | scienceqa | by_category/Weather and climate | 97 | 77.32 |
| vanilla_2b | scienceqa | by_category/Word usage and nuance | 45 | 60.00 |
| vanilla_2b | scienceqa | by_category/World religions | 1 | 100.00 |
| vanilla_2b | scienceqa | by_modality/image | 2017 | 85.62 |
| vanilla_2b | scienceqa | by_modality/text | 2224 | 80.53 |
| vanilla_2b | scienceqa | by_subject/language science | 1100 | 72.64 |
| vanilla_2b | scienceqa | by_subject/natural science | 2252 | 84.72 |
| vanilla_2b | scienceqa | by_subject/social science | 889 | 91.23 |
| vanilla_8b_teacher | scienceqa | by_category/20th century American history | 4 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Adaptations | 98 | 98.98 |
| vanilla_8b_teacher | scienceqa | by_category/Age of Exploration | 1 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Analyzing literature | 4 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Anatomy and physiology | 2 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Ancient Egypt and Kush | 1 | 0.00 |
| vanilla_8b_teacher | scienceqa | by_category/Ancient Mesopotamia | 2 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Animals | 8 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Astronomy | 37 | 78.38 |
| vanilla_8b_teacher | scienceqa | by_category/Atoms and molecules | 64 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Audience, purpose, and tone | 17 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Author's purpose and tone | 27 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Banking and finance | 1 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Basic economic principles | 116 | 96.55 |
| vanilla_8b_teacher | scienceqa | by_category/Capitalization | 18 | 94.44 |
| vanilla_8b_teacher | scienceqa | by_category/Categories | 11 | 81.82 |
| vanilla_8b_teacher | scienceqa | by_category/Cells | 17 | 94.12 |
| vanilla_8b_teacher | scienceqa | by_category/Chemical reactions | 43 | 97.67 |
| vanilla_8b_teacher | scienceqa | by_category/Cities | 12 | 83.33 |
| vanilla_8b_teacher | scienceqa | by_category/Classification | 161 | 98.76 |
| vanilla_8b_teacher | scienceqa | by_category/Classification and scientific names | 46 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Climate change | 1 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Colonial America | 38 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Comprehension strategies | 5 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Conservation and natural resources | 1 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Context clues | 5 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Creative techniques | 19 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Cultural celebrations | 2 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Descriptive details | 3 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Designing experiments | 134 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Developing and supporting arguments | 101 | 95.05 |
| vanilla_8b_teacher | scienceqa | by_category/Domain-specific vocabulary | 4 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Early 19th century American history | 2 | 50.00 |
| vanilla_8b_teacher | scienceqa | by_category/Earth events | 18 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Ecological interactions | 24 | 75.00 |
| vanilla_8b_teacher | scienceqa | by_category/Economics | 11 | 90.91 |
| vanilla_8b_teacher | scienceqa | by_category/Ecosystems | 78 | 85.90 |
| vanilla_8b_teacher | scienceqa | by_category/Editing and revising | 9 | 66.67 |
| vanilla_8b_teacher | scienceqa | by_category/Electricity | 1 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Engineering practices | 55 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/English colonies in North America | 39 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Force and motion | 58 | 96.55 |
| vanilla_8b_teacher | scienceqa | by_category/Formatting | 73 | 80.82 |
| vanilla_8b_teacher | scienceqa | by_category/Fossils | 22 | 95.45 |
| vanilla_8b_teacher | scienceqa | by_category/Genes to traits | 195 | 98.46 |
| vanilla_8b_teacher | scienceqa | by_category/Geography | 46 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Government | 16 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Greece | 1 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Heat and thermal energy | 42 | 97.62 |
| vanilla_8b_teacher | scienceqa | by_category/Heredity | 41 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Historical figures | 2 | 50.00 |
| vanilla_8b_teacher | scienceqa | by_category/Independent reading comprehension | 5 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Informational texts: level 1 | 5 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Kinetic and potential energy | 19 | 94.74 |
| vanilla_8b_teacher | scienceqa | by_category/Literary devices | 267 | 93.63 |
| vanilla_8b_teacher | scienceqa | by_category/Magnets | 110 | 78.18 |
| vanilla_8b_teacher | scienceqa | by_category/Maps | 46 | 95.65 |
| vanilla_8b_teacher | scienceqa | by_category/Materials | 167 | 98.20 |
| vanilla_8b_teacher | scienceqa | by_category/Medieval Asia | 2 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Mixtures | 6 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Oceania: geography | 51 | 78.43 |
| vanilla_8b_teacher | scienceqa | by_category/Oceans and continents | 19 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Oceans and continents	 | 4 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Opinion writing | 4 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Particle motion and energy | 41 | 82.93 |
| vanilla_8b_teacher | scienceqa | by_category/Persuasive strategies | 11 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Photosynthesis | 4 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Phrases and clauses | 32 | 78.12 |
| vanilla_8b_teacher | scienceqa | by_category/Physical Geography | 51 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Physical and chemical change | 52 | 98.08 |
| vanilla_8b_teacher | scienceqa | by_category/Plant reproduction | 6 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Plants | 12 | 83.33 |
| vanilla_8b_teacher | scienceqa | by_category/Plate tectonics | 3 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Poetry elements | 4 | 75.00 |
| vanilla_8b_teacher | scienceqa | by_category/Pronouns | 37 | 97.30 |
| vanilla_8b_teacher | scienceqa | by_category/Pronouns and antecedents | 18 | 94.44 |
| vanilla_8b_teacher | scienceqa | by_category/Read-alone texts | 2 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Reference skills | 140 | 47.86 |
| vanilla_8b_teacher | scienceqa | by_category/Research skills | 4 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Rhyming | 17 | 64.71 |
| vanilla_8b_teacher | scienceqa | by_category/Rocks and minerals | 46 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Scientific names | 32 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Sentences, fragments, and run-ons | 135 | 97.04 |
| vanilla_8b_teacher | scienceqa | by_category/Shades of meaning | 10 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Short and long vowels | 5 | 40.00 |
| vanilla_8b_teacher | scienceqa | by_category/Social studies skills | 12 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Solutions | 54 | 79.63 |
| vanilla_8b_teacher | scienceqa | by_category/State capitals | 312 | 99.68 |
| vanilla_8b_teacher | scienceqa | by_category/States | 18 | 94.44 |
| vanilla_8b_teacher | scienceqa | by_category/States of matter | 47 | 97.87 |
| vanilla_8b_teacher | scienceqa | by_category/Supply and demand | 27 | 74.07 |
| vanilla_8b_teacher | scienceqa | by_category/Text structure | 2 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/The American Revolution | 4 | 75.00 |
| vanilla_8b_teacher | scienceqa | by_category/The Americas: geography | 45 | 62.22 |
| vanilla_8b_teacher | scienceqa | by_category/The Antebellum period | 1 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/The Civil War and Reconstruction | 2 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Thermal energy | 40 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Topographic maps | 1 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Traits and heredity | 81 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Units and measurement | 200 | 93.50 |
| vanilla_8b_teacher | scienceqa | by_category/Velocity, acceleration, and forces | 86 | 88.37 |
| vanilla_8b_teacher | scienceqa | by_category/Verb tense | 52 | 94.23 |
| vanilla_8b_teacher | scienceqa | by_category/Visual elements | 9 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Water cycle | 2 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_category/Weather and climate | 97 | 88.66 |
| vanilla_8b_teacher | scienceqa | by_category/Word usage and nuance | 45 | 73.33 |
| vanilla_8b_teacher | scienceqa | by_category/World religions | 1 | 100.00 |
| vanilla_8b_teacher | scienceqa | by_modality/image | 2017 | 93.36 |
| vanilla_8b_teacher | scienceqa | by_modality/text | 2224 | 91.46 |
| vanilla_8b_teacher | scienceqa | by_subject/language science | 1100 | 86.09 |
| vanilla_8b_teacher | scienceqa | by_subject/natural science | 2252 | 94.63 |
| vanilla_8b_teacher | scienceqa | by_subject/social science | 889 | 94.38 |
| grace_w4 | mmmu | by_question_type/multiple-choice | 847 | 67.41 |
| grace_w4 | mmmu | by_question_type/open | 53 | 22.64 |
| grace_w4 | mmmu | by_subject/Accounting | 30 | 40.00 |
| grace_w4 | mmmu | by_subject/Agriculture | 30 | 66.67 |
| grace_w4 | mmmu | by_subject/Architecture_and_Engineering | 30 | 53.33 |
| grace_w4 | mmmu | by_subject/Art | 30 | 80.00 |
| grace_w4 | mmmu | by_subject/Art_Theory | 30 | 83.33 |
| grace_w4 | mmmu | by_subject/Basic_Medical_Science | 30 | 80.00 |
| grace_w4 | mmmu | by_subject/Biology | 30 | 60.00 |
| grace_w4 | mmmu | by_subject/Chemistry | 30 | 43.33 |
| grace_w4 | mmmu | by_subject/Clinical_Medicine | 30 | 76.67 |
| grace_w4 | mmmu | by_subject/Computer_Science | 30 | 53.33 |
| grace_w4 | mmmu | by_subject/Design | 30 | 83.33 |
| grace_w4 | mmmu | by_subject/Diagnostics_and_Laboratory_Medicine | 30 | 70.00 |
| grace_w4 | mmmu | by_subject/Economics | 30 | 63.33 |
| grace_w4 | mmmu | by_subject/Electronics | 30 | 53.33 |
| grace_w4 | mmmu | by_subject/Energy_and_Power | 30 | 53.33 |
| grace_w4 | mmmu | by_subject/Finance | 30 | 50.00 |
| grace_w4 | mmmu | by_subject/Geography | 30 | 53.33 |
| grace_w4 | mmmu | by_subject/History | 30 | 93.33 |
| grace_w4 | mmmu | by_subject/Literature | 30 | 93.33 |
| grace_w4 | mmmu | by_subject/Manage | 30 | 70.00 |
| grace_w4 | mmmu | by_subject/Marketing | 30 | 70.00 |
| grace_w4 | mmmu | by_subject/Materials | 30 | 46.67 |
| grace_w4 | mmmu | by_subject/Math | 30 | 70.00 |
| grace_w4 | mmmu | by_subject/Mechanical_Engineering | 30 | 46.67 |
| grace_w4 | mmmu | by_subject/Music | 30 | 30.00 |
| grace_w4 | mmmu | by_subject/Pharmacy | 30 | 70.00 |
| grace_w4 | mmmu | by_subject/Physics | 30 | 63.33 |
| grace_w4 | mmmu | by_subject/Psychology | 30 | 76.67 |
| grace_w4 | mmmu | by_subject/Public_Health | 30 | 70.00 |
| grace_w4 | mmmu | by_subject/Sociology | 30 | 80.00 |
| grace_w8 | mmmu | by_question_type/multiple-choice | 847 | 70.25 |
| grace_w8 | mmmu | by_question_type/open | 53 | 22.64 |
| grace_w8 | mmmu | by_subject/Accounting | 30 | 40.00 |
| grace_w8 | mmmu | by_subject/Agriculture | 30 | 73.33 |
| grace_w8 | mmmu | by_subject/Architecture_and_Engineering | 30 | 53.33 |
| grace_w8 | mmmu | by_subject/Art | 30 | 76.67 |
| grace_w8 | mmmu | by_subject/Art_Theory | 30 | 80.00 |
| grace_w8 | mmmu | by_subject/Basic_Medical_Science | 30 | 76.67 |
| grace_w8 | mmmu | by_subject/Biology | 30 | 60.00 |
| grace_w8 | mmmu | by_subject/Chemistry | 30 | 50.00 |
| grace_w8 | mmmu | by_subject/Clinical_Medicine | 30 | 76.67 |
| grace_w8 | mmmu | by_subject/Computer_Science | 30 | 60.00 |
| grace_w8 | mmmu | by_subject/Design | 30 | 90.00 |
| grace_w8 | mmmu | by_subject/Diagnostics_and_Laboratory_Medicine | 30 | 73.33 |
| grace_w8 | mmmu | by_subject/Economics | 30 | 70.00 |
| grace_w8 | mmmu | by_subject/Electronics | 30 | 53.33 |
| grace_w8 | mmmu | by_subject/Energy_and_Power | 30 | 66.67 |
| grace_w8 | mmmu | by_subject/Finance | 30 | 43.33 |
| grace_w8 | mmmu | by_subject/Geography | 30 | 56.67 |
| grace_w8 | mmmu | by_subject/History | 30 | 96.67 |
| grace_w8 | mmmu | by_subject/Literature | 30 | 93.33 |
| grace_w8 | mmmu | by_subject/Manage | 30 | 70.00 |
| grace_w8 | mmmu | by_subject/Marketing | 30 | 70.00 |
| grace_w8 | mmmu | by_subject/Materials | 30 | 70.00 |
| grace_w8 | mmmu | by_subject/Math | 30 | 60.00 |
| grace_w8 | mmmu | by_subject/Mechanical_Engineering | 30 | 60.00 |
| grace_w8 | mmmu | by_subject/Music | 30 | 30.00 |
| grace_w8 | mmmu | by_subject/Pharmacy | 30 | 66.67 |
| grace_w8 | mmmu | by_subject/Physics | 30 | 76.67 |
| grace_w8 | mmmu | by_subject/Psychology | 30 | 76.67 |
| grace_w8 | mmmu | by_subject/Public_Health | 30 | 66.67 |
| grace_w8 | mmmu | by_subject/Sociology | 30 | 86.67 |
| qat_w4 | mmmu | by_question_type/multiple-choice | 847 | 68.95 |
| qat_w4 | mmmu | by_question_type/open | 53 | 24.53 |
| qat_w4 | mmmu | by_subject/Accounting | 30 | 63.33 |
| qat_w4 | mmmu | by_subject/Agriculture | 30 | 73.33 |
| qat_w4 | mmmu | by_subject/Architecture_and_Engineering | 30 | 56.67 |
| qat_w4 | mmmu | by_subject/Art | 30 | 86.67 |
| qat_w4 | mmmu | by_subject/Art_Theory | 30 | 80.00 |
| qat_w4 | mmmu | by_subject/Basic_Medical_Science | 30 | 76.67 |
| qat_w4 | mmmu | by_subject/Biology | 30 | 60.00 |
| qat_w4 | mmmu | by_subject/Chemistry | 30 | 53.33 |
| qat_w4 | mmmu | by_subject/Clinical_Medicine | 30 | 80.00 |
| qat_w4 | mmmu | by_subject/Computer_Science | 30 | 60.00 |
| qat_w4 | mmmu | by_subject/Design | 30 | 76.67 |
| qat_w4 | mmmu | by_subject/Diagnostics_and_Laboratory_Medicine | 30 | 63.33 |
| qat_w4 | mmmu | by_subject/Economics | 30 | 76.67 |
| qat_w4 | mmmu | by_subject/Electronics | 30 | 53.33 |
| qat_w4 | mmmu | by_subject/Energy_and_Power | 30 | 63.33 |
| qat_w4 | mmmu | by_subject/Finance | 30 | 50.00 |
| qat_w4 | mmmu | by_subject/Geography | 30 | 63.33 |
| qat_w4 | mmmu | by_subject/History | 30 | 90.00 |
| qat_w4 | mmmu | by_subject/Literature | 30 | 93.33 |
| qat_w4 | mmmu | by_subject/Manage | 30 | 76.67 |
| qat_w4 | mmmu | by_subject/Marketing | 30 | 66.67 |
| qat_w4 | mmmu | by_subject/Materials | 30 | 33.33 |
| qat_w4 | mmmu | by_subject/Math | 30 | 66.67 |
| qat_w4 | mmmu | by_subject/Mechanical_Engineering | 30 | 46.67 |
| qat_w4 | mmmu | by_subject/Music | 30 | 23.33 |
| qat_w4 | mmmu | by_subject/Pharmacy | 30 | 63.33 |
| qat_w4 | mmmu | by_subject/Physics | 30 | 63.33 |
| qat_w4 | mmmu | by_subject/Psychology | 30 | 73.33 |
| qat_w4 | mmmu | by_subject/Public_Health | 30 | 66.67 |
| qat_w4 | mmmu | by_subject/Sociology | 30 | 90.00 |
| qat_w8 | mmmu | by_question_type/multiple-choice | 847 | 73.32 |
| qat_w8 | mmmu | by_question_type/open | 53 | 28.30 |
| qat_w8 | mmmu | by_subject/Accounting | 30 | 50.00 |
| qat_w8 | mmmu | by_subject/Agriculture | 30 | 73.33 |
| qat_w8 | mmmu | by_subject/Architecture_and_Engineering | 30 | 56.67 |
| qat_w8 | mmmu | by_subject/Art | 30 | 86.67 |
| qat_w8 | mmmu | by_subject/Art_Theory | 30 | 80.00 |
| qat_w8 | mmmu | by_subject/Basic_Medical_Science | 30 | 83.33 |
| qat_w8 | mmmu | by_subject/Biology | 30 | 63.33 |
| qat_w8 | mmmu | by_subject/Chemistry | 30 | 46.67 |
| qat_w8 | mmmu | by_subject/Clinical_Medicine | 30 | 86.67 |
| qat_w8 | mmmu | by_subject/Computer_Science | 30 | 70.00 |
| qat_w8 | mmmu | by_subject/Design | 30 | 90.00 |
| qat_w8 | mmmu | by_subject/Diagnostics_and_Laboratory_Medicine | 30 | 70.00 |
| qat_w8 | mmmu | by_subject/Economics | 30 | 73.33 |
| qat_w8 | mmmu | by_subject/Electronics | 30 | 56.67 |
| qat_w8 | mmmu | by_subject/Energy_and_Power | 30 | 66.67 |
| qat_w8 | mmmu | by_subject/Finance | 30 | 53.33 |
| qat_w8 | mmmu | by_subject/Geography | 30 | 66.67 |
| qat_w8 | mmmu | by_subject/History | 30 | 86.67 |
| qat_w8 | mmmu | by_subject/Literature | 30 | 100.00 |
| qat_w8 | mmmu | by_subject/Manage | 30 | 83.33 |
| qat_w8 | mmmu | by_subject/Marketing | 30 | 73.33 |
| qat_w8 | mmmu | by_subject/Materials | 30 | 60.00 |
| qat_w8 | mmmu | by_subject/Math | 30 | 63.33 |
| qat_w8 | mmmu | by_subject/Mechanical_Engineering | 30 | 60.00 |
| qat_w8 | mmmu | by_subject/Music | 30 | 33.33 |
| qat_w8 | mmmu | by_subject/Pharmacy | 30 | 70.00 |
| qat_w8 | mmmu | by_subject/Physics | 30 | 83.33 |
| qat_w8 | mmmu | by_subject/Psychology | 30 | 76.67 |
| qat_w8 | mmmu | by_subject/Public_Health | 30 | 70.00 |
| qat_w8 | mmmu | by_subject/Sociology | 30 | 86.67 |
| released_bf16 | mmmu | by_question_type/multiple-choice | 847 | 72.61 |
| released_bf16 | mmmu | by_question_type/open | 53 | 28.30 |
| released_bf16 | mmmu | by_subject/Accounting | 30 | 53.33 |
| released_bf16 | mmmu | by_subject/Agriculture | 30 | 70.00 |
| released_bf16 | mmmu | by_subject/Architecture_and_Engineering | 30 | 60.00 |
| released_bf16 | mmmu | by_subject/Art | 30 | 93.33 |
| released_bf16 | mmmu | by_subject/Art_Theory | 30 | 86.67 |
| released_bf16 | mmmu | by_subject/Basic_Medical_Science | 30 | 73.33 |
| released_bf16 | mmmu | by_subject/Biology | 30 | 66.67 |
| released_bf16 | mmmu | by_subject/Chemistry | 30 | 56.67 |
| released_bf16 | mmmu | by_subject/Clinical_Medicine | 30 | 86.67 |
| released_bf16 | mmmu | by_subject/Computer_Science | 30 | 70.00 |
| released_bf16 | mmmu | by_subject/Design | 30 | 76.67 |
| released_bf16 | mmmu | by_subject/Diagnostics_and_Laboratory_Medicine | 30 | 70.00 |
| released_bf16 | mmmu | by_subject/Economics | 30 | 73.33 |
| released_bf16 | mmmu | by_subject/Electronics | 30 | 60.00 |
| released_bf16 | mmmu | by_subject/Energy_and_Power | 30 | 60.00 |
| released_bf16 | mmmu | by_subject/Finance | 30 | 50.00 |
| released_bf16 | mmmu | by_subject/Geography | 30 | 60.00 |
| released_bf16 | mmmu | by_subject/History | 30 | 86.67 |
| released_bf16 | mmmu | by_subject/Literature | 30 | 96.67 |
| released_bf16 | mmmu | by_subject/Manage | 30 | 76.67 |
| released_bf16 | mmmu | by_subject/Marketing | 30 | 83.33 |
| released_bf16 | mmmu | by_subject/Materials | 30 | 50.00 |
| released_bf16 | mmmu | by_subject/Math | 30 | 66.67 |
| released_bf16 | mmmu | by_subject/Mechanical_Engineering | 30 | 56.67 |
| released_bf16 | mmmu | by_subject/Music | 30 | 30.00 |
| released_bf16 | mmmu | by_subject/Pharmacy | 30 | 76.67 |
| released_bf16 | mmmu | by_subject/Physics | 30 | 76.67 |
| released_bf16 | mmmu | by_subject/Psychology | 30 | 83.33 |
| released_bf16 | mmmu | by_subject/Public_Health | 30 | 66.67 |
| released_bf16 | mmmu | by_subject/Sociology | 30 | 83.33 |
| released_grace_w4 | mmmu | by_question_type/multiple-choice | 847 | 63.87 |
| released_grace_w4 | mmmu | by_question_type/open | 53 | 18.87 |
| released_grace_w4 | mmmu | by_subject/Accounting | 30 | 46.67 |
| released_grace_w4 | mmmu | by_subject/Agriculture | 30 | 66.67 |
| released_grace_w4 | mmmu | by_subject/Architecture_and_Engineering | 30 | 53.33 |
| released_grace_w4 | mmmu | by_subject/Art | 30 | 83.33 |
| released_grace_w4 | mmmu | by_subject/Art_Theory | 30 | 73.33 |
| released_grace_w4 | mmmu | by_subject/Basic_Medical_Science | 30 | 70.00 |
| released_grace_w4 | mmmu | by_subject/Biology | 30 | 60.00 |
| released_grace_w4 | mmmu | by_subject/Chemistry | 30 | 46.67 |
| released_grace_w4 | mmmu | by_subject/Clinical_Medicine | 30 | 66.67 |
| released_grace_w4 | mmmu | by_subject/Computer_Science | 30 | 53.33 |
| released_grace_w4 | mmmu | by_subject/Design | 30 | 83.33 |
| released_grace_w4 | mmmu | by_subject/Diagnostics_and_Laboratory_Medicine | 30 | 56.67 |
| released_grace_w4 | mmmu | by_subject/Economics | 30 | 66.67 |
| released_grace_w4 | mmmu | by_subject/Electronics | 30 | 53.33 |
| released_grace_w4 | mmmu | by_subject/Energy_and_Power | 30 | 53.33 |
| released_grace_w4 | mmmu | by_subject/Finance | 30 | 40.00 |
| released_grace_w4 | mmmu | by_subject/Geography | 30 | 56.67 |
| released_grace_w4 | mmmu | by_subject/History | 30 | 83.33 |
| released_grace_w4 | mmmu | by_subject/Literature | 30 | 93.33 |
| released_grace_w4 | mmmu | by_subject/Manage | 30 | 46.67 |
| released_grace_w4 | mmmu | by_subject/Marketing | 30 | 66.67 |
| released_grace_w4 | mmmu | by_subject/Materials | 30 | 50.00 |
| released_grace_w4 | mmmu | by_subject/Math | 30 | 53.33 |
| released_grace_w4 | mmmu | by_subject/Mechanical_Engineering | 30 | 43.33 |
| released_grace_w4 | mmmu | by_subject/Music | 30 | 43.33 |
| released_grace_w4 | mmmu | by_subject/Pharmacy | 30 | 66.67 |
| released_grace_w4 | mmmu | by_subject/Physics | 30 | 60.00 |
| released_grace_w4 | mmmu | by_subject/Psychology | 30 | 56.67 |
| released_grace_w4 | mmmu | by_subject/Public_Health | 30 | 73.33 |
| released_grace_w4 | mmmu | by_subject/Sociology | 30 | 70.00 |
| released_grace_w8 | mmmu | by_question_type/multiple-choice | 847 | 59.50 |
| released_grace_w8 | mmmu | by_question_type/open | 53 | 15.09 |
| released_grace_w8 | mmmu | by_subject/Accounting | 30 | 40.00 |
| released_grace_w8 | mmmu | by_subject/Agriculture | 30 | 66.67 |
| released_grace_w8 | mmmu | by_subject/Architecture_and_Engineering | 30 | 40.00 |
| released_grace_w8 | mmmu | by_subject/Art | 30 | 73.33 |
| released_grace_w8 | mmmu | by_subject/Art_Theory | 30 | 73.33 |
| released_grace_w8 | mmmu | by_subject/Basic_Medical_Science | 30 | 63.33 |
| released_grace_w8 | mmmu | by_subject/Biology | 30 | 53.33 |
| released_grace_w8 | mmmu | by_subject/Chemistry | 30 | 36.67 |
| released_grace_w8 | mmmu | by_subject/Clinical_Medicine | 30 | 60.00 |
| released_grace_w8 | mmmu | by_subject/Computer_Science | 30 | 53.33 |
| released_grace_w8 | mmmu | by_subject/Design | 30 | 76.67 |
| released_grace_w8 | mmmu | by_subject/Diagnostics_and_Laboratory_Medicine | 30 | 60.00 |
| released_grace_w8 | mmmu | by_subject/Economics | 30 | 66.67 |
| released_grace_w8 | mmmu | by_subject/Electronics | 30 | 43.33 |
| released_grace_w8 | mmmu | by_subject/Energy_and_Power | 30 | 50.00 |
| released_grace_w8 | mmmu | by_subject/Finance | 30 | 36.67 |
| released_grace_w8 | mmmu | by_subject/Geography | 30 | 53.33 |
| released_grace_w8 | mmmu | by_subject/History | 30 | 76.67 |
| released_grace_w8 | mmmu | by_subject/Literature | 30 | 90.00 |
| released_grace_w8 | mmmu | by_subject/Manage | 30 | 43.33 |
| released_grace_w8 | mmmu | by_subject/Marketing | 30 | 63.33 |
| released_grace_w8 | mmmu | by_subject/Materials | 30 | 40.00 |
| released_grace_w8 | mmmu | by_subject/Math | 30 | 56.67 |
| released_grace_w8 | mmmu | by_subject/Mechanical_Engineering | 30 | 46.67 |
| released_grace_w8 | mmmu | by_subject/Music | 30 | 36.67 |
| released_grace_w8 | mmmu | by_subject/Pharmacy | 30 | 63.33 |
| released_grace_w8 | mmmu | by_subject/Physics | 30 | 60.00 |
| released_grace_w8 | mmmu | by_subject/Psychology | 30 | 53.33 |
| released_grace_w8 | mmmu | by_subject/Public_Health | 30 | 63.33 |
| released_grace_w8 | mmmu | by_subject/Sociology | 30 | 66.67 |
| sft_2b | mmmu | by_question_type/multiple-choice | 847 | 45.22 |
| sft_2b | mmmu | by_question_type/open | 53 | 7.55 |
| sft_2b | mmmu | by_subject/Accounting | 30 | 16.67 |
| sft_2b | mmmu | by_subject/Agriculture | 30 | 30.00 |
| sft_2b | mmmu | by_subject/Architecture_and_Engineering | 30 | 30.00 |
| sft_2b | mmmu | by_subject/Art | 30 | 53.33 |
| sft_2b | mmmu | by_subject/Art_Theory | 30 | 70.00 |
| sft_2b | mmmu | by_subject/Basic_Medical_Science | 30 | 40.00 |
| sft_2b | mmmu | by_subject/Biology | 30 | 36.67 |
| sft_2b | mmmu | by_subject/Chemistry | 30 | 26.67 |
| sft_2b | mmmu | by_subject/Clinical_Medicine | 30 | 40.00 |
| sft_2b | mmmu | by_subject/Computer_Science | 30 | 36.67 |
| sft_2b | mmmu | by_subject/Design | 30 | 60.00 |
| sft_2b | mmmu | by_subject/Diagnostics_and_Laboratory_Medicine | 30 | 40.00 |
| sft_2b | mmmu | by_subject/Economics | 30 | 46.67 |
| sft_2b | mmmu | by_subject/Electronics | 30 | 30.00 |
| sft_2b | mmmu | by_subject/Energy_and_Power | 30 | 30.00 |
| sft_2b | mmmu | by_subject/Finance | 30 | 36.67 |
| sft_2b | mmmu | by_subject/Geography | 30 | 46.67 |
| sft_2b | mmmu | by_subject/History | 30 | 43.33 |
| sft_2b | mmmu | by_subject/Literature | 30 | 86.67 |
| sft_2b | mmmu | by_subject/Manage | 30 | 43.33 |
| sft_2b | mmmu | by_subject/Marketing | 30 | 43.33 |
| sft_2b | mmmu | by_subject/Materials | 30 | 33.33 |
| sft_2b | mmmu | by_subject/Math | 30 | 36.67 |
| sft_2b | mmmu | by_subject/Mechanical_Engineering | 30 | 43.33 |
| sft_2b | mmmu | by_subject/Music | 30 | 36.67 |
| sft_2b | mmmu | by_subject/Pharmacy | 30 | 50.00 |
| sft_2b | mmmu | by_subject/Physics | 30 | 36.67 |
| sft_2b | mmmu | by_subject/Psychology | 30 | 53.33 |
| sft_2b | mmmu | by_subject/Public_Health | 30 | 50.00 |
| sft_2b | mmmu | by_subject/Sociology | 30 | 63.33 |
| vanilla_2b | mmmu | by_question_type/multiple-choice | 847 | 41.91 |
| vanilla_2b | mmmu | by_question_type/open | 53 | 9.43 |
| vanilla_2b | mmmu | by_subject/Accounting | 30 | 33.33 |
| vanilla_2b | mmmu | by_subject/Agriculture | 30 | 33.33 |
| vanilla_2b | mmmu | by_subject/Architecture_and_Engineering | 30 | 26.67 |
| vanilla_2b | mmmu | by_subject/Art | 30 | 53.33 |
| vanilla_2b | mmmu | by_subject/Art_Theory | 30 | 70.00 |
| vanilla_2b | mmmu | by_subject/Basic_Medical_Science | 30 | 46.67 |
| vanilla_2b | mmmu | by_subject/Biology | 30 | 33.33 |
| vanilla_2b | mmmu | by_subject/Chemistry | 30 | 23.33 |
| vanilla_2b | mmmu | by_subject/Clinical_Medicine | 30 | 33.33 |
| vanilla_2b | mmmu | by_subject/Computer_Science | 30 | 46.67 |
| vanilla_2b | mmmu | by_subject/Design | 30 | 73.33 |
| vanilla_2b | mmmu | by_subject/Diagnostics_and_Laboratory_Medicine | 30 | 36.67 |
| vanilla_2b | mmmu | by_subject/Economics | 30 | 33.33 |
| vanilla_2b | mmmu | by_subject/Electronics | 30 | 26.67 |
| vanilla_2b | mmmu | by_subject/Energy_and_Power | 30 | 26.67 |
| vanilla_2b | mmmu | by_subject/Finance | 30 | 23.33 |
| vanilla_2b | mmmu | by_subject/Geography | 30 | 53.33 |
| vanilla_2b | mmmu | by_subject/History | 30 | 53.33 |
| vanilla_2b | mmmu | by_subject/Literature | 30 | 83.33 |
| vanilla_2b | mmmu | by_subject/Manage | 30 | 30.00 |
| vanilla_2b | mmmu | by_subject/Marketing | 30 | 36.67 |
| vanilla_2b | mmmu | by_subject/Materials | 30 | 16.67 |
| vanilla_2b | mmmu | by_subject/Math | 30 | 33.33 |
| vanilla_2b | mmmu | by_subject/Mechanical_Engineering | 30 | 26.67 |
| vanilla_2b | mmmu | by_subject/Music | 30 | 40.00 |
| vanilla_2b | mmmu | by_subject/Pharmacy | 30 | 43.33 |
| vanilla_2b | mmmu | by_subject/Physics | 30 | 30.00 |
| vanilla_2b | mmmu | by_subject/Psychology | 30 | 53.33 |
| vanilla_2b | mmmu | by_subject/Public_Health | 30 | 30.00 |
| vanilla_2b | mmmu | by_subject/Sociology | 30 | 50.00 |
| vanilla_8b_teacher | mmmu | by_question_type/multiple-choice | 847 | 53.72 |
| vanilla_8b_teacher | mmmu | by_question_type/open | 53 | 13.21 |
| vanilla_8b_teacher | mmmu | by_subject/Accounting | 30 | 53.33 |
| vanilla_8b_teacher | mmmu | by_subject/Agriculture | 30 | 53.33 |
| vanilla_8b_teacher | mmmu | by_subject/Architecture_and_Engineering | 30 | 36.67 |
| vanilla_8b_teacher | mmmu | by_subject/Art | 30 | 63.33 |
| vanilla_8b_teacher | mmmu | by_subject/Art_Theory | 30 | 86.67 |
| vanilla_8b_teacher | mmmu | by_subject/Basic_Medical_Science | 30 | 60.00 |
| vanilla_8b_teacher | mmmu | by_subject/Biology | 30 | 46.67 |
| vanilla_8b_teacher | mmmu | by_subject/Chemistry | 30 | 26.67 |
| vanilla_8b_teacher | mmmu | by_subject/Clinical_Medicine | 30 | 66.67 |
| vanilla_8b_teacher | mmmu | by_subject/Computer_Science | 30 | 53.33 |
| vanilla_8b_teacher | mmmu | by_subject/Design | 30 | 76.67 |
| vanilla_8b_teacher | mmmu | by_subject/Diagnostics_and_Laboratory_Medicine | 30 | 30.00 |
| vanilla_8b_teacher | mmmu | by_subject/Economics | 30 | 53.33 |
| vanilla_8b_teacher | mmmu | by_subject/Electronics | 30 | 36.67 |
| vanilla_8b_teacher | mmmu | by_subject/Energy_and_Power | 30 | 46.67 |
| vanilla_8b_teacher | mmmu | by_subject/Finance | 30 | 26.67 |
| vanilla_8b_teacher | mmmu | by_subject/Geography | 30 | 46.67 |
| vanilla_8b_teacher | mmmu | by_subject/History | 30 | 66.67 |
| vanilla_8b_teacher | mmmu | by_subject/Literature | 30 | 80.00 |
| vanilla_8b_teacher | mmmu | by_subject/Manage | 30 | 36.67 |
| vanilla_8b_teacher | mmmu | by_subject/Marketing | 30 | 53.33 |
| vanilla_8b_teacher | mmmu | by_subject/Materials | 30 | 46.67 |
| vanilla_8b_teacher | mmmu | by_subject/Math | 30 | 13.33 |
| vanilla_8b_teacher | mmmu | by_subject/Mechanical_Engineering | 30 | 36.67 |
| vanilla_8b_teacher | mmmu | by_subject/Music | 30 | 43.33 |
| vanilla_8b_teacher | mmmu | by_subject/Pharmacy | 30 | 70.00 |
| vanilla_8b_teacher | mmmu | by_subject/Physics | 30 | 53.33 |
| vanilla_8b_teacher | mmmu | by_subject/Psychology | 30 | 60.00 |
| vanilla_8b_teacher | mmmu | by_subject/Public_Health | 30 | 56.67 |
| vanilla_8b_teacher | mmmu | by_subject/Sociology | 30 | 60.00 |
| grace_w4 | seedbench_img | by_question_type/Instance Attributes | 4649 | 75.11 |
| grace_w4 | seedbench_img | by_question_type/Instance Identity | 1831 | 76.46 |
| grace_w4 | seedbench_img | by_question_type/Instance Interaction | 97 | 71.13 |
| grace_w4 | seedbench_img | by_question_type/Instance Location | 978 | 66.97 |
| grace_w4 | seedbench_img | by_question_type/Instances Counting | 2447 | 65.71 |
| grace_w4 | seedbench_img | by_question_type/Scene Understanding | 3158 | 78.40 |
| grace_w4 | seedbench_img | by_question_type/Spatial Relation | 657 | 58.30 |
| grace_w4 | seedbench_img | by_question_type/Text Understanding | 85 | 68.24 |
| grace_w4 | seedbench_img | by_question_type/Visual Reasoning | 331 | 77.04 |
| grace_w8 | seedbench_img | by_question_type/Instance Attributes | 4649 | 74.58 |
| grace_w8 | seedbench_img | by_question_type/Instance Identity | 1831 | 76.84 |
| grace_w8 | seedbench_img | by_question_type/Instance Interaction | 97 | 72.16 |
| grace_w8 | seedbench_img | by_question_type/Instance Location | 978 | 67.89 |
| grace_w8 | seedbench_img | by_question_type/Instances Counting | 2447 | 65.30 |
| grace_w8 | seedbench_img | by_question_type/Scene Understanding | 3158 | 77.61 |
| grace_w8 | seedbench_img | by_question_type/Spatial Relation | 657 | 58.75 |
| grace_w8 | seedbench_img | by_question_type/Text Understanding | 85 | 67.06 |
| grace_w8 | seedbench_img | by_question_type/Visual Reasoning | 331 | 77.95 |
| qat_w4 | seedbench_img | by_question_type/Instance Attributes | 4649 | 74.17 |
| qat_w4 | seedbench_img | by_question_type/Instance Identity | 1831 | 74.44 |
| qat_w4 | seedbench_img | by_question_type/Instance Interaction | 97 | 65.98 |
| qat_w4 | seedbench_img | by_question_type/Instance Location | 978 | 68.71 |
| qat_w4 | seedbench_img | by_question_type/Instances Counting | 2447 | 65.71 |
| qat_w4 | seedbench_img | by_question_type/Scene Understanding | 3158 | 77.90 |
| qat_w4 | seedbench_img | by_question_type/Spatial Relation | 657 | 58.45 |
| qat_w4 | seedbench_img | by_question_type/Text Understanding | 85 | 61.18 |
| qat_w4 | seedbench_img | by_question_type/Visual Reasoning | 331 | 78.25 |
| qat_w8 | seedbench_img | by_question_type/Instance Attributes | 4649 | 74.19 |
| qat_w8 | seedbench_img | by_question_type/Instance Identity | 1831 | 75.15 |
| qat_w8 | seedbench_img | by_question_type/Instance Interaction | 97 | 70.10 |
| qat_w8 | seedbench_img | by_question_type/Instance Location | 978 | 69.63 |
| qat_w8 | seedbench_img | by_question_type/Instances Counting | 2447 | 66.00 |
| qat_w8 | seedbench_img | by_question_type/Scene Understanding | 3158 | 78.21 |
| qat_w8 | seedbench_img | by_question_type/Spatial Relation | 657 | 58.30 |
| qat_w8 | seedbench_img | by_question_type/Text Understanding | 85 | 50.59 |
| qat_w8 | seedbench_img | by_question_type/Visual Reasoning | 331 | 76.13 |
| released_bf16 | seedbench_img | by_question_type/Instance Attributes | 4649 | 73.97 |
| released_bf16 | seedbench_img | by_question_type/Instance Identity | 1831 | 74.66 |
| released_bf16 | seedbench_img | by_question_type/Instance Interaction | 97 | 72.16 |
| released_bf16 | seedbench_img | by_question_type/Instance Location | 978 | 67.79 |
| released_bf16 | seedbench_img | by_question_type/Instances Counting | 2447 | 64.61 |
| released_bf16 | seedbench_img | by_question_type/Scene Understanding | 3158 | 78.31 |
| released_bf16 | seedbench_img | by_question_type/Spatial Relation | 657 | 58.45 |
| released_bf16 | seedbench_img | by_question_type/Text Understanding | 85 | 60.00 |
| released_bf16 | seedbench_img | by_question_type/Visual Reasoning | 331 | 78.85 |
| released_grace_w4 | seedbench_img | by_question_type/Instance Attributes | 4649 | 74.53 |
| released_grace_w4 | seedbench_img | by_question_type/Instance Identity | 1831 | 75.04 |
| released_grace_w4 | seedbench_img | by_question_type/Instance Interaction | 97 | 73.20 |
| released_grace_w4 | seedbench_img | by_question_type/Instance Location | 978 | 66.77 |
| released_grace_w4 | seedbench_img | by_question_type/Instances Counting | 2447 | 66.20 |
| released_grace_w4 | seedbench_img | by_question_type/Scene Understanding | 3158 | 77.52 |
| released_grace_w4 | seedbench_img | by_question_type/Spatial Relation | 657 | 52.97 |
| released_grace_w4 | seedbench_img | by_question_type/Text Understanding | 85 | 57.65 |
| released_grace_w4 | seedbench_img | by_question_type/Visual Reasoning | 331 | 77.04 |
| released_grace_w8 | seedbench_img | by_question_type/Instance Attributes | 4649 | 74.12 |
| released_grace_w8 | seedbench_img | by_question_type/Instance Identity | 1831 | 74.82 |
| released_grace_w8 | seedbench_img | by_question_type/Instance Interaction | 97 | 75.26 |
| released_grace_w8 | seedbench_img | by_question_type/Instance Location | 978 | 67.59 |
| released_grace_w8 | seedbench_img | by_question_type/Instances Counting | 2447 | 64.65 |
| released_grace_w8 | seedbench_img | by_question_type/Scene Understanding | 3158 | 78.06 |
| released_grace_w8 | seedbench_img | by_question_type/Spatial Relation | 657 | 52.82 |
| released_grace_w8 | seedbench_img | by_question_type/Text Understanding | 85 | 55.29 |
| released_grace_w8 | seedbench_img | by_question_type/Visual Reasoning | 331 | 77.34 |
| sft_2b | seedbench_img | by_question_type/Instance Attributes | 4649 | 74.23 |
| sft_2b | seedbench_img | by_question_type/Instance Identity | 1831 | 77.06 |
| sft_2b | seedbench_img | by_question_type/Instance Interaction | 97 | 73.20 |
| sft_2b | seedbench_img | by_question_type/Instance Location | 978 | 68.10 |
| sft_2b | seedbench_img | by_question_type/Instances Counting | 2447 | 66.82 |
| sft_2b | seedbench_img | by_question_type/Scene Understanding | 3158 | 78.28 |
| sft_2b | seedbench_img | by_question_type/Spatial Relation | 657 | 56.62 |
| sft_2b | seedbench_img | by_question_type/Text Understanding | 85 | 68.24 |
| sft_2b | seedbench_img | by_question_type/Visual Reasoning | 331 | 79.46 |
| vanilla_2b | seedbench_img | by_question_type/Instance Attributes | 4649 | 74.36 |
| vanilla_2b | seedbench_img | by_question_type/Instance Identity | 1831 | 77.55 |
| vanilla_2b | seedbench_img | by_question_type/Instance Interaction | 97 | 76.29 |
| vanilla_2b | seedbench_img | by_question_type/Instance Location | 978 | 67.38 |
| vanilla_2b | seedbench_img | by_question_type/Instances Counting | 2447 | 63.51 |
| vanilla_2b | seedbench_img | by_question_type/Scene Understanding | 3158 | 77.96 |
| vanilla_2b | seedbench_img | by_question_type/Spatial Relation | 657 | 57.08 |
| vanilla_2b | seedbench_img | by_question_type/Text Understanding | 85 | 63.53 |
| vanilla_2b | seedbench_img | by_question_type/Visual Reasoning | 331 | 77.95 |
| vanilla_8b_teacher | seedbench_img | by_question_type/Instance Attributes | 4649 | 77.54 |
| vanilla_8b_teacher | seedbench_img | by_question_type/Instance Identity | 1831 | 79.52 |
| vanilla_8b_teacher | seedbench_img | by_question_type/Instance Interaction | 97 | 77.32 |
| vanilla_8b_teacher | seedbench_img | by_question_type/Instance Location | 978 | 72.49 |
| vanilla_8b_teacher | seedbench_img | by_question_type/Instances Counting | 2447 | 69.31 |
| vanilla_8b_teacher | seedbench_img | by_question_type/Scene Understanding | 3158 | 80.40 |
| vanilla_8b_teacher | seedbench_img | by_question_type/Spatial Relation | 657 | 65.91 |
| vanilla_8b_teacher | seedbench_img | by_question_type/Text Understanding | 85 | 60.00 |
| vanilla_8b_teacher | seedbench_img | by_question_type/Visual Reasoning | 331 | 81.87 |
| grace_w4 | mmstar | by_category/coarse perception | 250 | 74.40 |
| grace_w4 | mmstar | by_category/fine-grained perception | 250 | 52.00 |
| grace_w4 | mmstar | by_category/instance reasoning | 250 | 70.00 |
| grace_w4 | mmstar | by_category/logical reasoning | 250 | 54.00 |
| grace_w4 | mmstar | by_category/math | 250 | 36.80 |
| grace_w4 | mmstar | by_category/science & technology | 250 | 45.20 |
| grace_w4 | mmstar | by_l2_category/biology & chemistry & physics | 146 | 49.32 |
| grace_w4 | mmstar | by_l2_category/code & sequence reasoning | 39 | 48.72 |
| grace_w4 | mmstar | by_l2_category/common reasoning | 101 | 52.48 |
| grace_w4 | mmstar | by_l2_category/cross-instance attribute reasoning | 89 | 57.30 |
| grace_w4 | mmstar | by_l2_category/cross-instance relation reasoning | 62 | 80.65 |
| grace_w4 | mmstar | by_l2_category/diagram reasoning | 110 | 57.27 |
| grace_w4 | mmstar | by_l2_category/electronics & energy & mechanical eng. | 46 | 43.48 |
| grace_w4 | mmstar | by_l2_category/geography & earth science & agriculture | 58 | 36.21 |
| grace_w4 | mmstar | by_l2_category/geometry | 116 | 29.31 |
| grace_w4 | mmstar | by_l2_category/image emotion | 31 | 93.55 |
| grace_w4 | mmstar | by_l2_category/image scene and topic | 141 | 63.12 |
| grace_w4 | mmstar | by_l2_category/image style & quality | 78 | 87.18 |
| grace_w4 | mmstar | by_l2_category/localization | 40 | 65.00 |
| grace_w4 | mmstar | by_l2_category/numeric commonsense and calculation | 48 | 50.00 |
| grace_w4 | mmstar | by_l2_category/object counting | 92 | 46.74 |
| grace_w4 | mmstar | by_l2_category/recognition | 118 | 51.69 |
| grace_w4 | mmstar | by_l2_category/single-instance reasoning | 99 | 74.75 |
| grace_w4 | mmstar | by_l2_category/statistical reasoning | 86 | 39.53 |
| grace_w8 | mmstar | by_category/coarse perception | 250 | 72.80 |
| grace_w8 | mmstar | by_category/fine-grained perception | 250 | 49.60 |
| grace_w8 | mmstar | by_category/instance reasoning | 250 | 64.40 |
| grace_w8 | mmstar | by_category/logical reasoning | 250 | 50.80 |
| grace_w8 | mmstar | by_category/math | 250 | 35.20 |
| grace_w8 | mmstar | by_category/science & technology | 250 | 46.40 |
| grace_w8 | mmstar | by_l2_category/biology & chemistry & physics | 146 | 50.00 |
| grace_w8 | mmstar | by_l2_category/code & sequence reasoning | 39 | 51.28 |
| grace_w8 | mmstar | by_l2_category/common reasoning | 101 | 47.52 |
| grace_w8 | mmstar | by_l2_category/cross-instance attribute reasoning | 89 | 51.69 |
| grace_w8 | mmstar | by_l2_category/cross-instance relation reasoning | 62 | 72.58 |
| grace_w8 | mmstar | by_l2_category/diagram reasoning | 110 | 53.64 |
| grace_w8 | mmstar | by_l2_category/electronics & energy & mechanical eng. | 46 | 43.48 |
| grace_w8 | mmstar | by_l2_category/geography & earth science & agriculture | 58 | 39.66 |
| grace_w8 | mmstar | by_l2_category/geometry | 116 | 32.76 |
| grace_w8 | mmstar | by_l2_category/image emotion | 31 | 87.10 |
| grace_w8 | mmstar | by_l2_category/image scene and topic | 141 | 61.70 |
| grace_w8 | mmstar | by_l2_category/image style & quality | 78 | 87.18 |
| grace_w8 | mmstar | by_l2_category/localization | 40 | 60.00 |
| grace_w8 | mmstar | by_l2_category/numeric commonsense and calculation | 48 | 41.67 |
| grace_w8 | mmstar | by_l2_category/object counting | 92 | 43.48 |
| grace_w8 | mmstar | by_l2_category/recognition | 118 | 50.85 |
| grace_w8 | mmstar | by_l2_category/single-instance reasoning | 99 | 70.71 |
| grace_w8 | mmstar | by_l2_category/statistical reasoning | 86 | 34.88 |
| qat_w4 | mmstar | by_category/coarse perception | 250 | 80.00 |
| qat_w4 | mmstar | by_category/fine-grained perception | 250 | 58.80 |
| qat_w4 | mmstar | by_category/instance reasoning | 250 | 78.40 |
| qat_w4 | mmstar | by_category/logical reasoning | 250 | 60.40 |
| qat_w4 | mmstar | by_category/math | 250 | 56.40 |
| qat_w4 | mmstar | by_category/science & technology | 250 | 49.20 |
| qat_w4 | mmstar | by_l2_category/biology & chemistry & physics | 146 | 50.68 |
| qat_w4 | mmstar | by_l2_category/code & sequence reasoning | 39 | 46.15 |
| qat_w4 | mmstar | by_l2_category/common reasoning | 101 | 68.32 |
| qat_w4 | mmstar | by_l2_category/cross-instance attribute reasoning | 89 | 71.91 |
| qat_w4 | mmstar | by_l2_category/cross-instance relation reasoning | 62 | 80.65 |
| qat_w4 | mmstar | by_l2_category/diagram reasoning | 110 | 58.18 |
| qat_w4 | mmstar | by_l2_category/electronics & energy & mechanical eng. | 46 | 58.70 |
| qat_w4 | mmstar | by_l2_category/geography & earth science & agriculture | 58 | 37.93 |
| qat_w4 | mmstar | by_l2_category/geometry | 116 | 57.76 |
| qat_w4 | mmstar | by_l2_category/image emotion | 31 | 96.77 |
| qat_w4 | mmstar | by_l2_category/image scene and topic | 141 | 69.50 |
| qat_w4 | mmstar | by_l2_category/image style & quality | 78 | 92.31 |
| qat_w4 | mmstar | by_l2_category/localization | 40 | 65.00 |
| qat_w4 | mmstar | by_l2_category/numeric commonsense and calculation | 48 | 60.42 |
| qat_w4 | mmstar | by_l2_category/object counting | 92 | 59.78 |
| qat_w4 | mmstar | by_l2_category/recognition | 118 | 55.93 |
| qat_w4 | mmstar | by_l2_category/single-instance reasoning | 99 | 82.83 |
| qat_w4 | mmstar | by_l2_category/statistical reasoning | 86 | 52.33 |
| qat_w8 | mmstar | by_category/coarse perception | 250 | 79.60 |
| qat_w8 | mmstar | by_category/fine-grained perception | 250 | 59.60 |
| qat_w8 | mmstar | by_category/instance reasoning | 250 | 78.80 |
| qat_w8 | mmstar | by_category/logical reasoning | 250 | 63.60 |
| qat_w8 | mmstar | by_category/math | 250 | 58.40 |
| qat_w8 | mmstar | by_category/science & technology | 250 | 54.40 |
| qat_w8 | mmstar | by_l2_category/biology & chemistry & physics | 146 | 56.16 |
| qat_w8 | mmstar | by_l2_category/code & sequence reasoning | 39 | 56.41 |
| qat_w8 | mmstar | by_l2_category/common reasoning | 101 | 66.34 |
| qat_w8 | mmstar | by_l2_category/cross-instance attribute reasoning | 89 | 67.42 |
| qat_w8 | mmstar | by_l2_category/cross-instance relation reasoning | 62 | 83.87 |
| qat_w8 | mmstar | by_l2_category/diagram reasoning | 110 | 63.64 |
| qat_w8 | mmstar | by_l2_category/electronics & energy & mechanical eng. | 46 | 65.22 |
| qat_w8 | mmstar | by_l2_category/geography & earth science & agriculture | 58 | 41.38 |
| qat_w8 | mmstar | by_l2_category/geometry | 116 | 65.52 |
| qat_w8 | mmstar | by_l2_category/image emotion | 31 | 93.55 |
| qat_w8 | mmstar | by_l2_category/image scene and topic | 141 | 70.92 |
| qat_w8 | mmstar | by_l2_category/image style & quality | 78 | 89.74 |
| qat_w8 | mmstar | by_l2_category/localization | 40 | 62.50 |
| qat_w8 | mmstar | by_l2_category/numeric commonsense and calculation | 48 | 54.17 |
| qat_w8 | mmstar | by_l2_category/object counting | 92 | 59.78 |
| qat_w8 | mmstar | by_l2_category/recognition | 118 | 58.47 |
| qat_w8 | mmstar | by_l2_category/single-instance reasoning | 99 | 85.86 |
| qat_w8 | mmstar | by_l2_category/statistical reasoning | 86 | 51.16 |
| released_bf16 | mmstar | by_category/coarse perception | 250 | 79.20 |
| released_bf16 | mmstar | by_category/fine-grained perception | 250 | 59.20 |
| released_bf16 | mmstar | by_category/instance reasoning | 250 | 79.60 |
| released_bf16 | mmstar | by_category/logical reasoning | 250 | 63.20 |
| released_bf16 | mmstar | by_category/math | 250 | 59.60 |
| released_bf16 | mmstar | by_category/science & technology | 250 | 53.60 |
| released_bf16 | mmstar | by_l2_category/biology & chemistry & physics | 146 | 56.16 |
| released_bf16 | mmstar | by_l2_category/code & sequence reasoning | 39 | 51.28 |
| released_bf16 | mmstar | by_l2_category/common reasoning | 101 | 65.35 |
| released_bf16 | mmstar | by_l2_category/cross-instance attribute reasoning | 89 | 70.79 |
| released_bf16 | mmstar | by_l2_category/cross-instance relation reasoning | 62 | 83.87 |
| released_bf16 | mmstar | by_l2_category/diagram reasoning | 110 | 65.45 |
| released_bf16 | mmstar | by_l2_category/electronics & energy & mechanical eng. | 46 | 58.70 |
| released_bf16 | mmstar | by_l2_category/geography & earth science & agriculture | 58 | 43.10 |
| released_bf16 | mmstar | by_l2_category/geometry | 116 | 62.07 |
| released_bf16 | mmstar | by_l2_category/image emotion | 31 | 90.32 |
| released_bf16 | mmstar | by_l2_category/image scene and topic | 141 | 71.63 |
| released_bf16 | mmstar | by_l2_category/image style & quality | 78 | 88.46 |
| released_bf16 | mmstar | by_l2_category/localization | 40 | 65.00 |
| released_bf16 | mmstar | by_l2_category/numeric commonsense and calculation | 48 | 58.33 |
| released_bf16 | mmstar | by_l2_category/object counting | 92 | 54.35 |
| released_bf16 | mmstar | by_l2_category/recognition | 118 | 61.02 |
| released_bf16 | mmstar | by_l2_category/single-instance reasoning | 99 | 84.85 |
| released_bf16 | mmstar | by_l2_category/statistical reasoning | 86 | 56.98 |
| released_grace_w4 | mmstar | by_category/coarse perception | 250 | 78.40 |
| released_grace_w4 | mmstar | by_category/fine-grained perception | 250 | 59.60 |
| released_grace_w4 | mmstar | by_category/instance reasoning | 250 | 75.60 |
| released_grace_w4 | mmstar | by_category/logical reasoning | 250 | 63.20 |
| released_grace_w4 | mmstar | by_category/math | 250 | 57.20 |
| released_grace_w4 | mmstar | by_category/science & technology | 250 | 54.00 |
| released_grace_w4 | mmstar | by_l2_category/biology & chemistry & physics | 146 | 54.11 |
| released_grace_w4 | mmstar | by_l2_category/code & sequence reasoning | 39 | 53.85 |
| released_grace_w4 | mmstar | by_l2_category/common reasoning | 101 | 66.34 |
| released_grace_w4 | mmstar | by_l2_category/cross-instance attribute reasoning | 89 | 69.66 |
| released_grace_w4 | mmstar | by_l2_category/cross-instance relation reasoning | 62 | 79.03 |
| released_grace_w4 | mmstar | by_l2_category/diagram reasoning | 110 | 63.64 |
| released_grace_w4 | mmstar | by_l2_category/electronics & energy & mechanical eng. | 46 | 60.87 |
| released_grace_w4 | mmstar | by_l2_category/geography & earth science & agriculture | 58 | 48.28 |
| released_grace_w4 | mmstar | by_l2_category/geometry | 116 | 63.79 |
| released_grace_w4 | mmstar | by_l2_category/image emotion | 31 | 96.77 |
| released_grace_w4 | mmstar | by_l2_category/image scene and topic | 141 | 68.09 |
| released_grace_w4 | mmstar | by_l2_category/image style & quality | 78 | 89.74 |
| released_grace_w4 | mmstar | by_l2_category/localization | 40 | 65.00 |
| released_grace_w4 | mmstar | by_l2_category/numeric commonsense and calculation | 48 | 52.08 |
| released_grace_w4 | mmstar | by_l2_category/object counting | 92 | 58.70 |
| released_grace_w4 | mmstar | by_l2_category/recognition | 118 | 58.47 |
| released_grace_w4 | mmstar | by_l2_category/single-instance reasoning | 99 | 78.79 |
| released_grace_w4 | mmstar | by_l2_category/statistical reasoning | 86 | 51.16 |
| released_grace_w8 | mmstar | by_category/coarse perception | 250 | 78.00 |
| released_grace_w8 | mmstar | by_category/fine-grained perception | 250 | 55.20 |
| released_grace_w8 | mmstar | by_category/instance reasoning | 250 | 75.20 |
| released_grace_w8 | mmstar | by_category/logical reasoning | 250 | 62.00 |
| released_grace_w8 | mmstar | by_category/math | 250 | 57.20 |
| released_grace_w8 | mmstar | by_category/science & technology | 250 | 54.00 |
| released_grace_w8 | mmstar | by_l2_category/biology & chemistry & physics | 146 | 52.74 |
| released_grace_w8 | mmstar | by_l2_category/code & sequence reasoning | 39 | 48.72 |
| released_grace_w8 | mmstar | by_l2_category/common reasoning | 101 | 64.36 |
| released_grace_w8 | mmstar | by_l2_category/cross-instance attribute reasoning | 89 | 69.66 |
| released_grace_w8 | mmstar | by_l2_category/cross-instance relation reasoning | 62 | 74.19 |
| released_grace_w8 | mmstar | by_l2_category/diagram reasoning | 110 | 64.55 |
| released_grace_w8 | mmstar | by_l2_category/electronics & energy & mechanical eng. | 46 | 63.04 |
| released_grace_w8 | mmstar | by_l2_category/geography & earth science & agriculture | 58 | 50.00 |
| released_grace_w8 | mmstar | by_l2_category/geometry | 116 | 64.66 |
| released_grace_w8 | mmstar | by_l2_category/image emotion | 31 | 96.77 |
| released_grace_w8 | mmstar | by_l2_category/image scene and topic | 141 | 67.38 |
| released_grace_w8 | mmstar | by_l2_category/image style & quality | 78 | 89.74 |
| released_grace_w8 | mmstar | by_l2_category/localization | 40 | 57.50 |
| released_grace_w8 | mmstar | by_l2_category/numeric commonsense and calculation | 48 | 52.08 |
| released_grace_w8 | mmstar | by_l2_category/object counting | 92 | 54.35 |
| released_grace_w8 | mmstar | by_l2_category/recognition | 118 | 55.08 |
| released_grace_w8 | mmstar | by_l2_category/single-instance reasoning | 99 | 80.81 |
| released_grace_w8 | mmstar | by_l2_category/statistical reasoning | 86 | 50.00 |
| sft_2b | mmstar | by_category/coarse perception | 250 | 70.00 |
| sft_2b | mmstar | by_category/fine-grained perception | 250 | 51.60 |
| sft_2b | mmstar | by_category/instance reasoning | 250 | 65.20 |
| sft_2b | mmstar | by_category/logical reasoning | 250 | 41.60 |
| sft_2b | mmstar | by_category/math | 250 | 47.60 |
| sft_2b | mmstar | by_category/science & technology | 250 | 38.80 |
| sft_2b | mmstar | by_l2_category/biology & chemistry & physics | 146 | 38.36 |
| sft_2b | mmstar | by_l2_category/code & sequence reasoning | 39 | 35.90 |
| sft_2b | mmstar | by_l2_category/common reasoning | 101 | 29.70 |
| sft_2b | mmstar | by_l2_category/cross-instance attribute reasoning | 89 | 52.81 |
| sft_2b | mmstar | by_l2_category/cross-instance relation reasoning | 62 | 74.19 |
| sft_2b | mmstar | by_l2_category/diagram reasoning | 110 | 54.55 |
| sft_2b | mmstar | by_l2_category/electronics & energy & mechanical eng. | 46 | 45.65 |
| sft_2b | mmstar | by_l2_category/geography & earth science & agriculture | 58 | 34.48 |
| sft_2b | mmstar | by_l2_category/geometry | 116 | 49.14 |
| sft_2b | mmstar | by_l2_category/image emotion | 31 | 80.65 |
| sft_2b | mmstar | by_l2_category/image scene and topic | 141 | 57.45 |
| sft_2b | mmstar | by_l2_category/image style & quality | 78 | 88.46 |
| sft_2b | mmstar | by_l2_category/localization | 40 | 55.00 |
| sft_2b | mmstar | by_l2_category/numeric commonsense and calculation | 48 | 45.83 |
| sft_2b | mmstar | by_l2_category/object counting | 92 | 51.09 |
| sft_2b | mmstar | by_l2_category/recognition | 118 | 50.85 |
| sft_2b | mmstar | by_l2_category/single-instance reasoning | 99 | 70.71 |
| sft_2b | mmstar | by_l2_category/statistical reasoning | 86 | 46.51 |
| vanilla_2b | mmstar | by_category/coarse perception | 250 | 66.40 |
| vanilla_2b | mmstar | by_category/fine-grained perception | 250 | 46.40 |
| vanilla_2b | mmstar | by_category/instance reasoning | 250 | 57.20 |
| vanilla_2b | mmstar | by_category/logical reasoning | 250 | 37.60 |
| vanilla_2b | mmstar | by_category/math | 250 | 28.00 |
| vanilla_2b | mmstar | by_category/science & technology | 250 | 32.40 |
| vanilla_2b | mmstar | by_l2_category/biology & chemistry & physics | 146 | 30.82 |
| vanilla_2b | mmstar | by_l2_category/code & sequence reasoning | 39 | 38.46 |
| vanilla_2b | mmstar | by_l2_category/common reasoning | 101 | 34.65 |
| vanilla_2b | mmstar | by_l2_category/cross-instance attribute reasoning | 89 | 49.44 |
| vanilla_2b | mmstar | by_l2_category/cross-instance relation reasoning | 62 | 53.23 |
| vanilla_2b | mmstar | by_l2_category/diagram reasoning | 110 | 40.00 |
| vanilla_2b | mmstar | by_l2_category/electronics & energy & mechanical eng. | 46 | 39.13 |
| vanilla_2b | mmstar | by_l2_category/geography & earth science & agriculture | 58 | 31.03 |
| vanilla_2b | mmstar | by_l2_category/geometry | 116 | 23.28 |
| vanilla_2b | mmstar | by_l2_category/image emotion | 31 | 74.19 |
| vanilla_2b | mmstar | by_l2_category/image scene and topic | 141 | 53.90 |
| vanilla_2b | mmstar | by_l2_category/image style & quality | 78 | 85.90 |
| vanilla_2b | mmstar | by_l2_category/localization | 40 | 52.50 |
| vanilla_2b | mmstar | by_l2_category/numeric commonsense and calculation | 48 | 33.33 |
| vanilla_2b | mmstar | by_l2_category/object counting | 92 | 39.13 |
| vanilla_2b | mmstar | by_l2_category/recognition | 118 | 50.00 |
| vanilla_2b | mmstar | by_l2_category/single-instance reasoning | 99 | 66.67 |
| vanilla_2b | mmstar | by_l2_category/statistical reasoning | 86 | 31.40 |
| vanilla_8b_teacher | mmstar | by_category/coarse perception | 250 | 72.80 |
| vanilla_8b_teacher | mmstar | by_category/fine-grained perception | 250 | 50.80 |
| vanilla_8b_teacher | mmstar | by_category/instance reasoning | 250 | 71.20 |
| vanilla_8b_teacher | mmstar | by_category/logical reasoning | 250 | 60.00 |
| vanilla_8b_teacher | mmstar | by_category/math | 250 | 52.80 |
| vanilla_8b_teacher | mmstar | by_category/science & technology | 250 | 47.20 |
| vanilla_8b_teacher | mmstar | by_l2_category/biology & chemistry & physics | 146 | 52.05 |
| vanilla_8b_teacher | mmstar | by_l2_category/code & sequence reasoning | 39 | 51.28 |
| vanilla_8b_teacher | mmstar | by_l2_category/common reasoning | 101 | 59.41 |
| vanilla_8b_teacher | mmstar | by_l2_category/cross-instance attribute reasoning | 89 | 68.54 |
| vanilla_8b_teacher | mmstar | by_l2_category/cross-instance relation reasoning | 62 | 75.81 |
| vanilla_8b_teacher | mmstar | by_l2_category/diagram reasoning | 110 | 63.64 |
| vanilla_8b_teacher | mmstar | by_l2_category/electronics & energy & mechanical eng. | 46 | 41.30 |
| vanilla_8b_teacher | mmstar | by_l2_category/geography & earth science & agriculture | 58 | 39.66 |
| vanilla_8b_teacher | mmstar | by_l2_category/geometry | 116 | 56.03 |
| vanilla_8b_teacher | mmstar | by_l2_category/image emotion | 31 | 64.52 |
| vanilla_8b_teacher | mmstar | by_l2_category/image scene and topic | 141 | 65.25 |
| vanilla_8b_teacher | mmstar | by_l2_category/image style & quality | 78 | 89.74 |
| vanilla_8b_teacher | mmstar | by_l2_category/localization | 40 | 70.00 |
| vanilla_8b_teacher | mmstar | by_l2_category/numeric commonsense and calculation | 48 | 52.08 |
| vanilla_8b_teacher | mmstar | by_l2_category/object counting | 92 | 45.65 |
| vanilla_8b_teacher | mmstar | by_l2_category/recognition | 118 | 48.31 |
| vanilla_8b_teacher | mmstar | by_l2_category/single-instance reasoning | 99 | 70.71 |
| vanilla_8b_teacher | mmstar | by_l2_category/statistical reasoning | 86 | 48.84 |

Acceptance thresholds remain for supervisor discussion. BF16 storage of quantized weights does not measure packed W4/W8 memory or speed.
