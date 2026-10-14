# Vanilla 2B / SFT / QAT / GRACE evaluation comparison

Source: `/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/experiments/constrained_32k_b128_p100352_t2048_seed42/evaluation`. Read 28 merged JSON files (not rank shards). Validated row counts, correct counts, primary accuracies, unique question identities and matching identities/gold labels across methods.

## Primary scores

| Suite / metric | N | Vanilla 2B (%) | SFT (%) | QAT (%) | GRACE (%) | GRACE−Vanilla 2B (pp) | GRACE−QAT (pp) | GRACE−SFT (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HallusionBench aAcc | 1129 | 59.88 | 64.13 | 56.95 | 60.05 | +0.18 | +3.10 | -4.07 |
| MMBench EN dev (per-row) | 4329 | 83.37 | 83.85 | 82.88 | 81.89 | -1.48 | -0.99 | -1.96 |
| ScienceQA test (all) | 4241 | 82.95 | 82.72 | 81.54 | 80.45 | -2.50 | -1.08 | -2.26 |
| AI2D test | 3088 | 74.81 | 74.94 | 72.93 | 72.41 | -2.40 | -0.52 | -2.53 |
| MMMU validation | 900 | 40.00 | 43.00 | 43.78 | 39.89 | -0.11 | -3.89 | -3.11 |
| SEED-Bench-IMG | 14233 | 72.46 | 73.06 | 73.03 | 72.59 | +0.13 | -0.44 | -0.47 |
| MMStar | 1500 | 44.67 | 52.47 | 50.13 | 47.47 | +2.80 | -2.67 | -5.00 |
| Unweighted mean (7 suites) | — | 65.45 | 67.74 | 65.89 | 64.96 | -0.48 | -0.93 | -2.77 |

All scores are percentages; deltas are percentage points (pp). The mean gives each suite equal weight, uses HallusionBench aAcc and full-test ScienceQA, and is a descriptive aggregate, not an official benchmark metric.

## Additional metrics

| Suite / metric | N | Vanilla 2B (%) | SFT (%) | QAT (%) | GRACE (%) | GRACE−Vanilla 2B (pp) | GRACE−QAT (pp) | GRACE−SFT (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HallusionBench qAcc | 455 | 26.37 | 31.87 | 24.18 | 26.37 | +0.00 | +2.20 | -5.49 |
| HallusionBench fAcc | 346 | 31.79 | 35.55 | 29.77 | 34.68 | +2.89 | +4.91 | -0.87 |
| ScienceQA test (image) | 2017 | 85.62 | 84.78 | 82.50 | 82.15 | -3.47 | -0.35 | -2.63 |
| ScienceQA test (text) | 2224 | 80.53 | 80.85 | 80.67 | 78.91 | -1.62 | -1.75 | -1.93 |

HallusionBench qAcc requires every answer in a question group to be correct; fAcc requires every answer in an image/figure group to be correct. N is the number of groups. ScienceQA image is the image-containing subset of the full test split.

## Execution and parsing

| Suite | Vanilla 2B errors / parse fails | SFT errors / parse fails | QAT errors / parse fails | GRACE errors / parse fails |
| --- | ---: | ---: | ---: | ---: |
| hallusionbench | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| mmbench | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| scienceqa | 0 / 1 | 0 / 0 | 0 / 4 | 0 / 6 |
| ai2d | 0 / 2 | 0 / 0 | 0 / 0 | 0 / 1 |
| mmmu | 0 / not reported | 0 / not reported | 0 / not reported | 0 / not reported |
| seedbench_img | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| mmstar | 0 / 1 | 0 / 0 | 0 / 0 | 0 / 0 |

Errors are counted from individual predictions. Parse failures remain in the accuracy denominator. MMMU does not report a separate parse-failure count.

## Interpretation limits

These are the repository's local evaluation protocols. MMBench uses per-row accuracy; MMMU uses the local open-answer scorer. Do not assume official leaderboard equivalence. Vanilla 2B, when present, is Qwen3-VL-2B-Instruct without additional fine-tuning, loaded in BF16. The default launcher uses the SFT processor to match its image budget; baseline result metadata records the actual processor and budget. SFT is the fine-tuned BF16 quality reference; QAT and GRACE use the matched W4G128 training/export setup described in README_constrained.md. These scores do not measure packed INT4 inference speed or deployment memory. One seed does not establish repeatability.

## All category breakdowns

| Suite / metric | N | Vanilla 2B (%) | SFT (%) | QAT (%) | GRACE (%) | GRACE−Vanilla 2B (pp) | GRACE−QAT (pp) | GRACE−SFT (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hallusionbench / by_category / VD | 591 | 59.39 | 62.94 | 57.02 | 59.73 | +0.34 | +2.71 | -3.21 |
| hallusionbench / by_category / VS | 538 | 60.41 | 65.43 | 56.88 | 60.41 | +0.00 | +3.53 | -5.02 |
| hallusionbench / by_modality / image | 951 | 61.20 | 65.40 | 58.78 | 62.46 | +1.26 | +3.68 | -2.94 |
| hallusionbench / by_modality / text | 178 | 52.81 | 57.30 | 47.19 | 47.19 | -5.62 | +0.00 | -10.11 |
| hallusionbench / by_subcategory / VD/figure | 80 | 71.25 | 77.50 | 68.75 | 67.50 | -3.75 | -1.25 | -10.00 |
| hallusionbench / by_subcategory / VD/illusion | 144 | 60.42 | 62.50 | 57.64 | 64.58 | +4.17 | +6.94 | +2.08 |
| hallusionbench / by_subcategory / VD/math | 108 | 47.22 | 53.70 | 43.52 | 52.78 | +5.56 | +9.26 | -0.93 |
| hallusionbench / by_subcategory / VD/ocr | 89 | 78.65 | 78.65 | 77.53 | 79.78 | +1.12 | +2.25 | +1.12 |
| hallusionbench / by_subcategory / VD/video | 170 | 50.59 | 54.12 | 48.82 | 45.88 | -4.71 | -2.94 | -8.24 |
| hallusionbench / by_subcategory / VS/chart | 206 | 51.94 | 61.65 | 47.57 | 54.37 | +2.43 | +6.80 | -7.28 |
| hallusionbench / by_subcategory / VS/map | 96 | 57.29 | 66.67 | 53.12 | 57.29 | +0.00 | +4.17 | -9.38 |
| hallusionbench / by_subcategory / VS/ocr | 81 | 58.02 | 62.96 | 61.73 | 59.26 | +1.23 | -2.47 | -3.70 |
| hallusionbench / by_subcategory / VS/table | 155 | 74.84 | 70.97 | 69.03 | 70.97 | -3.87 | +1.94 | +0.00 |
| mmbench / by_category / action_recognition | 215 | 96.28 | 94.88 | 94.88 | 94.42 | -1.86 | -0.47 | -0.47 |
| mmbench / by_category / attribute_comparison | 141 | 77.30 | 71.63 | 73.05 | 68.79 | -8.51 | -4.26 | -2.84 |
| mmbench / by_category / attribute_recognition | 264 | 92.80 | 94.70 | 92.80 | 91.67 | -1.14 | -1.14 | -3.03 |
| mmbench / by_category / celebrity_recognition | 396 | 93.43 | 95.20 | 93.94 | 92.42 | -1.01 | -1.52 | -2.78 |
| mmbench / by_category / function_reasoning | 304 | 92.11 | 92.11 | 91.78 | 90.79 | -1.32 | -0.99 | -1.32 |
| mmbench / by_category / future_prediction | 130 | 44.62 | 41.54 | 41.54 | 43.85 | -0.77 | +2.31 | +2.31 |
| mmbench / by_category / identity_reasoning | 176 | 97.16 | 97.16 | 96.59 | 96.59 | -0.57 | +0.00 | -0.57 |
| mmbench / by_category / image_emotion | 200 | 83.00 | 85.00 | 85.00 | 85.50 | +2.50 | +0.50 | +0.50 |
| mmbench / by_category / image_quality | 150 | 65.33 | 64.67 | 61.33 | 62.00 | -3.33 | +0.67 | -2.67 |
| mmbench / by_category / image_scene | 407 | 97.54 | 97.79 | 97.54 | 97.05 | -0.49 | -0.49 | -0.74 |
| mmbench / by_category / image_style | 212 | 92.92 | 93.87 | 92.45 | 90.57 | -2.36 | -1.89 | -3.30 |
| mmbench / by_category / image_topic | 140 | 94.29 | 90.71 | 90.00 | 92.14 | -2.14 | +2.14 | +1.43 |
| mmbench / by_category / nature_relation | 179 | 74.86 | 75.42 | 76.54 | 69.83 | -5.03 | -6.70 | -5.59 |
| mmbench / by_category / object_localization | 315 | 70.16 | 68.57 | 67.62 | 68.25 | -1.90 | +0.63 | -0.32 |
| mmbench / by_category / ocr | 156 | 91.67 | 90.38 | 87.18 | 86.54 | -5.13 | -0.64 | -3.85 |
| mmbench / by_category / physical_property_reasoning | 219 | 79.00 | 80.82 | 79.00 | 72.15 | -6.85 | -6.85 | -8.68 |
| mmbench / by_category / physical_relation | 94 | 68.09 | 79.79 | 76.60 | 76.60 | +8.51 | +0.00 | -3.19 |
| mmbench / by_category / social_relation | 172 | 90.12 | 92.44 | 95.35 | 94.19 | +4.07 | -1.16 | +1.74 |
| mmbench / by_category / spatial_relationship | 177 | 41.81 | 43.50 | 40.68 | 42.94 | +1.13 | +2.26 | -0.56 |
| mmbench / by_category / structuralized_imagetext_understanding | 282 | 76.24 | 78.72 | 75.53 | 74.82 | -1.42 | -0.71 | -3.90 |
| mmbench / by_l2_category / attribute_reasoning | 699 | 89.27 | 89.84 | 88.98 | 86.41 | -2.86 | -2.58 | -3.43 |
| mmbench / by_l2_category / coarse_perception | 1109 | 89.27 | 89.36 | 88.46 | 88.37 | -0.90 | -0.09 | -0.99 |
| mmbench / by_l2_category / finegrained_perception (cross-instance) | 533 | 73.17 | 71.67 | 71.11 | 70.54 | -2.63 | -0.56 | -1.13 |
| mmbench / by_l2_category / finegrained_perception (instance-level) | 1131 | 86.56 | 87.00 | 85.41 | 84.70 | -1.86 | -0.71 | -2.30 |
| mmbench / by_l2_category / logic_reasoning | 412 | 66.26 | 66.99 | 64.81 | 65.05 | -1.21 | +0.24 | -1.94 |
| mmbench / by_l2_category / relation_reasoning | 445 | 79.33 | 82.92 | 83.82 | 80.67 | +1.35 | -3.15 | -2.25 |
| scienceqa / by_category / 20th century American history | 4 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Adaptations | 98 | 84.69 | 89.80 | 87.76 | 82.65 | -2.04 | -5.10 | -7.14 |
| scienceqa / by_category / Age of Exploration | 1 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Analyzing literature | 4 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Anatomy and physiology | 2 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Ancient Egypt and Kush | 1 | 100.00 | 0.00 | 100.00 | 100.00 | +0.00 | +0.00 | +100.00 |
| scienceqa / by_category / Ancient Mesopotamia | 2 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Animals | 8 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Astronomy | 37 | 54.05 | 62.16 | 45.95 | 64.86 | +10.81 | +18.92 | +2.70 |
| scienceqa / by_category / Atoms and molecules | 64 | 98.44 | 98.44 | 95.31 | 95.31 | -3.12 | +0.00 | -3.12 |
| scienceqa / by_category / Audience, purpose, and tone | 17 | 100.00 | 94.12 | 94.12 | 94.12 | -5.88 | +0.00 | +0.00 |
| scienceqa / by_category / Author's purpose and tone | 27 | 85.19 | 74.07 | 88.89 | 85.19 | +0.00 | -3.70 | +11.11 |
| scienceqa / by_category / Banking and finance | 1 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Basic economic principles | 116 | 86.21 | 89.66 | 79.31 | 73.28 | -12.93 | -6.03 | -16.38 |
| scienceqa / by_category / Capitalization | 18 | 61.11 | 66.67 | 72.22 | 55.56 | -5.56 | -16.67 | -11.11 |
| scienceqa / by_category / Categories | 11 | 36.36 | 63.64 | 54.55 | 45.45 | +9.09 | -9.09 | -18.18 |
| scienceqa / by_category / Cells | 17 | 76.47 | 76.47 | 70.59 | 82.35 | +5.88 | +11.76 | +5.88 |
| scienceqa / by_category / Chemical reactions | 43 | 93.02 | 76.74 | 81.40 | 88.37 | -4.65 | +6.98 | +11.63 |
| scienceqa / by_category / Cities | 12 | 58.33 | 58.33 | 50.00 | 41.67 | -16.67 | -8.33 | -16.67 |
| scienceqa / by_category / Classification | 161 | 96.89 | 97.52 | 95.65 | 96.89 | +0.00 | +1.24 | -0.62 |
| scienceqa / by_category / Classification and scientific names | 46 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Climate change | 1 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Colonial America | 38 | 92.11 | 86.84 | 89.47 | 89.47 | -2.63 | +0.00 | +2.63 |
| scienceqa / by_category / Comprehension strategies | 5 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Conservation and natural resources | 1 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Context clues | 5 | 60.00 | 80.00 | 80.00 | 60.00 | +0.00 | -20.00 | -20.00 |
| scienceqa / by_category / Creative techniques | 19 | 68.42 | 68.42 | 63.16 | 73.68 | +5.26 | +10.53 | +5.26 |
| scienceqa / by_category / Cultural celebrations | 2 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Descriptive details | 3 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Designing experiments | 134 | 94.78 | 96.27 | 97.01 | 95.52 | +0.75 | -1.49 | -0.75 |
| scienceqa / by_category / Developing and supporting arguments | 101 | 87.13 | 88.12 | 85.15 | 85.15 | -1.98 | +0.00 | -2.97 |
| scienceqa / by_category / Domain-specific vocabulary | 4 | 100.00 | 75.00 | 100.00 | 100.00 | +0.00 | +0.00 | +25.00 |
| scienceqa / by_category / Early 19th century American history | 2 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Earth events | 18 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Ecological interactions | 24 | 58.33 | 58.33 | 58.33 | 50.00 | -8.33 | -8.33 | -8.33 |
| scienceqa / by_category / Economics | 11 | 81.82 | 72.73 | 54.55 | 63.64 | -18.18 | +9.09 | -9.09 |
| scienceqa / by_category / Ecosystems | 78 | 87.18 | 89.74 | 88.46 | 85.90 | -1.28 | -2.56 | -3.85 |
| scienceqa / by_category / Editing and revising | 9 | 55.56 | 55.56 | 55.56 | 55.56 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Electricity | 1 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Engineering practices | 55 | 100.00 | 98.18 | 96.36 | 96.36 | -3.64 | +0.00 | -1.82 |
| scienceqa / by_category / English colonies in North America | 39 | 94.87 | 89.74 | 89.74 | 89.74 | -5.13 | +0.00 | +0.00 |
| scienceqa / by_category / Force and motion | 58 | 86.21 | 81.03 | 79.31 | 79.31 | -6.90 | +0.00 | -1.72 |
| scienceqa / by_category / Formatting | 73 | 64.38 | 75.34 | 78.08 | 65.75 | +1.37 | -12.33 | -9.59 |
| scienceqa / by_category / Fossils | 22 | 59.09 | 63.64 | 72.73 | 72.73 | +13.64 | +0.00 | +9.09 |
| scienceqa / by_category / Genes to traits | 195 | 83.59 | 81.54 | 82.05 | 82.05 | -1.54 | +0.00 | +0.51 |
| scienceqa / by_category / Geography | 46 | 95.65 | 93.48 | 93.48 | 93.48 | -2.17 | +0.00 | +0.00 |
| scienceqa / by_category / Government | 16 | 81.25 | 81.25 | 81.25 | 81.25 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Greece | 1 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Heat and thermal energy | 42 | 92.86 | 100.00 | 95.24 | 90.48 | -2.38 | -4.76 | -9.52 |
| scienceqa / by_category / Heredity | 41 | 100.00 | 82.93 | 97.56 | 97.56 | -2.44 | +0.00 | +14.63 |
| scienceqa / by_category / Historical figures | 2 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Independent reading comprehension | 5 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Informational texts: level 1 | 5 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Kinetic and potential energy | 19 | 84.21 | 78.95 | 84.21 | 73.68 | -10.53 | -10.53 | -5.26 |
| scienceqa / by_category / Literary devices | 267 | 77.90 | 79.03 | 77.53 | 73.78 | -4.12 | -3.75 | -5.24 |
| scienceqa / by_category / Magnets | 110 | 41.82 | 44.55 | 41.82 | 47.27 | +5.45 | +5.45 | +2.73 |
| scienceqa / by_category / Maps | 46 | 86.96 | 84.78 | 86.96 | 80.43 | -6.52 | -6.52 | -4.35 |
| scienceqa / by_category / Materials | 167 | 92.81 | 94.01 | 91.62 | 89.82 | -2.99 | -1.80 | -4.19 |
| scienceqa / by_category / Medieval Asia | 2 | 50.00 | 50.00 | 50.00 | 50.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Mixtures | 6 | 83.33 | 83.33 | 83.33 | 50.00 | -33.33 | -33.33 | -33.33 |
| scienceqa / by_category / Oceania: geography | 51 | 82.35 | 68.63 | 64.71 | 70.59 | -11.76 | +5.88 | +1.96 |
| scienceqa / by_category / Oceans and continents | 19 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Oceans and continents	 | 4 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Opinion writing | 4 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Particle motion and energy | 41 | 95.12 | 80.49 | 80.49 | 90.24 | -4.88 | +9.76 | +9.76 |
| scienceqa / by_category / Persuasive strategies | 11 | 100.00 | 81.82 | 63.64 | 63.64 | -36.36 | +0.00 | -18.18 |
| scienceqa / by_category / Photosynthesis | 4 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Phrases and clauses | 32 | 59.38 | 56.25 | 37.50 | 43.75 | -15.62 | +6.25 | -12.50 |
| scienceqa / by_category / Physical Geography | 51 | 98.04 | 96.08 | 96.08 | 88.24 | -9.80 | -7.84 | -7.84 |
| scienceqa / by_category / Physical and chemical change | 52 | 94.23 | 92.31 | 90.38 | 90.38 | -3.85 | +0.00 | -1.92 |
| scienceqa / by_category / Plant reproduction | 6 | 83.33 | 83.33 | 83.33 | 83.33 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Plants | 12 | 91.67 | 91.67 | 91.67 | 91.67 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Plate tectonics | 3 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Poetry elements | 4 | 75.00 | 50.00 | 100.00 | 75.00 | +0.00 | -25.00 | +25.00 |
| scienceqa / by_category / Pronouns | 37 | 59.46 | 64.86 | 62.16 | 51.35 | -8.11 | -10.81 | -13.51 |
| scienceqa / by_category / Pronouns and antecedents | 18 | 44.44 | 55.56 | 44.44 | 33.33 | -11.11 | -11.11 | -22.22 |
| scienceqa / by_category / Read-alone texts | 2 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Reference skills | 140 | 46.43 | 50.71 | 47.86 | 49.29 | +2.86 | +1.43 | -1.43 |
| scienceqa / by_category / Research skills | 4 | 75.00 | 75.00 | 75.00 | 50.00 | -25.00 | -25.00 | -25.00 |
| scienceqa / by_category / Rhyming | 17 | 58.82 | 41.18 | 70.59 | 47.06 | -11.76 | -23.53 | +5.88 |
| scienceqa / by_category / Rocks and minerals | 46 | 97.83 | 95.65 | 97.83 | 95.65 | -2.17 | -2.17 | +0.00 |
| scienceqa / by_category / Scientific names | 32 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Sentences, fragments, and run-ons | 135 | 85.93 | 88.15 | 82.22 | 86.67 | +0.74 | +4.44 | -1.48 |
| scienceqa / by_category / Shades of meaning | 10 | 100.00 | 90.00 | 90.00 | 100.00 | +0.00 | +10.00 | +10.00 |
| scienceqa / by_category / Short and long vowels | 5 | 40.00 | 60.00 | 60.00 | 20.00 | -20.00 | -40.00 | -40.00 |
| scienceqa / by_category / Social studies skills | 12 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Solutions | 54 | 68.52 | 61.11 | 70.37 | 66.67 | -1.85 | -3.70 | +5.56 |
| scienceqa / by_category / State capitals | 312 | 98.08 | 96.79 | 91.35 | 89.42 | -8.65 | -1.92 | -7.37 |
| scienceqa / by_category / States | 18 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / States of matter | 47 | 89.36 | 87.23 | 89.36 | 89.36 | +0.00 | +0.00 | +2.13 |
| scienceqa / by_category / Supply and demand | 27 | 74.07 | 70.37 | 74.07 | 74.07 | +0.00 | +0.00 | +3.70 |
| scienceqa / by_category / Text structure | 2 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / The American Revolution | 4 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / The Americas: geography | 45 | 66.67 | 66.67 | 57.78 | 55.56 | -11.11 | -2.22 | -11.11 |
| scienceqa / by_category / The Antebellum period | 1 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / The Civil War and Reconstruction | 2 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Thermal energy | 40 | 70.00 | 70.00 | 70.00 | 77.50 | +7.50 | +7.50 | +7.50 |
| scienceqa / by_category / Topographic maps | 1 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Traits and heredity | 81 | 91.36 | 83.95 | 97.53 | 95.06 | +3.70 | -2.47 | +11.11 |
| scienceqa / by_category / Units and measurement | 200 | 86.00 | 87.00 | 90.00 | 88.50 | +2.50 | -1.50 | +1.50 |
| scienceqa / by_category / Velocity, acceleration, and forces | 86 | 52.33 | 54.65 | 59.30 | 55.81 | +3.49 | -3.49 | +1.16 |
| scienceqa / by_category / Verb tense | 52 | 78.85 | 76.92 | 76.92 | 76.92 | -1.92 | +0.00 | +0.00 |
| scienceqa / by_category / Visual elements | 9 | 100.00 | 100.00 | 88.89 | 88.89 | -11.11 | +0.00 | -11.11 |
| scienceqa / by_category / Water cycle | 2 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Weather and climate | 97 | 77.32 | 80.41 | 71.13 | 71.13 | -6.19 | +0.00 | -9.28 |
| scienceqa / by_category / Word usage and nuance | 45 | 60.00 | 66.67 | 62.22 | 66.67 | +6.67 | +4.44 | +0.00 |
| scienceqa / by_category / World religions | 1 | 100.00 | 100.00 | 0.00 | 0.00 | -100.00 | +0.00 | -100.00 |
| scienceqa / by_modality / image | 2017 | 85.62 | 84.78 | 82.50 | 82.15 | -3.47 | -0.35 | -2.63 |
| scienceqa / by_modality / text | 2224 | 80.53 | 80.85 | 80.67 | 78.91 | -1.62 | -1.75 | -1.93 |
| scienceqa / by_subject / language science | 1100 | 72.64 | 74.45 | 72.64 | 70.45 | -2.18 | -2.18 | -4.00 |
| scienceqa / by_subject / natural science | 2252 | 84.72 | 84.15 | 84.37 | 84.19 | -0.53 | -0.18 | +0.04 |
| scienceqa / by_subject / social science | 889 | 91.23 | 89.31 | 85.38 | 83.35 | -7.87 | -2.02 | -5.96 |
| mmmu / by_question_type / multiple-choice | 847 | 41.91 | 45.22 | 46.04 | 42.15 | +0.24 | -3.90 | -3.07 |
| mmmu / by_question_type / open | 53 | 9.43 | 7.55 | 7.55 | 3.77 | -5.66 | -3.77 | -3.77 |
| mmmu / by_subject / Accounting | 30 | 33.33 | 16.67 | 26.67 | 40.00 | +6.67 | +13.33 | +23.33 |
| mmmu / by_subject / Agriculture | 30 | 33.33 | 30.00 | 33.33 | 33.33 | +0.00 | +0.00 | +3.33 |
| mmmu / by_subject / Architecture_and_Engineering | 30 | 26.67 | 30.00 | 36.67 | 23.33 | -3.33 | -13.33 | -6.67 |
| mmmu / by_subject / Art | 30 | 53.33 | 53.33 | 63.33 | 63.33 | +10.00 | +0.00 | +10.00 |
| mmmu / by_subject / Art_Theory | 30 | 70.00 | 70.00 | 70.00 | 70.00 | +0.00 | +0.00 | +0.00 |
| mmmu / by_subject / Basic_Medical_Science | 30 | 46.67 | 40.00 | 53.33 | 46.67 | +0.00 | -6.67 | +6.67 |
| mmmu / by_subject / Biology | 30 | 33.33 | 36.67 | 30.00 | 33.33 | +0.00 | +3.33 | -3.33 |
| mmmu / by_subject / Chemistry | 30 | 23.33 | 26.67 | 13.33 | 16.67 | -6.67 | +3.33 | -10.00 |
| mmmu / by_subject / Clinical_Medicine | 30 | 33.33 | 40.00 | 43.33 | 36.67 | +3.33 | -6.67 | -3.33 |
| mmmu / by_subject / Computer_Science | 30 | 46.67 | 36.67 | 46.67 | 40.00 | -6.67 | -6.67 | +3.33 |
| mmmu / by_subject / Design | 30 | 73.33 | 60.00 | 66.67 | 63.33 | -10.00 | -3.33 | +3.33 |
| mmmu / by_subject / Diagnostics_and_Laboratory_Medicine | 30 | 36.67 | 40.00 | 43.33 | 40.00 | +3.33 | -3.33 | +0.00 |
| mmmu / by_subject / Economics | 30 | 33.33 | 46.67 | 43.33 | 33.33 | +0.00 | -10.00 | -13.33 |
| mmmu / by_subject / Electronics | 30 | 26.67 | 30.00 | 30.00 | 33.33 | +6.67 | +3.33 | +3.33 |
| mmmu / by_subject / Energy_and_Power | 30 | 26.67 | 30.00 | 36.67 | 33.33 | +6.67 | -3.33 | +3.33 |
| mmmu / by_subject / Finance | 30 | 23.33 | 36.67 | 30.00 | 23.33 | +0.00 | -6.67 | -13.33 |
| mmmu / by_subject / Geography | 30 | 53.33 | 46.67 | 56.67 | 53.33 | +0.00 | -3.33 | +6.67 |
| mmmu / by_subject / History | 30 | 53.33 | 43.33 | 50.00 | 36.67 | -16.67 | -13.33 | -6.67 |
| mmmu / by_subject / Literature | 30 | 83.33 | 86.67 | 80.00 | 76.67 | -6.67 | -3.33 | -10.00 |
| mmmu / by_subject / Manage | 30 | 30.00 | 43.33 | 40.00 | 33.33 | +3.33 | -6.67 | -10.00 |
| mmmu / by_subject / Marketing | 30 | 36.67 | 43.33 | 53.33 | 43.33 | +6.67 | -10.00 | +0.00 |
| mmmu / by_subject / Materials | 30 | 16.67 | 33.33 | 23.33 | 26.67 | +10.00 | +3.33 | -6.67 |
| mmmu / by_subject / Math | 30 | 33.33 | 36.67 | 50.00 | 26.67 | -6.67 | -23.33 | -10.00 |
| mmmu / by_subject / Mechanical_Engineering | 30 | 26.67 | 43.33 | 30.00 | 30.00 | +3.33 | +0.00 | -13.33 |
| mmmu / by_subject / Music | 30 | 40.00 | 36.67 | 30.00 | 33.33 | -6.67 | +3.33 | -3.33 |
| mmmu / by_subject / Pharmacy | 30 | 43.33 | 50.00 | 43.33 | 33.33 | -10.00 | -10.00 | -16.67 |
| mmmu / by_subject / Physics | 30 | 30.00 | 36.67 | 33.33 | 30.00 | +0.00 | -3.33 | -6.67 |
| mmmu / by_subject / Psychology | 30 | 53.33 | 53.33 | 50.00 | 40.00 | -13.33 | -10.00 | -13.33 |
| mmmu / by_subject / Public_Health | 30 | 30.00 | 50.00 | 50.00 | 46.67 | +16.67 | -3.33 | -3.33 |
| mmmu / by_subject / Sociology | 30 | 50.00 | 63.33 | 56.67 | 56.67 | +6.67 | +0.00 | -6.67 |
| seedbench_img / by_question_type / Instance Attributes | 4649 | 74.36 | 74.23 | 75.63 | 74.55 | +0.19 | -1.08 | +0.32 |
| seedbench_img / by_question_type / Instance Identity | 1831 | 77.55 | 77.06 | 76.68 | 76.73 | -0.82 | +0.05 | -0.33 |
| seedbench_img / by_question_type / Instance Interaction | 97 | 76.29 | 73.20 | 76.29 | 78.35 | +2.06 | +2.06 | +5.15 |
| seedbench_img / by_question_type / Instance Location | 978 | 67.38 | 68.10 | 68.61 | 68.81 | +1.43 | +0.20 | +0.72 |
| seedbench_img / by_question_type / Instances Counting | 2447 | 63.51 | 66.82 | 65.43 | 63.75 | +0.25 | -1.68 | -3.06 |
| seedbench_img / by_question_type / Scene Understanding | 3158 | 77.96 | 78.28 | 77.49 | 77.96 | +0.00 | +0.47 | -0.32 |
| seedbench_img / by_question_type / Spatial Relation | 657 | 57.08 | 56.62 | 56.62 | 56.93 | -0.15 | +0.30 | +0.30 |
| seedbench_img / by_question_type / Text Understanding | 85 | 63.53 | 68.24 | 67.06 | 65.88 | +2.35 | -1.18 | -2.35 |
| seedbench_img / by_question_type / Visual Reasoning | 331 | 77.95 | 79.46 | 76.44 | 78.55 | +0.60 | +2.11 | -0.91 |
| mmstar / by_category / coarse perception | 250 | 66.40 | 70.00 | 64.80 | 65.60 | -0.80 | +0.80 | -4.40 |
| mmstar / by_category / fine-grained perception | 250 | 46.40 | 51.60 | 49.20 | 44.80 | -1.60 | -4.40 | -6.80 |
| mmstar / by_category / instance reasoning | 250 | 57.20 | 65.20 | 61.60 | 61.20 | +4.00 | -0.40 | -4.00 |
| mmstar / by_category / logical reasoning | 250 | 37.60 | 41.60 | 43.20 | 41.20 | +3.60 | -2.00 | -0.40 |
| mmstar / by_category / math | 250 | 28.00 | 47.60 | 46.80 | 43.60 | +15.60 | -3.20 | -4.00 |
| mmstar / by_category / science & technology | 250 | 32.40 | 38.80 | 35.20 | 28.40 | -4.00 | -6.80 | -10.40 |
| mmstar / by_l2_category / biology & chemistry & physics | 146 | 30.82 | 38.36 | 34.93 | 23.97 | -6.85 | -10.96 | -14.38 |
| mmstar / by_l2_category / code & sequence reasoning | 39 | 38.46 | 35.90 | 33.33 | 38.46 | +0.00 | +5.13 | +2.56 |
| mmstar / by_l2_category / common reasoning | 101 | 34.65 | 29.70 | 36.63 | 39.60 | +4.95 | +2.97 | +9.90 |
| mmstar / by_l2_category / cross-instance attribute reasoning | 89 | 49.44 | 52.81 | 47.19 | 48.31 | -1.12 | +1.12 | -4.49 |
| mmstar / by_l2_category / cross-instance relation reasoning | 62 | 53.23 | 74.19 | 67.74 | 66.13 | +12.90 | -1.61 | -8.06 |
| mmstar / by_l2_category / diagram reasoning | 110 | 40.00 | 54.55 | 52.73 | 43.64 | +3.64 | -9.09 | -10.91 |
| mmstar / by_l2_category / electronics & energy & mechanical eng. | 46 | 39.13 | 45.65 | 45.65 | 52.17 | +13.04 | +6.52 | +6.52 |
| mmstar / by_l2_category / geography & earth science & agriculture | 58 | 31.03 | 34.48 | 27.59 | 20.69 | -10.34 | -6.90 | -13.79 |
| mmstar / by_l2_category / geometry | 116 | 23.28 | 49.14 | 48.28 | 47.41 | +24.14 | -0.86 | -1.72 |
| mmstar / by_l2_category / image emotion | 31 | 74.19 | 80.65 | 77.42 | 77.42 | +3.23 | +0.00 | -3.23 |
| mmstar / by_l2_category / image scene and topic | 141 | 53.90 | 57.45 | 55.32 | 53.19 | -0.71 | -2.13 | -4.26 |
| mmstar / by_l2_category / image style & quality | 78 | 85.90 | 88.46 | 76.92 | 83.33 | -2.56 | +6.41 | -5.13 |
| mmstar / by_l2_category / localization | 40 | 52.50 | 55.00 | 60.00 | 55.00 | +2.50 | -5.00 | +0.00 |
| mmstar / by_l2_category / numeric commonsense and calculation | 48 | 33.33 | 45.83 | 37.50 | 39.58 | +6.25 | +2.08 | -6.25 |
| mmstar / by_l2_category / object counting | 92 | 39.13 | 51.09 | 43.48 | 39.13 | +0.00 | -4.35 | -11.96 |
| mmstar / by_l2_category / recognition | 118 | 50.00 | 50.85 | 50.00 | 45.76 | -4.24 | -4.24 | -5.08 |
| mmstar / by_l2_category / single-instance reasoning | 99 | 66.67 | 70.71 | 70.71 | 69.70 | +3.03 | -1.01 | -1.01 |
| mmstar / by_l2_category / statistical reasoning | 86 | 31.40 | 46.51 | 50.00 | 40.70 | +9.30 | -9.30 | -5.81 |
