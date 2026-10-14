# Vanilla 2B / SFT / QAT / GRACE evaluation comparison

Source: `/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/experiments/constrained_32k_seed42/evaluation`. Read 28 merged JSON files (not rank shards). Validated row counts, correct counts, primary accuracies, unique question identities and matching identities/gold labels across methods.

## Primary scores

| Suite / metric | N | Vanilla 2B (%) | SFT (%) | QAT (%) | GRACE (%) | GRACE−Vanilla 2B (pp) | GRACE−QAT (pp) | GRACE−SFT (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HallusionBench aAcc | 1129 | 57.31 | 59.43 | 55.62 | 57.93 | +0.62 | +2.30 | -1.51 |
| MMBench EN dev (per-row) | 4329 | 82.26 | 82.95 | 81.54 | 80.78 | -1.48 | -0.76 | -2.17 |
| ScienceQA test (all) | 4241 | 80.90 | 81.35 | 80.71 | 79.86 | -1.04 | -0.85 | -1.49 |
| AI2D test | 3088 | 72.18 | 72.57 | 69.95 | 69.88 | -2.30 | -0.06 | -2.69 |
| MMMU validation | 900 | 40.67 | 44.33 | 41.89 | 39.11 | -1.56 | -2.78 | -5.22 |
| SEED-Bench-IMG | 14233 | 70.22 | 70.95 | 70.74 | 70.86 | +0.65 | +0.12 | -0.08 |
| MMStar | 1500 | 42.53 | 49.13 | 47.47 | 45.33 | +2.80 | -2.13 | -3.80 |
| Unweighted mean (7 suites) | — | 63.72 | 65.82 | 63.99 | 63.39 | -0.33 | -0.59 | -2.42 |

All scores are percentages; deltas are percentage points (pp). The mean gives each suite equal weight, uses HallusionBench aAcc and full-test ScienceQA, and is a descriptive aggregate, not an official benchmark metric.

## Additional metrics

| Suite / metric | N | Vanilla 2B (%) | SFT (%) | QAT (%) | GRACE (%) | GRACE−Vanilla 2B (pp) | GRACE−QAT (pp) | GRACE−SFT (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HallusionBench qAcc | 455 | 22.64 | 23.96 | 21.98 | 24.62 | +1.98 | +2.64 | +0.66 |
| HallusionBench fAcc | 346 | 30.92 | 30.92 | 30.35 | 34.10 | +3.18 | +3.76 | +3.18 |
| ScienceQA test (image) | 2017 | 81.31 | 81.56 | 80.22 | 79.67 | -1.64 | -0.55 | -1.88 |
| ScienceQA test (text) | 2224 | 80.53 | 81.16 | 81.16 | 80.04 | -0.49 | -1.12 | -1.12 |

HallusionBench qAcc requires every answer in a question group to be correct; fAcc requires every answer in an image/figure group to be correct. N is the number of groups. ScienceQA image is the image-containing subset of the full test split.

## Execution and parsing

| Suite | Vanilla 2B errors / parse fails | SFT errors / parse fails | QAT errors / parse fails | GRACE errors / parse fails |
| --- | ---: | ---: | ---: | ---: |
| hallusionbench | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| mmbench | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| scienceqa | 0 / 1 | 0 / 0 | 0 / 7 | 0 / 8 |
| ai2d | 0 / 2 | 0 / 0 | 0 / 1 | 0 / 0 |
| mmmu | 0 / not reported | 0 / not reported | 0 / not reported | 0 / not reported |
| seedbench_img | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| mmstar | 0 / 1 | 0 / 0 | 0 / 0 | 0 / 0 |

Errors are counted from individual predictions. Parse failures remain in the accuracy denominator. MMMU does not report a separate parse-failure count.

## Interpretation limits

These are the repository's local evaluation protocols. MMBench uses per-row accuracy; MMMU uses the local open-answer scorer. Do not assume official leaderboard equivalence. Vanilla 2B, when present, is Qwen3-VL-2B-Instruct without additional fine-tuning, loaded in BF16. The default launcher uses the SFT processor to match its image budget; baseline result metadata records the actual processor and budget. SFT is the fine-tuned BF16 quality reference; QAT and GRACE use the matched W4G128 training/export setup described in README_constrained.md. These scores do not measure packed INT4 inference speed or deployment memory. One seed does not establish repeatability.

## All category breakdowns

| Suite / metric | N | Vanilla 2B (%) | SFT (%) | QAT (%) | GRACE (%) | GRACE−Vanilla 2B (pp) | GRACE−QAT (pp) | GRACE−SFT (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hallusionbench / by_category / VD | 591 | 57.70 | 58.88 | 56.51 | 58.88 | +1.18 | +2.37 | +0.00 |
| hallusionbench / by_category / VS | 538 | 56.88 | 60.04 | 54.65 | 56.88 | +0.00 | +2.23 | -3.16 |
| hallusionbench / by_modality / image | 951 | 58.15 | 59.73 | 56.99 | 59.83 | +1.68 | +2.84 | +0.11 |
| hallusionbench / by_modality / text | 178 | 52.81 | 57.87 | 48.31 | 47.75 | -5.06 | -0.56 | -10.11 |
| hallusionbench / by_subcategory / VD/figure | 80 | 68.75 | 65.00 | 63.75 | 61.25 | -7.50 | -2.50 | -3.75 |
| hallusionbench / by_subcategory / VD/illusion | 144 | 60.42 | 56.94 | 58.33 | 61.81 | +1.39 | +3.47 | +4.86 |
| hallusionbench / by_subcategory / VD/math | 108 | 45.37 | 52.78 | 45.37 | 52.78 | +7.41 | +7.41 | +0.00 |
| hallusionbench / by_subcategory / VD/ocr | 89 | 80.90 | 79.78 | 76.40 | 84.27 | +3.37 | +7.87 | +4.49 |
| hallusionbench / by_subcategory / VD/video | 170 | 45.88 | 50.59 | 48.24 | 45.88 | +0.00 | -2.35 | -4.71 |
| hallusionbench / by_subcategory / VS/chart | 206 | 47.09 | 53.88 | 42.72 | 49.03 | +1.94 | +6.31 | -4.85 |
| hallusionbench / by_subcategory / VS/map | 96 | 58.33 | 60.42 | 57.29 | 56.25 | -2.08 | -1.04 | -4.17 |
| hallusionbench / by_subcategory / VS/ocr | 81 | 59.26 | 61.73 | 55.56 | 56.79 | -2.47 | +1.23 | -4.94 |
| hallusionbench / by_subcategory / VS/table | 155 | 67.74 | 67.10 | 68.39 | 67.74 | +0.00 | -0.65 | +0.65 |
| mmbench / by_category / action_recognition | 215 | 94.88 | 94.88 | 93.49 | 95.35 | +0.47 | +1.86 | +0.47 |
| mmbench / by_category / attribute_comparison | 141 | 75.18 | 70.21 | 71.63 | 70.92 | -4.26 | -0.71 | +0.71 |
| mmbench / by_category / attribute_recognition | 264 | 92.05 | 93.94 | 92.80 | 92.42 | +0.38 | -0.38 | -1.52 |
| mmbench / by_category / celebrity_recognition | 396 | 91.92 | 92.17 | 91.92 | 92.17 | +0.25 | +0.25 | +0.00 |
| mmbench / by_category / function_reasoning | 304 | 89.80 | 91.45 | 89.80 | 88.82 | -0.99 | -0.99 | -2.63 |
| mmbench / by_category / future_prediction | 130 | 41.54 | 36.15 | 39.23 | 43.08 | +1.54 | +3.85 | +6.92 |
| mmbench / by_category / identity_reasoning | 176 | 97.16 | 96.59 | 96.02 | 96.59 | -0.57 | +0.57 | +0.00 |
| mmbench / by_category / image_emotion | 200 | 85.00 | 86.00 | 85.50 | 85.00 | +0.00 | -0.50 | -1.00 |
| mmbench / by_category / image_quality | 150 | 66.00 | 71.33 | 58.67 | 64.00 | -2.00 | +5.33 | -7.33 |
| mmbench / by_category / image_scene | 407 | 98.28 | 98.28 | 98.53 | 98.53 | +0.25 | +0.00 | +0.25 |
| mmbench / by_category / image_style | 212 | 92.45 | 91.04 | 91.04 | 89.15 | -3.30 | -1.89 | -1.89 |
| mmbench / by_category / image_topic | 140 | 94.29 | 92.86 | 89.29 | 87.86 | -6.43 | -1.43 | -5.00 |
| mmbench / by_category / nature_relation | 179 | 73.18 | 75.98 | 74.86 | 70.95 | -2.23 | -3.91 | -5.03 |
| mmbench / by_category / object_localization | 315 | 67.30 | 69.21 | 66.35 | 66.98 | -0.32 | +0.63 | -2.22 |
| mmbench / by_category / ocr | 156 | 91.03 | 89.74 | 86.54 | 85.26 | -5.77 | -1.28 | -4.49 |
| mmbench / by_category / physical_property_reasoning | 219 | 72.15 | 75.34 | 73.52 | 63.47 | -8.68 | -10.05 | -11.87 |
| mmbench / by_category / physical_relation | 94 | 72.34 | 78.72 | 75.53 | 76.60 | +4.26 | +1.06 | -2.13 |
| mmbench / by_category / social_relation | 172 | 90.70 | 91.28 | 90.70 | 90.12 | -0.58 | -0.58 | -1.16 |
| mmbench / by_category / spatial_relationship | 177 | 41.24 | 38.42 | 38.98 | 40.11 | -1.13 | +1.13 | +1.69 |
| mmbench / by_category / structuralized_imagetext_understanding | 282 | 74.11 | 78.01 | 75.53 | 70.92 | -3.19 | -4.61 | -7.09 |
| mmbench / by_l2_category / attribute_reasoning | 699 | 86.12 | 87.70 | 86.27 | 82.83 | -3.29 | -3.43 | -4.86 |
| mmbench / by_l2_category / coarse_perception | 1109 | 89.90 | 90.35 | 88.19 | 88.28 | -1.62 | +0.09 | -2.07 |
| mmbench / by_l2_category / finegrained_perception (cross-instance) | 533 | 71.86 | 69.61 | 69.61 | 70.54 | -1.31 | +0.94 | +0.94 |
| mmbench / by_l2_category / finegrained_perception (instance-level) | 1131 | 84.97 | 85.85 | 84.26 | 84.26 | -0.71 | +0.00 | -1.59 |
| mmbench / by_l2_category / logic_reasoning | 412 | 63.83 | 64.81 | 64.08 | 62.14 | -1.70 | -1.94 | -2.67 |
| mmbench / by_l2_category / relation_reasoning | 445 | 79.78 | 82.47 | 81.12 | 79.55 | -0.22 | -1.57 | -2.92 |
| scienceqa / by_category / 20th century American history | 4 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Adaptations | 98 | 84.69 | 90.82 | 85.71 | 82.65 | -2.04 | -3.06 | -8.16 |
| scienceqa / by_category / Age of Exploration | 1 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Analyzing literature | 4 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Anatomy and physiology | 2 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Ancient Egypt and Kush | 1 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Ancient Mesopotamia | 2 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Animals | 8 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Astronomy | 37 | 54.05 | 56.76 | 54.05 | 64.86 | +10.81 | +10.81 | +8.11 |
| scienceqa / by_category / Atoms and molecules | 64 | 98.44 | 98.44 | 93.75 | 93.75 | -4.69 | +0.00 | -4.69 |
| scienceqa / by_category / Audience, purpose, and tone | 17 | 100.00 | 94.12 | 94.12 | 88.24 | -11.76 | -5.88 | -5.88 |
| scienceqa / by_category / Author's purpose and tone | 27 | 85.19 | 74.07 | 85.19 | 77.78 | -7.41 | -7.41 | +3.70 |
| scienceqa / by_category / Banking and finance | 1 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Basic economic principles | 116 | 74.14 | 84.48 | 80.17 | 79.31 | +5.17 | -0.86 | -5.17 |
| scienceqa / by_category / Capitalization | 18 | 61.11 | 77.78 | 83.33 | 72.22 | +11.11 | -11.11 | -5.56 |
| scienceqa / by_category / Categories | 11 | 36.36 | 90.91 | 63.64 | 63.64 | +27.27 | +0.00 | -27.27 |
| scienceqa / by_category / Cells | 17 | 76.47 | 76.47 | 70.59 | 82.35 | +5.88 | +11.76 | +5.88 |
| scienceqa / by_category / Chemical reactions | 43 | 93.02 | 74.42 | 81.40 | 93.02 | +0.00 | +11.63 | +18.60 |
| scienceqa / by_category / Cities | 12 | 66.67 | 58.33 | 50.00 | 66.67 | +0.00 | +16.67 | +8.33 |
| scienceqa / by_category / Classification | 161 | 96.89 | 97.52 | 96.27 | 95.65 | -1.24 | -0.62 | -1.86 |
| scienceqa / by_category / Classification and scientific names | 46 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Climate change | 1 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Colonial America | 38 | 89.47 | 86.84 | 84.21 | 84.21 | -5.26 | +0.00 | -2.63 |
| scienceqa / by_category / Comprehension strategies | 5 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Conservation and natural resources | 1 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Context clues | 5 | 80.00 | 60.00 | 60.00 | 100.00 | +20.00 | +40.00 | +40.00 |
| scienceqa / by_category / Creative techniques | 19 | 68.42 | 52.63 | 68.42 | 68.42 | +0.00 | +0.00 | +15.79 |
| scienceqa / by_category / Cultural celebrations | 2 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Descriptive details | 3 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Designing experiments | 134 | 94.78 | 97.01 | 95.52 | 97.01 | +2.24 | +1.49 | +0.00 |
| scienceqa / by_category / Developing and supporting arguments | 101 | 87.13 | 87.13 | 88.12 | 82.18 | -4.95 | -5.94 | -4.95 |
| scienceqa / by_category / Domain-specific vocabulary | 4 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Early 19th century American history | 2 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Earth events | 18 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Ecological interactions | 24 | 66.67 | 66.67 | 58.33 | 66.67 | +0.00 | +8.33 | +0.00 |
| scienceqa / by_category / Economics | 11 | 81.82 | 63.64 | 54.55 | 54.55 | -27.27 | +0.00 | -9.09 |
| scienceqa / by_category / Ecosystems | 78 | 80.77 | 85.90 | 87.18 | 84.62 | +3.85 | -2.56 | -1.28 |
| scienceqa / by_category / Editing and revising | 9 | 55.56 | 55.56 | 44.44 | 55.56 | +0.00 | +11.11 | +0.00 |
| scienceqa / by_category / Electricity | 1 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Engineering practices | 55 | 98.18 | 98.18 | 98.18 | 98.18 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / English colonies in North America | 39 | 92.31 | 94.87 | 94.87 | 84.62 | -7.69 | -10.26 | -10.26 |
| scienceqa / by_category / Force and motion | 58 | 86.21 | 77.59 | 79.31 | 81.03 | -5.17 | +1.72 | +3.45 |
| scienceqa / by_category / Formatting | 73 | 64.38 | 72.60 | 75.34 | 71.23 | +6.85 | -4.11 | -1.37 |
| scienceqa / by_category / Fossils | 22 | 54.55 | 50.00 | 54.55 | 54.55 | +0.00 | +0.00 | +4.55 |
| scienceqa / by_category / Genes to traits | 195 | 83.59 | 81.03 | 82.05 | 83.08 | -0.51 | +1.03 | +2.05 |
| scienceqa / by_category / Geography | 46 | 86.96 | 86.96 | 80.43 | 80.43 | -6.52 | +0.00 | -6.52 |
| scienceqa / by_category / Government | 16 | 81.25 | 81.25 | 81.25 | 81.25 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Greece | 1 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Heat and thermal energy | 42 | 92.86 | 100.00 | 95.24 | 90.48 | -2.38 | -4.76 | -9.52 |
| scienceqa / by_category / Heredity | 41 | 100.00 | 90.24 | 97.56 | 97.56 | -2.44 | +0.00 | +7.32 |
| scienceqa / by_category / Historical figures | 2 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Independent reading comprehension | 5 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Informational texts: level 1 | 5 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Kinetic and potential energy | 19 | 84.21 | 89.47 | 68.42 | 68.42 | -15.79 | +0.00 | -21.05 |
| scienceqa / by_category / Literary devices | 267 | 77.90 | 79.40 | 77.53 | 72.66 | -5.24 | -4.87 | -6.74 |
| scienceqa / by_category / Magnets | 110 | 39.09 | 46.36 | 43.64 | 43.64 | +4.55 | +0.00 | -2.73 |
| scienceqa / by_category / Maps | 46 | 80.43 | 80.43 | 76.09 | 78.26 | -2.17 | +2.17 | -2.17 |
| scienceqa / by_category / Materials | 167 | 89.82 | 91.62 | 91.02 | 89.22 | -0.60 | -1.80 | -2.40 |
| scienceqa / by_category / Medieval Asia | 2 | 50.00 | 50.00 | 50.00 | 50.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Mixtures | 6 | 83.33 | 83.33 | 66.67 | 66.67 | -16.67 | +0.00 | -16.67 |
| scienceqa / by_category / Oceania: geography | 51 | 64.71 | 56.86 | 66.67 | 64.71 | +0.00 | -1.96 | +7.84 |
| scienceqa / by_category / Oceans and continents | 19 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Oceans and continents	 | 4 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Opinion writing | 4 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Particle motion and energy | 41 | 80.49 | 75.61 | 75.61 | 90.24 | +9.76 | +14.63 | +14.63 |
| scienceqa / by_category / Persuasive strategies | 11 | 100.00 | 81.82 | 81.82 | 63.64 | -36.36 | -18.18 | -18.18 |
| scienceqa / by_category / Photosynthesis | 4 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Phrases and clauses | 32 | 59.38 | 50.00 | 53.12 | 31.25 | -28.12 | -21.88 | -18.75 |
| scienceqa / by_category / Physical Geography | 51 | 78.43 | 72.55 | 70.59 | 68.63 | -9.80 | -1.96 | -3.92 |
| scienceqa / by_category / Physical and chemical change | 52 | 94.23 | 92.31 | 92.31 | 94.23 | +0.00 | +1.92 | +1.92 |
| scienceqa / by_category / Plant reproduction | 6 | 83.33 | 66.67 | 66.67 | 66.67 | -16.67 | +0.00 | +0.00 |
| scienceqa / by_category / Plants | 12 | 91.67 | 91.67 | 91.67 | 91.67 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Plate tectonics | 3 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Poetry elements | 4 | 75.00 | 50.00 | 100.00 | 75.00 | +0.00 | -25.00 | +25.00 |
| scienceqa / by_category / Pronouns | 37 | 59.46 | 64.86 | 67.57 | 56.76 | -2.70 | -10.81 | -8.11 |
| scienceqa / by_category / Pronouns and antecedents | 18 | 44.44 | 55.56 | 50.00 | 50.00 | +5.56 | +0.00 | -5.56 |
| scienceqa / by_category / Read-alone texts | 2 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Reference skills | 140 | 46.43 | 46.43 | 50.00 | 51.43 | +5.00 | +1.43 | +5.00 |
| scienceqa / by_category / Research skills | 4 | 75.00 | 75.00 | 75.00 | 50.00 | -25.00 | -25.00 | -25.00 |
| scienceqa / by_category / Rhyming | 17 | 58.82 | 41.18 | 64.71 | 64.71 | +5.88 | +0.00 | +23.53 |
| scienceqa / by_category / Rocks and minerals | 46 | 97.83 | 95.65 | 97.83 | 97.83 | +0.00 | +0.00 | +2.17 |
| scienceqa / by_category / Scientific names | 32 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Sentences, fragments, and run-ons | 135 | 85.93 | 87.41 | 82.96 | 85.19 | -0.74 | +2.22 | -2.22 |
| scienceqa / by_category / Shades of meaning | 10 | 100.00 | 90.00 | 100.00 | 100.00 | +0.00 | +0.00 | +10.00 |
| scienceqa / by_category / Short and long vowels | 5 | 40.00 | 80.00 | 60.00 | 20.00 | -20.00 | -40.00 | -60.00 |
| scienceqa / by_category / Social studies skills | 12 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Solutions | 54 | 62.96 | 57.41 | 70.37 | 59.26 | -3.70 | -11.11 | +1.85 |
| scienceqa / by_category / State capitals | 312 | 96.79 | 95.83 | 90.06 | 88.78 | -8.01 | -1.28 | -7.05 |
| scienceqa / by_category / States | 18 | 94.44 | 88.89 | 88.89 | 88.89 | -5.56 | +0.00 | +0.00 |
| scienceqa / by_category / States of matter | 47 | 89.36 | 91.49 | 89.36 | 89.36 | +0.00 | +0.00 | -2.13 |
| scienceqa / by_category / Supply and demand | 27 | 74.07 | 74.07 | 77.78 | 74.07 | +0.00 | -3.70 | +0.00 |
| scienceqa / by_category / Text structure | 2 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / The American Revolution | 4 | 100.00 | 75.00 | 100.00 | 75.00 | -25.00 | -25.00 | +0.00 |
| scienceqa / by_category / The Americas: geography | 45 | 40.00 | 37.78 | 37.78 | 42.22 | +2.22 | +4.44 | +4.44 |
| scienceqa / by_category / The Antebellum period | 1 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / The Civil War and Reconstruction | 2 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Thermal energy | 40 | 70.00 | 70.00 | 72.50 | 77.50 | +7.50 | +5.00 | +7.50 |
| scienceqa / by_category / Topographic maps | 1 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Traits and heredity | 81 | 91.36 | 86.42 | 97.53 | 97.53 | +6.17 | +0.00 | +11.11 |
| scienceqa / by_category / Units and measurement | 200 | 86.00 | 90.00 | 90.50 | 89.50 | +3.50 | -1.00 | -0.50 |
| scienceqa / by_category / Velocity, acceleration, and forces | 86 | 50.00 | 55.81 | 56.98 | 61.63 | +11.63 | +4.65 | +5.81 |
| scienceqa / by_category / Verb tense | 52 | 78.85 | 78.85 | 80.77 | 75.00 | -3.85 | -5.77 | -3.85 |
| scienceqa / by_category / Visual elements | 9 | 88.89 | 88.89 | 77.78 | 88.89 | +0.00 | +11.11 | +0.00 |
| scienceqa / by_category / Water cycle | 2 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| scienceqa / by_category / Weather and climate | 97 | 73.20 | 76.29 | 65.98 | 64.95 | -8.25 | -1.03 | -11.34 |
| scienceqa / by_category / Word usage and nuance | 45 | 60.00 | 66.67 | 62.22 | 60.00 | +0.00 | -2.22 | -6.67 |
| scienceqa / by_category / World religions | 1 | 100.00 | 0.00 | 0.00 | 0.00 | -100.00 | +0.00 | +0.00 |
| scienceqa / by_modality / image | 2017 | 81.31 | 81.56 | 80.22 | 79.67 | -1.64 | -0.55 | -1.88 |
| scienceqa / by_modality / text | 2224 | 80.53 | 81.16 | 81.16 | 80.04 | -0.49 | -1.12 | -1.12 |
| scienceqa / by_subject / language science | 1100 | 72.64 | 73.73 | 74.18 | 70.64 | -2.00 | -3.55 | -3.09 |
| scienceqa / by_subject / natural science | 2252 | 83.48 | 84.01 | 83.70 | 84.19 | +0.71 | +0.49 | +0.18 |
| scienceqa / by_subject / social science | 889 | 84.59 | 84.03 | 81.21 | 80.31 | -4.27 | -0.90 | -3.71 |
| mmmu / by_question_type / multiple-choice | 847 | 42.62 | 46.64 | 44.16 | 41.32 | -1.30 | -2.83 | -5.31 |
| mmmu / by_question_type / open | 53 | 9.43 | 7.55 | 5.66 | 3.77 | -5.66 | -1.89 | -3.77 |
| mmmu / by_subject / Accounting | 30 | 30.00 | 36.67 | 26.67 | 33.33 | +3.33 | +6.67 | -3.33 |
| mmmu / by_subject / Agriculture | 30 | 36.67 | 30.00 | 43.33 | 43.33 | +6.67 | +0.00 | +13.33 |
| mmmu / by_subject / Architecture_and_Engineering | 30 | 23.33 | 36.67 | 40.00 | 16.67 | -6.67 | -23.33 | -20.00 |
| mmmu / by_subject / Art | 30 | 53.33 | 53.33 | 50.00 | 53.33 | +0.00 | +3.33 | +0.00 |
| mmmu / by_subject / Art_Theory | 30 | 73.33 | 63.33 | 60.00 | 60.00 | -13.33 | +0.00 | -3.33 |
| mmmu / by_subject / Basic_Medical_Science | 30 | 50.00 | 50.00 | 53.33 | 46.67 | -3.33 | -6.67 | -3.33 |
| mmmu / by_subject / Biology | 30 | 33.33 | 40.00 | 30.00 | 33.33 | +0.00 | +3.33 | -6.67 |
| mmmu / by_subject / Chemistry | 30 | 23.33 | 33.33 | 20.00 | 16.67 | -6.67 | -3.33 | -16.67 |
| mmmu / by_subject / Clinical_Medicine | 30 | 36.67 | 36.67 | 40.00 | 40.00 | +3.33 | +0.00 | +3.33 |
| mmmu / by_subject / Computer_Science | 30 | 53.33 | 36.67 | 53.33 | 36.67 | -16.67 | -16.67 | +0.00 |
| mmmu / by_subject / Design | 30 | 70.00 | 63.33 | 66.67 | 66.67 | -3.33 | +0.00 | +3.33 |
| mmmu / by_subject / Diagnostics_and_Laboratory_Medicine | 30 | 36.67 | 36.67 | 46.67 | 36.67 | +0.00 | -10.00 | +0.00 |
| mmmu / by_subject / Economics | 30 | 30.00 | 46.67 | 36.67 | 33.33 | +3.33 | -3.33 | -13.33 |
| mmmu / by_subject / Electronics | 30 | 13.33 | 26.67 | 23.33 | 30.00 | +16.67 | +6.67 | +3.33 |
| mmmu / by_subject / Energy_and_Power | 30 | 30.00 | 33.33 | 40.00 | 36.67 | +6.67 | -3.33 | +3.33 |
| mmmu / by_subject / Finance | 30 | 16.67 | 36.67 | 36.67 | 33.33 | +16.67 | -3.33 | -3.33 |
| mmmu / by_subject / Geography | 30 | 56.67 | 46.67 | 43.33 | 53.33 | -3.33 | +10.00 | +6.67 |
| mmmu / by_subject / History | 30 | 50.00 | 50.00 | 40.00 | 40.00 | -10.00 | +0.00 | -10.00 |
| mmmu / by_subject / Literature | 30 | 90.00 | 90.00 | 80.00 | 76.67 | -13.33 | -3.33 | -13.33 |
| mmmu / by_subject / Manage | 30 | 30.00 | 33.33 | 40.00 | 30.00 | +0.00 | -10.00 | -3.33 |
| mmmu / by_subject / Marketing | 30 | 40.00 | 50.00 | 46.67 | 40.00 | +0.00 | -6.67 | -10.00 |
| mmmu / by_subject / Materials | 30 | 20.00 | 36.67 | 23.33 | 23.33 | +3.33 | +0.00 | -13.33 |
| mmmu / by_subject / Math | 30 | 36.67 | 36.67 | 40.00 | 20.00 | -16.67 | -20.00 | -16.67 |
| mmmu / by_subject / Mechanical_Engineering | 30 | 33.33 | 40.00 | 26.67 | 26.67 | -6.67 | +0.00 | -13.33 |
| mmmu / by_subject / Music | 30 | 40.00 | 40.00 | 30.00 | 36.67 | -3.33 | +6.67 | -3.33 |
| mmmu / by_subject / Pharmacy | 30 | 43.33 | 56.67 | 40.00 | 30.00 | -13.33 | -10.00 | -26.67 |
| mmmu / by_subject / Physics | 30 | 26.67 | 30.00 | 26.67 | 30.00 | +3.33 | +3.33 | +0.00 |
| mmmu / by_subject / Psychology | 30 | 43.33 | 53.33 | 50.00 | 43.33 | +0.00 | -6.67 | -10.00 |
| mmmu / by_subject / Public_Health | 30 | 46.67 | 50.00 | 46.67 | 53.33 | +6.67 | +6.67 | +3.33 |
| mmmu / by_subject / Sociology | 30 | 53.33 | 56.67 | 56.67 | 53.33 | +0.00 | -3.33 | -3.33 |
| seedbench_img / by_question_type / Instance Attributes | 4649 | 71.97 | 72.32 | 72.81 | 72.90 | +0.92 | +0.09 | +0.58 |
| seedbench_img / by_question_type / Instance Identity | 1831 | 74.11 | 73.68 | 73.78 | 73.95 | -0.16 | +0.16 | +0.27 |
| seedbench_img / by_question_type / Instance Interaction | 97 | 72.16 | 74.23 | 74.23 | 71.13 | -1.03 | -3.09 | -3.09 |
| seedbench_img / by_question_type / Instance Location | 978 | 65.44 | 65.64 | 67.79 | 67.69 | +2.25 | -0.10 | +2.04 |
| seedbench_img / by_question_type / Instances Counting | 2447 | 60.97 | 63.87 | 62.57 | 61.67 | +0.69 | -0.90 | -2.21 |
| seedbench_img / by_question_type / Scene Understanding | 3158 | 77.17 | 76.95 | 76.03 | 77.49 | +0.32 | +1.46 | +0.54 |
| seedbench_img / by_question_type / Spatial Relation | 657 | 52.51 | 55.25 | 54.19 | 54.03 | +1.52 | -0.15 | -1.22 |
| seedbench_img / by_question_type / Text Understanding | 85 | 60.00 | 70.59 | 69.41 | 55.29 | -4.71 | -14.12 | -15.29 |
| seedbench_img / by_question_type / Visual Reasoning | 331 | 77.34 | 77.64 | 75.83 | 76.74 | -0.60 | +0.91 | -0.91 |
| mmstar / by_category / coarse perception | 250 | 66.40 | 68.40 | 63.60 | 66.80 | +0.40 | +3.20 | -1.60 |
| mmstar / by_category / fine-grained perception | 250 | 40.00 | 47.20 | 42.80 | 42.80 | +2.80 | +0.00 | -4.40 |
| mmstar / by_category / instance reasoning | 250 | 56.80 | 62.00 | 58.00 | 57.20 | +0.40 | -0.80 | -4.80 |
| mmstar / by_category / logical reasoning | 250 | 35.20 | 41.60 | 42.80 | 38.40 | +3.20 | -4.40 | -3.20 |
| mmstar / by_category / math | 250 | 29.20 | 41.20 | 46.00 | 40.40 | +11.20 | -5.60 | -0.80 |
| mmstar / by_category / science & technology | 250 | 27.60 | 34.40 | 31.60 | 26.40 | -1.20 | -5.20 | -8.00 |
| mmstar / by_l2_category / biology & chemistry & physics | 146 | 28.77 | 32.88 | 28.77 | 23.29 | -5.48 | -5.48 | -9.59 |
| mmstar / by_l2_category / code & sequence reasoning | 39 | 43.59 | 46.15 | 41.03 | 41.03 | -2.56 | +0.00 | -5.13 |
| mmstar / by_l2_category / common reasoning | 101 | 32.67 | 34.65 | 37.62 | 41.58 | +8.91 | +3.96 | +6.93 |
| mmstar / by_l2_category / cross-instance attribute reasoning | 89 | 47.19 | 52.81 | 43.82 | 48.31 | +1.12 | +4.49 | -4.49 |
| mmstar / by_l2_category / cross-instance relation reasoning | 62 | 53.23 | 62.90 | 58.06 | 56.45 | +3.23 | -1.61 | -6.45 |
| mmstar / by_l2_category / diagram reasoning | 110 | 34.55 | 46.36 | 48.18 | 34.55 | +0.00 | -13.64 | -11.82 |
| mmstar / by_l2_category / electronics & energy & mechanical eng. | 46 | 32.61 | 45.65 | 47.83 | 47.83 | +15.22 | +0.00 | +2.17 |
| mmstar / by_l2_category / geography & earth science & agriculture | 58 | 20.69 | 29.31 | 25.86 | 17.24 | -3.45 | -8.62 | -12.07 |
| mmstar / by_l2_category / geometry | 116 | 25.00 | 38.79 | 46.55 | 39.66 | +14.66 | -6.90 | +0.86 |
| mmstar / by_l2_category / image emotion | 31 | 77.42 | 77.42 | 77.42 | 77.42 | +0.00 | +0.00 | +0.00 |
| mmstar / by_l2_category / image scene and topic | 141 | 53.19 | 56.03 | 53.19 | 55.32 | +2.13 | +2.13 | -0.71 |
| mmstar / by_l2_category / image style & quality | 78 | 85.90 | 87.18 | 76.92 | 83.33 | -2.56 | +6.41 | -3.85 |
| mmstar / by_l2_category / localization | 40 | 55.00 | 57.50 | 55.00 | 60.00 | +5.00 | +5.00 | +2.50 |
| mmstar / by_l2_category / numeric commonsense and calculation | 48 | 35.42 | 41.67 | 35.42 | 45.83 | +10.42 | +10.42 | +4.17 |
| mmstar / by_l2_category / object counting | 92 | 36.96 | 45.65 | 36.96 | 36.96 | +0.00 | +0.00 | -8.70 |
| mmstar / by_l2_category / recognition | 118 | 37.29 | 44.92 | 43.22 | 41.53 | +4.24 | -1.69 | -3.39 |
| mmstar / by_l2_category / single-instance reasoning | 99 | 67.68 | 69.70 | 70.71 | 65.66 | -2.02 | -5.05 | -4.04 |
| mmstar / by_l2_category / statistical reasoning | 86 | 31.40 | 44.19 | 51.16 | 38.37 | +6.98 | -12.79 | -5.81 |
