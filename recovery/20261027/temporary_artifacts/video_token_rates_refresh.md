# Saved video token rates

Updated 2026-10-25 14:38:10 SGT. Validated **17/50** selected-population model/suite pairs.

Also measured **6 incomplete pairs** with latest-attempt coverage strictly above 0%.

**2 partial pairs failed validation; their coverage is shown with rates withheld.**

Source matrices: `video_evaluation/video_resilient_v3_f185cede8bfd/full`, `video_evaluation/concurrent_grace_w4_onwards/full`, `video_evaluation/video_unsaturated_v2_3a74c7231e27/full`.

**VideoMME now means the 1,800 short/medium questions; the 900 long questions are excluded.** A complete VideoMME row covers this declared subset, not the original 2,700 questions.

The existing filename is retained for IDE links; the update time above identifies this snapshot. Each model/suite is counted once across matrices. Prefer a certified completion; otherwise use the largest saved prefix from one latest attempt. Retries and directories are never concatenated.

Compatible older non-VideoMME results remain eligible when they preserve evaluated progress. The original 2,700-question VideoMME population is excluded from this combined table. JSON/CSV record the source matrix for every selected row.

All completed runs passed the existing evaluator’s artifact hashes, selected ordered question coverage, source/input identities, exact token accounting, saved-score/summary checks, context bounds, and EOS checks. Selected source populations also match the frozen dataset manifests. No inference was run.

## Tokens per source video second

**Each rate is sum(tokens) / sum(video duration in seconds).** Repeated videos count once per evaluated question because their tokens are processed again. These duration-weighted rates normalize token use for clip length; they do not measure inference speed. CSV/JSON also retain the arithmetic mean, median and p95 of per-question rates.

Text input includes the question/options, timestamps, vision boundary tokens and chat template. The timestamp/boundary sequence grows with video length even though subtitles and audio are disabled. Visual-token rates also depend on processed frame geometry and the sampling/temporal patch policy.

Rows labelled **incomplete (partial rates)** use only the latest attempt’s saved questions; they remain incomplete. Other incomplete rows have no rates. **100% saved coverage without a valid completion certificate remains incomplete.** Counts may change after this snapshot. Retries are never concatenated. Partial rows passed ordered-prefix, source/input identity, token accounting, saved-score, context and EOS checks, and their selected source manifests match the frozen datasets. The parsed prediction snapshot has a separate digest; it has no full-run completion certificate. Any unfinished final JSONL record is excluded.

**Partial populations can be biased by question order.** The reduced VideoMME run visits short, then medium videos. These are observed-prefix rates, not estimates for the missing questions.

| Model | Task | Questions | Visual / video s | Text input / video s | Total input / video s | Text output / video s | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| vanilla_2b | tempcompass | 1580 | 40.860 | 12.792 | 53.652 | 0.267 | complete |
| vanilla_2b | videomme (short + medium) | 1800 | 32.240 | 9.892 | 42.132 | 0.015 | complete |
| vanilla_2b | videommmu_adaptation | 300 | 33.435 | 10.189 | 43.624 | 0.016 | complete |
| vanilla_2b | videommmu_comprehension | 300 | 33.435 | 10.244 | 43.678 | 0.013 | complete |
| vanilla_2b | videommmu_perception | 300 | 33.435 | 10.237 | 43.672 | 0.012 | complete |
| vanilla_8b_teacher | tempcompass | 1580 | 40.860 | 12.792 | 53.652 | 0.336 | complete |
| vanilla_8b_teacher | videomme (short + medium) | 1800 | 32.240 | 9.892 | 42.132 | 0.026 | complete |
| vanilla_8b_teacher | videommmu_adaptation | 178/300 (59.3%) | — | — | — | — | incomplete (non-EOS; rates withheld) |
| vanilla_8b_teacher | videommmu_comprehension | 178/300 (59.3%) | 33.649 | 10.217 | 43.866 | 0.008 | incomplete (partial rates) |
| vanilla_8b_teacher | videommmu_perception | 178/300 (59.3%) | 33.649 | 10.257 | 43.906 | 0.010 | incomplete (partial rates) |
| sft_2b | tempcompass | 1580 | 40.860 | 12.792 | 53.652 | 0.167 | complete |
| sft_2b | videomme (short + medium) | 1800 | 32.240 | 9.892 | 42.132 | 0.007 | complete |
| sft_2b | videommmu_adaptation | 300/300 (100.0%) | — | — | — | — | incomplete (non-EOS; rates withheld) |
| sft_2b | videommmu_comprehension | 300 | 33.435 | 10.244 | 43.678 | 0.004 | complete |
| sft_2b | videommmu_perception | 300 | 33.435 | 10.237 | 43.672 | 0.004 | complete |
| released_bf16 | tempcompass | 1580 | 40.860 | 12.792 | 53.652 | 0.167 | complete |
| released_bf16 | videomme (short + medium) | 524/1800 (29.1%) | 32.406 | 9.714 | 42.120 | 0.025 | incomplete (partial rates) |
| released_bf16 | videommmu_adaptation | 0/300 (0.0%) | — | — | — | — | incomplete |
| released_bf16 | videommmu_comprehension | 0/300 (0.0%) | — | — | — | — | incomplete |
| released_bf16 | videommmu_perception | 0/300 (0.0%) | — | — | — | — | incomplete |
| released_grace_w4 | tempcompass | 0/1580 (0.0%) | — | — | — | — | incomplete |
| released_grace_w4 | videomme (short + medium) | 0/1800 (0.0%) | — | — | — | — | incomplete |
| released_grace_w4 | videommmu_adaptation | 0/300 (0.0%) | — | — | — | — | incomplete |
| released_grace_w4 | videommmu_comprehension | 0/300 (0.0%) | — | — | — | — | incomplete |
| released_grace_w4 | videommmu_perception | 2/300 (0.7%) | 36.795 | 10.184 | 46.979 | 0.003 | incomplete (partial rates) |
| released_grace_w8 | tempcompass | 0/1580 (0.0%) | — | — | — | — | incomplete |
| released_grace_w8 | videomme (short + medium) | 0/1800 (0.0%) | — | — | — | — | incomplete |
| released_grace_w8 | videommmu_adaptation | 0/300 (0.0%) | — | — | — | — | incomplete |
| released_grace_w8 | videommmu_comprehension | 0/300 (0.0%) | — | — | — | — | incomplete |
| released_grace_w8 | videommmu_perception | 0/300 (0.0%) | — | — | — | — | incomplete |
| qat_w4 | tempcompass | 1580 | 40.860 | 12.792 | 53.652 | 0.167 | complete |
| qat_w4 | videomme (short + medium) | 1800 | 32.240 | 9.892 | 42.132 | 0.007 | complete |
| qat_w4 | videommmu_adaptation | 160/300 (53.3%) | 33.690 | 10.229 | 43.919 | 0.005 | incomplete (partial rates) |
| qat_w4 | videommmu_comprehension | 300 | 33.435 | 10.244 | 43.678 | 0.004 | complete |
| qat_w4 | videommmu_perception | 300 | 33.435 | 10.237 | 43.672 | 0.004 | complete |
| grace_w4 | tempcompass | 1580 | 40.860 | 12.792 | 53.652 | 0.167 | complete |
| grace_w4 | videomme (short + medium) | 1056/1800 (58.7%) | 32.524 | 9.814 | 42.338 | 0.014 | incomplete (partial rates) |
| grace_w4 | videommmu_adaptation | 0/300 (0.0%) | — | — | — | — | incomplete |
| grace_w4 | videommmu_comprehension | 0/300 (0.0%) | — | — | — | — | incomplete |
| grace_w4 | videommmu_perception | 0/300 (0.0%) | — | — | — | — | incomplete |
| qat_w8 | tempcompass | 0/1580 (0.0%) | — | — | — | — | incomplete |
| qat_w8 | videomme (short + medium) | 0/1800 (0.0%) | — | — | — | — | incomplete |
| qat_w8 | videommmu_adaptation | 0/300 (0.0%) | — | — | — | — | incomplete |
| qat_w8 | videommmu_comprehension | 0/300 (0.0%) | — | — | — | — | incomplete |
| qat_w8 | videommmu_perception | 0/300 (0.0%) | — | — | — | — | incomplete |
| grace_w8 | tempcompass | 0/1580 (0.0%) | — | — | — | — | incomplete |
| grace_w8 | videomme (short + medium) | 0/1800 (0.0%) | — | — | — | — | incomplete |
| grace_w8 | videommmu_adaptation | 0/300 (0.0%) | — | — | — | — | incomplete |
| grace_w8 | videommmu_comprehension | 0/300 (0.0%) | — | — | — | — | incomplete |
| grace_w8 | videommmu_perception | 0/300 (0.0%) | — | — | — | — | incomplete |

## VideoMME duration groups

| Model | Group | Questions | Mean video seconds | Mean visual tokens | Mean text input tokens | Visual / video s | Text input / video s | Total input / video s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vanilla_2b | short | 900 | 80.666 | 2606.400 | 781.763 | 32.311 | 9.691 | 42.002 |
| vanilla_2b | medium | 900 | 516.814 | 16656.107 | 5128.578 | 32.228 | 9.923 | 42.152 |
| vanilla_8b_teacher | short | 900 | 80.666 | 2606.400 | 781.763 | 32.311 | 9.691 | 42.002 |
| vanilla_8b_teacher | medium | 900 | 516.814 | 16656.107 | 5128.578 | 32.228 | 9.923 | 42.152 |
| sft_2b | short | 900 | 80.666 | 2606.400 | 781.763 | 32.311 | 9.691 | 42.002 |
| sft_2b | medium | 900 | 516.814 | 16656.107 | 5128.578 | 32.228 | 9.923 | 42.152 |
| qat_w4 | short | 900 | 80.666 | 2606.400 | 781.763 | 32.311 | 9.691 | 42.002 |
| qat_w4 | medium | 900 | 516.814 | 16656.107 | 5128.578 | 32.228 | 9.923 | 42.152 |

## Tokens per elapsed processing second

Denominator: sum of saved per-question `elapsed_seconds`. It includes video extraction/probing, decoding/preprocessing, CPU/GPU transfer, generation and scoring. The first question includes model loading; worker startup and final summary are excluded. This is an observed end-to-end processing rate, not isolated prefill or decode throughput. Shared-machine conditions and per-attempt initialization costs affect comparisons. Imported predictions retain the original inference times, not the CPU import duration.

| Model | Task | Questions | Visual / elapsed s | Text input / elapsed s | Total input / elapsed s | Text output / elapsed s | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| vanilla_2b | tempcompass | 1580 | 995.001 | 311.510 | 1306.511 | 6.506 | complete |
| vanilla_2b | videomme (short + medium) | 1800 | 440.691 | 135.218 | 575.909 | 0.199 | complete |
| vanilla_2b | videommmu_adaptation | 300 | 315.503 | 96.145 | 411.649 | 0.153 | complete |
| vanilla_2b | videommmu_comprehension | 300 | 315.425 | 96.639 | 412.064 | 0.121 | complete |
| vanilla_2b | videommmu_perception | 300 | 317.385 | 97.179 | 414.564 | 0.113 | complete |
| vanilla_8b_teacher | tempcompass | 1580 | 814.539 | 255.012 | 1069.551 | 6.700 | complete |
| vanilla_8b_teacher | videomme (short + medium) | 1800 | 525.136 | 161.128 | 686.264 | 0.419 | complete |
| sft_2b | tempcompass | 1580 | 1067.175 | 334.106 | 1401.281 | 4.364 | complete |
| sft_2b | videomme (short + medium) | 1800 | 571.209 | 175.265 | 746.473 | 0.119 | complete |
| sft_2b | videommmu_comprehension | 300 | 298.842 | 91.558 | 390.400 | 0.035 | complete |
| sft_2b | videommmu_perception | 300 | 298.555 | 91.414 | 389.969 | 0.035 | complete |
| released_bf16 | tempcompass | 1580 | 1049.254 | 328.495 | 1377.749 | 4.290 | complete |
| qat_w4 | tempcompass | 1580 | 945.212 | 295.922 | 1241.134 | 3.865 | complete |
| qat_w4 | videomme (short + medium) | 1800 | 436.997 | 134.084 | 571.081 | 0.091 | complete |
| qat_w4 | videommmu_comprehension | 300 | 213.974 | 65.557 | 279.531 | 0.025 | complete |
| qat_w4 | videommmu_perception | 300 | 220.071 | 67.383 | 287.454 | 0.026 | complete |
| grace_w4 | tempcompass | 1580 | 733.495 | 229.639 | 963.133 | 2.999 | complete |

## Partial runs above the coverage threshold

Rates below describe the saved prefix of the latest attempt, with the same ratio-of-sums definition. They are excluded from the complete-run count.

| Model | Task | Questions | Visual / video s | Text input / video s | Total input / video s | Text output / video s | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| vanilla_8b_teacher | videommmu_comprehension | 178/300 (59.3%) | 33.649 | 10.217 | 43.866 | 0.008 | incomplete (partial rates) |
| vanilla_8b_teacher | videommmu_perception | 178/300 (59.3%) | 33.649 | 10.257 | 43.906 | 0.010 | incomplete (partial rates) |
| released_bf16 | videomme (short + medium) | 524/1800 (29.1%) | 32.406 | 9.714 | 42.120 | 0.025 | incomplete (partial rates) |
| released_grace_w4 | videommmu_perception | 2/300 (0.7%) | 36.795 | 10.184 | 46.979 | 0.003 | incomplete (partial rates) |
| qat_w4 | videommmu_adaptation | 160/300 (53.3%) | 33.690 | 10.229 | 43.919 | 0.005 | incomplete (partial rates) |
| grace_w4 | videomme (short + medium) | 1056/1800 (58.7%) | 32.524 | 9.814 | 42.338 | 0.014 | incomplete (partial rates) |

| Model | Task | Questions | Visual / elapsed s | Text input / elapsed s | Total input / elapsed s | Text output / elapsed s | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| vanilla_8b_teacher | videommmu_comprehension | 178/300 (59.3%) | 320.504 | 97.320 | 417.824 | 0.075 | incomplete (partial rates) |
| vanilla_8b_teacher | videommmu_perception | 178/300 (59.3%) | 306.597 | 93.461 | 400.057 | 0.089 | incomplete (partial rates) |
| released_bf16 | videomme (short + medium) | 524/1800 (29.1%) | 558.431 | 167.391 | 725.822 | 0.432 | incomplete (partial rates) |
| released_grace_w4 | videommmu_perception | 2/300 (0.7%) | 324.334 | 89.765 | 414.100 | 0.024 | incomplete (partial rates) |
| qat_w4 | videommmu_adaptation | 160/300 (53.3%) | 297.529 | 90.333 | 387.862 | 0.043 | incomplete (partial rates) |
| grace_w4 | videomme (short + medium) | 1056/1800 (58.7%) | 521.961 | 157.495 | 679.455 | 0.230 | incomplete (partial rates) |

## Partial runs with validation failures

These saved predictions do not qualify for uncensored token-rate measurements. Their raw coverage does not certify a complete result.

| Model | Task | Questions | Visual / video s | Text input / video s | Total input / video s | Text output / video s | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| vanilla_8b_teacher | videommmu_adaptation | 178/300 (59.3%) | — | — | — | — | incomplete (non-EOS; rates withheld) |
| sft_2b | videommmu_adaptation | 300/300 (100.0%) | — | — | — | — | incomplete (non-EOS; rates withheld) |

- `vanilla_8b_teacher/videommmu_adaptation`: Partial population contains non-EOS generations.
- `sft_2b/videommmu_adaptation`: Partial population contains non-EOS generations.

## Historical smoke runs

One-question source-resolution smoke runs are separate previews, excluded from the full-run count. Their rates use the same definitions but a different spatial policy.

| Model | Task | Questions | Visual / video s | Text input / video s | Total input / video s | Text output / video s | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| grace_w4 | tempcompass | 1 | 193.455 | 14.545 | 208.000 | 0.291 | complete |
| grace_w8 | tempcompass | 1 | 193.455 | 14.545 | 208.000 | 0.291 | complete |
| qat_w4 | tempcompass | 1 | 193.455 | 14.545 | 208.000 | 0.291 | complete |
| qat_w8 | tempcompass | 1 | 193.455 | 14.545 | 208.000 | 0.291 | complete |
| released_bf16 | tempcompass | 1 | 193.455 | 14.545 | 208.000 | 0.291 | complete |
| released_grace_w4 | tempcompass | 1 | 193.455 | 14.545 | 208.000 | 0.291 | complete |
| released_grace_w8 | tempcompass | 1 | 193.455 | 14.545 | 208.000 | 0.291 | complete |
| sft_2b | tempcompass | 1 | 193.455 | 14.545 | 208.000 | 0.291 | complete |
| vanilla_2b | tempcompass | 1 | 193.455 | 14.545 | 208.000 | 0.291 | complete |
| vanilla_8b_teacher | tempcompass | 1 | 193.455 | 14.545 | 208.000 | 0.291 | complete |

## Files

- `video_token_rates_refresh.json`: definitions, validation hashes, all run states, duration/category statistics, retry aliases and smoke results.
- `video_token_rates_refresh.csv`: all benchmark pairs; status and measurement scope distinguish complete, partial and unmeasured runs.
- `video_token_rates_refresh_questions.csv`: exact counts and both rates for measured questions, labelled by run status and scope.
- `video_token_rates_refresh_categories.csv`: category aggregates labelled by run status and scope.
- `video_token_rates_refresh_attempts.csv`: all attempt states, observed coverage and saved failure/block evidence.
