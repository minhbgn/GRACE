# Saved video token rates

Updated 2026-10-25 14:58:37 SGT. Validated **17/50** selected-population model/suite pairs.

Also measured **6 incomplete pairs** with latest-attempt coverage strictly above 0%.

**2 partial pairs failed validation; their coverage is shown with rates withheld.**

Source matrices: `video_evaluation/video_resilient_v3_f185cede8bfd/full`, `video_evaluation/concurrent_grace_w4_onwards/full`, `video_evaluation/video_unsaturated_v2_3a74c7231e27/full`.

**VideoMME now means the 1,800 short/medium questions; the 900 long questions are excluded.** A complete VideoMME row covers this declared subset, not the original 2,700 questions.

The existing filename is retained for IDE links; the update time above identifies this snapshot. Each model/suite is counted once across matrices. Prefer a certified completion; otherwise use the largest saved prefix from one latest attempt. Retries and directories are never concatenated.

Compatible older non-VideoMME results remain eligible when they preserve evaluated progress. The original 2,700-question VideoMME population is excluded from this combined table. JSON/CSV record the source matrix for every selected row.

All completed runs passed the existing evaluator’s artifact hashes, selected ordered question coverage, source/input identities, exact token accounting, saved-score/summary checks, context bounds, and EOS checks. Selected source populations also match the frozen dataset manifests. No inference was run.

## Average tokens per evaluated question

These are arithmetic means of saved per-question token counts, **not tokens per second**. Visual input counts the video tokens. Text input includes question/options, timestamps and chat markers; text output includes generated IDs and EOS. **Total text = text input + text output; total input = visual input + text input; all tokens = visual input + total text.** Columns are rounded independently to two decimal places. Repeated questions for the same video are separate evaluations. Partial means use only the observed validated prefix; invalid or unobserved populations have no means.

| Model | Benchmark | Questions | Mean question only | Mean visual input | Mean text input | Mean text output | Mean total text | Mean total input | Mean all tokens | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vanilla_2b | tempcompass | 1580/1580 | 9.90 | 489.12 | 153.13 | 3.20 | 156.33 | 642.25 | 645.45 | complete |
| vanilla_2b | videomme (short + medium) | 1800/1800 | 14.50 | 9,631.25 | 2,955.17 | 4.36 | 2,959.53 | 12,586.42 | 12,590.78 | complete |
| vanilla_2b | videommmu_adaptation | 300/300 | 47.05 | 16,939.89 | 5,162.19 | 8.20 | 5,170.39 | 22,102.09 | 22,110.28 | complete |
| vanilla_2b | videommmu_comprehension | 300/300 | 59.04 | 16,939.89 | 5,190.00 | 6.47 | 5,196.48 | 22,129.90 | 22,136.37 | complete |
| vanilla_2b | videommmu_perception | 300/300 | 35.39 | 16,939.89 | 5,186.78 | 6.06 | 5,192.83 | 22,126.67 | 22,132.73 | complete |
| vanilla_8b_teacher | tempcompass | 1580/1580 | 9.90 | 489.12 | 153.13 | 4.02 | 157.15 | 642.25 | 646.27 | complete |
| vanilla_8b_teacher | videomme (short + medium) | 1800/1800 | 14.50 | 9,631.25 | 2,955.17 | 7.68 | 2,962.85 | 12,586.42 | 12,594.10 | complete |
| vanilla_8b_teacher | videommmu_adaptation | 178/300 | — | — | — | — | — | — | — | incomplete (validation failed; means withheld) |
| vanilla_8b_teacher | videommmu_comprehension | 178/300 | 48.48 | 16,400.04 | 4,979.83 | 3.82 | 4,983.65 | 21,379.87 | 21,383.69 | incomplete (observed prefix) |
| vanilla_8b_teacher | videommmu_perception | 178/300 | 36.83 | 16,400.04 | 4,999.29 | 4.79 | 5,004.07 | 21,399.33 | 21,404.12 | incomplete (observed prefix) |
| sft_2b | tempcompass | 1580/1580 | 9.90 | 489.12 | 153.13 | 2.00 | 155.13 | 642.25 | 644.25 | complete |
| sft_2b | videomme (short + medium) | 1800/1800 | 14.50 | 9,631.25 | 2,955.17 | 2.00 | 2,957.17 | 12,586.42 | 12,588.42 | complete |
| sft_2b | videommmu_adaptation | 300/300 | — | — | — | — | — | — | — | incomplete (validation failed; means withheld) |
| sft_2b | videommmu_comprehension | 300/300 | 59.04 | 16,939.89 | 5,190.00 | 2.00 | 5,192.00 | 22,129.90 | 22,131.90 | complete |
| sft_2b | videommmu_perception | 300/300 | 35.39 | 16,939.89 | 5,186.78 | 2.00 | 5,188.78 | 22,126.67 | 22,128.67 | complete |
| released_bf16 | tempcompass | 1580/1580 | 9.90 | 489.12 | 153.13 | 2.00 | 155.13 | 642.25 | 644.25 | complete |
| released_bf16 | videomme (short + medium) | 524/1800 | 13.99 | 2,586.34 | 775.26 | 2.00 | 777.26 | 3,361.60 | 3,363.60 | incomplete (observed prefix) |
| released_bf16 | videommmu_adaptation | 0/300 | — | — | — | — | — | — | — | incomplete |
| released_bf16 | videommmu_comprehension | 0/300 | — | — | — | — | — | — | — | incomplete |
| released_bf16 | videommmu_perception | 0/300 | — | — | — | — | — | — | — | incomplete |
| released_grace_w4 | tempcompass | 0/1580 | — | — | — | — | — | — | — | incomplete |
| released_grace_w4 | videomme (short + medium) | 0/1800 | — | — | — | — | — | — | — | incomplete |
| released_grace_w4 | videommmu_adaptation | 0/300 | — | — | — | — | — | — | — | incomplete |
| released_grace_w4 | videommmu_comprehension | 0/300 | — | — | — | — | — | — | — | incomplete |
| released_grace_w4 | videommmu_perception | 2/300 | 30.50 | 27,440.00 | 7,594.50 | 2.00 | 7,596.50 | 35,034.50 | 35,036.50 | incomplete (observed prefix) |
| released_grace_w8 | tempcompass | 0/1580 | — | — | — | — | — | — | — | incomplete |
| released_grace_w8 | videomme (short + medium) | 0/1800 | — | — | — | — | — | — | — | incomplete |
| released_grace_w8 | videommmu_adaptation | 0/300 | — | — | — | — | — | — | — | incomplete |
| released_grace_w8 | videommmu_comprehension | 0/300 | — | — | — | — | — | — | — | incomplete |
| released_grace_w8 | videommmu_perception | 0/300 | — | — | — | — | — | — | — | incomplete |
| qat_w4 | tempcompass | 1580/1580 | 9.90 | 489.12 | 153.13 | 2.00 | 155.13 | 642.25 | 644.25 | complete |
| qat_w4 | videomme (short + medium) | 1800/1800 | 14.50 | 9,631.25 | 2,955.17 | 2.00 | 2,957.17 | 12,586.42 | 12,588.42 | complete |
| qat_w4 | videommmu_adaptation | 160/300 | 54.96 | 16,489.65 | 5,006.44 | 2.38 | 5,008.82 | 21,496.09 | 21,498.47 | incomplete (observed prefix) |
| qat_w4 | videommmu_comprehension | 300/300 | 59.04 | 16,939.89 | 5,190.00 | 2.00 | 5,192.00 | 22,129.90 | 22,131.90 | complete |
| qat_w4 | videommmu_perception | 300/300 | 35.39 | 16,939.89 | 5,186.78 | 2.00 | 5,188.78 | 22,126.67 | 22,128.67 | complete |
| grace_w4 | tempcompass | 1580/1580 | 9.90 | 489.12 | 153.13 | 2.00 | 155.13 | 642.25 | 644.25 | complete |
| grace_w4 | videomme (short + medium) | 1056/1800 | 14.16 | 4,546.55 | 1,371.86 | 2.00 | 1,373.86 | 5,918.40 | 5,920.40 | incomplete (observed prefix) |
| grace_w4 | videommmu_adaptation | 0/300 | — | — | — | — | — | — | — | incomplete |
| grace_w4 | videommmu_comprehension | 0/300 | — | — | — | — | — | — | — | incomplete |
| grace_w4 | videommmu_perception | 0/300 | — | — | — | — | — | — | — | incomplete |
| qat_w8 | tempcompass | 0/1580 | — | — | — | — | — | — | — | incomplete |
| qat_w8 | videomme (short + medium) | 0/1800 | — | — | — | — | — | — | — | incomplete |
| qat_w8 | videommmu_adaptation | 0/300 | — | — | — | — | — | — | — | incomplete |
| qat_w8 | videommmu_comprehension | 0/300 | — | — | — | — | — | — | — | incomplete |
| qat_w8 | videommmu_perception | 0/300 | — | — | — | — | — | — | — | incomplete |
| grace_w8 | tempcompass | 0/1580 | — | — | — | — | — | — | — | incomplete |
| grace_w8 | videomme (short + medium) | 0/1800 | — | — | — | — | — | — | — | incomplete |
| grace_w8 | videommmu_adaptation | 0/300 | — | — | — | — | — | — | — | incomplete |
| grace_w8 | videommmu_comprehension | 0/300 | — | — | — | — | — | — | — | incomplete |
| grace_w8 | videommmu_perception | 0/300 | — | — | — | — | — | — | — | incomplete |

## Question length versus full text input

**Question only** counts the annotation question stem, excluding choices, task instructions, timestamps and chat formatting. **Question + options** adds answer choices; **task prompt** also includes evaluation instructions. These full-annotation means cover all 1,580 TempCompass questions, all 300 questions in each Video-MMMU split, and 900 questions per VideoMME length. They do not require completed inference. The per-model tables above instead use the observed validated questions. CPU tokenization uses `add_special_tokens=False`; every model tokenizer was checked to produce identical counts for all three payloads.

| Benchmark | Video length | Questions | Mean question only | Mean question + options | Mean task prompt | Mean full text input (Vanilla 2B) | Video/chat residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tempcompass | all | 1580 | 9.90 | 33.59 | 41.56 | 153.13 | 111.57 |
| videomme | short | 900 | 13.76 | 42.83 | 49.83 | 781.76 | 731.93 |
| videomme | medium | 900 | 15.25 | 49.93 | 56.93 | 5,128.58 | 5,071.65 |
| videomme | long | 900 | 17.85 | 63.27 | 70.27 | 26,103.32 | 26,033.06 |
| videommmu_perception | all | 300 | 35.39 | 182.24 | 203.08 | 5,186.78 | 4,983.70 |
| videommmu_comprehension | all | 300 | 59.04 | 185.49 | 206.30 | 5,190.00 | 4,983.70 |
| videommmu_adaptation | all | 300 | 47.05 | 142.00 | 178.49 | 5,162.19 | 4,983.70 |

**Why text input scales with video length:** `input_counts()` in `evaluation/protocols/video_protocol.py` reconstructs one timestamp and vision boundary pair for every temporal video patch group: `<12.0 seconds><|vision_start|>[visual tokens]<|vision_end|>`. With 2 fps sampling and two frames per temporal patch, there is approximately one such group per video second. Longer videos therefore add many nonvisual text tokens, even when the question stays short. Larger timestamp values can also need more tokenizer pieces. No subtitles or audio are being counted. The residual column is full text input minus the separately tokenized task prompt; it includes timestamps, vision boundaries, chat markers and BPE boundary differences. The full-input reference uses certified complete Vanilla 2B results; the long row is historical and excluded from current scores.

## VideoMME average tokens by video length

Each length has 900 questions in the original population. Current runs evaluate short and medium only. Partial rows use only observed questions in that length; missing means are not zero or estimates. The status describes the parent model/suite run, even if one length already has 900 saved answers.

| Model | Video length | Questions | Mean question only | Mean visual input | Mean text input | Mean text output | Mean total text | Mean total input | Mean all tokens | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vanilla_2b | short | 900/900 | 13.76 | 2,606.40 | 781.76 | 3.96 | 785.72 | 3,388.16 | 3,392.12 | complete |
| vanilla_2b | medium | 900/900 | 15.25 | 16,656.11 | 5,128.58 | 4.76 | 5,133.33 | 21,784.68 | 21,789.44 | complete |
| vanilla_2b | long | excluded | — | — | — | — | — | — | — | excluded from current protocol |
| vanilla_8b_teacher | short | 900/900 | 13.76 | 2,606.40 | 781.76 | 6.73 | 788.49 | 3,388.16 | 3,394.89 | complete |
| vanilla_8b_teacher | medium | 900/900 | 15.25 | 16,656.11 | 5,128.58 | 8.63 | 5,137.21 | 21,784.68 | 21,793.32 | complete |
| vanilla_8b_teacher | long | excluded | — | — | — | — | — | — | — | excluded from current protocol |
| sft_2b | short | 900/900 | 13.76 | 2,606.40 | 781.76 | 2.00 | 783.76 | 3,388.16 | 3,390.16 | complete |
| sft_2b | medium | 900/900 | 15.25 | 16,656.11 | 5,128.58 | 2.00 | 5,130.58 | 21,784.68 | 21,786.68 | complete |
| sft_2b | long | excluded | — | — | — | — | — | — | — | excluded from current protocol |
| released_bf16 | short | 524/900 | 13.99 | 2,586.34 | 775.26 | 2.00 | 777.26 | 3,361.60 | 3,363.60 | incomplete (observed prefix) |
| released_bf16 | medium | —/900 | — | — | — | — | — | — | — | no validated observations |
| released_bf16 | long | excluded | — | — | — | — | — | — | — | excluded from current protocol |
| released_grace_w4 | short | —/900 | — | — | — | — | — | — | — | no validated observations |
| released_grace_w4 | medium | —/900 | — | — | — | — | — | — | — | no validated observations |
| released_grace_w4 | long | excluded | — | — | — | — | — | — | — | excluded from current protocol |
| released_grace_w8 | short | —/900 | — | — | — | — | — | — | — | no validated observations |
| released_grace_w8 | medium | —/900 | — | — | — | — | — | — | — | no validated observations |
| released_grace_w8 | long | excluded | — | — | — | — | — | — | — | excluded from current protocol |
| qat_w4 | short | 900/900 | 13.76 | 2,606.40 | 781.76 | 2.00 | 783.76 | 3,388.16 | 3,390.16 | complete |
| qat_w4 | medium | 900/900 | 15.25 | 16,656.11 | 5,128.58 | 2.00 | 5,130.58 | 21,784.68 | 21,786.68 | complete |
| qat_w4 | long | excluded | — | — | — | — | — | — | — | excluded from current protocol |
| grace_w4 | short | 900/900 | 13.76 | 2,606.40 | 781.76 | 2.00 | 783.76 | 3,388.16 | 3,390.16 | incomplete (observed prefix) |
| grace_w4 | medium | 156/900 | 16.51 | 15,739.69 | 4,776.26 | 2.00 | 4,778.26 | 20,515.95 | 20,517.95 | incomplete (observed prefix) |
| grace_w4 | long | excluded | — | — | — | — | — | — | — | excluded from current protocol |
| qat_w8 | short | —/900 | — | — | — | — | — | — | — | no validated observations |
| qat_w8 | medium | —/900 | — | — | — | — | — | — | — | no validated observations |
| qat_w8 | long | excluded | — | — | — | — | — | — | — | excluded from current protocol |
| grace_w8 | short | —/900 | — | — | — | — | — | — | — | no validated observations |
| grace_w8 | medium | —/900 | — | — | — | — | — | — | — | no validated observations |
| grace_w8 | long | excluded | — | — | — | — | — | — | — | excluded from current protocol |

### Historical long videos (excluded from current scores)

The long-video means below come only from validated older 2,700-question attempts. They are separate evidence and are **not included** in the current benchmark means or completion count. Partial historical means cover only the observed long questions; models with no validated long observations have no historical row. Source attempts and hashes are recorded in JSON.

| Model | Video length | Questions | Mean question only | Mean visual input | Mean text input | Mean text output | Mean total text | Mean total input | Mean all tokens | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vanilla_2b | long | 900/900 | 17.85 | 78,936.19 | 26,103.32 | 4.73 | 26,108.06 | 105,039.51 | 105,044.24 | historical; excluded from current scores; complete |
| sft_2b | long | 527/900 | 17.92 | 79,471.20 | 26,288.52 | 2.00 | 26,290.52 | 105,759.71 | 105,761.71 | historical; excluded from current scores; incomplete (observed prefix) |
| qat_w4 | long | 51/900 | 17.98 | 80,521.41 | 26,650.67 | 2.00 | 26,652.67 | 107,172.08 | 107,174.08 | historical; excluded from current scores; incomplete (observed prefix) |

## Tokens per source video second

**Each rate is sum(tokens) / sum(video duration in seconds).** Repeated videos count once per evaluated question because their tokens are processed again. These duration-weighted rates normalize token use for clip length; they do not measure inference speed. CSV/JSON also retain the arithmetic mean, median and p95 of per-question rates.

Text input includes the question/options, timestamps, vision boundary tokens and chat template. The timestamp/boundary sequence grows with video length even though subtitles and audio are disabled. Visual-token rates also depend on processed frame geometry and the sampling/temporal patch policy.

Rows labelled **incomplete (partial rates)** use only the latest attempt’s saved questions; they remain incomplete. Other incomplete rows have no rates. **100% saved coverage without a valid completion certificate remains incomplete.** Counts may change after this snapshot. Retries are never concatenated. Partial rows passed ordered-prefix, source/input identity, token accounting, saved-score, context and EOS checks, and their selected source manifests match the frozen datasets. The parsed prediction snapshot has a separate digest; it has no full-run completion certificate. Any unfinished final JSONL record is excluded.

**Partial populations can be biased by question order.** These are observed-prefix rates, not estimates for the missing questions. VideoMME visits short, then medium videos.

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

- `question_tokens_preview.json`: definitions, validation hashes, all run states, duration/category statistics, retry aliases and smoke results.
- `question_tokens_preview.csv`: all benchmark pairs; status and measurement scope distinguish complete, partial and unmeasured runs.
- `question_tokens_preview_questions.csv`: exact counts and both rates for current measured questions, labelled by run status and scope.
- `question_tokens_preview_categories.csv`: current category means/rates and separately labelled historical long-video means/rates.
- `question_tokens_preview_attempts.csv`: all attempt states, observed coverage and saved failure/block evidence.
