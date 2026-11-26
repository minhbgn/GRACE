# What the completed token-reduction experiment tells us

Updated **2026-11-25 (Asia/Singapore)**. All **120/120 QAT W4 configurations** are complete: 70 image and 50 video jobs. This report replaces the November 16 interpretation, uses the completed Video-MMMU splits, and replaces the two original-policy VideoMME resolution rows with the November 23/24 matched reruns. The [November 16 report](experiment_report_20261116.md) remains available for the detailed implementation explanations and integration plan.

**VideoMME resolution-r25 is the strongest measured accuracy/compute tradeoff among these simple settings:** 61.056% versus 61.500% at r100 (−0.444 percentage points), using 23.646% of the visual tokens and 13.748 recorded model-hours versus 45.240. Resolution-r10 reaches 56.833% (−4.667 pp) at 8.232% actual retention and 7.772 model-hours. These are full 1,800-question results.

The separate 50-question timing sweep estimates **15.819 h at r25 and 9.070 h at r10**, respectively **15.06% and 16.70% above** those full-run model totals. Within the full reruns, the selected questions estimate total time within 1%; most of the discrepancy comes from slower vision/embedding in the separate timing sweep. Its sampling intervals therefore should not be read as guarantees for another run.

For uniform/merge, the completed timing sweep supplies a consistent single-GPU estimate absent from the old mixed-scheduling totals. Projected model speedups are **1.86× / 3.12× / 4.97× for uniform** and **1.84× / 3.01× / 4.78× for merge** at r50/r25/r10. Full-set accuracy losses are appreciably larger than resolution at r25/r10.

![Full-set VideoMME accuracy versus projected runtime](reports/videomme_tradeoff_20261125.png)

Horizontal error bars show sampling-only 95% intervals for runtime; accuracy comes from the full sets. Lines connect measured settings and do not estimate intermediate settings. [PDF](reports/videomme_tradeoff_20261125.pdf).

## 1. Completion, source selection and scope

| Suite | Completed / 10 |
| --- | --- |
| hallusionbench | 10/10 |
| mmbench | 10/10 |
| scienceqa | 10/10 |
| ai2d | 10/10 |
| mmmu | 10/10 |
| seedbench_img | 10/10 |
| mmstar | 10/10 |
| tempcompass | 10/10 |
| videomme | 10/10 |
| videommmu_perception | 10/10 |
| videommmu_comprehension | 10/10 |
| videommmu_adaptation | 10/10 |

The matrix covers 29,420 image rows plus 4,280 video questions per configuration: **337,000 scored question inferences** across the ten settings, excluding smoke tests, timing repeats and superseded attempts. The 50-question timing sweep adds 500 measured requests and one excluded full-input warmup per configuration. VideoMME covers 900 short + 900 medium questions; long videos, audio and subtitles are excluded. Each Video-MMMU split covers 300 questions; TempCompass covers 1,580.

Source precedence is explicit, rather than selecting the newest directory by its timestamp:

- Images: `results/full`.
- VideoMME resolution-r25/r10: `results_video_matched_exclusive/full`.
- Other video rows: `results_video_retry_gpu0_scheduler_resume/full`, falling back to the original completed row when the repaired counterpart does not exist.
- Timing projections: `results_videomme_timing50_seed42_phased_20261123` only. The preparation drafts and unfinished older timing directories are excluded.

Three Video-MMMU resolution-r10 rows still use the original spatial/execution policy. Their scores are reported, but baseline deltas and speedup claims are withheld. Every VideoMME setting now has the same adaptive starting policy. The older VideoMME r25/r10 scores remain preserved in the original artifacts and are superseded in current comparisons.

This is a frozen-checkpoint, inference-only **CE-only LoRA-QAT W4** experiment; no reducer, scales or weights were trained in this sweep. Vanilla/BF16 matrices remain incomplete, so the results measure reduction within QAT W4 rather than an INT4-versus-BF16 advantage.

## 2. What each method changes

Resolution resizes the whole image/frame before vision encoding, saving vision and decoder work. Uniform encodes the full input and selects the lower-middle token from each evenly sized contiguous sequence bin. Merge uses the same bins but averages their embeddings, including aligned DeepStack features. Both compact the visual sequence before decoder layer 0 and preserve selected original M-RoPE coordinates. They keep all temporal groups; none of these methods drops frames or uses question-dependent importance.

`r025` means keep 25%, not remove 25%. Spatial grid rounding makes actual resolution retention lower than its nominal budget. Uniform and merge have identical token counts. Method comparisons at a nominal budget are not exact equal-token comparisons.

The checkpoint uses packed Triton W4A16 for 196 decoder linears, group size 128; vision, embeddings and head remain floating point. Evaluation uses batch one, greedy EOS generation, seed 42, FlashAttention 2, a 256-output-token cap, and 589,824 maximum pixels per image/frame. Videos retain full duration at approximately 2 fps, with two frames per temporal group. The repaired video path streams vision one temporal group at a time, stages features on CPU, and chunks prefill at 512 tokens while retaining the full GPU KV cache.

All repaired settings derive reduction from one common adaptive baseline with a 131,072 decoder-token operational cap. The cap adapts 846/1,800 VideoMME questions (all medium) and 251/300 questions in each Video-MMMU split. TempCompass never activates it. VideoMME r100 already retains only about 50.14% of the original-policy visual-token total. These percentages describe the adaptive baseline, rather than native full spatial detail for every long clip.

Details: [method definitions](experiment_report_20261116.md#2-resolution-uniform-and-merge-in-plain-terms), [video execution](experiment_report_20261116.md#3-the-video-choices-that-are-easy-to-miss), and [checkpoint training report](../released_bf16_init_32k_b128_p589824_t4096_seed42/experiment_report.md).

## 3. Image results: all seven suites

| Suite | baseline_r100 | res_r050 | res_r025 | res_r010 | uniform_r050 | uniform_r025 | uniform_r010 | merge_r050 | merge_r025 | merge_r010 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hallusionbench | 64.92 | 64.48 | 63.77 | 62.27 | 63.60 | 62.80 | 59.61 | 63.95 | 61.56 | 59.96 |
| mmbench | 89.26 | 87.63 | 84.97 | 78.61 | 87.97 | 83.68 | 75.26 | 88.92 | 83.42 | 69.85 |
| scienceqa | 83.49 | 82.86 | 82.10 | 79.56 | 82.36 | 81.00 | 78.68 | 83.31 | 81.68 | 79.04 |
| ai2d | 81.22 | 79.76 | 78.56 | 75.06 | 79.79 | 76.13 | 72.18 | 80.76 | 77.62 | 72.93 |
| mmmu | 67.78 | 66.67 | 66.11 | 63.00 | 65.22 | 63.44 | 61.67 | 66.00 | 63.56 | 61.00 |
| seedbench_img | 75.30 | 73.86 | 72.63 | 69.87 | 73.80 | 71.43 | 63.85 | 74.00 | 70.91 | 61.29 |
| mmstar | 67.27 | 64.87 | 63.13 | 59.20 | 64.13 | 60.73 | 56.07 | 65.53 | 62.40 | 56.87 |

Image results are unchanged. At r50, merge has the highest reduced score on five of seven suites; resolution wins HallusionBench and MMMU. At r25 and r10, resolution scores highest among the three methods on every image suite. These are descriptive single-checkpoint results, without a significance claim for small differences.

Normal-answer model-time ratios against the same suite's r100:

| Suite | res_r050 | res_r025 | res_r010 | uniform_r050 | uniform_r025 | uniform_r010 | merge_r050 | merge_r025 | merge_r010 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hallusionbench | 1.468× | 1.826× | 2.212× | 1.410× | 1.778× | 2.105× | 1.408× | 1.775× | 2.107× |
| mmbench | 1.257× | 1.405× | 1.551× | 1.231× | 1.395× | 1.526× | 1.234× | 1.394× | 1.527× |
| scienceqa | 1.140× | 1.198× | 1.263× | 1.106× | 1.191× | 1.259× | 1.106× | 1.190× | 1.257× |
| ai2d | 1.397× | 1.672× | 1.960× | 1.334× | 1.642× | 1.898× | 1.336× | 1.636× | 1.895× |
| mmmu | 1.330× | 1.530× | 1.733× | 1.289× | 1.507× | 1.704× | 1.291× | 1.495× | 1.698× |
| seedbench_img | 1.592× | 2.080× | 2.498× | 1.471× | 1.991× | 2.384× | 1.470× | 1.991× | 2.373× |
| mmstar | 1.290× | 1.447× | 1.635× | 1.261× | 1.424× | 1.604× | 1.265× | 1.421× | 1.602× |

Three-question, fixed-32-output-token profile ratios:

| Suite | res_r050 | res_r025 | res_r010 | uniform_r050 | uniform_r025 | uniform_r010 | merge_r050 | merge_r025 | merge_r010 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hallusionbench | 1.088× | 1.130× | 1.161× | 1.066× | 1.118× | 1.138× | 1.070× | 1.114× | 1.146× |
| mmbench | 1.005× | 1.010× | 1.012× | 1.004× | 1.010× | 1.006× | 1.006× | 1.011× | 1.008× |
| scienceqa | 1.017× | 1.024× | 1.031× | 1.015× | 1.021× | 1.024× | 1.020× | 1.021× | 1.027× |
| ai2d | 1.035× | 1.037× | 1.065× | 1.025× | 1.045× | 1.071× | 1.033× | 1.044× | 1.065× |
| mmmu | 1.026× | 1.026× | 1.029× | 1.007× | 1.031× | 1.030× | 1.012× | 1.028× | 1.030× |
| seedbench_img | 1.086× | 1.123× | 1.149× | 1.073× | 1.119× | 1.145× | 1.078× | 1.121× | 1.146× |
| mmstar | 1.033× | 1.042× | 1.068× | 1.031× | 1.047× | 1.063× | 1.038× | 1.044× | 1.067× |

The different speedup ranges reflect different output workloads: normal image answers are usually two tokens, while the diagnostic forces 32. The three-question profile is not a representative latency distribution. Token reduction mostly saves input work, so its relative benefit shrinks when generating a longer answer.

## 4. TempCompass: uniform-r25 remains a useful simple control

| Configuration | Score % | Delta pp | Mean visual | Retained % | Model s | Observed time ratio | 32-token ratio | Peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_r100 | 61.329 | +0.000 | 2,716.353 | 100.000 | 3.021 | 1.000× | 1.000× | 3.160 |
| resolution_r050 | 60.570 | -0.759 | 1,252.280 | 46.102 | 1.679 | 1.800× | 1.224× | 2.563 |
| resolution_r025 | 60.190 | -1.139 | 657.394 | 24.201 | 1.132 | 2.668× | 1.302× | 2.336 |
| resolution_r010 | 58.354 | -2.975 | 222.729 | 8.200 | 0.727 | 4.153× | 1.389× | 2.171 |
| uniform_r050 | 61.835 | +0.506 | 1,358.177 | 50.000 | 1.791 | 1.687× | 1.169× | 2.614 |
| uniform_r025 | 61.646 | +0.316 | 681.326 | 25.082 | 1.173 | 2.575× | 1.282× | 2.345 |
| uniform_r010 | 59.367 | -1.962 | 273.156 | 10.056 | 0.804 | 3.755× | 1.324× | 2.261 |
| merge_r050 | 61.582 | +0.253 | 1,358.177 | 50.000 | 1.788 | 1.690× | 1.184× | 2.614 |
| merge_r025 | 61.329 | +0.000 | 681.326 | 25.082 | 1.169 | 2.583× | 1.295× | 2.345 |
| merge_r010 | 56.329 | -5.000 | 273.156 | 10.056 | 0.807 | 3.744× | 1.342× | 2.261 |

Uniform-r25 retains 25.082% of visual tokens, scores 61.646% versus 61.329% at r100 (+0.316 pp, five net additional correct answers), and has an observed normal-answer time ratio of 2.575×. Merge-r25 matches the baseline score. Resolution-r25 loses 1.139 pp, while merge-r10 loses 5.000 pp. All 1,580 questions per setting complete with no capped answers.

These small positive differences are not evidence that pruning reliably improves accuracy. Category effects also differ: uniform-r25 improves speed and order scores but lowers attribute-change accuracy; the full category table is in [the generated tables](reports/completed_tables_20261125.md). TempCompass is scored with the deterministic local MC fallback, not an API judge.

## 5. VideoMME: the matched full-population results

All rows below now use matching temporal coverage and adaptive baseline geometry, verified per question. All answers have two generated tokens and no output-cap hits. Retention is relative to adaptive r100. The model-time ratios are ratios of recorded timers across dates/cards, not repeated-run controlled speed benchmarks. In particular, the six uniform/merge full totals contain a replayed mixture of shared-GPU and one-worker timings; use section 7 for their consistent timing estimates.

| Configuration | Score % | Delta pp | Mean visual | Retained % | Model s | Observed time ratio | 32-token ratio | Peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_r100 | 61.500 | +0.000 | 78,494.013 | 100.000 | 90.479 | 1.000× | 1.000× | 16.170 |
| resolution_r050 | 61.222 | -0.278 | 35,840.322 | 45.660 | 43.943 | 2.059× | 1.982× | 9.150 |
| resolution_r025 | 61.056 | -0.444 | 18,560.920 | 23.646 | 27.497 | 3.291× | 3.106× | 6.091 |
| resolution_r010 | 56.833 | -4.667 | 6,461.872 | 8.232 | 15.545 | 5.821× | 5.589× | 4.181 |
| uniform_r050 | 60.611 | -0.889 | 39,279.495 | 50.041 | 65.611 | 1.379× | 0.721× | 9.506 |
| uniform_r025 | 57.444 | -4.056 | 19,689.425 | 25.084 | 46.465 | 1.947× | 1.040× | 6.248 |
| uniform_r010 | 52.944 | -8.556 | 7,942.640 | 10.119 | 35.431 | 2.554× | 1.438× | 4.948 |
| merge_r050 | 60.444 | -1.056 | 39,279.495 | 50.041 | 64.730 | 1.398× | 0.755× | 9.506 |
| merge_r025 | 56.722 | -4.778 | 19,689.425 | 25.084 | 45.221 | 2.001× | 1.063× | 6.248 |
| merge_r010 | 48.611 | -12.889 | 7,942.640 | 10.119 | 34.436 | 2.627× | 1.482× | 4.948 |

Resolution-r25 loses only eight net correct answers against r100: 94 errors are fixed and 102 new errors appear. Relative to resolution-r50 it loses three correct answers (−0.167 pp), while recorded model time falls 21.971 → 13.748 h (37.4% lower). Peak PyTorch allocation is 6.091 GiB versus 16.170 at r100, a 62.3% reduction; this excludes reserved memory and other process/device usage.

At r25, resolution scores **3.611 pp above uniform** and **4.333 pp above merge**, while retaining fewer visual tokens (23.646% versus 25.084%). At r10 it scores **3.889 pp above uniform** and **8.222 pp above merge**, again with fewer tokens (8.232% versus 10.119%). Resolution-r10 sacrifices 4.667 pp against baseline and uses 4.181 GiB peak allocation. A strict accuracy requirement may still favor r25 or r50.

Short/medium accuracy and paired error swaps against r100:

| Configuration | short | medium | Fixes | New errors |
| --- | --- | --- | --- | --- |
| baseline_r100 | 68.556 | 54.444 | 0 | 0 |
| resolution_r050 | 68.667 | 53.778 | 69 | 74 |
| resolution_r025 | 68.444 | 53.667 | 94 | 102 |
| resolution_r010 | 64.444 | 49.222 | 87 | 171 |
| uniform_r050 | 66.889 | 54.333 | 53 | 69 |
| uniform_r025 | 63.556 | 51.333 | 64 | 137 |
| uniform_r010 | 57.333 | 48.556 | 71 | 225 |
| merge_r050 | 67.333 | 53.556 | 50 | 69 |
| merge_r025 | 62.333 | 51.111 | 74 | 160 |
| merge_r010 | 51.333 | 45.889 | 77 | 309 |

The r25 loss is concentrated in medium videos: short changes −0.111 pp and medium −0.778 pp. At r10 both degrade (−4.111 and −5.222 pp). Uniform/merge at r10 hurt short accuracy particularly strongly, so sensitivity cannot be inferred from duration alone.

### How the reruns change the old interpretation

| Setting | Old original-policy accuracy % | Matched rerun accuracy % | Old mean visual | Matched mean visual | Matched delta vs r100 pp |
| --- | --- | --- | --- | --- | --- |
| resolution_r025 | 61.722 | 61.056 | 38,893.043 | 18,560.920 | −0.444 |
| resolution_r010 | 59.000 | 56.833 | 13,557.088 | 6,461.872 | −4.667 |

The original r25 used more tokens than repaired r50 (38,893 versus 35,840 per question), despite its smaller nominal label. The rerun resolves that mismatch. It supports r25 as a strong tradeoff at a genuinely smaller budget; it does not support the old suggestion of an accuracy improvement over adaptive r100.

## 6. Does the resolution subset predict the new full reruns?

The pinned seed-42 subset contains 50 distinct videos: 25 short and 25 medium, five videos from each original-visual-token quintile within each duration, with one randomly selected question per video. Each observation has expansion weight 36. The projected full model time is `36 × sum(50 measured model times)`; no extrapolation of accuracy is used.

Two checks answer different questions. **Same-run backtesting** takes the selected IDs from the full-run telemetry and expands those saved times. **Fresh projection error** compares the independently measured 50-question sweep against the full run. Both use the same selected questions and spatial/frame geometry.

| Configuration | Recorded full model h | Fresh subset projection h [95% CI] | Fresh error vs full | Same-run subset projection h | Same-run error |
| --- | --- | --- | --- | --- | --- |
| baseline_r100 | 45.240 | 46.27 [45.73, 46.76] | +2.27% | 45.806 | +1.25% |
| resolution_r050 | 21.971 | 23.31 [22.75, 23.90] | +6.09% | 22.186 | +0.98% |
| resolution_r025 | 13.748 | 15.82 [15.04, 16.58] | +15.06% | 13.738 | -0.07% |
| resolution_r010 | 7.772 | 9.07 [8.56, 9.60] | +16.70% | 7.723 | -0.63% |

For r25/r10, same-run sampling error is −0.07%/−0.63%, but fresh total error is +15.06%/+16.70%. Their full totals lie below the fresh 95% intervals. Both fail the older screening threshold of ≤10% overall error; medium errors also exceed its ≤15% duration threshold:

| Setting | Fresh short error | Fresh medium error | Same-run short error | Same-run medium error |
| --- | --- | --- | --- | --- |
| resolution_r025 | +6.01% | +17.90% | -1.49% | +0.37% |
| resolution_r010 | +4.88% | +19.63% | -5.63% | +0.61% |

The component timers locate most of the shift in vision/embedding, rather than prefill. The following compares **the same 50 IDs** expanded by 36 in each run:

| Setting | Full-run selected model h | Fresh selected model h | Vision/embedding h: full → fresh | Prefill h: full → fresh |
| --- | --- | --- | --- | --- |
| resolution_r025 | 13.738 | 15.819 | 3.611 → 5.657 | 10.031 → 10.057 |
| resolution_r010 | 7.723 | 9.070 | 3.240 → 4.565 | 4.425 → 4.446 |

Vision/embedding grows 56.6% at r25 and 40.9% at r10; prefill grows only 0.26% and 0.47%. Full reruns used GPU 2 and the timing sweep GPU 4. Host load, transfers, clocks and card differences are plausible contributors, but these artifacts do not establish a cause. Matching geometry and matching selected predictions rule out a changed workload or answer length for these two comparisons.

The inference is that this subset represents the resolution full-run workload well, while absolute runtime changes across executions. It is not validated to ±10% on the fresh r25/r10 runs. The old [projection check](results_videomme_timing50_seed42_phased_20261123/projection_check.md) predates these reruns and compares against the original-policy r25/r10; it remains a historical phase gate. This report's [current timing audit](reports/timing_analysis_20261125.json) is the updated comparison. No seed reselection or post-hoc correction factor has been applied.

## 7. Uniform and merge extrapolation, including request time

All ten timing configurations finished on the same exclusive A100 (physical GPU 4), sequentially, with batch one and no answer replay. Saved GPU observations contain one process per configuration. The first full-input generation is used as warmup and excluded. Accuracy below comes from the full 1,800-question runs, never the small timing sample.

**Model time** sums synchronized vision/embedding, reduction, prefill and decode. **Request time** additionally includes media preprocessing, input staging and measured generation-call overhead. Startup, warmup, harness setup/scoring, durable checkpoint writes and unmeasured filesystem context-manager overhead are excluded. These are estimated steady-state serial totals, not observed full-run wall-clock durations.

Hours and speedups include paired within-stratum bootstrap 95% intervals (4,000 draws):

| Configuration | Full accuracy % | Delta pp | Projected model h [95% CI] | Projected request h [95% CI] | Model speedup [95% CI] | Request speedup [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_r100 | 61.500 | +0.000 | 46.27 [45.73, 46.76] | 56.69 [55.59, 57.72] | 1.00× [1.00, 1.00] | 1.00× [1.00, 1.00] |
| resolution_r050 | 61.222 | -0.278 | 23.31 [22.75, 23.90] | 36.67 [35.35, 38.10] | 1.98× [1.94, 2.03] | 1.55× [1.49, 1.60] |
| resolution_r025 | 61.056 | -0.444 | 15.82 [15.04, 16.58] | 32.09 [30.42, 33.68] | 2.92× [2.79, 3.07] | 1.77× [1.68, 1.86] |
| resolution_r010 | 56.833 | -4.667 | 9.07 [8.56, 9.60] | 24.12 [22.89, 25.34] | 5.10× [4.83, 5.39] | 2.35× [2.24, 2.47] |
| uniform_r050 | 60.611 | -0.889 | 24.81 [24.45, 25.15] | 34.94 [33.99, 35.83] | 1.86× [1.85, 1.88] | 1.62× [1.60, 1.64] |
| uniform_r025 | 57.444 | -4.056 | 14.81 [14.51, 15.09] | 24.91 [23.97, 25.79] | 3.12× [3.08, 3.17] | 2.28× [2.23, 2.32] |
| uniform_r010 | 52.944 | -8.556 | 9.31 [9.01, 9.62] | 21.52 [20.55, 22.44] | 4.97× [4.82, 5.11] | 2.63× [2.55, 2.72] |
| merge_r050 | 60.444 | -1.056 | 25.11 [24.74, 25.45] | 36.35 [35.38, 37.28] | 1.84× [1.83, 1.86] | 1.56× [1.54, 1.58] |
| merge_r025 | 56.722 | -4.778 | 15.38 [14.96, 15.82] | 27.42 [26.28, 28.57] | 3.01× [2.94, 3.08] | 2.07× [2.01, 2.13] |
| merge_r010 | 48.611 | -12.889 | 9.69 [9.31, 10.07] | 21.66 [20.84, 22.48] | 4.78× [4.60, 4.95] | 2.62× [2.55, 2.69] |

Uniform-r25 projects **14.81 model-hours / 24.91 request-hours** at 57.444% accuracy. Merge-r25 projects **15.38 / 27.42 h** at 56.722%. At r10, uniform projects **9.31 / 21.52 h** at 52.944%, and merge **9.69 / 21.66 h** at 48.611%. In this sweep uniform has lower projected model and request time than merge at every budget, as well as higher full-set accuracy. Small timing gaps do not establish a repeatable runtime ranking.

For comparison, resolution-r25 projects **15.82 / 32.09 h** at 61.056%. Thus it preserves considerably more accuracy but is not the fastest projected request at that nominal budget in this execution. Its unexpectedly slower vision component, and the difference from its full-run timing, should be considered before making a method-level runtime claim. At r10, resolution projects **9.07 / 24.12 h** at 56.833%; uniform's request estimate is lower even though its model estimate is slightly higher. Request measurements capture costs that model timers omit.

Relative to the 56.69 h projected r100 request total, uniform request speedups are about **1.62× / 2.28× / 2.63×**, and merge about **1.56× / 2.07× / 2.62×**. The 3–5× model speedups at small budgets therefore should not be described as end-to-end speedups.

### Why these estimates differ from the historical uniform/merge totals

| Configuration | Old recorded model h | Fresh projected model h | Fresh minus old | Historical same-run sampling error |
| --- | --- | --- | --- | --- |
| uniform_r050 | 32.805 | 24.809 | -24.37% | -2.26% |
| uniform_r025 | 23.233 | 14.805 | -36.27% | -6.48% |
| uniform_r010 | 17.715 | 9.314 | -47.42% | -15.41% |
| merge_r050 | 32.365 | 25.106 | -22.43% | -2.15% |
| merge_r025 | 22.610 | 15.375 | -32.00% | -4.93% |
| merge_r010 | 17.218 | 9.688 | -43.73% | -13.18% |

The old full totals mix shared-GPU and one-worker timers. For example, uniform-r10 retains 1,032 shared-GPU records and only 768 one-worker records; merge-r10 retains 1,050 and 750. The fresh projected totals are 22–47% below the old totals, consistent with this scheduling difference. They should not be evaluated as prediction failures against a controlled standalone ground truth, because those old totals are not that ground truth.

At r10, even same-run expansion underestimates the mixed historical totals by 15.41% for uniform and 13.18% for merge, mostly in medium questions. Good resolution backtests do not prove unbiased uniform/merge sampling under a different scheduling history. The fixed subset avoids choosing a favorable seed, but only five videos per stratum remain a small sample.

Intervals quantify sampling variation conditional on this execution. They exclude run-to-run hardware/host variation and do not imply that full accuracy has a bootstrap interval here. In fact, none of the ten historical full-model totals falls within the corresponding fresh sampling-only interval. Nine configurations reproduce all 50 selected answer strings; merge-r10 changes one. Historical full-set accuracy remains authoritative. Request-level full-population telemetry was not saved for the reruns, so request-hour projections cannot be backtested against measured full request totals.

## 8. Video-MMMU: all three splits are now complete

The following table covers all 30 split/configuration pairs, with 300 questions each. “Original” resolution-r10 rows use a different spatial/execution policy; their baseline ratios are intentionally unavailable. Other model-time ratios are descriptive across recorded runs/cards. Generated answer length varies, unlike VideoMME, so runtime differences also include changed output work.

| Split | Configuration | Origin | Score % | Cap hits | Mean visual | Model s | Observed ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| videommmu_perception | baseline_r100 | recovery | 44.000 | 2 | 111,817.810 | 136.198 | 1.000× |
| videommmu_perception | resolution_r050 | recovery | 45.333 | 0 | 50,910.127 | 67.093 | 2.030× |
| videommmu_perception | resolution_r025 | recovery | 38.333 | 0 | 26,176.910 | 41.660 | 3.269× |
| videommmu_perception | resolution_r010 | original | 38.000 | 1 | 24,842.913 | 26.388 | N/A |
| videommmu_perception | uniform_r050 | recovery | 41.333 | 1 | 55,961.920 | 73.938 | 1.842× |
| videommmu_perception | uniform_r025 | recovery | 34.333 | 1 | 28,044.453 | 44.895 | 3.034× |
| videommmu_perception | uniform_r010 | recovery | 34.000 | 1 | 11,360.583 | 29.300 | 4.648× |
| videommmu_perception | merge_r050 | recovery | 41.333 | 0 | 55,961.920 | 74.198 | 1.836× |
| videommmu_perception | merge_r025 | recovery | 33.000 | 0 | 28,044.453 | 45.163 | 3.016× |
| videommmu_perception | merge_r010 | recovery | 31.667 | 0 | 11,360.583 | 28.800 | 4.729× |
| videommmu_comprehension | baseline_r100 | recovery | 23.667 | 5 | 111,765.337 | 136.143 | 1.000× |
| videommmu_comprehension | resolution_r050 | recovery | 20.333 | 3 | 50,870.200 | 66.804 | 2.038× |
| videommmu_comprehension | resolution_r025 | recovery | 21.000 | 1 | 26,170.220 | 42.341 | 3.215× |
| videommmu_comprehension | resolution_r010 | original | 21.667 | 4 | 24,842.913 | 26.929 | N/A |
| videommmu_comprehension | uniform_r050 | recovery | 23.000 | 3 | 55,936.520 | 74.717 | 1.822× |
| videommmu_comprehension | uniform_r025 | recovery | 23.667 | 4 | 28,028.897 | 46.456 | 2.931× |
| videommmu_comprehension | uniform_r010 | recovery | 22.667 | 3 | 11,353.467 | 29.938 | 4.547× |
| videommmu_comprehension | merge_r050 | recovery | 22.667 | 4 | 55,936.520 | 74.456 | 1.829× |
| videommmu_comprehension | merge_r025 | recovery | 21.000 | 3 | 28,028.897 | 45.380 | 3.000× |
| videommmu_comprehension | merge_r010 | recovery | 23.000 | 5 | 11,353.467 | 29.214 | 4.660× |
| videommmu_adaptation | baseline_r100 | recovery | 27.000 | 31 | 111,769.780 | 140.694 | 1.000× |
| videommmu_adaptation | resolution_r050 | recovery | 28.667 | 29 | 50,900.850 | 71.252 | 1.975× |
| videommmu_adaptation | resolution_r025 | recovery | 26.333 | 35 | 26,166.887 | 46.038 | 3.056× |
| videommmu_adaptation | resolution_r010 | original | 27.667 | 40 | 24,842.913 | 31.452 | N/A |
| videommmu_adaptation | uniform_r050 | recovery | 28.667 | 32 | 55,938.347 | 77.982 | 1.804× |
| videommmu_adaptation | uniform_r025 | recovery | 29.667 | 43 | 28,029.187 | 51.318 | 2.742× |
| videommmu_adaptation | uniform_r010 | recovery | 26.667 | 29 | 11,355.843 | 34.104 | 4.125× |
| videommmu_adaptation | merge_r050 | recovery | 27.000 | 31 | 55,938.347 | 78.465 | 1.793× |
| videommmu_adaptation | merge_r025 | recovery | 29.667 | 40 | 28,029.187 | 50.720 | 2.774× |
| videommmu_adaptation | merge_r010 | recovery | 24.333 | 39 | 11,355.843 | 35.542 | 3.959× |

Perception favors resolution-r50: 45.333% versus 44.000% at baseline (+1.333 pp). Stronger reduction degrades it: resolution-r25 is 38.333%, uniform-r25 34.333%, and merge-r25 33.000%. The unmatched original r10 cannot be placed on the same retention curve.

Comprehension is lower and relatively flat: baseline 23.667%, uniform-r25 23.667%, uniform-r10 22.667%, merge-r10 23.000%. Resolution-r50/r25 score 20.333%/21.000%. Keeping more visual input does not consistently improve this split for this checkpoint, but differences of a few questions at n=300 do not establish equivalence or a reliable improvement.

Adaptation baseline is 27.000%; uniform-r25 and merge-r25 both reach 29.667% (+2.667 pp, about eight additional correct answers). This result has a substantial generation-cap limitation: **31/300 capped answers at baseline, 43 at uniform-r25, 40 at merge-r25**, and 29–43 across the matched settings. Outputs at the cap stay in the score; neither dropping them nor calling these uncensored reasoning scores would be justified. Merge-r10 falls to 24.333% with 39 cap hits.

The 16 newly completed configurations since November 16 are seven comprehension rows and nine adaptation rows. The two matched VideoMME reruns replace previously completed but policy-mismatched results; they are not additional matrix cells. These completed splits strengthen the case for task-dependent reduction choices, rather than a single winning method.

## 9. Token counts and complete accuracy-retention curves

| Suite | Rows | Visual sum (exact) | Mean visual | Mean text | Mean total input | Mean input + output |
| --- | --- | --- | --- | --- | --- | --- |
| hallusionbench | 1129 | 380,159 | 336.721878 | 41.483614 | 378.205492 | 380.205492 |
| mmbench | 4329 | 653,110 | 150.868561 | 75.057750 | 225.926311 | 227.926311 |
| scienceqa | 4241 | 323,985 | 76.393539 | 79.534308 | 155.927847 | 157.927847 |
| ai2d | 3088 | 842,370 | 272.788212 | 53.295337 | 326.083549 | 328.083549 |
| mmmu | 900 | 250,776 | 278.640000 | 107.396667 | 386.036667 | 388.204444 |
| seedbench_img | 14233 | 7,772,322 | 546.077566 | 55.012717 | 601.090283 | 603.090283 |
| mmstar | 1500 | 285,192 | 190.128000 | 77.406000 | 267.534000 | 269.534000 |
| tempcompass | 1580 | 4,291,838 | 2,716.353165 | 153.131013 | 2,869.484177 | 2,871.484177 |
| videomme | 1800 | 141,289,224 | 78,494.013333 | 2,959.166667 | 81,453.180000 | 81,455.180000 |
| videommmu_perception | 300 | 33,545,343 | 111,817.810000 | 5,178.966667 | 116,996.776667 | 117,001.306667 |
| videommmu_comprehension | 300 | 33,529,601 | 111,765.336667 | 5,182.033333 | 116,947.370000 | 116,955.280000 |
| videommmu_adaptation | 300 | 33,530,934 | 111,769.780000 | 5,160.696667 | 116,930.476667 | 116,974.426667 |

Counts are per evaluated question, including repeated questions on the same video and text-only image rows. Visual embeddings exclude prompt text, timestamps and boundary markers. The source-frame rate and actual visual-token counts vary with aspect ratio, clip duration, adaptive cap and grid rounding; 576 is a spatial ceiling per image or two-frame temporal group, not a whole-video token budget.

| Suite | baseline_r100 | res_r050 | res_r025 | res_r010 | uniform_r050 | uniform_r025 | uniform_r010 | merge_r050 | merge_r025 | merge_r010 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hallusionbench | 336.722 | 157.317 | 79.570 | 28.800 | 168.461 | 84.392 | 34.023 | 168.461 | 84.392 | 34.023 |
| mmbench | 150.869 | 68.048 | 35.817 | 12.532 | 75.473 | 37.802 | 15.526 | 75.473 | 37.802 | 15.526 |
| scienceqa | 76.394 | 34.996 | 17.180 | 6.623 | 38.304 | 19.248 | 7.810 | 38.304 | 19.248 | 7.810 |
| ai2d | 272.788 | 125.861 | 64.426 | 22.930 | 136.519 | 68.427 | 27.633 | 136.519 | 68.427 | 27.633 |
| mmmu | 278.640 | 128.134 | 65.732 | 23.324 | 139.427 | 69.887 | 28.222 | 139.427 | 69.887 | 28.222 |
| seedbench_img | 546.078 | 260.031 | 129.535 | 49.876 | 273.244 | 136.673 | 55.018 | 273.244 | 136.673 | 55.018 |
| mmstar | 190.128 | 86.625 | 45.109 | 15.844 | 95.131 | 47.663 | 19.431 | 95.131 | 47.663 | 19.431 |
| tempcompass | 2,716.353 | 1,252.280 | 657.394 | 222.729 | 1,358.177 | 681.326 | 273.156 | 1,358.177 | 681.326 | 273.156 |
| videomme | 78,494.013 | 35,840.322 | 18,560.920 | 6,461.872 | 39,279.495 | 19,689.425 | 7,942.640 | 39,279.495 | 19,689.425 | 7,942.640 |
| videommmu_perception | 111,817.810 | 50,910.127 | 26,176.910 | 24,842.913 | 55,961.920 | 28,044.453 | 11,360.583 | 55,961.920 | 28,044.453 | 11,360.583 |
| videommmu_comprehension | 111,765.337 | 50,870.200 | 26,170.220 | 24,842.913 | 55,936.520 | 28,028.897 | 11,353.467 | 55,936.520 | 28,028.897 | 11,353.467 |
| videommmu_adaptation | 111,769.780 | 50,900.850 | 26,166.887 | 24,842.913 | 55,938.347 | 28,029.187 | 11,355.843 | 55,938.347 | 28,029.187 | 11,355.843 |

![Measured accuracy changes versus actual retention for all twelve suites](reports/accuracy_retention_20261125.png)

Each panel has its own vertical scale. The three original-policy Video-MMMU resolution-r10 points are omitted from matched curves; both VideoMME reruns are included. [PDF](reports/accuracy_retention_20261125.pdf). Baseline throughput, source-time rates, category scores and other generated tables are available in [the complete table artifact](reports/completed_tables_20261125.md).

## 10. Research decisions supported by these results

Use **VideoMME resolution-r25** as the main simple accuracy-preserving control for a stronger reducer: it removes about 76% of adaptive-baseline visual tokens for a 0.444 pp loss. Include resolution-r50 for a gentler budget and resolution-r10 for an aggressive budget. Match actual token counts when comparing a new selector, since equal nominal ratios do not give equal counts.

Keep **TempCompass uniform-r25** as a task-specific control and **image merge-r50 / resolution-r25** as useful image controls. Video-MMMU perception favors a milder resolution budget, while comprehension/adaptation have low absolute accuracy and adaptation has many capped outputs. These results do not support treating one reduction rule as universally best.

The uniform-versus-merge accuracy gap at aggressive VideoMME budgets supports testing a content-aware selector that preserves distinctive embeddings. It does not prove that a learned selector or DART will outperform resolution. The previous [LLMC+/joint-QAT integration plan](experiment_report_20261116.md#11-a-concrete-next-step-integration-plan) remains a proposed next experiment, separate from this inference-only sweep.

For runtime planning, use the completed single-GPU projected model/request totals with the execution-shift limitation established by r25/r10. Do not scale the uniform/merge estimates by a fitted resolution correction: the shift is concentrated in vision/embedding, whose workload differs by method. A same-card repeat would be needed to estimate repeatability or claim a stable absolute latency advantage. No additional GPU work is required for this report.

## 11. Evidence and reproducibility

The audit verifies every selected completion's artifact SHA-256 values, successful exit, validation payload, ordered full telemetry coverage, positive packed-INT4 calls, all 28 decoder-layer visual counts, telemetry sums/means, fixed-output profile aggregates, 196 converted modules and integer-code verification. It verifies the original/recovery protocol chain, saved reuse markers and current pinned evaluation/extension source hashes. VideoMME/TempCompass correctness is independently counted from saved scorer records; Video-MMMU uses the saved validated local scorer results. This is an artifact/telemetry audit, not a regeneration of answers or an independent benchmark scorer implementation.

The timing audit separately verifies all ten 50-question completions, protocol/subset hashes, per-question IDs, positive packed-INT4 telemetry, GPU observations, excluded warmups, summed request components and matching full/subset geometry/token counts. It recomputes all bootstrap projections and requires exact agreement with the saved extrapolation JSON. It then backtests the same selected IDs against the newly selected full runs.

- [Full metrics and provenance JSON](reports/completed_analysis_20261125.json) / [CSV](reports/completed_analysis_20261125.csv).
- [Timing comparison and component decomposition JSON](reports/timing_analysis_20261125.json) / [CSV](reports/timing_analysis_20261125.csv).
- [All generated tables](reports/completed_tables_20261125.md).
- [Original timing-sweep extrapolation](results_videomme_timing50_seed42_phased_20261123/extrapolation.md).
- [Historical November 16 interpretation](experiment_report_20261116.md); the November 5/9 [report](experiment_report.md) and progress sections are historical snapshots.

Rebuild from the experiment directory (the analysis scripts do not use GPUs or change run artifacts):

```bash
/home/air/miniconda3/envs/grace/bin/python reports/compile_latest_analysis.py
/home/air/miniconda3/envs/grace/bin/python reports/compile_timing_analysis.py
/home/air/miniconda3/envs/fm/bin/python reports/plot_latest_analysis.py
/home/air/miniconda3/envs/grace/bin/python reports/write_latest_report.py
```

The plotting command uses the existing `fm` environment because it has matplotlib. The historical compilers keep their original source-selection and narrative assumptions and should not be used to overwrite this completed-matrix analysis.
