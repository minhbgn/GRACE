# GRACE experiments: consolidated findings

Updated **2026-11-25, Asia/Singapore**. The completed studies favor **CE-only QAT over matched GRACE on the seven-suite image mean**. The latest inference study finds that simple spatial resolution reduction preserves substantially more accuracy at small budgets than uniform selection or contiguous-bin merging, especially on VideoMME. These are single-seed observations under the recorded local protocols.

The released `Qwen3-VL-2B-GRACE-BF16` initialization comes from the authors' **ordinary BF16 SFT training**, without GDKD/RCKA. QAT continues that model with CE-only LoRA-QAT. GRACE uses the same initialization and adds GDKD, RCKA and the adaptive controller. Its name does not imply that the shared initialization was distilled.

## How the experiments developed

| Study | Initialization and budget | Main observation | Evidence now available |
| --- | --- | --- | --- |
| `constrained_32k_seed42` | Vanilla 2B; 49 visual-token ceiling, 1,024 total tokens; batch 64, 512 updates | SFT/QAT/GRACE means 65.82/63.99/63.39; GRACE−QAT −0.59 pp | Historical report |
| `constrained_32k_b128_p100352_t2048_seed42` | Vanilla 2B; 98-token ceiling, 2,048 total tokens; batch 128, 256 updates | SFT/QAT/GRACE 67.74/65.89/64.96; GRACE−QAT −0.93 pp | Historical report |
| `released_bf16_init_32k_b128_p100352_t2048_seed42` | Released BF16 SFT; 98-token ceiling; W4/W8; batch 128, 256 updates | Stronger initialization raises scores; QAT still leads GRACE. Original short-generation/per-row MMBench means differ from the later local protocol | Historical report and protocol/audit notes |
| `reproduction_eval_only` | Existing 98-token exports and five frozen references; native processors, greedy 256 | Local QAT W4/W8 73.77/75.54; GRACE 72.61/73.34. Released BF16 77.40 | Report; raw predictions absent locally |
| `released_bf16_init_32k_b128_p589824_t2048_seed42` | Earlier name/preparation for the larger-budget recipe | Contains superseded memory/preparation evidence; its scaffold is not another completed production comparison | Historical scaffold/readme; use the `t4096` study |
| `released_bf16_init_32k_b128_p589824_t4096_seed42` | Released BF16 SFT; 576-token ceiling, 4,096 total tokens; W4/W8; batch 128, 256 updates | QAT W4/W8 75.77/77.37; GRACE 74.56/74.86; GRACE−QAT −1.22/−2.51 pp | Checkpoints and reports; original evaluation predictions absent locally |
| `qat_int4_token_reduction_576_seed42` | Frozen larger-budget QAT W4; packed W4A16; ten settings × twelve suites | 120/120 full configurations, 337,000 scored inferences; latest matched VideoMME r25 is a useful accuracy/compute tradeoff | All selected raw artifacts verified, plus ten complete timing configurations |

The first three historical evaluation recipes use different prompts/parsers, short answer caps and per-row MMBench. Their means must not be substituted for the later native greedy-256/circular-MMBench results. Scores from different protocols remain separate in [the merged results](../reports/merged_results.json). Historical entries are marked `report_only`; that label describes this checkout's evidence, not a claim that the original studies were incomplete.

## Larger-budget QAT and GRACE controls

| Method | 98-token mean % | 576-token mean % | Change pp | GRACE−QAT at 576 pp |
| --- | ---: | ---: | ---: | ---: |
| QAT W4 | 73.77 | 75.77 | +2.01 | — |
| GRACE W4 | 72.61 | 74.56 | +1.95 | −1.22 |
| QAT W8 | 75.54 | 77.37 | +1.83 | — |
| GRACE W8 | 73.34 | 74.86 | +1.52 | −2.51 |

These are equal-weight means of seven benchmark metrics, not pooled accuracy. Deltas use unrounded source scores; rounded cells can differ slightly in subtraction. QAT W8 is close to released BF16's 77.40 mean, but processor budgets differ, so this does not establish precision equivalence.

The larger-budget recipe improves 27 of 28 model/task comparisons. GRACE wins HallusionBench, ScienceQA and SEED against matched QAT, and loses MMBench, AI2D, MMMU and MMStar at both precisions. MMStar is the largest deficit, especially math; all local MMStar answers are valid choices, so extraction failures do not explain the gap. Excluding MMStar as a diagnostic still leaves GRACE behind by 0.10 pp at W4 and 0.73 pp at W8. The seven-suite mean remains the primary outcome.

All four production arms used 32,768 examples, seed 42, two A100s, microbatch one, accumulation 64, LoRA rank/alpha 32/64, group size 128 and 256 updates. The 576 ceiling is a maximum, not a constant input length: mean visual count is 254.64 and mean total length 478.91; maximum total length is 2,209. Against the 98-token controls, one example was replaced and 99 positions differ. Training/evaluation image budgets and the total training length limit changed together, so the old/new comparison does not isolate resolution alone.

QAT took about 1.96 elapsed hours with 9.55 GiB peak reserved per GPU; GRACE about 2.50 hours with 27.13 GiB. These are training measurements. The earlier evaluation loaded rounded/baked exports in BF16 and does not establish packed low-bit deployment speed. The later token-reduction experiment separately verifies packed INT4 execution.

The historical evaluation-only audit found missing MMBench completion markers caused by JSON integer/string histogram keys. Its report independently validated the saved scores, while retaining the missing-marker distinction. Those original prediction files are absent locally now, so this cleanup preserves the report and does not manufacture completion markers. [Detailed control report](released_bf16_init_32k_b128_p589824_t4096_seed42/experiment_report.md), [evaluation-only report](reproduction_eval_only/experiment_report.md).

## Completed token-reduction sweep

Resolution changes image/frame geometry before vision encoding. Uniform selects one representative from each contiguous token bin; merge averages the embeddings in those bins, including aligned DeepStack features. Uniform/merge encode the original visual input before compacting the decoder sequence. Neither removes temporal groups. Ratios denote retained tokens, with resolution rounding below the nominal target.

The sweep uses packed Triton W4A16 on 196 decoder linears, batch one, greedy EOS, a 256-token output cap, 589,824 maximum pixels and 2 fps full-duration video. Repaired video jobs stream vision by temporal group and chunk prefill at 512 tokens. Their common adaptive starting policy enforces a 131,072-token operational cap. Retention is relative to that adaptive baseline, which already lowers spatial detail for long clips.

| VideoMME configuration | Full accuracy % | Change vs baseline pp | Actual visual retention % | Recorded model hours |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 61.500 | 0.000 | 100.000 | 45.240 |
| Resolution r50 | 61.222 | −0.278 | 45.660 | 21.971 |
| Resolution r25 | 61.056 | −0.444 | 23.646 | 13.748 |
| Resolution r10 | 56.833 | −4.667 | 8.232 | 7.772 |
| Uniform r25 | 57.444 | −4.056 | 25.084 | Mixed/replayed timings; use serial projection |
| Merge r25 | 56.722 | −4.778 | 25.084 | Mixed/replayed timings; use serial projection |

All rows score the full 1,800 short/medium questions; long videos, audio and subtitles are excluded. The newest r25/r10 matched reruns supersede the original-policy 61.722/59.000 scores. Resolution r25 loses eight net correct answers versus baseline while reducing recorded model time substantially. These across-run timer ratios are observations, not controlled repeated speed benchmarks.

At r25/r10, resolution has the highest score of these three methods on every image suite. At r50, merge leads five of seven suites. TempCompass uniform-r25 scores 61.646 versus baseline 61.329 with about 25.08% retention; five net additional correct answers do not establish a reliable accuracy improvement. TempCompass uses a local MC fallback, not an API judge.

All three Video-MMMU splits finish at all ten settings, but their three resolution-r10 rows retain the original spatial/execution policy. Their scores remain visible, while deltas and speedups against adaptive baselines are withheld. Some Video-MMMU outputs hit the generation cap; per-row counts are in the merged JSON and [detailed report](qat_int4_token_reduction_576_seed42/REPORT.md). Vanilla/BF16 token-reduction matrices remain incomplete; this study does not establish an INT4-versus-BF16 gain.

## Timing evidence and its limits

The separate seed-42 timing sweep completes all ten configurations on one exclusive A100. It measures 50 distinct videos, balanced across short/medium and visual-token quintiles, with expansion weight 36, one excluded full-input warmup and 4,000 paired bootstrap draws. Full-population accuracy always comes from the full runs.

| Method at r25 | Projected model hours | Projected request hours | Full accuracy % |
| --- | ---: | ---: | ---: |
| Resolution | 15.82 | 32.09 | 61.056 |
| Uniform | 14.81 | 24.91 | 57.444 |
| Merge | 15.38 | 27.42 | 56.722 |

Model time covers synchronized model components. Request time adds measured preprocessing, staging and generation overhead. Startup, warmup, harness/scoring and durable writes are excluded. Uniform's projected model speedups are 1.86×/3.12×/4.97× at r50/r25/r10; merge's are 1.84×/3.01×/4.78×. Request speedups are smaller.

The fresh resolution-r25/r10 projections exceed the full rerun model totals by 15.06%/16.70%, even though backtesting the selected IDs within those full runs is within 1%. Most of the shift is in vision/embedding, not prefill. Card/load/transfer differences are plausible, but the artifacts do not identify a cause. Bootstrap intervals quantify sampling within an execution and do not guarantee runtime on another card or run. [Detailed analysis and plots](qat_int4_token_reduction_576_seed42/REPORT.md).

## Historical video and diagnostic reports

The earlier 256-pixel local video matrix had only five completed model/task pairs out of 50 planned: TempCompass for vanilla 2B/teacher/local SFT, and Video-MMMU perception/comprehension for vanilla 2B. OOMs and partial prefixes are coverage evidence, not full-population scores. None established a completed local QAT/GRACE video comparison. Later 720p lmms harness preparation/validation is a separate protocol and does not supersede the packed-INT4 sweep's completed results.

Memory-screen reports test synthetic long inputs and short diagnostic training; they are feasibility evidence rather than production resource measurements. Startup/runtime investigations, concurrency analyses and recovery notes explain implementation and scheduling repairs. The active runtime preserves those repairs; scheduler-specific scripts and obsolete report builders are archived. [Historical video report](released_bf16_init_32k_b128_p589824_t4096_seed42/experiment_report_video.md).

## Evidence and reproduction

[The command reference](README.md) covers all suites, controls, timing, environment setup and fresh rerun comparison. [Merged JSON](../reports/merged_results.json) contains 120 artifact-verified rows, ten timing configurations and 215 protocol-labelled report-only scores. [CSV](../reports/merged_results.csv) contains all score rows with an `evidence` column. The [catalog](../evaluation/protocols/result_catalog.json) pins explicit completion markers, including the two latest VideoMME replacements; it never selects by modification time.

Every source document used during consolidation is listed with its original path and SHA-256 in [the report-source index](../reports/sources/index.json). The index links all 39 original experiment reports/readmes/diagnostic notes: token-reduction documents are in its `reports/history/documents/`, and the remaining sources are in [the repository report archive](../reports/sources/). Superseded interpretations and pending scaffolds remain preserved. Consult these for detailed category tables and historical hypotheses. The consolidated findings above take precedence over superseded narratives; model artifacts and immutable source reports are unchanged.

Cleanup verification rehashes all relocated files, selected completion artifacts, pinned evaluator sources, telemetry totals, paired populations and timing geometry. It reproduces the current report's metrics without GPU inference. A fresh full inference rerun was not performed during cleanup; the documented commands preserve the selected protocols and compare any rerun against the frozen scores.
