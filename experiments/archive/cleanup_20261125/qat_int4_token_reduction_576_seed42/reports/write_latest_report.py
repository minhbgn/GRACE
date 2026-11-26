"""Render the November 25 synthesis from the two audited analysis artifacts."""
import json
from pathlib import Path
from compile_completed_analysis import table, CONFIGS, IMAGES, VIDEOS

HERE = Path(__file__).resolve().parent
EXP = HERE.parent


def main():
    data = json.loads((HERE/'completed_analysis_20261125.json').read_text())
    timing = json.loads((HERE/'timing_analysis_20261125.json').read_text())
    by = {(j['suite'],j['config']):j for j in data['jobs']}
    tb = {e['config']:e for e in timing['rows']}
    blocks = {}
    for part in (HERE/'completed_tables_20261125.md').read_text().split('<!-- ')[1:]:
        name, text = part.split(' -->\n',1)
        blocks[name] = text.strip()
    lines = []
    def add(text):
        lines.append(text.strip()+'\n')
    def section(title, text):
        add('## '+title+'\n\n'+text)
    def ci(e, metric, speed=False):
        value=e[metric]
        if speed:
            lo,hi=value['speedup_ci95']
            return f"{value['speedup']:.2f}× [{lo:.2f}, {hi:.2f}]"
        lo,hi=value['full_seconds_ci95']
        return f"{value['full_seconds']/3600:.2f} [{lo/3600:.2f}, {hi/3600:.2f}]"

    add('''# What the completed token-reduction experiment tells us

Updated **2026-11-25 (Asia/Singapore)**. All **120/120 QAT W4 configurations** are complete: 70 image and 50 video jobs. This report replaces the November 16 interpretation, uses the completed Video-MMMU splits, and replaces the two original-policy VideoMME resolution rows with the November 23/24 matched reruns. The [November 16 report](experiment_report_20261116.md) remains available for the detailed implementation explanations and integration plan.

**VideoMME resolution-r25 is the strongest measured accuracy/compute tradeoff among these simple settings:** 61.056% versus 61.500% at r100 (−0.444 percentage points), using 23.646% of the visual tokens and 13.748 recorded model-hours versus 45.240. Resolution-r10 reaches 56.833% (−4.667 pp) at 8.232% actual retention and 7.772 model-hours. These are full 1,800-question results.

The separate 50-question timing sweep estimates **15.819 h at r25 and 9.070 h at r10**, respectively **15.06% and 16.70% above** those full-run model totals. Within the full reruns, the selected questions estimate total time within 1%; most of the discrepancy comes from slower vision/embedding in the separate timing sweep. Its sampling intervals therefore should not be read as guarantees for another run.

For uniform/merge, the completed timing sweep supplies a consistent single-GPU estimate absent from the old mixed-scheduling totals. Projected model speedups are **1.86× / 3.12× / 4.97× for uniform** and **1.84× / 3.01× / 4.78× for merge** at r50/r25/r10. Full-set accuracy losses are appreciably larger than resolution at r25/r10.

![Full-set VideoMME accuracy versus projected runtime](reports/videomme_tradeoff_20261125.png)

Horizontal error bars show sampling-only 95% intervals for runtime; accuracy comes from the full sets. Lines connect measured settings and do not estimate intermediate settings. [PDF](reports/videomme_tradeoff_20261125.pdf).''')

    section('1. Completion, source selection and scope', blocks['inventory']+'''

The matrix covers 29,420 image rows plus 4,280 video questions per configuration: **337,000 scored question inferences** across the ten settings, excluding smoke tests, timing repeats and superseded attempts. The 50-question timing sweep adds 500 measured requests and one excluded full-input warmup per configuration. VideoMME covers 900 short + 900 medium questions; long videos, audio and subtitles are excluded. Each Video-MMMU split covers 300 questions; TempCompass covers 1,580.

Source precedence is explicit, rather than selecting the newest directory by its timestamp:

- Images: `results/full`.
- VideoMME resolution-r25/r10: `results_video_matched_exclusive/full`.
- Other video rows: `results_video_retry_gpu0_scheduler_resume/full`, falling back to the original completed row when the repaired counterpart does not exist.
- Timing projections: `results_videomme_timing50_seed42_phased_20261123` only. The preparation drafts and unfinished older timing directories are excluded.

Three Video-MMMU resolution-r10 rows still use the original spatial/execution policy. Their scores are reported, but baseline deltas and speedup claims are withheld. Every VideoMME setting now has the same adaptive starting policy. The older VideoMME r25/r10 scores remain preserved in the original artifacts and are superseded in current comparisons.

This is a frozen-checkpoint, inference-only **CE-only LoRA-QAT W4** experiment; no reducer, scales or weights were trained in this sweep. Vanilla/BF16 matrices remain incomplete, so the results measure reduction within QAT W4 rather than an INT4-versus-BF16 advantage.''')

    section('2. What each method changes', '''Resolution resizes the whole image/frame before vision encoding, saving vision and decoder work. Uniform encodes the full input and selects the lower-middle token from each evenly sized contiguous sequence bin. Merge uses the same bins but averages their embeddings, including aligned DeepStack features. Both compact the visual sequence before decoder layer 0 and preserve selected original M-RoPE coordinates. They keep all temporal groups; none of these methods drops frames or uses question-dependent importance.

`r025` means keep 25%, not remove 25%. Spatial grid rounding makes actual resolution retention lower than its nominal budget. Uniform and merge have identical token counts. Method comparisons at a nominal budget are not exact equal-token comparisons.

The checkpoint uses packed Triton W4A16 for 196 decoder linears, group size 128; vision, embeddings and head remain floating point. Evaluation uses batch one, greedy EOS generation, seed 42, FlashAttention 2, a 256-output-token cap, and 589,824 maximum pixels per image/frame. Videos retain full duration at approximately 2 fps, with two frames per temporal group. The repaired video path streams vision one temporal group at a time, stages features on CPU, and chunks prefill at 512 tokens while retaining the full GPU KV cache.

All repaired settings derive reduction from one common adaptive baseline with a 131,072 decoder-token operational cap. The cap adapts 846/1,800 VideoMME questions (all medium) and 251/300 questions in each Video-MMMU split. TempCompass never activates it. VideoMME r100 already retains only about 50.14% of the original-policy visual-token total. These percentages describe the adaptive baseline, rather than native full spatial detail for every long clip.

Details: [method definitions](experiment_report_20261116.md#2-resolution-uniform-and-merge-in-plain-terms), [video execution](experiment_report_20261116.md#3-the-video-choices-that-are-easy-to-miss), and [checkpoint training report](../released_bf16_init_32k_b128_p589824_t4096_seed42/experiment_report.md).''')

    section('3. Image results: all seven suites', blocks['image_accuracy']+'''

Image results are unchanged. At r50, merge has the highest reduced score on five of seven suites; resolution wins HallusionBench and MMMU. At r25 and r10, resolution scores highest among the three methods on every image suite. These are descriptive single-checkpoint results, without a significance claim for small differences.

Normal-answer model-time ratios against the same suite's r100:

'''+blocks['image_speed_natural']+'''

Three-question, fixed-32-output-token profile ratios:

'''+blocks['image_speed_32']+'''

The different speedup ranges reflect different output workloads: normal image answers are usually two tokens, while the diagnostic forces 32. The three-question profile is not a representative latency distribution. Token reduction mostly saves input work, so its relative benefit shrinks when generating a longer answer.''')

    section('4. TempCompass: uniform-r25 remains a useful simple control', blocks['tempcompass_results']+'''

Uniform-r25 retains 25.082% of visual tokens, scores 61.646% versus 61.329% at r100 (+0.316 pp, five net additional correct answers), and has an observed normal-answer time ratio of 2.575×. Merge-r25 matches the baseline score. Resolution-r25 loses 1.139 pp, while merge-r10 loses 5.000 pp. All 1,580 questions per setting complete with no capped answers.

These small positive differences are not evidence that pruning reliably improves accuracy. Category effects also differ: uniform-r25 improves speed and order scores but lowers attribute-change accuracy; the full category table is in [the generated tables](reports/completed_tables_20261125.md). TempCompass is scored with the deterministic local MC fallback, not an API judge.''')

    section('5. VideoMME: the matched full-population results', '''All rows below now use matching temporal coverage and adaptive baseline geometry, verified per question. All answers have two generated tokens and no output-cap hits. Retention is relative to adaptive r100. The model-time ratios are ratios of recorded timers across dates/cards, not repeated-run controlled speed benchmarks. In particular, the six uniform/merge full totals contain a replayed mixture of shared-GPU and one-worker timings; use section 7 for their consistent timing estimates.

'''+blocks['videomme_results']+'''

Resolution-r25 loses only eight net correct answers against r100: 94 errors are fixed and 102 new errors appear. Relative to resolution-r50 it loses three correct answers (−0.167 pp), while recorded model time falls 21.971 → 13.748 h (37.4% lower). Peak PyTorch allocation is 6.091 GiB versus 16.170 at r100, a 62.3% reduction; this excludes reserved memory and other process/device usage.

At r25, resolution scores **3.611 pp above uniform** and **4.333 pp above merge**, while retaining fewer visual tokens (23.646% versus 25.084%). At r10 it scores **3.889 pp above uniform** and **8.222 pp above merge**, again with fewer tokens (8.232% versus 10.119%). Resolution-r10 sacrifices 4.667 pp against baseline and uses 4.181 GiB peak allocation. A strict accuracy requirement may still favor r25 or r50.

Short/medium accuracy and paired error swaps against r100:

'''+blocks['videomme_accuracy_groups']+'''

The r25 loss is concentrated in medium videos: short changes −0.111 pp and medium −0.778 pp. At r10 both degrade (−4.111 and −5.222 pp). Uniform/merge at r10 hurt short accuracy particularly strongly, so sensitivity cannot be inferred from duration alone.

### How the reruns change the old interpretation

'''+table(['Setting','Old original-policy accuracy %','Matched rerun accuracy %','Old mean visual','Matched mean visual','Matched delta vs r100 pp'],[
        ['resolution_r025','61.722','61.056','38,893.043','18,560.920','−0.444'],
        ['resolution_r010','59.000','56.833','13,557.088','6,461.872','−4.667']])+'''

The original r25 used more tokens than repaired r50 (38,893 versus 35,840 per question), despite its smaller nominal label. The rerun resolves that mismatch. It supports r25 as a strong tradeoff at a genuinely smaller budget; it does not support the old suggestion of an accuracy improvement over adaptive r100.''')

    section('6. Does the resolution subset predict the new full reruns?', '''The pinned seed-42 subset contains 50 distinct videos: 25 short and 25 medium, five videos from each original-visual-token quintile within each duration, with one randomly selected question per video. Each observation has expansion weight 36. The projected full model time is `36 × sum(50 measured model times)`; no extrapolation of accuracy is used.

Two checks answer different questions. **Same-run backtesting** takes the selected IDs from the full-run telemetry and expands those saved times. **Fresh projection error** compares the independently measured 50-question sweep against the full run. Both use the same selected questions and spatial/frame geometry.

'''+table(['Configuration','Recorded full model h','Fresh subset projection h [95% CI]','Fresh error vs full','Same-run subset projection h','Same-run error'],[
        [c,f"{tb[c]['full_recorded_model_hours']:.3f}",ci(tb[c],'model'),
         f"{tb[c]['fresh_projection_relative_error']:+.2%}",
         f"{tb[c]['historical_sampling_backtest']['projected_seconds']/3600:.3f}",
         f"{tb[c]['historical_sampling_backtest']['relative_error']:+.2%}"]
        for c in CONFIGS[:4]])+'''

For r25/r10, same-run sampling error is −0.07%/−0.63%, but fresh total error is +15.06%/+16.70%. Their full totals lie below the fresh 95% intervals. Both fail the older screening threshold of ≤10% overall error; medium errors also exceed its ≤15% duration threshold:

'''+table(['Setting','Fresh short error','Fresh medium error','Same-run short error','Same-run medium error'],[
        [c,f"{tb[c]['fresh_duration_relative_errors']['short']:+.2%}",
         f"{tb[c]['fresh_duration_relative_errors']['medium']:+.2%}",
         f"{tb[c]['historical_sampling_backtest']['by_duration']['short']['relative_error']:+.2%}",
         f"{tb[c]['historical_sampling_backtest']['by_duration']['medium']['relative_error']:+.2%}"]
        for c in ('resolution_r025','resolution_r010')])+'''

The component timers locate most of the shift in vision/embedding, rather than prefill. The following compares **the same 50 IDs** expanded by 36 in each run:

'''+table(['Setting','Full-run selected model h','Fresh selected model h','Vision/embedding h: full → fresh','Prefill h: full → fresh'],[
        [c,f"{tb[c]['component_comparison']['model_seconds']['full_run_subset_expanded_hours']:.3f}",
         f"{tb[c]['component_comparison']['model_seconds']['new_subset_expanded_hours']:.3f}",
         f"{tb[c]['component_comparison']['vision_and_embedding_seconds']['full_run_subset_expanded_hours']:.3f} → {tb[c]['component_comparison']['vision_and_embedding_seconds']['new_subset_expanded_hours']:.3f}",
         f"{tb[c]['component_comparison']['prefill_seconds']['full_run_subset_expanded_hours']:.3f} → {tb[c]['component_comparison']['prefill_seconds']['new_subset_expanded_hours']:.3f}"]
        for c in ('resolution_r025','resolution_r010')])+'''

Vision/embedding grows 56.6% at r25 and 40.9% at r10; prefill grows only 0.26% and 0.47%. Full reruns used GPU 2 and the timing sweep GPU 4. Host load, transfers, clocks and card differences are plausible contributors, but these artifacts do not establish a cause. Matching geometry and matching selected predictions rule out a changed workload or answer length for these two comparisons.

The inference is that this subset represents the resolution full-run workload well, while absolute runtime changes across executions. It is not validated to ±10% on the fresh r25/r10 runs. The old [projection check](results_videomme_timing50_seed42_phased_20261123/projection_check.md) predates these reruns and compares against the original-policy r25/r10; it remains a historical phase gate. This report's [current timing audit](reports/timing_analysis_20261125.json) is the updated comparison. No seed reselection or post-hoc correction factor has been applied.''')

    section('7. Uniform and merge extrapolation, including request time', '''All ten timing configurations finished on the same exclusive A100 (physical GPU 4), sequentially, with batch one and no answer replay. Saved GPU observations contain one process per configuration. The first full-input generation is used as warmup and excluded. Accuracy below comes from the full 1,800-question runs, never the small timing sample.

**Model time** sums synchronized vision/embedding, reduction, prefill and decode. **Request time** additionally includes media preprocessing, input staging and measured generation-call overhead. Startup, warmup, harness setup/scoring, durable checkpoint writes and unmeasured filesystem context-manager overhead are excluded. These are estimated steady-state serial totals, not observed full-run wall-clock durations.

Hours and speedups include paired within-stratum bootstrap 95% intervals (4,000 draws):

'''+table(['Configuration','Full accuracy %','Delta pp','Projected model h [95% CI]','Projected request h [95% CI]','Model speedup [95% CI]','Request speedup [95% CI]'],[
        [c,f"{by['videomme',c]['score_percent']:.3f}",f"{by['videomme',c]['delta_pp']:+.3f}",
         ci(tb[c],'model'),ci(tb[c],'request'),ci(tb[c],'model',True),ci(tb[c],'request',True)]
        for c in CONFIGS])+'''

Uniform-r25 projects **14.81 model-hours / 24.91 request-hours** at 57.444% accuracy. Merge-r25 projects **15.38 / 27.42 h** at 56.722%. At r10, uniform projects **9.31 / 21.52 h** at 52.944%, and merge **9.69 / 21.66 h** at 48.611%. In this sweep uniform has lower projected model and request time than merge at every budget, as well as higher full-set accuracy. Small timing gaps do not establish a repeatable runtime ranking.

For comparison, resolution-r25 projects **15.82 / 32.09 h** at 61.056%. Thus it preserves considerably more accuracy but is not the fastest projected request at that nominal budget in this execution. Its unexpectedly slower vision component, and the difference from its full-run timing, should be considered before making a method-level runtime claim. At r10, resolution projects **9.07 / 24.12 h** at 56.833%; uniform's request estimate is lower even though its model estimate is slightly higher. Request measurements capture costs that model timers omit.

Relative to the 56.69 h projected r100 request total, uniform request speedups are about **1.62× / 2.28× / 2.63×**, and merge about **1.56× / 2.07× / 2.62×**. The 3–5× model speedups at small budgets therefore should not be described as end-to-end speedups.

### Why these estimates differ from the historical uniform/merge totals

'''+table(['Configuration','Old recorded model h','Fresh projected model h','Fresh minus old','Historical same-run sampling error'],[
        [c,f"{tb[c]['full_recorded_model_hours']:.3f}",f"{tb[c]['model']['full_seconds']/3600:.3f}",
         f"{tb[c]['fresh_projection_relative_error']:+.2%}",
         f"{tb[c]['historical_sampling_backtest']['relative_error']:+.2%}"]
        for c in CONFIGS[4:]])+'''

The old full totals mix shared-GPU and one-worker timers. For example, uniform-r10 retains 1,032 shared-GPU records and only 768 one-worker records; merge-r10 retains 1,050 and 750. The fresh projected totals are 22–47% below the old totals, consistent with this scheduling difference. They should not be evaluated as prediction failures against a controlled standalone ground truth, because those old totals are not that ground truth.

At r10, even same-run expansion underestimates the mixed historical totals by 15.41% for uniform and 13.18% for merge, mostly in medium questions. Good resolution backtests do not prove unbiased uniform/merge sampling under a different scheduling history. The fixed subset avoids choosing a favorable seed, but only five videos per stratum remain a small sample.

Intervals quantify sampling variation conditional on this execution. They exclude run-to-run hardware/host variation and do not imply that full accuracy has a bootstrap interval here. In fact, none of the ten historical full-model totals falls within the corresponding fresh sampling-only interval. Nine configurations reproduce all 50 selected answer strings; merge-r10 changes one. Historical full-set accuracy remains authoritative. Request-level full-population telemetry was not saved for the reruns, so request-hour projections cannot be backtested against measured full request totals.''')

    section('8. Video-MMMU: all three splits are now complete', '''The following table covers all 30 split/configuration pairs, with 300 questions each. “Original” resolution-r10 rows use a different spatial/execution policy; their baseline ratios are intentionally unavailable. Other model-time ratios are descriptive across recorded runs/cards. Generated answer length varies, unlike VideoMME, so runtime differences also include changed output work.

'''+blocks['videommmu_results']+'''

Perception favors resolution-r50: 45.333% versus 44.000% at baseline (+1.333 pp). Stronger reduction degrades it: resolution-r25 is 38.333%, uniform-r25 34.333%, and merge-r25 33.000%. The unmatched original r10 cannot be placed on the same retention curve.

Comprehension is lower and relatively flat: baseline 23.667%, uniform-r25 23.667%, uniform-r10 22.667%, merge-r10 23.000%. Resolution-r50/r25 score 20.333%/21.000%. Keeping more visual input does not consistently improve this split for this checkpoint, but differences of a few questions at n=300 do not establish equivalence or a reliable improvement.

Adaptation baseline is 27.000%; uniform-r25 and merge-r25 both reach 29.667% (+2.667 pp, about eight additional correct answers). This result has a substantial generation-cap limitation: **31/300 capped answers at baseline, 43 at uniform-r25, 40 at merge-r25**, and 29–43 across the matched settings. Outputs at the cap stay in the score; neither dropping them nor calling these uncensored reasoning scores would be justified. Merge-r10 falls to 24.333% with 39 cap hits.

The 16 newly completed configurations since November 16 are seven comprehension rows and nine adaptation rows. The two matched VideoMME reruns replace previously completed but policy-mismatched results; they are not additional matrix cells. These completed splits strengthen the case for task-dependent reduction choices, rather than a single winning method.''')

    section('9. Token counts and complete accuracy-retention curves', blocks['baseline_tokens']+'''

Counts are per evaluated question, including repeated questions on the same video and text-only image rows. Visual embeddings exclude prompt text, timestamps and boundary markers. The source-frame rate and actual visual-token counts vary with aspect ratio, clip duration, adaptive cap and grid rounding; 576 is a spatial ceiling per image or two-frame temporal group, not a whole-video token budget.

'''+blocks['all_token_counts']+'''

![Measured accuracy changes versus actual retention for all twelve suites](reports/accuracy_retention_20261125.png)

Each panel has its own vertical scale. The three original-policy Video-MMMU resolution-r10 points are omitted from matched curves; both VideoMME reruns are included. [PDF](reports/accuracy_retention_20261125.pdf). Baseline throughput, source-time rates, category scores and other generated tables are available in [the complete table artifact](reports/completed_tables_20261125.md).''')

    section('10. Research decisions supported by these results', '''Use **VideoMME resolution-r25** as the main simple accuracy-preserving control for a stronger reducer: it removes about 76% of adaptive-baseline visual tokens for a 0.444 pp loss. Include resolution-r50 for a gentler budget and resolution-r10 for an aggressive budget. Match actual token counts when comparing a new selector, since equal nominal ratios do not give equal counts.

Keep **TempCompass uniform-r25** as a task-specific control and **image merge-r50 / resolution-r25** as useful image controls. Video-MMMU perception favors a milder resolution budget, while comprehension/adaptation have low absolute accuracy and adaptation has many capped outputs. These results do not support treating one reduction rule as universally best.

The uniform-versus-merge accuracy gap at aggressive VideoMME budgets supports testing a content-aware selector that preserves distinctive embeddings. It does not prove that a learned selector or DART will outperform resolution. The previous [LLMC+/joint-QAT integration plan](experiment_report_20261116.md#11-a-concrete-next-step-integration-plan) remains a proposed next experiment, separate from this inference-only sweep.

For runtime planning, use the completed single-GPU projected model/request totals with the execution-shift limitation established by r25/r10. Do not scale the uniform/merge estimates by a fitted resolution correction: the shift is concentrated in vision/embedding, whose workload differs by method. A same-card repeat would be needed to estimate repeatability or claim a stable absolute latency advantage. No additional GPU work is required for this report.''')

    section('11. Evidence and reproducibility', '''The audit verifies every selected completion's artifact SHA-256 values, successful exit, validation payload, ordered full telemetry coverage, positive packed-INT4 calls, all 28 decoder-layer visual counts, telemetry sums/means, fixed-output profile aggregates, 196 converted modules and integer-code verification. It verifies the original/recovery protocol chain, saved reuse markers and current pinned evaluation/extension source hashes. VideoMME/TempCompass correctness is independently counted from saved scorer records; Video-MMMU uses the saved validated local scorer results. This is an artifact/telemetry audit, not a regeneration of answers or an independent benchmark scorer implementation.

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

The plotting command uses the existing `fm` environment because it has matplotlib. The historical compilers keep their original source-selection and narrative assumptions and should not be used to overwrite this completed-matrix analysis.''')
    (EXP/'experiment_report_completed_video.md').write_text('\n'.join(lines))
    print(EXP/'experiment_report_completed_video.md')


if __name__ == '__main__':
    main()
