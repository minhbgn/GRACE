# QAT W4: packed INT4 and visual-token reduction

**Latest follow-up — 2026-11-25:** [What the completed token-reduction experiment tells us](experiment_report_completed_video.md). All **120/120 QAT configurations** are verified, including every Video-MMMU split and matched VideoMME resolution-r25/r10 reruns. The report compares subset runtime projections against those reruns and reports all uniform/merge model/request extrapolations. The [November 16 report](experiment_report_20261116.md) preserves the earlier interpretation and integration plan. The November 9 and November 5 snapshots below are historical.

<!-- CURRENT_VIDEO_PROGRESS_START -->
## Historical video progress and token-reduction effects — 2026-11-09

Snapshot at 13:11:18 Asia/Singapore. CPU-only artifact review; the evaluation continues unchanged.

**QAT W4: 87/120 complete: 70/70 image jobs and 17/50 video jobs.** The video total is 12 new recovery completions plus 5 reused original passes; 33 video configurations remain uncommitted. Vanilla 2B/released BF16 comparisons remain pending.

| Suite | Complete / planned | Remaining |
| --- | --- | --- |
| tempcompass | 10/10 | 0 |
| videomme | 4/10 | 6 |
| videommmu_perception | 1/10 | 9 |
| videommmu_comprehension | 1/10 | 9 |
| videommmu_adaptation | 1/10 | 9 |

**Current attempt:** `videomme/uniform_r050` — 296/1800 completed question records (16.4%). This is progress only; no partial benchmark score is reported.

The controller output lock is held by the running controller. Every reported completion passed artifact SHA-256, exit, validation and full-population telemetry checks. Completed QAT telemetry has ordered question indices and positive fused INT4 calls. Accuracy comparisons below also verify saved question identities and independently count per-question correctness.

### TempCompass: all naive reduction settings completed

All 1,580 questions completed for each configuration, with zero adapted clips and zero capped answers. These are matched-input-policy comparisons with deterministic local MC scoring, not API-judged official scores. The deltas are reduced minus baseline in percentage points.

| Configuration | Score % | Delta pp | Actual visual retained | Mean model s | Model speedup | 32-token speedup | Max allocated GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_r100 | 61.329 | +0.000 | 100.0% | 3.021 | 1.00× | 1.00× | 3.160 |
| resolution_r050 | 60.570 | -0.759 | 46.1% | 1.679 | 1.80× | 1.22× | 2.563 |
| resolution_r025 | 60.190 | -1.139 | 24.2% | 1.132 | 2.67× | 1.30× | 2.336 |
| resolution_r010 | 58.354 | -2.975 | 8.2% | 0.727 | 4.15× | 1.39× | 2.171 |
| uniform_r050 | 61.835 | +0.506 | 50.0% | 1.791 | 1.69× | 1.17× | 2.614 |
| uniform_r025 | 61.646 | +0.316 | 25.1% | 1.173 | 2.58× | 1.28× | 2.345 |
| uniform_r010 | 59.367 | -1.962 | 10.1% | 0.804 | 3.76× | 1.32× | 2.261 |
| merge_r050 | 61.582 | +0.253 | 50.0% | 1.788 | 1.69× | 1.18× | 2.614 |
| merge_r025 | 61.329 | +0.000 | 25.1% | 1.169 | 2.58× | 1.30× | 2.345 |
| merge_r010 | 56.329 | -5.000 | 10.1% | 0.807 | 3.74× | 1.34× | 2.261 |

**Observed TempCompass result:** uniform retention at 50% and 25% slightly exceeds baseline accuracy (+0.506 and +0.316 pp), while merge at 25% matches the baseline score. Uniform 25% retains about one quarter of the visual tokens and reduces mean full-population model time from 3.021 to 1.173 s/question (2.58×). Merge 25% is similar at 1.169 s/question (2.58×). At 10%, all methods lose accuracy; merge has the largest loss (−5.000 pp). Simple token reduction therefore has a useful 25–50% region on this suite, while aggressive merging has a substantial accuracy cost at 10%.

The small positive score differences do not establish a statistically reliable accuracy improvement. Uniform 25% fixes 52 baseline errors and introduces 47 new errors (net +5); merge 25% fixes and introduces 71 each. Equal aggregate scores do not mean identical answers.

### VideoMME: completed comparable runs

Population: 1,800 short+medium questions (900 each), excluding long videos; no subtitles/audio or API judge. The repaired baseline is already adaptive: 846/1,800 questions (47.0%) require a smaller spatial grid. Across all questions it retains 50.14% of the original uncapped policy’s total visual tokens. “100%” below means the full adaptive baseline, not the uncapped 576-token-per-group input.

| Configuration | Score % | Delta pp | Mean visual tokens | Mean model s | Model speedup | 32-token speedup | Model hours | Attempt hours | Max allocated GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_r100 | 61.500 | +0.000 | 78,494 | 90.48 | 1.00× | 1.00× | 45.24 | 55.40 | 16.17 |
| resolution_r050 | 61.222 | -0.278 | 35,840 | 43.94 | 2.06× | 1.98× | 21.97 | 30.49 | 9.15 |

**Observed VideoMME result:** 50% resolution gives 61.222% versus 61.500% (−0.278 pp, net five fewer correct answers). Achieved visual retention is 45.66% because grids are patch-aligned. Mean model time falls 90.48 → 43.94 s (2.06×), attempt time 55.40 → 30.49 h (1.82×), and peak allocation 16.17 → 9.15 GiB (43.4% lower). Both runs generate two tokens per answer and have zero cap hits, so the speed difference does not come from shorter answers.

| Video duration | Questions | Baseline % | Resolution 50% % | Delta pp |
| --- | --- | --- | --- | --- |
| short | 900 | 68.556 | 68.667 | +0.111 |
| medium | 900 | 54.444 | 53.778 | -0.667 |

The reduction fixes 69 baseline errors and introduces 74 new errors. The small aggregate change conceals 143 changed correctness outcomes; it is not proof of equivalent predictions.

### Why VideoMME takes days

| Baseline measured component | Hours | Share of model time |
| --- | --- | --- |
| Decoder prefill | 41.155 | 90.97% |
| Vision + embedding | 3.821 | 8.45% |
| Reduction + host staging | 0.205 | 0.45% |
| Autoregressive decode | 0.058 | 0.13% |

VideoMME’s baseline averages 78,494 visual plus 2,959 text tokens per question; its two-token answers cost little compared with prefilling that long context. Prefill consumes 41.15 of 45.24 measured model hours (91.0%). The 512-token prefill chunks and one-temporal-group vision chunks bound memory, but add repeated kernel/transfer and Python-loop work. All 1,800 questions are evaluated independently; repeated questions about the same clip do not share precomputed vision features or a prompt KV cache.

Model-only baseline p50/p95 latency is 73.05/146.31 s/question. The 55.40 h attempt time also includes about 10.16 h outside the recorded normal-answer model timers: media decoding/preprocessing, setup, profiling, I/O and other uninstrumented overhead. The residual cannot be attributed to any single component from these records. Attempt time is estimated from the attempt-directory timestamp to the exit.json modification time, and is affected by host scheduling/contention. Only one job runs at a time on GPU 1.

Uniform/merge still encode the full adaptive spatial input before reducing decoder tokens, so they retain the full vision/preprocessing cost. Resolution reduction can save work in both vision and the decoder. The first three questions used for 32-token profiles are a small, nonrepresentative sample; their speedups also use longer outputs than the two-token benchmark answers. Keep fixed-output profile and full-population normal-answer speedups separate. No stable remaining-time estimate is inferred from the active partial run.

### Reused original-policy results

| Suite | Configuration | Score % | Questions | Cap hits |
| --- | --- | --- | --- | --- |
| videomme | resolution_r025 | 61.722 | 1800 | 0 |
| videomme | resolution_r010 | 59.000 | 1800 | 0 |
| videommmu_perception | resolution_r010 | 38.000 | 300 | 1 |
| videommmu_comprehension | resolution_r010 | 21.667 | 300 | 4 |
| videommmu_adaptation | resolution_r010 | 27.667 | 300 | 40 |

These five passes are preserved and were not rerun. They use the original spatial/runtime policy, so no accuracy delta or timing speedup against the adaptive recovery baseline is assigned. In particular, VideoMME original resolution 25% at 61.722% cannot be read as an improvement over the repaired baseline under an identical input policy. Video-MMMU still lacks repaired full-population baselines.

### Assessment so far

Naive reduction can preserve most accuracy while materially reducing long-video prefill cost: TempCompass uniform/merge at 25% and VideoMME resolution at 50% are promising observed tradeoffs. This is not evidence that every suite tolerates the same retention. The completed image experiment already shows task-dependent losses, and TempCompass merge at 10% loses five points. VideoMME uniform/merge and repaired Video-MMMU results are still needed before ranking methods across video suites. At 25% retention, the image experiment favored resolution on all seven suites, whereas TempCompass favors uniform selection over resolution (61.646% versus 60.190%). Achieved token budgets differ slightly, as shown above; no single naive method is best everywhere. No INT4-versus-BF16 conclusion is available until matched baseline models are evaluated.

Machine evidence: `reports/video_progress.json` contains verified completion provenance, paired correctness counts and the active attempt snapshot; `reports/video_progress.csv` contains the completed metrics. This synthesis does not modify the running evaluation, its pinned protocol, or completed artifacts.

<!-- CURRENT_VIDEO_PROGRESS_END -->

## Historical report — 2026-11-05

Prepared 2026-11-05 (Asia/Singapore). Snapshot of `results/full`; all scores below are percentages.

**Recovery follow-up:** [GPU-1-only video recovery](VIDEO_RECOVERY_GPU1.md) is prepared for the 45 failed QAT video jobs. It reuses the five passed jobs, repairs the TempCompass mapping, bounds vision/prefill working memory, and applies the user-approved adaptive spatial cap to oversized clips with full duration/2 fps preserved. Its results and policy are separate from this original snapshot. No GPU recovery run has started.

**QAT W4 has 75/120 verified completions: 70/70 image jobs and 5/50 video jobs.** Every configuration was attempted, but an attempted job is not necessarily a successful evaluation. The remaining video jobs failed: 35 CUDA OOMs and 10 missing-YAML failures. Vanilla 2B and released BF16 comparisons remain pending; no claim about INT4 versus BF16 accuracy, memory saving or speedup is supported yet.

## Completion and evidence

| Model | Verified complete / planned | Incomplete attempts | Unstarted |
| --- | --- | --- | --- |
| qat_w4 | 75/120 | 45 | 0 |
| vanilla_2b | 8/120 | 1 | 111 |
| released_bf16 | 0/120 | 0 | 120 |

Vanilla 2B already has eight completed HallusionBench configurations. Its `merge_r025` attempt has `validated.json` but no `exit.json` or `COMPLETE.json`, so it is uncommitted and will restart; `merge_r010` is unstarted. Those partial baseline results are retained in the machine summary, but are not used for cross-model conclusions here.

The compiler independently verified every completed artifact SHA-256, successful exit and saved validation payload. For all completed QAT jobs it also checked raw telemetry coverage/order, positive fused INT4 calls on every question, all 28 decoder layers seeing the retained token count, timing/token/memory aggregates, three fixed-output profile samples, 196 packed modules and integer-code verification. All pinned evaluation source hashes match the saved protocol. This is an artifact/telemetry audit, not an independent rerun of generation or benchmark scoring.

Evidence: [full runner summary](results/full/comparison.md), [summary JSON](results/full/comparison.json), [completion audit](reports/completion_audit.json), [completed metrics CSV](reports/completed_metrics.csv), [frozen protocol](results/full/protocol.json), and per-job `COMPLETE.json` / attempt artifacts. Smoke results are excluded.

## Checkpoint and evaluation protocol

- Checkpoint: `../released_bf16_init_32k_b128_p589824_t4096_seed42/w4g128/runs/qat`; training completed step 256 (CE loss 0.596977; training time 6,779 s). W4, group size 128, merged LoRA rank 32/alpha 64 on 112 attention linears; 196 language linears are quantized.
- Runtime: strict packed Triton W4A16, FP16 activation tiles/scales with FP32 accumulation. Vision, embeddings and head remain floating point. This is weight-only INT4 with floating-point arithmetic inside the kernel. Both saved GPU numerical gates passed with maximum absolute error 0.0078125.
- QAT sidecar SHA-256: `3383915972b9685a2cc51d900684de09144475f6b13a2c34244ffb8bfc1e0379`. The provenance file describes the source safetensors as floating point; `runtime.json` and telemetry establish packed execution.
- Maximum image/frame area 589,824 pixels: at most 576 visual tokens per image or video two-frame temporal group, not per whole video. Video sampling is 2 fps without duration/frame truncation. Smaller images retain their actual baseline count.
- One shared 100% baseline, then resolution/uniform/merge at 50%, 25%, 10%. Resolution reduces input pixels; uniform/merge first encode the full visual input and then compact tokens before decoder layer 0. Merge is deterministic contiguous-bin averaging, not a learned token selector.
- FlashAttention 2; seed 42; greedy single-beam decoding; 256-token answer cap; batch size one. Fixed-output timing uses the first three questions, one warm-up and three timed repeats of exactly 32 output tokens.
- Saved environment: Python 3.11.0, torch 2.5.1+cu121, transformers 5.9.0, Triton 3.1.0, lmms-eval 0.7.1. Results were dispatched to physical GPUs 0 and 2; failure logs identify 39.39 GiB devices. Performance across different cards or concurrent workloads is not controlled by this report.

## Image accuracy

All seven image populations completed in all ten configurations: 29,420 rows per configuration, 294,200 scored question inferences in total. No completed image answer hit the output cap. HallusionBench includes text-only rows and uses row accuracy; ScienceQA also retains text-only rows. MMBench uses circular accuracy on its supplied rotations. The other suites use their saved deterministic local primary metric. No aggregate accuracy across unlike benchmarks is calculated.

| Suite | Rows | 100% | Res 50% | Res 25% | Res 10% | Uniform 50% | Uniform 25% | Uniform 10% | Merge 50% | Merge 25% | Merge 10% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hallusionbench | 1129 | 64.92 | 64.48 | 63.77 | 62.27 | 63.60 | 62.80 | 59.61 | 63.95 | 61.56 | 59.96 |
| mmbench | 4329 | 89.26 | 87.63 | 84.97 | 78.61 | 87.97 | 83.68 | 75.26 | 88.92 | 83.42 | 69.85 |
| scienceqa | 4241 | 83.49 | 82.86 | 82.10 | 79.56 | 82.36 | 81.00 | 78.68 | 83.31 | 81.68 | 79.04 |
| ai2d | 3088 | 81.22 | 79.76 | 78.56 | 75.06 | 79.79 | 76.13 | 72.18 | 80.76 | 77.62 | 72.93 |
| mmmu | 900 | 67.78 | 66.67 | 66.11 | 63.00 | 65.22 | 63.44 | 61.67 | 66.00 | 63.56 | 61.00 |
| seedbench_img | 14233 | 75.30 | 73.86 | 72.63 | 69.87 | 73.80 | 71.43 | 63.85 | 74.00 | 70.91 | 61.29 |
| mmstar | 1500 | 67.27 | 64.87 | 63.13 | 59.20 | 64.13 | 60.73 | 56.07 | 65.53 | 62.40 | 56.87 |

### Accuracy cost and resource change

The ranges below span the seven image suites, without averaging their scores. Loss is each suite’s 100% score minus its reduced score (percentage points). Memory saving compares the maximum per-question allocated GPU memory within each suite. Actual retention is the ratio of summed retained to summed original visual tokens, excluding text.

| Configuration | Accuracy loss pp | Actual retained | 32-token speedup | Peak allocation saving |
| --- | --- | --- | --- | --- |
| resolution_r050 | 0.44–2.40 | 45.1–47.6% | 1.005–1.088× | 0.44–5.01% |
| resolution_r025 | 1.15–4.30 | 22.5–23.7% | 1.010–1.130× | 0.58–6.45% |
| resolution_r010 | 2.66–10.65 | 8.3–9.1% | 1.012–1.161× | 0.68–7.03% |
| uniform_r050 | 1.13–3.13 | 50.0–50.1% | 1.004–1.073× | 0.34–1.73% |
| uniform_r025 | 2.13–6.53 | 25.0–25.2% | 1.010–1.119× | 0.35–1.73% |
| uniform_r010 | 4.81–14.00 | 10.1–10.3% | 1.006–1.145× | 0.35–1.73% |
| merge_r050 | 0.19–1.78 | 50.0–50.1% | 1.006–1.078× | 0.34–1.73% |
| merge_r025 | 1.82–5.84 | 25.0–25.2% | 1.011–1.121× | 0.35–1.73% |
| merge_r010 | 4.46–19.42 | 10.1–10.3% | 1.008–1.146× | 0.35–1.73% |

**Observed tradeoff:** at 50% retention, merge preserves accuracy best on five of seven image suites (loss 0.19–1.78 pp), while resolution is best on HallusionBench and MMMU. At 25% and 10%, resolution has the highest score among the three reduction methods on every image suite. At 10%, resolution loses 2.66–10.65 pp; merge loses as much as 19.42 pp on MMBench. These are descriptive outcomes from one checkpoint/seed; small differences are not significance tests.

For follow-up evaluation, 50% merge is a useful accuracy-preserving candidate, and 25% resolution is a useful stronger-reduction candidate. Neither is a universal optimum; select using per-suite accuracy requirements. Resolution retains about 45–48% for a 50% target and 8–9% for a 10% target because of patch alignment; method comparisons at the same nominal target therefore do not have exactly equal token budgets.

## Timing and GPU memory

| Suite | Mean visual tokens at 100% | Mean answer model time s | 32-output-token time s | Max allocated GiB |
| --- | --- | --- | --- | --- |
| hallusionbench | 336.7 | 0.4604 | 3.0763 | 2.152 |
| mmbench | 150.9 | 0.3491 | 2.8160 | 2.122 |
| scienceqa | 76.4 | 0.2728 | 2.7733 | 2.126 |
| ai2d | 272.8 | 0.4248 | 2.8067 | 2.157 |
| mmmu | 278.6 | 0.4875 | 2.7542 | 2.289 |
| seedbench_img | 546.1 | 0.6374 | 3.0746 | 2.160 |
| mmstar | 190.1 | 0.3812 | 2.8062 | 2.180 |

The observed image fixed-output gains are modest: about 1.004–1.161× over the same packed QAT model at full input. These do not establish a speedup over BF16. Removing 90% of visual tokens does not remove 90% of total work: text and autoregressive decoding remain, and uniform/merge still pay the full vision-encoder cost. Image peak allocation is about 2.12–2.29 GiB at full input; savings from token reduction are small compared with the token-count reductions. The report does not infer quantization memory savings without matched BF16 runs.

Model timing includes synchronized GPU vision/embedding, reduction, decoder prefill and decode. It excludes checkpoint load/packing, CPU media decoding/preprocessing, transfer and file I/O. Mean normal-answer time varies with output length; use fixed-output timing for the within-model comparisons. Only three fixed questions per suite were profiled, so these estimates are exploratory and have no representative latency distribution or uncertainty estimate. Peak allocated memory is PyTorch allocation, not total/reserved VRAM or whole-process memory.

## Video results and failures

Only complete full-population jobs are scored below. Partial predictions from failed jobs are excluded. All full-token video baselines failed, so neither video accuracy deltas versus full input nor baseline-relative speedups are available.

| Suite / population | Configuration | Questions | Score % | Actual retained | Mean model s | Max allocated GiB | Capped answers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| videomme | resolution_r025 | 1800 | 61.72 | 24.84% | 37.531 | 29.835 | 0 |
| videomme | resolution_r010 | 1800 | 59.00 | 8.66% | 14.469 | 13.915 | 0 |
| videommmu_perception | resolution_r010 | 300 | 38.00 | 8.67% | 26.388 | 25.283 | 1 |
| videommmu_comprehension | resolution_r010 | 300 | 21.67 | 8.67% | 26.929 | 25.270 | 4 |
| videommmu_adaptation | resolution_r010 | 300 | 27.67 | 8.67% | 31.452 | 25.291 | 40 |

VideoMME is the short+medium 1,800-question population, excluding long videos; Video-MMMU uses all 300 questions in each split. Video-MMMU adaptation has 40/300 capped answers (13.3%), comprehension 4/300 and perception 1/300; these outputs remain in the score and limit interpretation. There is no audio/subtitle input or API judge.

| Suite | Complete configurations | Failed configurations / reason |
| --- | --- | --- |
| tempcompass | None | 10 / missing YAML |
| videomme | resolution_r025, resolution_r010 | 8 / CUDA OOM |
| videommmu_perception | resolution_r010 | 9 / CUDA OOM |
| videommmu_comprehension | resolution_r010 | 9 / CUDA OOM |
| videommmu_adaptation | resolution_r010 | 9 / CUDA OOM |

The TempCompass failure is deterministic: the pinned runner requests `third_party/lmms-eval-v0.7.1/lmms_eval/tasks/tempcompass/multi-choice.yaml`, while the clean pinned harness contains `tempcompass_mc.yaml`. These jobs failed before answering any questions. A future repaired evaluation must also verify the MC task adapter/metric and local deterministic scorer. Editing the runner changes a pinned source hash, so a repaired protocol needs a separate output base; do not rewrite the current `protocol.json`.

OOMs include vision-encoder allocations before token selection. Uniform/merge therefore cannot prevent those full-input vision peaks. Most lower-resolution video configurations also exceeded the roughly 40 GiB card capacity. Freeing a busy GPU is necessary for launch but does not by itself resolve these recorded per-process OOMs. Retry unchanged on a sufficiently larger idle card, or implement a separately validated memory optimization in a new protocol/output. Do not silently reduce fps, cap duration, drop questions or change the population.

## Resume from the last committed job

From the workspace root, after the selected GPUs are available and ownership has been checked:

```bash
export LMMS_PYTHON=/home/air/miniconda3/envs/grace/bin/python
EXP=reproduction/GRACE/experiments/qat_int4_token_reduction_576_seed42

# CPU-only plan: continue the two baseline models on all image suites.
bash "$EXP/run_experiment.sh" plan --models vanilla_2b,released_bf16 \
  --suites hallusionbench,mmbench,scienceqa,ai2d,mmmu,seedbench_img,mmstar \
  --output "$EXP/results"

# Launch only once physical GPUs 0 and 2 are idle; use --gpu 2 if only one is available.
bash "$EXP/run_experiment.sh" run --gpus 0,2 --reserve-gpus \
  --models vanilla_2b,released_bf16 \
  --suites hallusionbench,mmbench,scienceqa,ai2d,mmmu,seedbench_img,mmstar \
  --output "$EXP/results"

# Refresh the entire matrix summary without using a GPU.
bash "$EXP/run_experiment.sh" summarize --output "$EXP/results"
```

This baseline-image continuation plans 140 jobs, skips eight verified completions and has 132 remaining at this snapshot. Completed jobs are verified by hashes and skipped. A job without `COMPLETE.json` restarts in a new attempt from its first question; there is no per-question continuation, even if predictions or `validated.json` survived. Old attempts remain intact. `--output` takes the parent results directory: the runner appends `/full`; do not pass `results/full`. Do not use `--smoke` to resume full results.

To resume the original entire matrix instead (including all failed QAT video jobs):

```bash
bash "$EXP/run_experiment.sh" run --gpus 0,2 --reserve-gpus \
  --models qat_w4,vanilla_2b,released_bf16 --output "$EXP/results"
```

At this snapshot the full continuation has 277 uncommitted jobs (45 QAT, 112 vanilla, 120 released BF16). It retries failed video jobs; it does not repair them. Selecting only `--models vanilla_2b,released_bf16` with no suite filter continues both baseline image and video matrices (232 uncommitted jobs). The controller reserves idle GPUs and runs one independent job per card; multiple GPUs do not pool their VRAM.

Keep the original evaluation Python/packages, source files, dataset assets, policy, checkpoints and output base. Subset selection and available GPU indices may change; source/policy/environment changes cause the existing protocol pin to reject reuse. Do not remove completion markers, edit pinned hashes, or launch two controllers against this output. No GPU jobs were launched while preparing this report.

## Reproduce this synthesis

The report compiler contains the interpretation for this November 5 snapshot and rejects a changed completion inventory. After more jobs complete, the runner’s `summarize` action remains usable; update the compiler’s narrative and snapshot check before rebuilding the experiment report.

```bash
export LMMS_PYTHON=/home/air/miniconda3/envs/grace/bin/python
EXP=reproduction/GRACE/experiments/qat_int4_token_reduction_576_seed42
bash "$EXP/run_experiment.sh" summarize
python3 "$EXP/compile_report.py" --date 2026-11-05
```

## Evaluation infrastructure follow-up — 2026-11-09

The QAT video recovery launcher now supports `--parallel-evaluation N` (default
1), memory admission/OOM backoff, answer checkpoints for new attempts, and adding
GPUs through a live pool file. Completed original evaluations remain valid.
Legacy in-flight attempts cannot resume their partial telemetry as answers; let
the current pair finish before switching controllers. The saved 16.17 GiB peak
belongs to the **baseline**; completed resolution 50% peaked at 9.15 GiB, while
uniform 50% had observed 6.09 GiB over its first 351 questions at inspection.
Shared-GPU timing is not standalone latency. CPU scheduler/reservation tests,
preflight, and an interrupted-answer replay through the real VideoMME harness
passed; GPU concurrency/throughput has not been tested. Implementation and
commands: [parallel evaluation guide](../../evaluation/docs/README_parallel_video_evaluation.md).
