# QAT W4 video comparison

## Video progress and token-reduction effects — 2026-11-09

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
