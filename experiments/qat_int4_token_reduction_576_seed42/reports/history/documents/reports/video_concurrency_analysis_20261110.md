# Six-worker VideoMME speed and completion estimate

Snapshot: 2026-11-10 14:18:37 Asia/Singapore.

Six currently active VideoMME configurations only; excludes 27 pending Video-MMMU configurations.

| Configuration | GPU | Saved / 1800 | Lifetime q/h | Mean model s/q | Medium model s/q | Naive remaining h | Workload-adjusted remaining h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| merge_r010 | 1 | 1046 | 44.0 | 37.5 | 115.9 | 17.2 | 46–62 |
| merge_r025 | 0 | 989 | 41.6 | 43.9 | 135.3 | 19.5 | 57–79 |
| merge_r050 | 1 | 924 | 38.8 | 53.6 | 160.4 | 22.6 | 69–99 |
| uniform_r010 | 0 | 1029 | 43.2 | 38.6 | 128.0 | 17.8 | 51–69 |
| uniform_r025 | 1 | 980 | 41.2 | 44.4 | 137.3 | 19.9 | 59–82 |
| uniform_r050 | 0 | 923 | 38.8 | 53.5 | 174.6 | 22.6 | 72–103 |

## Timing evidence

Single-worker resolution50: 30.49 wall hours, 43.94 model seconds/question, 59.04 questions/hour over all 1800 questions.

| Component, same first 443 Uniform50 questions | Single worker seconds | Three workers/GPU seconds | Slowdown |
| --- | --- | --- | --- |
| model_seconds | 23.553 | 52.621 | 2.23x |
| vision_and_embedding_seconds | 2.444 | 15.942 | 6.52x |
| prefill_seconds | 20.838 | 36.234 | 1.74x |

## Method and limits

- Lifetime wall throughput includes setup, profiling, preprocessing, and checkpoint writes.
- Checkpoint rows contain no per-answer timestamps. Recent measured window uses this session’s earlier 13:58:59 SGT snapshot; few completions make this window noisy.
- Rows 0..899 are short videos and rows 900..1799 medium, verified against standalone sample docs.
- Parallel model remainder: scale each remaining baseline component by current-medium component total / matched baseline-medium component total.
- Outside-model remainder: elapsed minus model timers and estimated profiling; extrapolate by either total pixels or sampled frames. These two scenarios are a sensitivity analysis, not a confidence interval.
- Completion scenarios assume broadly similar contention and no interruptions; future queued workers and changing load can alter them.
- Exploratory solo times interpolate/extrapolate decoder prefill between standalone baseline and resolution50 by decoder token count, use full-resolution baseline vision and baseline residual weighted by frames/pixels. These are not measured solo Uniform/Merge medium runtimes.
- Paired Uniform50 comparison matches the exact first 443 video sampling records. Legacy run was interrupted and has no COMPLETE marker; its telemetry is used only for timing evidence.

Exploratory serial queue remaining hours per GPU: {"0": [52.953823135866756, 56.927015184568155], "1": [52.84948674710364, 56.80058781251138]}. This estimate requires a matched-medium A/B check before treating it as a speed guarantee.

## Host observations and recommendation

128 logical CPUs; observed load average 208.38. Each worker had 362–375 total threads. GPU 0 utilization averaged 66.25% and GPU 1 averaged 10% across eight samples over about 40 seconds; this short window is not representative of an entire run. GPUs 0 and 1 contained only the six evaluation workers. The video reader already uses one decoding thread, while frame resize and preprocessing use PyTorch CPU operations. High total thread counts alone do not establish oversubscription.

Prefer a controlled one-worker-per-GPU trial on the remaining medium-video phase. Evidence points to CPU/vision contention, but matched-medium A/B measurement is needed to establish aggregate throughput advantage. No processes were changed.
