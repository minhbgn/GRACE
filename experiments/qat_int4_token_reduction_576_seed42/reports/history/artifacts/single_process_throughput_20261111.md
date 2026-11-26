**The single-worker execution uses the same inference implementation and settings as the pre-parallel serial recovery runs. Its measured decoder throughput is consistent with that implementation. End-to-end runtime is not literally identical: the new answer-checkpoint wrapper adds measurable I/O, and two workers still share the host.** Comparing different benchmarks cannot establish strict latency equality on identical inputs.

Measured on **2026-11-11, 14:23:49–14:33:49 Asia/Singapore**, without changing the workers or launching GPU inference. Controller PID 1410209 uses `--gpus 0,1 --parallel-evaluation 1 --evaluation-memory-gib 36`. Host GPU process inspection confirmed just one evaluation worker on each of those GPUs: Uniform50 PID 1412576 on GPU0 and Uniform10 PID 3569420 on GPU1. There are two evaluation processes in total.

**Direct live throughput**

The observer read only new complete answer-checkpoint lines, polling once/second. Every pre-existing/replayed answer was excluded. Both workers were already in the medium-video portion. Boundary questions can begin before or end after the fixed interval; the second rate below uses only complete answer-to-answer cycles and avoids counting the initial partial question.

| Configuration | New answers in 600 seconds | Fixed-window throughput | Complete answer cycles | Cycle throughput | Mean cycle seconds/question |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform50 | 5, saved count 1592 → 1597 | **30.0 questions/hour** | 4 | **33.8 questions/hour** | **106.4** |
| Uniform10 | 7, saved count 1158 → 1165 | **42.0 questions/hour** | 6 | **43.4 questions/hour** | **82.9** |

Individual observed cycles ranged from 96–113 seconds for Uniform50 and 54–128 seconds for Uniform10. This is a small, varying-workload window, not a confidence interval or a stable completion forecast. Together the workers completed 12 new questions in ten minutes, or 72 questions/hour during this interval.

For the complete interarrival cycles, model timers averaged 78.0 seconds/question for Uniform50 and 42.5 seconds/question for Uniform10. The corresponding wall-minus-model residuals are approximately **28.4 and 40.5 seconds/question**. These residuals combine decoding/CPU input preparation, request handling, checkpointing and other work outside the model timer; they do not measure preprocessing alone.

**Comparison with TempCompass counterparts**

The TempCompass reference runs finished under the old serial recovery code on November 5. Each completed all 1,580 questions. For the medium VideoMME comparison, only answers computed after the current serial resume are included: the first 926/984/1032 restored answers for Uniform50/25/10 are excluded. This broader sample is less sensitive than a ten-minute window to any particular clip.

| Configuration | TempCompass historical wall questions/hour | Fresh medium VideoMME attempt-average questions/hour | Fresh medium questions |
| --- | ---: | ---: | ---: |
| Uniform50 | 1588.7 | 28.1 | 672 |
| Uniform25, completed | 2210.5 | 38.2 | 816 |
| Uniform10 | 2845.6 | 52.9 | 133 |

The broader VideoMME averages differ from the live window because the clip mix changes. Historical wall duration is reconstructed from the attempt-name start timestamp and exit.json modification time. The fresh VideoMME attempt averages include setup/replay time in the denominator but exclude restored answers from the numerator. TempCompass includes setup and initial fixed-output profiling too. These averages are useful context, not a controlled overhead comparison.

VideoMME is much larger per question: the fresh segments average **1,025–1,065 sampled frames**, compared with **24.4** for TempCompass, approximately **42–44 times as many**. Uniform50 averages about **57,704 visual + 5,086 text/timestamp tokens** versus **1,358 + 153**. Mean model time is 78.04 versus 1.79 seconds. All compared answers contain two generated tokens. The much lower questions/hour therefore does not imply a change in inference implementation.

A more informative consistency check divides total decoder input tokens (retained visual plus text) by total prefill seconds:

| Configuration | Old serial TempCompass input tokens/prefill second | Fresh single-worker medium VideoMME | Difference |
| --- | ---: | ---: | ---: |
| Uniform50 | **1079.8** | **1032.3** | −4.4% |
| Uniform25 | **1056.5** | **1068.6** | +1.1% |
| Uniform10 | **1009.3** | **1086.3** | +7.6% |

These rates are close despite the roughly fortyfold larger inputs. They are consistent with similar decoder execution efficiency, rather than a large new decoder regression. They are **prefill rates**, not generated-token throughput, end-to-end throughput, or isolated kernel benchmarks. Attention lengths, spatial grids, host load and hardware scheduling differ; linear token normalization cannot prove strict performance equivalence.

**Execution-path and hidden-change check**

The audit rehashed all pinned sources: **25 original sources, three serial recovery sources/tests, and four parallel extension sources all match their recorded hashes**. The original matrix worker, token-reduction runtime, video preprocessing, INT4 kernel, and streamed-memory runtime are unchanged relative to the manifests used by the TempCompass results.

Comparing the actual old TempCompass Uniform50 job payload with the current VideoMME Uniform50 payload yields exactly six differing top-level fields: `suite`, `gpu`, `controller_pid`, and the added `extension_code`, `resume_provenance`, `scheduling`. The entire policy and execution dictionaries, method/retention, model path, QAT sidecar hash, packed-cache path and smoke limit are equal. In particular:

- Batch size remains one, with the same strict packed INT4 W4A16 path and FlashAttention2.
- The seed, greedy generation, output reserve and scoring adapter are inherited unchanged.
- Video sampling remains 2 fps over the full duration.
- Streamed vision still processes one temporal group per chunk; CPU feature staging and 512-token decoder prefill remain the same.
- The adaptive shared-baseline cap remains 131,072 tokens. The scheduler's 36 GiB estimate does not set this cap or alter tensor shapes.
- The reduction operation still occurs in the same `MemoryMeasuredGenerator.once`. The new `Generator` subclass only stores a reference during construction; it does not override `once` or generation.

The old serial worker called `retry.install_worker(execution)` followed by `matrix.worker(job_path)`. The current worker still calls those functions, with an outer wrapper around harness `generate_until`. This wrapper checks the request identity, replays any saved prefix, and saves each fresh answer afterward. After the prefix, it calls the original adapter, which performs the same video preparation and model generation. The added parent-watcher thread and scheduler polling are outside that inference implementation.

**Known new overhead, measured separately**

A CPU-only microbenchmark exercised the real `checkpoint_adapter` and real filesystem writes, comparing dummy generation with and without the wrapper. It used five alternating-order trials of 100 new answers on the **same reports filesystem**, with a real medium-video telemetry/sampling payload of about 7.4 KiB and a real source-row identity. Temporary benchmark files were removed automatically afterward.

- Plain telemetry/sampling writing: approximately **0.32 ms/question**.
- Wrapped path including identity/checksum work, tail reads and durable answer `fsync`: approximately **10.56 ms/question**.
- Incremental wrapper cost: **10.24 ms/question**, around **0.01%** of the observed 83–106-second answer cycles.

This is a local microbenchmark, not a profiler injected into the real workers. Actual filesystem stalls can vary, and the benchmark does not include scheduler interference or startup replay. Nevertheless, the measured new per-answer persistence cost is far too small to explain tens of seconds of residual wall time. Startup checkpoint scanning/replay, GPU validation gates and completion hashing have separate one-time costs.

No explicit CPU thread cap or affinity setting was added by the extension. Running one worker per GPU removes sibling GPU contention, but two workers can still contend for CPU and memory bandwidth. The old TempCompass run's complete host-load/thread state was not recorded, so identical historical host conditions cannot be verified.

**Conclusion supported by the measurements**

The inference implementation and configuration are unchanged relative to the serial recovery version that produced TempCompass. Normalized prefill throughput shows no large unexplained regression. The observed raw questions/hour difference is primarily consistent with the much larger medium-video workload and substantial host-side work. There is a small, explicit new checkpoint overhead, so the whole execution path is not byte-for-byte identical. Strict equality of elapsed runtime would require running the same selected medium questions through both entrypoints under controlled resource conditions; a cross-suite comparison alone cannot establish it.

Evidence: [live observations](live_throughput_20261111.json), [combined calculations and source/job checks](single_process_throughput_20261111.json), [checkpoint microbenchmark](checkpoint_overhead_20261111.json), [read-only live observer](measure_live_throughput_20261111.py), and [comparison script](compare_single_process_throughput_20261111.py).
