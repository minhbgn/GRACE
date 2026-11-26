**The current evaluator is suitable for comparing methods on a controlled runtime subset. The Uniform/resolution throughput ratio is not invariant across input lengths, and the available data do not establish that the new wrapper causes the larger medium-video gap.**

This addresses the within-suite ratio, rather than comparing raw throughput between benchmarks. For the same paired questions, define R = throughput(Uniform) / throughput(resolution) = sum(time_resolution) / sum(time_uniform). R below one means Uniform is slower. Use ratios of total time over identical populations, rather than averaging individual-question ratios.

| Nominal retention | TempCompass, old serial recovery | Medium VideoMME, fresh single-worker answers |
| --- | ---: | ---: |
| 50% | **0.937** | **0.864** |
| 25% | **0.965** | No comparable repaired resolution run |
| 10% | **0.904** | No comparable repaired resolution run |

These are **model throughput ratios**, excluding process setup/teardown and also excluding CPU video preprocessing. TempCompass includes all 1,580 questions. The VideoMME comparison uses 682 matched medium-video questions, indices 926–1607 inclusive, from the fresh Uniform50 segment. Question IDs, frame indices and shared adaptive-baseline geometry/text counts were verified. The methods were measured on different dates, so host contention is not experimentally controlled. The reused original VideoMME Resolution25/10 passes have a different input policy and cannot fill the empty cells.

At 50%, Uniform's throughput is therefore 6.3% lower than resolution's on TempCompass and 13.6% lower on this medium-video segment. Its extra mean model time is 0.112 seconds/question on TempCompass and 10.646 seconds/question on VideoMME. For VideoMME, about 5.185 seconds is additional vision/embedding time and 5.264 seconds additional prefill time; reduction/staging adds about 0.182 seconds.

There is no expectation that R must stay constant with unchanged code. Resolution reduces spatial inputs before vision; Uniform retains the shared baseline vision input and reduces embeddings afterward. Longer clips change the balance among per-group vision calls/transfers, decoder prefill, and text/timestamp tokens. Actual retained tokens also differ because of spatial-grid rounding. These effects, plus uncontrolled historical host load, can change R. The measurements support a different observed ratio, not a causal attribution of that difference solely to clip length or the wrapper.

All 32 pinned source hashes were rechecked and match. The original inference/memory/preprocessing path remains unchanged. The new checkpoint/request wrapper operates outside `model_seconds`; its separately measured incremental persistence cost was approximately 10 ms/answer with a representative medium payload. It is not a process-startup constant, and serialization volume can grow with input metadata, but it is small relative to the observed medium-video inference times. The wrapper cannot explain the table through direct inclusion in its timers, though indirect resource effects remain possible.

For runtime measurement, use the existing streamed/chunked inference path with these controls:

1. Freeze explicit question/clip IDs for every method and retention. Select across duration/sampled-frame count, baseline token count, aspect ratio and adaptive-cap activation. Report short/medium or size-stratum ratios; weight strata to represent the intended population. A subset estimates that defined population, not a universal speed ratio.
2. Use one evaluation at a time for the cleanest host/GPU comparison; fix CPU thread settings, attention, chunk sizes and generation settings. Alternate method order across repeats. Exclude model loading, packing/JIT warmup, initialization and final scoring/hashing from the measured interval.
3. Report both existing `model_seconds` and a per-question input-to-answer timer. The latter should surround video input preparation through answer generation, before checkpoint writes. Input-dependent decoding/resizing, transfers and reduction should remain in the end-to-end measurement. Label checkpoint cost separately.
4. Recompute every timed sample. Use a fresh benchmark namespace or an explicit no-replay path; restored telemetry is historical evidence, not a new measurement. Warm up outside the timed samples and report output length; use the same fixed generation length when measuring controlled decoding rather than natural answer latency.
5. The current `--smoke N` selects a prefix, not a representative stratified subset. The fixed-output profiler also covers only the first three questions. A runtime subset needs an explicit-ID manifest and a separate benchmark entrypoint/subset option, without editing sources pinned by the active run. A practical starting pilot is 24–48 distinct clips per suite, followed by more clips/repeats if stratum ratios remain unstable; this is a pilot size, not a precision guarantee.

Constant setup/teardown can be excluded as proposed. Input-dependent method costs should remain visible. This yields a fair relative-runtime comparison **for this evaluator and its declared streaming/chunking/input policy**; it does not establish equivalence to a different unchunked runtime.

The [paired calculations](within_suite_runtime_ratios_20261111.json) contain the exact snapshot, component means, source hashes and limitations. No GPU benchmark was launched and no active evaluation source was changed for this comparison.
