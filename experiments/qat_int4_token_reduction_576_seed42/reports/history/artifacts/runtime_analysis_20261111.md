**Uniform reduction has a real runtime disadvantage relative to resolution reduction, and the earlier three-workers-per-GPU run added a separate, larger contention penalty.** The evidence does not indicate that multiprocessing changed the reduction algorithm or INT4 inference path. Current workers are advancing with one evaluation per GPU.

Evidence snapshot: **2026-11-11 14:10 Asia/Singapore**. GPU sampling was at 14:08. This audit added reports only; it did not restart jobs or alter evaluation code, policies, or existing results.

**Process status**

Controller PID **1410209** is running `run_parallel_video_evaluation.py run --gpus 0,1 --parallel-evaluation 1 --evaluation-memory-gib 36`. The concurrency option is a maximum **per GPU**, so this means two simultaneous evaluations in total.

| Evaluation | Physical GPU | Worker PID | Durable answers | State |
| --- | ---: | ---: | ---: | --- |
| VideoMME uniform_r050 | 0 | 1412576 | 1585 / 1800 | Active; last answer 14:07:52 SGT |
| VideoMME uniform_r025 | 1 previously | — | 1800 / 1800 | Complete; score 57.444%; last answer 12:03:36 SGT |
| VideoMME uniform_r010 | 1 | 3569420 | 1146 / 1800 | Active; last answer 14:08:12 SGT |
| VideoMME merge_r050 | — | — | 927 / 1800 | Saved prefix, waiting for a worker slot |
| VideoMME merge_r025 | — | — | 992 / 1800 | Saved prefix, waiting for a worker slot |
| VideoMME merge_r010 | — | — | 1050 / 1800 | Saved prefix, waiting for a worker slot |

All ten TempCompass configurations are complete. Across the 50 video configuration/suite pairs, 18 are complete: 13 repaired completions plus five reused original passes. Two are active and 30 await dispatch, including 27 Video-MMMU configurations. The generic `failed/incomplete` status in comparison.md does not distinguish an active attempt from an interrupted attempt waiting to resume.

GPU ownership was resolved from `cwd:/raid/lvchengtao/Minh/Distillation`: GPUs 0 and 1 each contained just the corresponding evaluation worker. Other jobs occupied GPUs 2 and 4. In six samples over 10.6 seconds, GPUs 0 and 1 were both at 0% utilization, while their workers consumed approximately 46.3 and 48.2 CPU cores respectively. They had 374 and 369 threads and access to all 128 logical CPUs. The controller consumed under 1% of one CPU core. This establishes active CPU work during that window, consistent with video preprocessing/host operations; it does not identify a particular Python call or establish full-run average GPU utilization. Thread count alone is not evidence of oversubscription. No `watch_eval_health.py` process was visible in the host process snapshot.

The IDE's `uniform_r025/attempt_20261109T061933_501523Z/run.log` is historical. It contains one saved answer. The completed attempt is `uniform_r025/attempt_20261110T064111_150483Z/`; the currently active Uniform10 attempt began on November 11.

**Why the methods differ**

The relevant paths are [adaptive_video_geometry](../../../evaluation/protocols/video_memory_runtime.py), [build_inputs](../../../evaluation/protocols/video_protocol.py), and [prepare_features/compact](../../../evaluation/protocols/token_reduction_runtime.py).

1. **Resolution reduction acts before preprocessing and vision inference.** It lowers each frame's spatial grid, then builds the video tensor at that size. The vision encoder, embeddings, host transfers, and decoder consequently process fewer spatial tokens. Video frame sampling and duration remain the same.
2. **Uniform reduction acts after vision inference.** It builds inputs and runs vision at the shared adaptive baseline resolution, then retains representative visual embeddings before decoder layer 0. It reduces decoder work and KV length, but retains the baseline vision and larger host-input work. `group_plan` chooses representative positions within contiguous spatial bins; it does not skip video frames. Merge follows the same full-vision path, then averages bin embeddings and aligned DeepStack features.
3. **Equal nominal retention does not mean equal token count.** Resolution uses an aspect-preserving patch-aligned grid that can undershoot the target. Uniform rounds the retained count up within each visual group. This difference materially favors resolution's runtime at the same nominal fraction.
4. **Wall time includes substantial work outside the model timer.** Video reading, CPU frame resize/stacking/processor work, request handling, and answer checkpoint writes occur outside `MemoryMeasuredGenerator.once`. That timer includes streamed vision, feature compaction/host staging, decoder prefill, decoding, and waits within those operations. Its components are elapsed times, not isolated GPU kernel times.

**Completed TempCompass evidence, before multiprocessing**

All ten runs finished on November 5, under the serial recovery controller. Their jobs have no parallel scheduling metadata or checkpoint wrapper. They share the same 1,580-question identity, use the same streamed vision / 512-token prefill runtime, and report **zero adaptive spatial-cap interventions**. Every question generated two tokens, so answer length is controlled in these comparisons.

| Configuration | Actual visual retention | Mean visual tokens | Model s/question | Vision + embeddings s/question | Prefill s/question | Approx. suite wall minutes | Score % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_r100 | 100.00% | 2716.35 | 3.0208 | 0.3085 | 2.6248 | 92.01 | 61.329 |
| resolution_r050 | 46.10% | 1252.28 | 1.6786 | 0.2894 | 1.3046 | 56.37 | 60.570 |
| uniform_r050 | 50.00% | 1358.18 | 1.7909 | 0.3036 | 1.3996 | 59.67 | 61.835 |
| merge_r050 | 50.00% | 1358.18 | 1.7878 | 0.2990 | 1.3996 | 59.25 | 61.582 |
| resolution_r025 | 24.20% | 657.39 | 1.1322 | 0.2781 | 0.7705 | 41.28 | 60.190 |
| uniform_r025 | 25.08% | 681.33 | 1.1731 | 0.2976 | 0.7898 | 42.89 | 61.646 |
| merge_r025 | 25.08% | 681.33 | 1.1695 | 0.2927 | 0.7896 | 42.77 | 61.329 |
| resolution_r010 | 8.20% | 222.73 | 0.7274 | 0.2699 | 0.3748 | 30.23 | 58.354 |
| uniform_r010 | 10.06% | 273.16 | 0.8044 | 0.2976 | 0.4223 | 33.31 | 59.367 |
| merge_r010 | 10.06% | 273.16 | 0.8069 | 0.2983 | 0.4222 | 33.16 | 56.329 |

Uniform's model latency is **6.69%, 3.62%, and 10.58% higher** than resolution's at nominal 50%, 25%, and 10%. Its token counts are respectively **8.46%, 3.64%, and 22.64% higher**. At nominal 50%, the 0.1123-second gap consists mainly of 0.0949 seconds more prefill and 0.0142 seconds more vision/embedding time. The additional reduction/staging time is only 0.0032 seconds. These observations establish a disadvantage that predates parallel execution; they do not isolate a pure algorithm effect at exactly equal actual token counts.

Uniform and Merge have almost identical model times at each retention, despite different selection/averaging operations. This further argues against the selection operation itself being the main bottleneck. Both remain faster than the unreduced baseline. Uniform's TempCompass accuracy is also 1.01–1.46 percentage points higher than resolution at the nominal matched fractions, so the measurements reflect a speed/accuracy tradeoff.

**The additional multiprocessing penalty**

The strongest comparison uses **the exact same first 443 Uniform50 VideoMME questions**, first under the legacy serial recovery worker and then under three workers per GPU. The audit verified identical complete video sampling records, input/output token counts, adaptation metadata, and generated token IDs.

| Mean elapsed component, seconds/question | Serial recovery | Three workers/GPU | Ratio |
| --- | ---: | ---: | ---: |
| Model total | 23.5533 | 52.6212 | 2.23× |
| Vision + embeddings | 2.4438 | 15.9416 | 6.52× |
| Prefill | 20.8379 | 36.2339 | 1.74× |
| Reduction + feature staging | 0.1810 | 0.2139 | 1.18× |
| Decode | 0.0895 | 0.1782 | 1.99× |

This is a same-method slowdown, so changing from resolution to uniform cannot explain it. Resource sharing across three full model/CPU preprocessing processes per GPU is the leading explanation. The timing evidence cannot apportion GPU scheduling, CPU bandwidth/thread contention, and host transfers precisely. The serial attempt was interrupted and is used only as timing evidence, not as a completed score.

These figures establish a per-evaluation latency penalty, **not** a definitive aggregate throughput comparison between concurrency settings. A controlled matched-workload trial would be needed to determine the best total completed-questions/hour setting.

The controller log records six simultaneous workers beginning November 9 at about 14:30 SGT, then one worker per GPU after the November 10 restart at about 14:41 SGT. The second restart changed the admission budget from 12 to 36 GiB. That budget is a scheduler estimate, not a tensor allocation, decoder token cap, or model setting.

**Fresh single-worker VideoMME evidence**

Exclude restored prefixes before examining current timings. The following use zero-based question ranges and compare each Uniform segment to the same questions in the completed repaired Resolution50 run. Frame indices and shared adaptive baseline geometry were verified.

| Fresh segment | New questions | Uniform model s/question | Resolution50 on those questions | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Uniform50, indices 926–1584 | 659 | 78.04 | 67.41 | Uniform50 is 15.8% slower |
| Uniform25, indices 984–1799 | 816 | 48.68 | 67.55 | Greater decoder reduction makes Uniform25 faster than Resolution50 |
| Uniform10, indices 1032–1145 | 114 | 29.86 | 66.91 | Same qualification; different nominal fractions |

For the Uniform50 segment, vision/embedding time is 16.61 vs 11.44 seconds, and prefill is 60.81 vs 55.55 seconds. Mean visual counts are 57,672 vs 52,688. Both the full-resolution vision path and extra retained decoder tokens contribute to the remaining gap. Measurements were taken on different dates with shared host resources, so 15.8% is an observed gap rather than a controlled causal estimate.

The completed Uniform25 marker's overall mean of 46.47 model seconds/question mixes **984 historical mostly short-video answers** with **816 fresh medium-video answers**. Its first-three-question fixed-output profiles were replayed from the earlier run too. Neither aggregate describes a clean current single-worker benchmark. VideoMME rows 0–899 are short and 900–1799 medium; comparing early short-video throughput with later medium-video throughput creates another large apparent slowdown.

**Before/after code comparison and runtime implications**

There is no committed Git revision for these local additions. The comparison uses the retained serial runner, current extension sources, pinned manifests, and actual job payloads. **All 25 original source hashes, all three recovery source/test hashes, and all four parallel extension hashes match the recorded manifests.** Thus the retained pre-parallel inference and recovery implementations can be compared directly with the extension. A [controller source diff](serial_vs_parallel_controller_20261111.diff) is provided; it compares the two implementations, not two historical Git commits.

| Area | Serial recovery implementation | Parallel extension | Runtime consequence |
| --- | --- | --- | --- |
| Launcher | Serial `run_token_reduction_video_retry.py` | Shell wrapper now dispatches `run_parallel_video_evaluation.py`; serial implementation retained | Selects the new scheduler and checkpoint worker |
| Work partition | One GPU1 worker, jobs sequential | Configurable physical GPUs and per-GPU worker limit; each process handles a complete suite/configuration | Duplicates models, CUDA contexts, host inputs and CPU thread pools; one evaluation is not distributed across GPUs |
| CLI / queue | Suite/output and recovery execution controls | Adds GPU lists/live pool file, concurrency, memory/headroom budgets, configuration filter and separate smoke namespaces | Changes which jobs can overlap; no automatic frame/token policy reduction |
| Admission | Whole-card preparation hold and idle handoff per job | Conservative per-worker peak estimate; completed matching peak ×1.25 +1 GiB with minimum 2 GiB, otherwise fallback; reserves future peak headroom | May delay/reorder dispatch; admission budget does not change computation |
| GPU ownership | GPU1 lock and reservation | Per-GPU locks; one holder released at first handoff, reacquired only after all siblings exit; optional GPUs join/drain | Avoids a full-card holder competing with a sibling; setup/gates add startup time |
| Scheduling loop | Wait for one subprocess | Polls GPU state and workers, normally every 0.2 seconds plus query time; allows smaller fitting jobs to pass; detects unexpected GPU occupants | Adds controller overhead; sampled controller CPU was small; queue wait is not model latency |
| Memory failures | Attempt fails, later rerun starts job again | Memory pressure lowers concurrency; recognized CUDA OOM lowers it and requeues when concurrency >1; serial OOM/non-OOM errors remain failures | Failed current question and setup can repeat, but committed answers survive |
| Worker entrypoint | Installs recovery adapter, calls `matrix.worker` | `parallel_video_worker.py` verifies extension hashes, calls the same recovery installer and `matrix.worker`, wraps harness `generate_until` | Same model computation for fresh requests; adds Python request identity checks and checkpoint handling |
| Answer persistence | Telemetry exists, but only COMPLETE jobs can be skipped | Fsynced checksummed answer + telemetry + sampling after each new answer; reads only the tail of telemetry/sampling to save it | Extra I/O/serialization after inference; outside model timer. Its cost has not been separately measured |
| Resume | Interrupted job restarts at question one | Scans/validates past checkpoint files, takes longest compatible prefix, checks request identity/order, replays before decoding; rescoring covers every question | Saves repeated inference; startup reads and replay writes cost time. Old legacy telemetry alone cannot be resumed |
| Replay telemetry | No replay | Copies old telemetry/sampling and writes a new checkpoint; increments generator index to preserve profiling position | Completion means and fixed-output samples may mix concurrency settings; wall-time denominators must exclude replayed answers |
| Lifecycle | Reservation handoff watches controller; subprocess handling | Adds a daemon checking parent once/second; scheduler terminates owned workers on interruption and handles pool removal by draining | Small monitoring overhead; reduces orphan workers; interrupted in-flight question can repeat |
| Provenance / completion | Pins original/recovery protocol, verifies reused passes | Also pins extension source hashes and protocol/model/dataset resume identity, records scheduling in jobs/markers | CPU/hash and completion-time disk overhead; safeguards compatible replay |
| Summary | Serial recovery can report same-runtime fixed-output speedups | Parallel summary suppresses speedups and records contention/replay caveat | Changes presentation; no inference cost. Active and historical attempts share generic incomplete status |
| Supporting files | Serial recovery validation and docs | Parallel scheduler/checkpoint tests, CPU harness integration, README; optional external health watcher | Tests and documentation do not run per question; no watcher was active in this snapshot |

The model, strict packed INT4 Triton W4A16 kernels, FlashAttention2, batch size one, greedy generation, 2 fps/full-duration sampling, retention logic, scoring, packed-weight cache, allocator environment, streamed temporal-group size one, 512-token prefill chunks, and 131,072-token adaptive baseline cap are inherited from the serial recovery implementation. The extension introduces no explicit CPU thread limit or affinity setting. Existing per-worker thread pools therefore multiply with the number of processes.

**Changes that preceded multiprocessing must be kept separate.** The recovery runtime already fixed TempCompass's YAML path, streamed vision one temporal group at a time, staged features on CPU, chunked decoder prefill, and introduced the shared adaptive spatial baseline. Those repairs can change runtime relative to the original non-recovery runs, and spatial adaptation can change input content. They were already present in the completed serial TempCompass and repaired Resolution50 runs. For VideoMME, 846/1,800 questions in repaired baseline/Resolution50/Uniform25 used adaptation. The reused original Resolution25/Resolution10 results use the original policy, so their timings cannot isolate a reduction-method effect against repaired Uniform results.

**Practical interpretation**

Keeping one worker per GPU is consistent with the observed contention problem, although the best aggregate throughput setting is not proven. The next useful optimization experiment is a small isolated, matched medium-video run that measures preprocessing wall time, model components, checkpoint I/O, and actual token counts while varying CPU thread limits. That would distinguish CPU preprocessing/thread overhead from vision/decoder work. Benchmark resolution and uniform at matched actual token counts as well as matched nominal fractions. Changing input policy or deleting checkpoints is unnecessary for that experiment.

For current diagnosis, the evidence supports three distinct causes: an inherent full-vision/host-input cost for Uniform, fewer actual tokens in resolution's rounded grids, and the earlier parallel contention penalty. Longer medium-video inputs and replayed timing aggregates complicate raw log comparisons. The large earlier slowdown should not be attributed solely to the token selection algorithm.

**Audit artifacts and verification**

- [Machine-readable calculations](runtime_analysis_20261111.json): TempCompass timings, exact 443-question pairing, fresh-prefix segmentation, frame/adaptive-grid checks, and source hashes.
- [Host process/GPU samples](runtime_host_snapshot_20261111.json): short CPU/GPU utilization sample and selected thread/GPU environment settings.
- [Reproducible CPU-only audit](analyze_runtime_20261111.py): verified 32 pinned source hashes and 144 artifacts across the 13 repaired completion markers, including every TempCompass result. It asserts ordered telemetry and matched question/frame identities.
- [Serial/parallel controller diff](serial_vs_parallel_controller_20261111.diff).

The existing CPU-only parallel scheduler/checkpoint test suite was rerun: **12 tests passed**. No GPU benchmark or training process was launched for this audit.

Approximate historical wall times use attempt-name timestamps and exit.json modification times. They include setup/profiling and are secondary to per-question component evidence. The existing November 10 concurrency report is useful historical evidence; its completion forecasts predate the current serial resume and should not be used as current ETAs.
