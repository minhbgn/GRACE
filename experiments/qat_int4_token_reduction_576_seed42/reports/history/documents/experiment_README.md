# QAT INT4 × visual-token reduction: 576-token image/video experiment

<!-- CURRENT_VIDEO_PROGRESS_START -->
Latest review: **2026-11-25**. QAT W4 has **120/120** verified completions, including all Video-MMMU splits and the matched VideoMME resolution-r25/r10 reruns. Read the [completed experiment report](experiment_report_completed_video.md) for all accuracy results, the resolution subset-versus-full comparison, and uniform/merge model/request runtime extrapolations with confidence intervals. Resolution-r25 scores **61.056%** versus **61.500%** at r100, retaining **23.646%** of visual tokens. Its 50-question runtime projection is **15.06% above** the full rerun; r10's is **16.70% above**. Rebuild with `reports/compile_latest_analysis.py`, `reports/compile_timing_analysis.py`, `reports/plot_latest_analysis.py`, and `reports/write_latest_report.py` using the interpreters listed in the report. The [November 16 report](experiment_report_20261116.md) preserves the detailed implementation explanations and integration plan. The older compilers and progress sections below are historical snapshots.
<!-- CURRENT_VIDEO_PROGRESS_END -->
Created 2026-10-26. Evaluation only; no training and no changes to previous experiment outputs.

## Current progress — 2026-11-05

**Video recovery prepared:** use [VIDEO_RECOVERY_GPU1.md](VIDEO_RECOVERY_GPU1.md) and `run_remaining_qat_videos_gpu1.sh` for the **45 failed QAT video jobs on GPU 1 only**. It verifies and reuses the five passed video jobs, fixes the TempCompass mapping, streams vision/decoder work, and uses the user-approved adaptive spatial cap for oversized clips while preserving 2 fps/full duration. The repaired protocol has its own output directory and explicit reuse manifest. No GPU jobs were launched during preparation. The original evaluation commands below retain their historical input policy.

See the [QAT W4 experiment report](experiment_report.md) for verified results, limitations and exact resume commands. The original full matrix under `results/full` has **75/120 completed QAT jobs**: all **70/70 image jobs** and **5/50 video jobs**. The other video jobs failed: **35 CUDA OOMs** and **10 TempCompass missing-YAML errors**. All configurations were attempted; the full video evaluation is not complete. Eight Vanilla 2B HallusionBench jobs also completed before interruption; released BF16 has no full-matrix completions yet.

The CPU-only `summarize` action now regenerates `results/full/comparison.md` and `comparison.json`. `python3 compile_report.py` from this directory independently audits saved artifacts/raw QAT telemetry and writes `experiment_report.md`, `reports/completion_audit.json`, and `reports/completed_metrics.csv`. Both baseline checkpoint directories are now present. No GPU work is needed to read or rebuild this synthesis.

To continue the baseline image evaluations once GPUs are available, use the report's `--models vanilla_2b,released_bf16` and seven-image-suite command with the existing `--output "$EXP/results"`. Completed jobs are hash-verified and skipped; incomplete jobs restart from the first question. Video retries require attention to the recorded OOMs and TempCompass mapping failure. Keep the pinned evaluation code and protocol unchanged when resuming this output directory.

## Historical progress — 2026-11-01

The experiment asks: **how much visual information can we remove before accuracy drops, and does combining this with QAT INT4 reduce inference time and GPU memory?** This is evaluation only. No new training is required.

- **QAT training is complete:** the October 30 run reached step 256; its exported weights and `qat_quantized_weights.bin` are present. The released BF16 checkpoint is also present.
- **GPU smoke now passes:** QAT INT4 on GPU 0 and released BF16 on GPU 2 each completed one full-token AI2D question plus timing profiles. Both CUDA packed-kernel gates pass (maximum absolute error 0.0078125). QAT telemetry records 392 fused INT4 calls. Results are under `results_reserved_gpu02/smoke_1/`. Full benchmark/reduction results remain pending; one-question scores and timings are not evidence of full-population accuracy or speedup.
- **Current preflight:** all 12 suites pass `check --models qat_w4,released_bf16` using the surviving Conda Python. The complete three-model check stops because `models/Qwen3-VL-2B-Instruct/` is missing. Restore that exact 2B model to run the full comparison; the available 8B model is not the requested baseline.
- **Environment:** `.venv-lmms-eval` is absent. Set `LMMS_PYTHON=/home/air/miniconda3/envs/grace/bin/python`; this environment passed the preflight above.
- **Two-GPU scheduling is implemented:** launchers now default to physical GPUs **0 and 2**. One independent job runs on each GPU; the next job goes to whichever GPU finishes first. Results stay in one directory/report. This can approach twice the throughput for balanced jobs, but actual savings depend on video lengths, loading, CPU/I/O contention and first-time packing. It does not make an individual question use both GPUs or double its memory capacity.
- **GPU availability at initial inspection:** GPU 0 was occupied by this workspace's `scripts/poll_gpu_hold.py` reservation (PID 1697111, also holding GPU 2); GPU 1 by Raymond's CARLA process (PID 1232144; cwd `/home/raymond/vbev-reproduction-20261031/carla/CarlaUE4/Binaries/Linux`). Neither was idle. No processes were stopped. Release the reservation and arrange for GPU 1 to be free before launching; the runner rejects occupied cards.

Plain-language design: compare **three models**, each with the full visual input and with **50%, 25%, or 10% retained**. Shrink the pixels (`resolution`), keep evenly spaced visual tokens (`uniform`), or average neighboring visual features (`merge`). Text is preserved. Start from at most **576 visual tokens per image**; videos use the same spatial ceiling per two-frame group at 2 fps. Measure accuracy, actual token counts, time and memory on seven image suites plus five video populations. That makes **360 jobs**, roughly **1.01 million question inferences**, plus timing repeats.

## Requested design

| Axis | Settings |
| --- | --- |
| Models, in order | QAT W4 with realized packed INT4 kernels; Vanilla Qwen3-VL 2B; released BF16 initialization |
| Image benchmarks | HallusionBench, MMBench EN dev, ScienceQA test, AI2D, MMMU validation, SEED-Bench image, MMStar |
| Video benchmarks | TempCompass MC; VideoMME short + medium; Video-MMMU perception, comprehension, adaptation |
| Baseline visual constraint | **576 merged visual tokens per image**, corresponding to maximum image area 589,824 pixels; minimum area 4,096 pixels |
| Video extension | Same maximum area per frame; at 2 fps, up to 576 merged visual tokens per two-frame temporal patch group, not 576 tokens for an entire video |
| Retained fractions | 100%, 50%, 25%, 10% of each input's actual baseline visual count |
| Methods | Resolution reduction; uniform selection; deterministic spatial token merging |
| Metrics | Benchmark performance; original/retained visual tokens; input/output text tokens; inference timing; fixed-output speedup; peak allocated GPU memory |

The user confirmed that “576px” means the **576-visual-token training ceiling**, not a 576-pixel image edge. All three models receive the same image area constraint. Small source images are not artificially expanded to 576 tokens. Videos are downsampled to the corresponding spatial budget; this replaces the earlier 720p video policy for this independent experiment. Whole-video coverage and the 2 fps policy are preserved. No maximum frame count, duration-dependent resolution, text truncation, or automatic population dropping is introduced. The training 4,096-token total sequence cap is not used to truncate inference; native model context is checked after physical reduction.

The 100% setting is shared across all three methods. It is executed once as `baseline_r100`; each of the three methods then has 50%, 25%, and 10% jobs. The default matrix therefore contains **3 models × 12 suites × 10 configurations = 360 jobs**, rather than duplicating the identical full-token baseline. With the current local populations (29,420 image rows and 4,280 video questions), that is 1,011,000 question inferences before repeated timing profiles; use the subset/smoke controls before committing to the full matrix.

## Local SFT selection check — 2026-11-01

This matrix does **not** evaluate any locally trained SFT checkpoint. Its only model IDs are `qat_w4`, `vanilla_2b` and `released_bf16`; the two-model smoke selects `qat_w4,released_bf16`. The separate released-BF16 576-token experiment's current image/video launchers now use an `evaluation.json` without the previous local `sft_2b` baseline. Released BF16 remains a valid reference.

## Exact source checkpoints

Paths below are relative to `reproduction/GRACE/` and are resolved by `experiment.json`:

| Model ID | Path | Execution |
| --- | --- | --- |
| `qat_w4` | `experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/w4g128/runs/qat` | Pack the trained G128 W4 codes/scales; strict fused Triton W4A16 GEMM |
| `vanilla_2b` | `models/Qwen3-VL-2B-Instruct` | BF16 |
| `released_bf16` | `models/Qwen3-VL-2B-GRACE-BF16` | BF16; matched starting checkpoint for QAT |

This does not run AWQ calibration or replace the trained quantizer with a new PTQ quantizer. Packing uses the QAT sidecar `qat_quantized_weights.bin`. Integer-code equality is checked for every converted module. The 196 eligible language-model linears are packed; the vision encoder, embeddings, head and other excluded modules remain floating point.

### What “realized INT4” means here

The installed environment lacks `awq_ext`. The experiment includes a strict Triton backend that reads the existing AWQ packing layout directly. Weights occupy packed 4-bit storage and are unpacked/dequantized **inside each GEMM tile**, using FP16 tensor-core operands and FP32 accumulation. This is **weight-only W4A16**, not integer-only arithmetic and not native INT4×INT4 tensor-core multiplication. It does not materialize the complete BF16/FP16 weight matrix during a linear forward.

There is no CPU/dense fallback. Kernel or integer-code verification failure aborts the job. Each inference records the number of fused INT4 calls. `run` first performs a numerical CUDA kernel test on the selected idle GPU before starting QAT jobs. Performance is specific to this correctness-first backend and must not be presented as the best possible AWQ/Marlin performance or as a guaranteed speedup over BF16.

The first QAT worker packs on CPU and caches the packed module state under `runtime_cache/packed_qat_w4/`. Later workers reuse it. Cache keys include the source sidecar hash and packing source; a file lock and atomic rename protect cache creation. Packing and checkpoint load time are outside inference measurements. FP16 scale storage and arithmetic differences may change outputs slightly; integer-code equality alone is not a guarantee of identical end-to-end predictions.

## Reduction semantics

Reduction occurs before decoder layer 0. Every decoder layer and its KV cache see the compacted physical sequence; tokens are not merely masked. Text, question tokens, timestamps and vision-boundary markers are retained.

- **`resolution`**: determine the actual baseline grid, reduce its area toward the requested fraction, resize source pixels, and encode the smaller input. This saves vision-encoder work as well as decoder work. Grids must remain patch aligned, so achieved fractions can differ from requested fractions; record both. It may go below the training minimum area at low retention.
- **`uniform`**: encode at the full baseline spatial budget, then keep evenly distributed representatives from each contiguous visual group. For video, timestamp-separated temporal groups remain present. It saves decoder work while paying for full vision encoding.
- **`merge`**: encode at the full baseline budget, divide each visual group into contiguous bins, and replace each bin with its mean feature. Apply the same merge to all aligned DeepStack features. Use the middle member's original M-RoPE coordinate as the representative coordinate. This is a deterministic spatial merge baseline, **not** a reproduction of ToMe, DART, FastV or an LLMC+ method, and has no learned importance score.

Uniform/merge budgets round up per image or temporal group, retaining at least one token per group. A 10% target may therefore retain slightly more than 10% for small inputs. Original multimodal coordinates are preserved; causal/cache positions use compact sequence indices. The manual greedy cache loop implements one beam, repetition penalty 1, tokenizer EOS, and at most 256 generated tokens for both image and video evaluation. Capped generations remain visible in results and summaries.

At 100%, no compaction/averaging occurs. A tiny Qwen3-VL integration test verifies that this decoder path matches Hugging Face generation, including DeepStack and cache behavior. Full-checkpoint/GPU equivalence remains a validation task.

## Populations and scoring

The image path reuses the previous local scripts, dataset iteration, prompt builders and deterministic scorers. ScienceQA and HallusionBench retain their text-only rows; text-only rows have zero visual tokens. MMBench uses the existing circular metric over the supplied rotations; a small smoke sample may not have a defined circular primary score.

Video uses the clean pinned lmms-eval harness and its existing local media adapters. TempCompass MC uses the explicit deterministic local choice fallback; unresolved counts wrong, and this is **not** an official API-judged score. VideoMME uses 1,800 short/medium questions, excluding 900 long questions. Each Video-MMMU split has 300 questions. No subtitle/audio input or external judge is used. API-only tasks without a local fallback remain outside this matrix.

Baseline 576-per-group inputs can still exceed context or memory for sufficiently long videos. Such failures remain failed attempts; they do not produce completion markers. Reduction may make an otherwise blocked example runnable, but no speedup against an unavailable full-token baseline is reported. Full-population benchmark results and speedup comparisons must have the same covered questions. Do not silently filter the test set to inputs that happened to fit.

## GPU reservations and parser repair — 2026-11-01 follow-up

The duplicated `--processor_path` and protocol-argument declarations were removed from all seven image evaluators, along with their duplicated metadata entries/imports/protocol application. All seven scripts now pass `--help`.

The user selected **GPUs 0 and 2**, both free, leaving GPU 1's CARLA process alone. These are now the launcher's default pair. `run` reserves VRAM by default (`--reserve-gpus`; disable with `--no-reserve-gpus`). Per-GPU file locks are acquired before CPU preflight. Separate holder processes occupy most VRAM, leaving 512 MiB per GPU, through dataset checks, provenance hashing, and CPU model loading/INT4 packing. The worker signals readiness immediately before moving its model to CUDA; the controller then releases that GPU's holder, checks it is idle, and permits the transfer. The same handoff protects kernel-test initialization. Between jobs the runner reacquires a reservation; normal exit or a preflight error releases it. Holder stdin EOF also releases memory if the controller dies.

This is a **best-effort VRAM reservation**, not enforced scheduler exclusivity: there is a brief release/allocation handoff, and unrelated programs can still attempt to use a GPU. Existing foreign jobs are rejected, not automatically stopped.

The previous `results/smoke_1` pins the old sources. The command below successfully completed both one-question jobs and released both GPU reservations; rerunning it verifies and skips the completions:

```bash
EXP=reproduction/GRACE/experiments/qat_int4_token_reduction_576_seed42
export LMMS_PYTHON=/home/air/miniconda3/envs/grace/bin/python
bash "$EXP/run_experiment.sh" run --gpus 0,2 --reserve-gpus \
  --models qat_w4,released_bf16 --suites ai2d --retentions 1 --smoke 1 \
  --output "$EXP/results_reserved_gpu02"
```

Seven CPU scheduler/reservation regression tests pass, covering concurrency, cleanup, readiness ordering and controller-death detection. The surviving Conda environment now also has the pinned `decord==0.6.0`; version recording and its import pass.

## Execution

From the workspace root, use the surviving environment:

```bash
EXP=reproduction/GRACE/experiments/qat_int4_token_reduction_576_seed42
export LMMS_PYTHON=/home/air/miniconda3/envs/grace/bin/python
bash "$EXP/run_experiment.sh" plan

# Runnable CPU preflight for the two checkpoints currently present.
bash "$EXP/run_experiment.sh" check --models qat_w4,released_bf16
```

The simplest GPU smoke path, **once both GPUs are idle**, is:

```bash
# One AI2D question per available model, full visual input.
# The run automatically tests the packed CUDA kernel on BOTH GPUs first.
bash "$EXP/run_experiment.sh" run --gpus 0,2 \
  --models qat_w4,released_bf16 --suites ai2d --retentions 1 --smoke 1

# Next: all reductions on one image and one video question per model.
bash "$EXP/run_experiment.sh" run --gpus 0,2 \
  --models qat_w4,released_bf16 --suites ai2d,tempcompass --smoke 1
```

Restore/download `Qwen/Qwen3-VL-2B-Instruct` (weights and processor/tokenizer files) into `reproduction/GRACE/models/Qwen3-VL-2B-Instruct` before the complete three-model run. For example, with network access, from the workspace root:

```bash
"$LMMS_PYTHON" - <<'PYMODEL'
from huggingface_hub import HfApi, snapshot_download
repo = "Qwen/Qwen3-VL-2B-Instruct"
revision = HfApi().model_info(repo).sha
print("Downloading", repo, "at", revision, flush=True)
snapshot_download(repo_id=repo, revision=revision,
                  local_dir="reproduction/GRACE/models/Qwen3-VL-2B-Instruct")
PYMODEL

bash "$EXP/run_experiment.sh" check
# Validate all three models on an image and video before the full matrix.
bash "$EXP/run_experiment.sh" run --gpus 0,2 --suites ai2d,tempcompass --smoke 1
# Full matrix; GPUs 0 and 2 are also the default if --gpus is omitted.
bash "$EXP/run_experiment.sh" run --gpus 0,2
bash "$EXP/run_experiment.sh" summarize
```

Reports are `results/full/comparison.md` and `comparison.json`; smoke runs use `results/smoke_1/`. Rerun the same command to resume verified completions. Long full-token videos may still exceed context or memory; they remain failed attempts.

`run_images.sh` and `run_videos.sh` accept the same options and default GPU pair. Use one controller to distribute its selected matrix across both GPUs. Separate concurrent controllers still require separate `--output` bases and disjoint GPUs because the output directory and each GPU are exclusively locked. Jobs are dispatched in the selected model/suite/configuration order; completion order may differ with concurrency. The packed-QAT cache is shared and protected by its existing file lock.

`plan` prints paths, GPUs and jobs without loading models. `check` validates selected local populations, checkpoint metadata, harness identity and Triton imports on CPU. It does not prove CUDA correctness or memory feasibility. `kernel-check --gpus 0,2` explicitly tests both GPUs without running benchmarks. `--gpu 0` retains single-GPU operation. GPU indices match `nvidia-smi` because workers set `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

`--models qat_w4,vanilla_2b,released_bf16`, `--methods resolution,uniform,merge`,
`--retentions 1,.5,.25,.1`, `--suites ...`, and `--output PATH` select subsets or a new result base. Explicit model dispatch order is honored. The launcher defaults to `.venv-lmms-eval/bin/python`; that environment is currently missing, so use the `LMMS_PYTHON` override above. Do not change the source checkpoint directories or their contents. Smoke/full outputs are separate; code, policy, experiment, environment, sources and model provenance are pinned. Changed settings require a new output base. Already completed jobs are skipped only after artifact hash verification; incomplete jobs restart in a fresh attempt.

## Timing and reporting

For every question, `telemetry.jsonl` records:

- Original baseline visual tokens, physically retained visual tokens, actual retained ratio, and decoder-layer visual counts.
- Input text tokens and generated token IDs/count, including EOS; output cutoff counts remain visible.
- Vision/embedding time, reduction overhead, decoder prefill time, decode time, total model time and final KV length.
- Peak allocated GPU memory and actual fused INT4 call count.

Total model time includes GPU vision encoding, selector/merger and decoder, measured with CUDA synchronization. It excludes checkpoint load/packing, CPU video decoding/image preprocessing, host-to-device transfer and file I/O. It is not end-to-end video upload/response latency. The manual decode path computes next-token logits only for the last position.

Normal benchmark answers stop at EOS. To control for changing answer lengths, the first three questions per job additionally receive one warm-up and three timed runs generating exactly 32 tokens (EOS ignored only in profiling). These profiling answers do not enter benchmark scoring. Fixed-output time includes vision and reduction on every repeat. Use `--profile-samples`, `--profile-repeats`, and `--profile-tokens` to declare a different protocol; use `--profile-samples 0` to disable profiling.

`comparison.json` / `comparison.md` combine per-suite performance, token counts, timings, cutoff counts and fixed-output speedup relative to the same model's `baseline_r100`. The full JSON retains text token counts and component timings. No heterogeneous cross-benchmark accuracy mean is invented. Default timing samples are a functional starting point, not a representative latency study; use a larger pinned stratified sample for final claims and report latency distributions/uncertainty from the raw records.

## Implementation and validation

- Concurrent dispatch: `evaluation/protocols/gpu_job_queue.py`; regressions: `evaluation/tests/test_gpu_job_queue.py`.
- Controller/worker: `evaluation/runners/run_token_reduction_matrix.py`.
- Budgeting, compaction, DeepStack alignment and greedy decoding: `evaluation/protocols/token_reduction_runtime.py`.
- Packed kernel: `evaluation/protocols/int4_gemm.py`.
- CPU regressions: `evaluation/tests/test_token_reduction.py`.
- CUDA numerical gate / offline compilation: `evaluation/tests/check_int4_kernel.py`.

```bash
"$LMMS_PYTHON" -m unittest discover \
  -s reproduction/GRACE/evaluation/tests -p test_token_reduction.py -v
"$LMMS_PYTHON" \
  reproduction/GRACE/evaluation/tests/check_int4_kernel.py --compile-only
```

See `validation.json` for historical validation and the November 1 follow-up. The new scheduler passed three CPU-only regression tests covering simultaneous dispatch, exactly-once execution, per-GPU serialization, failure propagation and single-GPU compatibility; planning and shell/Python syntax checks also passed. Current two-model/all-suite CPU preflight passes. The subsequent two-model AI2D GPU smoke and both numerical CUDA gates also pass; full benchmark accuracy, memory saving and speedup remain unmeasured. The kernel is based on tiled matrix multiplication as described in the [official Triton tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html); it adds AWQ-layout INT4 unpacking within each tile.

## Parallel video recovery and resumable answers — 2026-11-09

`run_remaining_qat_videos_gpu1.sh` now uses the additive parallel controller,
defaulting to GPU 1 and `--parallel-evaluation 1`. It accepts multiple GPUs,
memory-aware per-GPU concurrency, OOM backoff and a live GPU-pool file. New
attempts checkpoint individual answers; legacy interrupted attempts still
restart from their first question. Original pinned computation and verified
completions are preserved. See the [parallel evaluation guide](../../evaluation/docs/README_parallel_video_evaluation.md)
for commands, hold/release ownership, timing limitations and switching from an
already running legacy controller. CPU tests/preflight and real-harness replay
integration passed; no GPU workload was launched for this implementation.
