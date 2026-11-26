# QAT INT4 × visual-token reduction: 576-token image/video experiment

Created 2026-10-26. Evaluation only; no training and no changes to previous experiment outputs.

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

The user confirmed that “576px” means the **576-visual-token training ceiling**, not a 576-pixel image edge. All three models receive the same image area constraint. Small source images are not artificially expanded to 576 tokens. Videos are downsampled to the corresponding spatial budget; this replaces the earlier 720p video policy for this independent experiment. Whole-video coverage and the 2 fps policy are preserved. No maximum frame count, duration-dependent resolution, text truncation, or automatic population dropping is introduced.

The 100% setting is shared across all three methods. It is executed once as `baseline_r100`; each of the three methods then has 50%, 25%, and 10% jobs. The default matrix therefore contains **3 models × 12 suites × 10 configurations = 360 jobs**, rather than duplicating the identical full-token baseline.

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

## Execution

From the workspace root:

```bash
EXP=reproduction/GRACE/experiments/qat_int4_token_reduction_576_seed42
bash "$EXP/run_experiment.sh" plan
bash "$EXP/run_experiment.sh" check
```

`plan` prints the model directories and full matrix without loading models. `check` validates checkpoint metadata, all selected annotation populations/assets, pinned harness identity, and Triton imports on CPU. It does not claim CUDA correctness or GPU memory feasibility.

Inspect GPU ownership and select an idle compute GPU before either GPU command. The runner rejects occupied cards and uses the existing per-GPU lock. GPU 2 below is an example, not a reservation:

```bash
# Numerical packed-kernel comparison against dequantized FP16 GEMM.
bash "$EXP/run_experiment.sh" kernel-check --gpu 2

# First validate one image question, one condition, all three models.
bash "$EXP/run_experiment.sh" run --gpu 2 --suites ai2d --retentions 1 --smoke 1

# Exercise all reductions on one image and one video question per model.
bash "$EXP/run_experiment.sh" run --gpu 2 --suites ai2d,tempcompass --smoke 1

# Full requested matrix. Some full-token videos may be blocked by context/memory.
bash "$EXP/run_experiment.sh" run --gpu 2
bash "$EXP/run_experiment.sh" summarize
```

`--models qat_w4,vanilla_2b,released_bf16`, `--methods resolution,uniform,merge`,
`--retentions 1,.5,.25,.1`, `--suites ...`, and `--output PATH` select subsets or a new result base. Explicit model order is honored. The launcher uses the existing `.venv-lmms-eval/bin/python`; set `LMMS_PYTHON` to use a compatible isolated environment. Do not change the source checkpoint directories or their contents. Smoke/full outputs are separate; code, policy, experiment, environment, sources and model provenance are pinned. Changed settings require a new output base. Already completed jobs are skipped only after artifact hash verification; incomplete jobs restart in a fresh attempt.

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

- Controller/worker: `evaluation/runners/run_token_reduction_matrix.py`.
- Budgeting, compaction, DeepStack alignment and greedy decoding: `evaluation/protocols/token_reduction_runtime.py`.
- Packed kernel: `evaluation/protocols/int4_gemm.py`.
- CPU regressions: `evaluation/tests/test_token_reduction.py`.
- CUDA numerical gate / offline compilation: `evaluation/tests/check_int4_kernel.py`.

```bash
reproduction/GRACE/.venv-lmms-eval/bin/python -m unittest discover \
  -s reproduction/GRACE/evaluation/tests -p test_token_reduction.py -v
reproduction/GRACE/.venv-lmms-eval/bin/python \
  reproduction/GRACE/evaluation/tests/check_int4_kernel.py --compile-only
```

See `validation.json` for the actual validation status. No full benchmark accuracy, memory saving or speedup is claimed until the corresponding GPU jobs have run. All compute cards were occupied during setup, so GPU numerical validation and full-checkpoint smoke inference were not run.
