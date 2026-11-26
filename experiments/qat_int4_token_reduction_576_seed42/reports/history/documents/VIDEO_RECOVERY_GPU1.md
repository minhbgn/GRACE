# Resume the remaining QAT W4 video jobs on GPU 1

Prepared 2026-11-05. **No GPU job was launched during preparation.** The launcher is restricted to physical GPU **1**, model **`qat_w4`**, and the five existing video suites. It cannot dispatch image jobs or either BF16 baseline.

From the workspace root:

```bash
EXP=reproduction/GRACE/experiments/qat_int4_token_reduction_576_seed42

# Read-only plan, no GPU allocation.
bash "$EXP/run_remaining_qat_videos_gpu1.sh" plan

# Launch later, once physical GPU 1 is idle.
bash "$EXP/run_remaining_qat_videos_gpu1.sh" run

# Run the same command again after interruption to resume committed jobs.
bash "$EXP/run_remaining_qat_videos_gpu1.sh" run

# CPU-only combined status: original passes plus repaired completions.
bash "$EXP/run_remaining_qat_videos_gpu1.sh" summarize
```

The script defaults to `/home/air/miniconda3/envs/grace/bin/python`; `LMMS_PYTHON` may select an environment with the same pinned Python/package versions. It defaults to `plan`, so invoking it without an action does not start work. GPU indices follow `nvidia-smi` through `CUDA_DEVICE_ORDER=PCI_BUS_ID`. `--gpu 0` and `--gpu 2` are rejected. The controller checks that GPU 1 is idle, holds its existing file lock and reserves VRAM during CPU model loading; it never stops another process.

## Exactly what runs

| Suite | Already passed; reused | Failed; queued for retry |
| --- | ---: | ---: |
| TempCompass MC | 0 | 10 |
| VideoMME short + medium | 2 | 8 |
| Video-MMMU perception | 1 | 9 |
| Video-MMMU comprehension | 1 | 9 |
| Video-MMMU adaptation | 1 | 9 |
| Total | **5** | **45** |

Reused passes are VideoMME `resolution_r025` / `resolution_r010` and the three Video-MMMU `resolution_r010` jobs. Their `COMPLETE.json` and every recorded artifact hash are verified. They are referenced from their existing directories and are **never dispatched**. The recovery output records their source marker hashes in `reuse_manifest.json`. No original completion marker, prediction, timing record or pinned evaluation source is changed.

After each successful retry, a new `COMPLETE.json` is committed. Subsequent invocations verify it and skip that job too. A failed/interrupted attempt has no completion marker and restarts from its first question in a new attempt directory. This is job-level continuation, not per-question continuation. The controller updates the combined summary after every attempted job.

The default recovery output is:

```text
results_video_retry_gpu1/full/
  protocol.json
  reuse_manifest.json
  comparison.md
  comparison.json
  qat_w4/<suite>/<configuration>/attempt_*/...
```

Original results stay under `results/full`. Use this new launcher for recovery; the original `run_videos.sh` still implements the original pinned protocol.

## TempCompass repair

The clean pinned lmms-eval checkout already contains `tempcompass/tempcompass_mc.yaml`, its included template, and task utilities. The original runner requested the nonexistent `tempcompass/multi-choice.yaml`. The recovery runner uses the existing correct MC YAML; downloading a different version is unnecessary. The upstream checkout remains clean and its task files are hashed in the recovery protocol.

The deterministic local MC scorer remains in use: an unresolved answer counts wrong and no API judge is called. An offline integration test constructs the real task with two local annotation rows, supplies their gold answers and obtains 100%, including the upstream percent aggregation. This remains a local deterministic score, not an API-judged TempCompass score.

## Memory and context repair

The original evaluation already used batch size **one**. Its video tensor sent all sampled frames through vision at once, followed by a single decoder prefill. Failed attempts include large vision positional-embedding/attention allocations and large decoder activations.

The recovery runtime:

1. Keeps the complete preprocessed video tensor on CPU and transfers **one temporal patch group (two sampled frames)** at a time. Qwen3-VL vision attention partitions at those temporal groups; spatial positions and all DeepStack features are preserved.
2. Stages the decoder embeddings and aligned DeepStack features on CPU. Decoder prefill processes **512 tokens at a time**, keeping the causal KV cache and original multimodal positions. All retained tokens remain in the cache; there is no cache eviction or text truncation.
3. Uses the explicitly approved **adaptive spatial cap** for oversized clips, as described below. Frame indices, 2 fps sampling and whole-video coverage remain unchanged.

Chunking alone cannot cover the original full-input design. Saved complete low-resolution results show original visual counts up to **552,288** for VideoMME and **1,169,280** for Video-MMMU, before text. The model's native context is **262,144**. The QAT decoder's BF16 KV storage alone costs about **112 KiB per token** (28 layers × 8 KV heads × 128 head dimension × K/V × 2 bytes), so million-token inputs also exceed a 40 GiB GPU by a large margin.

### Declared adaptive spatial baseline

The recovery protocol caps the **adaptive baseline prompt plus the 256-token generation reserve at 131,072 tokens**. This leaves memory headroom on the roughly 40 GiB cards. The cap includes the exact question, template and timestamp tokens; it is not a visual-token-only limit.

For each question, the runner computes the original 576-token-per-group spatial grid, exact text count and all sampled frame indices before materializing the video tensor. If necessary, it lowers the per-group spatial grid to fit the cap. It then applies the requested baseline / resolution / uniform / merge configuration to that adaptive baseline. The adaptive baseline is identical for that clip across repaired methods/retentions. Grid alignment may retain fewer tokens than the nominal fraction. If even one visual token per temporal group plus text and the output reserve cannot fit, the job fails explicitly rather than dropping frames/questions.

Each question records the original and adaptive baseline token counts, both spatial grids, whether adaptation occurred, exact text count, decoder cap, requested retention and retained fraction relative to the original policy. Evidence is in `video_inputs.jsonl` and `telemetry.jsonl`. New completion summaries also report the number of adapted questions and maximum input sizes.

**Interpretation:** adapted jobs use a different spatial input policy from the five reused original passes. The combined summary labels their origins. Original scores/timings remain valid for their original inputs, but comparing them with adapted jobs does not isolate token-reduction effects. Baseline-relative speedups are only computed when both jobs completed under the repaired execution policy. Runtime timings also include the new host staging, so they are not interchangeable with old timings.

For an optional subset, retain the same output/protocol, for example:

```bash
bash "$EXP/run_remaining_qat_videos_gpu1.sh" run --suites tempcompass
```

Changing `--decoder-token-cap` or `--prefill-chunk-tokens` changes the pinned recovery protocol and requires a fresh `--output` base. The five verified original passes will still be reused in a new base. Do not edit `protocol.json` to bypass the check.

## Validation and first launch

Nine CPU regression/integration tests pass: streamed vision/DeepStack equivalence across multiple clips, chunked prefill answers/cache lengths/EOS, end-to-end reduction alignment, host staging, adaptive frame/budget invariants, correct YAML/local scoring, real TempCompass harness scoring, original-pass hash verification and resume skip behavior. CPU preflight also verifies the original source hashes, model/sidecar provenance, all five dataset fingerprints and the evaluation environment.

On the future GPU-1 launch, before any benchmark job starts, the controller runs the packed INT4 numerical gate and repeats the memory-runtime equivalence tests with a tiny Qwen3-VL model using CUDA BF16/FlashAttention 2. A failing gate aborts the launch. Logs are `int4_kernel_validation_gpu1.log` and `memory_equivalence_gpu1.log` in the recovery output.

Full-checkpoint CUDA memory feasibility has **not** been measured during preparation, because the GPU run is deferred to the user. The first launch performs those GPU checks; inspect a failed attempt's `run.log` if any remaining limit is reached.

To repeat CPU validation only:

```bash
/home/air/miniconda3/envs/grace/bin/python -m unittest discover \
  -s reproduction/GRACE/evaluation/tests -p test_video_memory_runtime.py -v
bash "$EXP/run_remaining_qat_videos_gpu1.sh" check
```
