# Offline video evaluation

`run_video_matrix.py` evaluates every model in the selected experiment JSON,
including both vanilla models, SFT, released BF16/W4/W8, and local LoRA-QAT and
LoRA-GRACE W4/W8. It uses Transformers inference on the existing unpacked
checkpoints. It never starts training or calls a paid judge.

The default experiment is `released_bf16_init_32k_b128_p589824_t4096_seed42`.
Video evaluation does **not** inherit its 576-image-token training ceiling,
4,096-token training sequence length, or the individual checkpoints' native
adaptive video pixel budgets.

## Selected input policy

- Sample **2 frames per second**, uniformly over the entire video, including
  the first and last frames. Round the requested frame count up and never
  exceed the number of source frames. There is **no maximum frame count**.
- Retain source spatial resolution, rounded to the nearest 32-pixel patch grid
  (minimum 32 per dimension). This small grid rounding is recorded explicitly.
- Disable processor resizing and its internal frame sampler. The processor's
  per-video pixel bound is set above the actual padded video's total pixels.
  This avoids Qwen's normal duration-dependent spatial reduction.
- Pass explicit source frame indices and source fps so timestamps are real.
  Only temporal patch padding duplicates the last sampled frame, as required
  by Qwen's two-frame temporal patches.
- Disable tokenizer truncation and padding. Use each model's real native context
  from `text_config.max_position_embeddings`, not the training sequence length.
  If input plus the generation allowance exceeds it, stop and record a blocker;
  do not shorten the video, shrink frames, drop prompt text, or omit the question.
- Greedy generation, BF16 loading, one beam, repetition penalty 1. The default
  output allowance is 4,096 tokens. Any generation that does not finish at EOS
  prevents certification. Increase `--max-new-tokens` in a separate protocol if
  this occurs; the retained diagnostic outputs are not uncensored results.
- No subtitles or audio. Inputs and sampling are identical across models. The
  scripts preserve native chat templates and verify matching input identities
  before placing completed models in the same comparison.

2 fps is the installed Qwen3-VL video processor's default sampling rate. Its
native maximum of 768 frames and adaptive total-pixel budget are **bypassed**.
2 fps remains a declared temporal sampling choice: it can miss short events;
this is not every-frame evaluation. `--fps 0` selects every source frame.
`--longest-side N` allows an explicitly separate spatial-resolution experiment.
Policy changes automatically produce different output directories.

Visual tokens are determined by the input representation, not chosen by the
language model. These checkpoints do not implement adaptive visual-token
selection. Identical videos under the same policy should therefore have equal
visual-token counts for vanilla/QAT/GRACE. Accuracy and generated-text length
can differ. Removing processor ceilings prevents saturation from biasing the
input means, but cannot turn those means into a model-specific token demand.

## Data and scoring

Supported local populations under `evaluation/video_benchmarks/`:

| Suite ID | Annotation path under its `data/` | Questions |
| --- | --- | ---: |
| `tempcompass` | `multi-choice/test-*.parquet` | 1,580 |
| `videomme` | `videomme/test-*.parquet` | 2,700 |
| `videommmu_perception` | `Perception/test-*.parquet` | 300 |
| `videommmu_comprehension` | `Comprehension/test-*.parquet` | 300 |
| `videommmu_adaptation` | `Adaptation/test-*.parquet` | 300 |

Paths in the table describe `<benchmark>/data/<annotation path>`; pass
`--data-root` to change the common benchmark root. Extracted MP4s and videos
inside the downloaded ZIP chunks are both supported. ZIPs are indexed without
bulk extraction; only the current video is extracted into a temporary directory
and cleaned up afterward. `--scratch` can select a filesystem for that temporary
file. Extraction never downloads additional data.

Empty/unavailable suites are listed but excluded. By default, a partially
downloaded suite is visibly blocked and excluded **as a whole**. Explicitly
requesting it with `--suites` fails. Missing/corrupt videos during processing
also fail the job. No result can pass on a reduced question denominator.

Scores use the existing `deterministic_local_v1` choice parser: ambiguous or
unresolved answers count wrong. Video-MMMU open responses use the pinned MMMU
typed candidate parser/matcher. Its Adaptation prompt follows the vendored
task's convention that the question image is at the end of the full video;
the separate question-only videos are excluded. Perception and Comprehension
ask the model to ignore the final quiz question. Report micro accuracy and
category diagnostics separately for each set. This is a declared local
protocol, not an official leaderboard replay or judge-equivalent score.

## Commands

From the workspace root:

```bash
RUN=reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/run_video_evaluation.sh

# Inspect populations and all ten model mappings; no model loading.
bash "$RUN" plan
bash "$RUN" check

# CPU-only: probe video metadata and compute exact prompt token counts.
# Full preflight includes context-overflow cases in the reported means.
bash "$RUN" prepare

# Choose a free GPU using gpuwho; GPU IDs below use PCI order.
# One example per model, isolated from full results.
bash "$RUN" run --suites tempcompass --smoke 1 --gpu 2

# Full TempCompass matrix for all ten checkpoints.
bash "$RUN" run --suites tempcompass --gpu 2
bash "$RUN" summarize --suites tempcompass
```

`--methods` selects comma-separated experiment IDs; default all ten. `--suites`
selects comma-separated IDs from the table. `LMMS_PYTHON` selects a different
Python executable; default the existing `.venv-lmms-eval/bin/python`.
`--attn sdpa` creates a separate protocol if FlashAttention 2 is unavailable.
Inspect logs and blockers before requesting a full matrix. An input fitting
context does not guarantee its vision encoder fits GPU memory; OOMs fail the
job without automatic downsampling.

Output defaults to `<experiment>/video_evaluation/video_unsaturated_v1_<policy_hash>/`
with separate `full/` and `smoke_N/` directories. The runner pins source code,
package versions, experiment configuration, source populations and policy.
Per-model provenance hashes include checkpoint weights. Changed provenance
rejects resume; use a fresh `--output` base after changes. Only one writer can
hold an output directory at a time. Failed attempts remain reviewable; rerunning
creates a new attempt. Completed jobs are revalidated before skipping them.

`comparison.md` / `comparison.json` report score and question-weighted means for
visual input, text input, and text output. Text input includes timestamps, chat
markers, and other nonvisual prompt tokens. Text output counts generated IDs,
including EOS; total text is input text plus output text. Each prediction saves
the raw output, generated token IDs, finish reason, exact input counts, source
frame indices/fps/resolution, processed grid, and peak allocated GPU memory.
Actual processor grids and placeholder counts must match CPU preflight.

Only jobs with complete population coverage, recomputed scores, consistent token
accounting, matching question/input identities, valid file hashes, and zero
generation cutoffs receive `COMPLETE.json`. Smoke scores are previews.

## Validation on 2026-10-17

CPU source/model preflight passed for all ten checkpoints. Six focused tests
cover uncapped long-video sampling, archive cleanup, duration-independent frame
resolution, score/telemetry tampering, output censoring, and actual Qwen processor
counts with odd/even temporal padding:

```bash
reproduction/GRACE/.venv-lmms-eval/bin/python -m unittest discover \
  -s reproduction/GRACE/evaluation/tests -p test_video_evaluation.py -v
```

The full CPU input audit at 2 fps/source resolution found:

| Population | Mean visual tokens | Maximum | Context overflows |
| --- | ---: | ---: | ---: |
| TempCompass, 1,580 questions | 2,875.36 | 10,752 | 0 |
| Each Video-MMMU set, 300 questions | 1,000,586.80 | 10,118,400 | 261 |

The context is 262,144 tokens, reserving 4,096 for output. Video-MMMU cannot be
fully evaluated under this policy on these checkpoints. Its overflow questions
must remain blocked unless a separate lower-resolution/rate policy is approved.
Video-MME currently lacks **253 of 900** referenced videos and is blocked.
LongVideoBench has no local annotations; MVBench/MLVU are unavailable locally.
No partial-population Video-MME/Video-MMMU score is presented as a full result.

GPU smoke validation passed for **all ten checkpoints** on one TempCompass
question each, with identical input-token hashes and EOS-complete generation.
A separate LoRA-GRACE W4 check successfully processed and generated from
**10,997 input tokens** (10,752 visual + 245 text), exceeding its saved tokenizer's
4,096-token training setting without truncation. The model's native context is
262,144; the tokenizer's generic length warning did not prevent inference.
A real Video-MME ZIP member was also probed and cleaned up successfully.

These are implementation checks; **full GPU benchmark scoring has not run**.
The compact, preserved evidence is
[video_evaluation_validation.json](../experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/reports/video_evaluation_validation.json).
