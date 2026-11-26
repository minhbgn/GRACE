# lmms-eval video evaluation (current protocol)

Download the required local assets with the [all-in-one downloader](README_download_all_data.md),
including Video-MME short/medium clips without downloading long-video payloads.

The separate [QAT INT4 × token-reduction study](../../experiments/qat_int4_token_reduction_576_seed42/README.md) uses the shared 576-visual-token image budget extended to video. Its own launch scripts and outputs are independent of the 720p protocol below.

The experiment’s `run_video_evaluation.sh` invokes the same central Python file
as image lmms evaluation: `evaluation/runners/run_lmms_matrix.py --video`.
Experiment bash scripts choose checkpoints, GPUs and output locations. Both the
video controller and its isolated worker entry point live in that Python file.
The pinned lmms-eval checkout is unchanged.

Default video model order is `qat_w4`, `qat_w8`, `vanilla_2b`, then the remaining
experiment models in their declared order. Each model attempts all selected
suites before the next model. An explicit `--methods` list overrides this order.
The short/medium wrapper now follows this same full-model default (override with
`VIDEO_METHODS` or `--methods`). The parallel wrapper attempts the three priority
models sequentially before starting its two remaining-model groups; QAT models
use the local GPU and vanilla 2B uses the reference GPU. Failed priority pairs
remain reported as failures and do not prevent the other priority models being
attempted. Completed pairs still skip after artifact verification.

New runs default to **720p**: downsample to a 720-pixel short edge and a
1280-pixel long edge, preserving aspect ratio and never upscaling. Dimensions
round to the checkpoint’s spatial patch grid (for example 1280×704). The former
policy capped the *long edge* at 256 pixels. Sampling remains 2 fps across the
whole clip, including both endpoints, without the stock adapter’s 32-frame cap.
Inputs reserve 4096 output tokens and use each checkpoint’s native context.
Context overflows and OOMs fail explicitly; there is no automatic downsampling
or truncation. Higher resolution can exceed context even after long exclusion.

| Population | Questions | Scoring |
| --- | ---: | --- |
| TempCompass multi-choice | 1580 | Explicit local deterministic fallback using `local_scoring.choice`; unresolved is wrong |
| VideoMME short + medium | 1800 | Pinned lmms-eval local scorer; 900 long questions excluded |
| Video-MMMU perception | 300 | Pinned lmms-eval local MC/open scorer |
| Video-MMMU comprehension | 300 | Pinned lmms-eval local MC/open scorer |
| Video-MMMU adaptation | 300 | Pinned lmms-eval local MC/open scorer |

TempCompass MC’s upstream parser can call an API judge. This runner substitutes
the existing deterministic local choice parser and labels the result as a local
fallback, so it must not be presented as the official judged metric. Captioning
is skipped because no local captioning scorer was found. TempCompass yes/no and
caption matching have no annotations in the current local population and are
skipped. LongVideoBench and VideoMME long are excluded. These decisions are saved
in plans, protocol manifests and comparisons. No API key is needed, and HTTP
requests through `requests` are blocked during evaluation, even if a key exists.
Other suites retain the pinned harness’s task prompts, parsers and aggregation.
No subtitles or audio are supplied.

```bash
RUN=reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/run_video_evaluation.sh
bash "$RUN" plan
bash "$RUN" check --methods vanilla_2b
bash "$RUN" prepare --methods vanilla_2b     # CPU token/context preflight
bash "$RUN" run --methods vanilla_2b --smoke 1 --gpu 2
bash "$RUN" run --gpu 2                    # all configured models, all five populations
bash "$RUN" summarize
```

Choose an idle physical GPU before running. `check` builds real local harness
tasks and validates checkpoint files without loading weights. `prepare` probes
all selected inputs and exits unsuccessfully on decode errors or context
oversize; use `--smoke N` for a smaller isolated preflight. `--shortest-side` and
`--longest-side` set alternate ceilings, and `--fps`, `--max-new-tokens` and
`--attn` are pinned protocol controls. For the historical 256-pixel long-edge
geometry, explicitly pass `--shortest-side 256 --longest-side 256`.

New outputs use `video_evaluation/lmms_video_v1_<policy_hash>/{full,smoke_N}`.
Historical local results are preserved and cannot resume into this namespace.
A completed pair is skipped only after artifact hashes validate. An incomplete
pair restarts in a fresh attempt; the old resilient local runner’s automatic
partial-prefix recovery is not used by the lmms controller. Per-GPU locks and
occupancy checks prevent launch on an occupied card.

Each attempt saves harness `results.json` and `samples.jsonl`, plus durable
`telemetry.jsonl` with frame indices, processed dimensions, exact visual/text
input counts, output token IDs, EOS status, synchronized generation time,
end-to-end time, and peak allocated GPU memory. End-to-end time for the first
question includes lazy weight loading; use generation time for inference
comparisons. QAT W4/W8 exports are loaded as merged BF16, so these are not packed
low-bit latency measurements. Completion requires full selected coverage,
finite correctly scaled metrics and EOS-complete generations. Model/tokenized
input identities and pinned source/data provenance guard comparisons.

CPU validation (no GPU or judge):

```bash
reproduction/GRACE/.venv-lmms-eval/bin/python -m unittest discover \
  -s reproduction/GRACE/evaluation/tests -p 'test_lmms*.py' -v
reproduction/GRACE/.venv-lmms-eval/bin/python \
  reproduction/GRACE/evaluation/tests/cpu_lmms_video_integration.py
```

The [2026-10-26 CPU validation](../../experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/reports/lmms_video_720p_validation_20261026/validation.md)
passed all five harness paths and real adapter tokenization with a fake CPU
decoder. Its one-question-per-suite 720p preflight found 312576 visual tokens
for the first Video-MMMU clip in each split, exceeding the native 262144 context.
Full 720p baseline evaluation therefore cannot complete those populations under
the unchanged 2 fps policy; prepare reports that blocker without changing inputs.
No GPU inference was run for this protocol update.

## Suggested additional suites (not registered in this wrapper yet)

These tasks exist in the pinned lmms-eval checkout, but adding their names to
`--suites` alone will not work. The current wrapper supports only the five
populations listed above. Recommendations below do not download or enable new
benchmarks automatically.

| Suggested addition | Pinned harness name | Why add it | Local evaluation |
| --- | --- | --- | --- |
| TVBench | `tvbench` (10 individual tasks) | Temporal order, counting and motion; particularly useful for checking what token reduction loses | Deterministic MC accuracy, no judge |
| Perception Test validation MC | `perceptiontest_val_mc` | Short clips, averaging 23 seconds; fine visual details and memory | Local labelled validation accuracy, no judge |
| NExT-QA MC test | `nextqa_mc_test` | Clips average 44 seconds; causal and temporal action reasoning | Local MC exact match; replace and label the pinned parser’s random unresolved fallback |
| MVBench | `mvbench` (20 individual tasks) | Broad temporal skill coverage and common comparison point | Local MC scoring, no judge; document aggregation differences |
| EgoSchema labelled subset (optional medium-duration stress test) | `egoschema_subset` | Fixed three-minute egocentric clips; 500 publicly labelled questions | Local subset accuracy; full set requires benchmark submission |

I would start with **TVBench + Perception Test validation + NExT-QA MC**, then
add MVBench for wider comparison. EgoSchema is a later memory stress test; its
paper calls it long-form, although its fixed three-minute clips fall below the
very long clips being excluded here. Clip duration is not a GPU-fit guarantee.

Primary references: [TVBench](https://github.com/daniel-cores/tvbench),
[Perception Test](https://deepmind.google/blog/measuring-perception-in-ai-models/),
[NExT-QA](https://doc-doc.github.io/docs/nextqa.html),
[MVBench](https://arxiv.org/abs/2311.17005),
[EgoSchema and its public subset](https://github.com/egoschema/EgoSchema).

### How to add them while retaining this protocol

1. Download annotations and media in a separate preparation step and pin the
   repository revision and annotation hashes. The existing downloader currently
   targets the original suites and needs additional dataset entries. Keep raw
   annotation fields for the upstream prompt and scorer, and index video paths
   with unambiguous IDs. MVBench/TVBench require their nested per-task media
   layout; TVBench’s Action Antonym videos require a separate NTU RGB+D download.
2. Extend `protocols/video_protocol.py` (`SUITES` and `load_suite`) with canonical
   rows and expected population coverage. Each row must retain a stable question
   ID, video ID, local asset, prompt, options, gold answer, question type and
   category. Index by relative path/subtask where basenames collide. Preserve
   benchmark-provided clip boundaries and frame timing; do not treat a frame
   directory as an MP4 without a declared conversion policy.
3. In the central `runners/run_lmms_matrix.py`, extend `VIDEO_TASKS`,
   `VIDEO_NOTES`, `video_documents`, and `video_task`. Match the upstream split:
   Perception Test uses `validation`; NExT-QA MC and EgoSchema subset use `test`;
   MVBench entries use `train` as their evaluation annotation split. The current
   local `video_task.download` hardcodes a `test` dataset, so it must become split
   aware before these additions. Generalize the current three-family annotation
   routing in `video_documents` rather than using its Video-MMMU default branch.
4. Register **individual tasks** for MVBench/TVBench, because `VIDEO_TASKS`
   expects one numeric primary metric per task, not a group YAML. Preserve
   per-task scores and add an explicit aggregate for the selected/full group.
   The current primary metric entries would be:

   | Task | YAML relative to `lmms_eval/tasks/` | Metric key | Fraction-to-percent multiplier |
   | --- | --- | --- | ---: |
   | Perception Test val MC | `perceptiontest/val/perceptiontest_mc.yaml` | `accuracy,none` | 100 |
   | NExT-QA MC test | `nextqa/nextqa_mc_test.yaml` | `exact_match,none` | 100 |
   | EgoSchema subset | `egoschema/egoschema_subset.yaml` | `score,none` | 100 |
   | Each MVBench subtask | `mvbench/mvbench_<subtask>.yaml` | `mvbench_accuracy,none` | 1 |
   | Each TVBench subtask | `tvbench/tvbench_<subtask>.yaml` | `tvbench_acc,none` | 100 |

   NExT-QA’s utility import also tries to download NLTK resources for its
   open-ended evaluator. Prepare those dependencies separately or use a scoped
   MC-only loader to avoid downloads during offline evaluation. Its unresolved
   MC fallback randomly selects an answer; use the existing deterministic local
   choice parser with unresolved counted wrong, and label the difference.
   MVBench’s pinned aggregation excludes empty raw predictions from its
   denominator. Either retain and disclose that metric or additionally report a
   clearly labelled full-population accuracy; do not silently change it.
5. Add a CPU harness integration using real local annotations and dummy answers,
   then run `check`, a small `prepare --smoke N`, and a GPU smoke test before a
   full run. Keep the 720p / 2 fps policy, explicit subset notes, network guard,
   token telemetry and context-failure checks. Any duration-filtered population
   must have its own pinned IDs/counts and subset label.

After an adapter is registered, use its new wrapper suite ID in this sequence
(the suite names here illustrate the recommended IDs; they are not enabled yet):

```bash
bash "$RUN" check --suites perceptiontest_val_mc --methods qat_w4,qat_w8,vanilla_2b
bash "$RUN" prepare --suites perceptiontest_val_mc --methods qat_w4 --smoke 2
bash "$RUN" run --suites perceptiontest_val_mc --methods qat_w4,qat_w8,vanilla_2b --smoke 2 --gpu 2
```

The material below documents the **historical local 256-pixel protocol** and
its saved results; it does not describe the new lmms 720p measurements.

---

# Offline video evaluation

`run_video_matrix_resilient.py` evaluates every model in the selected experiment JSON,
including both vanilla models, SFT, released BF16/W4/W8, and local LoRA-QAT and
LoRA-GRACE W4/W8. It uses Transformers inference on the existing unpacked
checkpoints. It never starts training or calls a paid judge.

The experiment's `run_video_evaluation.sh` now selects this resilient v3 runner.
The original v2 runner/worker remain available for historical runs already in
progress. New runs use a separate output namespace and do not append to v2
results. The video input policy and scoring are unchanged.

## Recovery and memory fixes (2026-10-23)

The default worker environment includes
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before Torch is imported.
This fixed the reproduced QAT W4 MLP OOM on VideoMME `618-1` without changing
its 151,218 input tokens. `--cuda-alloc-conf` selects another explicitly pinned
allocator setting for a separate comparison.

The historical zero-prediction local-model failures were SIGKILLs. Python
cannot prevent a kernel or external process from sending SIGKILL. The new
runner makes those failures recoverable and diagnosable:

- Each prediction is flushed and synced to disk. Rerunning the **same command**
  copies a validated, ordered prefix into a new attempt and starts at the next
  question. Only the final incomplete JSON write may be dropped. Source/model
  provenance, input identity, scores, EOS completion, context and population
  order are checked; changed code or protocol requires a new output directory.
- A SIGKILL triggers **one automatic retry** from saved progress by default.
  `--sigkill-retries 0` disables it. A repeated kill stops that matrix rather
  than launching all remaining checkpoint/suite pairs into the same condition.
- `worker.json`, `events.jsonl`, `status.json`, and parent-written `exit.json`
  retain the PID, last phase/question, signal/return code, allocator setting and
  host/cgroup memory counters even when the worker cannot write an exception.
  `resume.json` records which attempt supplied the retained predictions.
- A per-GPU lock coordinates resilient matrices across output directories.
  Before every worker, the runner rejects an occupied GPU, including jobs from
  older runners and other software. The lock is cooperative; it cannot reserve
  hardware against unrelated programs or prevent an external kill.

These checkpoints remain **merged BF16 research exports** at inference time.
Their QAT W4/W8 metadata does not enable packed low-bit inference in this
runner. Runtime provenance now records that loading format explicitly.

To run the four local checkpoints on a free PCI GPU, from the workspace root:

```bash
EXP=reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42
bash "$EXP/run_video_evaluation.sh" run --gpu 0 \
  --methods qat_w4,grace_w4,qat_w8,grace_w8
```

For a short startup check, append `--suites tempcompass --smoke 3`. Use a tmux
session for a full run that must outlive an SSH connection. Choose a free GPU
immediately before launch; no GPU processes are automatically killed.

The default experiment is `released_bf16_init_32k_b128_p589824_t4096_seed42`.
Video evaluation does **not** inherit its 576-image-token training ceiling,
4,096-token training sequence length, or the individual checkpoints' native
adaptive video pixel budgets.

## Default input policy — all suites

- Sample **2 frames per second**, uniformly over the entire video, including
  the first and last frames. Round the requested frame count up and never
  exceed the number of source frames. There is **no maximum frame count**.
- Use a **256 px maximum longest side for every suite, including TempCompass**.
  Preserve aspect ratio before rounding to the 32-pixel patch grid (minimum 32
  per dimension); smaller source frames are not deliberately enlarged. This is
  spatial downsampling, with no temporal clipping or fixed per-video token cap.
- Disable processor resizing and its internal frame sampler. The processor's
  per-video pixel bound is set above the actual padded video's total pixels.
  This avoids Qwen's normal duration-dependent spatial reduction.
- Pass explicit source frame indices and source fps so timestamps are real.
  Only temporal patch padding duplicates the last sampled frame, as required
  by Qwen's two-frame temporal patches.
- Disable tokenizer truncation and padding. Use each model's real native context
  from `text_config.max_position_embeddings`, not the training sequence length.
  If input plus the generation allowance exceeds it, stop and record a blocker;
  do not automatically shorten the video, reduce the configured resolution,
  drop prompt text, or omit the question.
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
The default is `--longest-side 256`. Higher-resolution experiments must pass an
explicit value, such as `--longest-side 320`; `--longest-side 0` explicitly selects
source resolution. The same selected value applies to every suite in the run.
Policy changes automatically produce different output directories.

Visual tokens are determined by the input representation, not chosen by the
language model. These checkpoints do not implement adaptive visual-token
selection. Identical videos under the same policy should therefore have equal
visual-token counts for vanilla/QAT/GRACE. Accuracy and generated-text length
can differ. Bypassing adaptive per-video budgets prevents a fixed total-token
ceiling from saturating the input means; the selected frame resolution remains
part of the declared input policy, rather than model-specific token demand.

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
file. Extraction never downloads additional data. Extracted MP4s take precedence
over copies in ZIP archives, so finishing extraction needs no code or data-path
override. Video-MME's current extracted layout is
`evaluation/video_benchmarks/VideoMME/data/data/<videoID>.mp4`; annotations remain
in the neighboring `videomme/` directory.

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

### Skip VideoMME's 900 long questions, all models on GPU 2

The reduced-population launcher keeps **1,800 VideoMME short/medium questions**
and excludes the final 900 questions (`601-1` through `900-3`). It checks the
annotation count and duration labels before selecting this subset. TempCompass
and all three Video-MMMU populations remain unchanged: **4,280 questions per
model, 42,800 across all ten models**, sequentially on GPU 2.

From the workspace root:

```bash
RUN=reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/run_video_evaluation_short_medium.sh
bash "$RUN" plan --gpu 2 --methods ""
bash "$RUN" check --gpu 2 --methods ""
# Run in tmux to survive SSH disconnection:
bash "$RUN" run --gpu 2 --methods ""
# Rerun the same command to resume; summarize without loading models:
bash "$RUN" summarize --methods ""
```

With `--methods ""`, `plan` lists the selected populations, all ten model paths, 50 model/suite
pairs and the output directory. `check` also validates checkpoint provenance.
The runner refuses to launch a worker if GPU 2 is occupied. `--methods` and
`--suites` remain available for explicit smaller runs; `--smoke 1` applies
after the long-question exclusion. For just the four local models, append
`--methods grace_w4,grace_w8,qat_w4,qat_w8`.

The equivalent Python command is:

```bash
reproduction/GRACE/.venv-lmms-eval/bin/python \
  reproduction/GRACE/evaluation/runners/run_video_matrix_resilient.py run \
  --gpu 2 --skip-videomme-long
```

The subset is pinned in the policy and dataset manifests, including excluded
question IDs and the original full-population identity. Its policy hash selects
a separate output directory automatically. Use a fresh `--output` if overriding
the default; previous full-population runs cannot be resumed into this subset.
The `full/` directory here means the full **selected** population. Comparison
Markdown and JSON explicitly label the VideoMME short/medium subset; its score
is not a 2,700-question VideoMME score. Sampling, inference and SIGKILL recovery
retain the resilient runner defaults above.

### Skip previously completed evaluations

Within the same output directory, `run` automatically validates and skips each
model/suite pair with `COMPLETE.json`. Interrupted pairs resume saved progress.
Historical directories require explicit reuse because the subset policy has a
different output hash. To reuse the previous full v2 run across all suites:

```bash
EXP=reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42
RUN="$EXP/run_video_evaluation_short_medium.sh"
OLD="$EXP/video_evaluation/video_unsaturated_v2_3a74c7231e27/full"

# List completion-marker candidates without writing or loading models:
bash "$RUN" plan --methods "" --reuse-completed-from "$OLD"
# Import compatible completed pairs on CPU; leave other pairs pending:
bash "$RUN" reuse --methods "" --reuse-completed-from "$OLD"
# Run only remaining pairs on GPU 2:
bash "$RUN" run --gpu 2 --methods ""
```

Alternatively, `bash "$RUN" run --gpu 2 --methods "" --reuse-completed-from "$OLD"` imports and then
evaluates remaining pairs in one command. Repeat the option to search multiple
source `full/` directories. Reuse requires matching model file hashes, source
question identities, inference settings, computation code and saved package
versions. Smoke results and incompatible/tampered completions are rejected.
Only runtime orchestration metadata and the declared population may differ.
For completed 2,700-question VideoMME runs, reuse selects the 1,800 requested
predictions and recomputes accuracy, categories and token statistics on CPU.
Source artifacts remain unchanged; `reuse.json` records their origin and policy.
Partial historical attempts are not imported by this option. Once imported,
normal summaries include reused and newly evaluated pairs together.

### Reuse short/medium predictions from unfinished VideoMME runs

An unfinished 2,700-question run can still cover the entire selected subset.
**At least 1,800 predictions** are sufficient when the first 1,800 records
match every short/medium question in order and pass the provenance, score,
token-accounting, context and EOS checks. No full-run `COMPLETE.json` or
`summary.json` is required for this dedicated CPU importer:

```bash
EXP=reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42
bash "$EXP/reuse_partial_videomme.sh" plan
bash "$EXP/reuse_partial_videomme.sh" import
# Imported subsets now have completion markers in the existing short/medium run:
bash "$EXP/run_video_evaluation_short_medium.sh" run --gpu 2 --methods ""
```

The importer scans the experiment's saved full-run directories for every model,
including older retry attempts. It selects a compatible complete prefix from
one attempt per model, recomputes the subset metrics, preserves source files and
records prefix/job/processor hashes in `reuse.json`. A malformed excluded
long-question tail is ignored; missing, reordered, duplicate, corrupt or
non-EOS predictions within the selected subset prevent reuse. Existing target
completions are skipped. This utility does not change the target's pinned
computation or run identity, and it takes the same matrix lock as evaluation.
An active target matrix must finish or be stopped before importing.

Use `--target /path/to/full` to select an initialized target explicitly,
`--source /path/to/full` (repeatable) to restrict sources, and `--methods` to
restrict models. The default target is the unique initialized short/medium run.
`partial_videomme_reuse.json` records imported/pending models and rejection
reasons; `comparison.md` continues to include all suites and all ten models.

### Continue from GRACE W4 on GPU 0

In `reports/video_token_rates_20261023.md`, the model rows from `grace_w4`
onwards are **`grace_w4`, `qat_w8`, `grace_w8`**. The existing launcher now defaults to all
five suites for those three models, sequentially on PCI GPU 0:

```bash
EXP=reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42
RUN="$EXP/run_video_evaluation_short_medium.sh"
bash "$RUN" plan
# Run inside tmux to survive an SSH disconnection:
bash "$RUN" run
# Same command resumes progress and skips completed pairs:
bash "$RUN" summarize
```

This selects 15 model/suite pairs and 4,280 questions per model, including only
1,800 VideoMME short/medium questions. The October 23 token-rate report is a
historical full-population snapshot; the live results are in the existing
short/medium run's `comparison.md`. This launcher uses that same output
directory, so previous validated imports remain available. It does not change
the pinned Python computation or invalidate resume.

The defaults are configured together at the top of that script. Explicit overrides remain available:

```bash
bash "$EXP/run_video_evaluation_short_medium.sh" run \
  --gpu 0 --methods grace_w4,qat_w8,grace_w8
```

`VIDEO_GPU` and `VIDEO_METHODS` can override the defaults of GPU 0 and
`grace_w4,qat_w8,grace_w8`; explicit `--gpu` / `--methods` take precedence.
The runner requires a functioning NVIDIA driver and rejects an occupied GPU
before starting a worker. To regenerate the comparison for all ten models
afterwards, use `bash "$EXP/run_video_evaluation_short_medium.sh" summarize --methods ""`.

### Refresh the saved token-rate table

The CPU auditor combines the current short/medium run and its concurrent GPU 0
output. Compatible older non-VideoMME progress remains eligible; the original
2,700-question VideoMME population is excluded. Each model/suite is counted
once, with its source recorded in JSON/CSV. All nonempty valid partial prefixes
can be included with `--include-partial-over 0`; prefixes containing non-EOS
outputs are shown with coverage but without rates. The existing report filename
is retained, with an updated Singapore timestamp inside:

```bash
EXP=reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42
reproduction/GRACE/.venv-lmms-eval/bin/python \
  reproduction/GRACE/evaluation/audits/audit_video_token_rates.py \
  --experiment "$EXP" \
  --output-prefix "$EXP/reports/video_token_rates_20261023" \
  --include-partial-over 0
```

Repeat `--matrix /path/to/full` to explicitly select compatible source matrices.
This audit reads prediction snapshots without loading models or modifying runs.

The report also shows arithmetic mean visual input, text input, text output,
total text (input plus output), total input (visual plus text) and all tokens
per question, for every benchmark and each VideoMME length. Short/medium use
the current selected population; long videos are excluded from that population.
Validated historical long-video means appear separately, with their observed
coverage, and do not affect current benchmark means or completion counts.
JSON and the category CSV retain those historical long groups with explicit
scope labels. Means from partial runs describe only the observed questions.

The question-length section separates annotation question stems, questions with
choices, and the full task prompt including instructions. It reports full
annotation means for TempCompass, every Video-MMMU split, and all three
VideoMME lengths, even when inference is incomplete. The per-model tables also
show question-only means on their validated observed samples. Counts are
computed on CPU with special-token insertion disabled, and all model tokenizer
variants are checked for matching counts. Full text input additionally includes
timestamps and vision/chat markers: roughly one timestamp/boundary group per
video second under the 2 fps, two-frame temporal patch policy. The residual
column subtracts independently tokenized task-prompt length from complete
Vanilla 2B text input, so it also includes token-boundary effects.

### Master input-token table across image and video suites

[Master suite tokens](../../experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/reports/master_suite_tokens.md)
combines all seven image suites with TempCompass, VideoMME short/medium and
the three Video-MMMU splits. Means are per question over each whole population.
Text means nonvisual **input** tokens, including formatting and timestamps;
generated answer tokens are excluded. VideoMME long is excluded.

Image values use the QAT W4 native 576-token-per-image ceiling and the saved
whole-suite visual audit; text inputs are reconstructed with the real prompt
builders and chat template. Video values use complete Vanilla 2B reference
populations at 2 fps / 256 px and can project input usage under that same policy.
The CSV retains raw-precision means and the JSON records source hashes and
cross-model checks. Reproduce with the existing image auditor:

```bash
EXP=reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42
reproduction/GRACE/.venv-lmms-eval/bin/python \
  reproduction/GRACE/evaluation/audits/audit_image_visual_tokens.py \
  --processor "$EXP/w4g128/runs/qat" \
  --visual-cache "$EXP/reports/inference_image_visual_tokens.json" \
  --video-report "$EXP/reports/video_token_rates_20261023.json" \
  --master-output-prefix "$EXP/reports/master_suite_tokens"
```

### All questions

From the workspace root, this **single full-run command** evaluates all ten models
on TempCompass, Video-MME and all three Video-MMMU sets at **2 fps / 256 px**:

```bash
bash reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/run_video_evaluation.sh run --gpu 2
```

GPU 2 is an example: choose an available PCI GPU index with `gpuwho`.
Omitting `--suites` and `--methods` selects all five available tasks and all ten
checkpoints (50 model/task pairs), sequentially on one GPU. There is no separate
TempCompass configuration. Generation remains greedy with 4,096 output tokens
and each checkpoint's native 262,144-token context.

Optional planning, CPU preparation, a one-question-per-task smoke run and summary
use the same defaults and output base:

```bash
RUN=reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/run_video_evaluation.sh
bash "$RUN" plan
bash "$RUN" check
bash "$RUN" prepare
bash "$RUN" run --smoke 1 --gpu 2
bash "$RUN" summarize
```

Later experiments select resolution explicitly; the policy hash automatically
separates their outputs:

```bash
bash "$RUN" run --longest-side 320 --gpu 2
# Explicit source-resolution input for a historical-policy comparison:
bash "$RUN" prepare --longest-side 0
```

Full-population projections from the saved source-size/frame-count audits show
**zero context overflows across all 5,180 questions at 256 px**, including the
4,096-token output reserve. The largest proposed input is 152,914 tokens. These
are CPU projections, not a fresh full decode or GPU-memory guarantee. See
[resolution feasibility](../../experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/reports/video_resolution_feasibility_20261017.json).
The longest vanilla-8B input still needs memory profiling before committing to
the full matrix; the runner records OOM failures without reducing inputs.

Video-MME's 900 videos / 2,700 questions are complete and extracted; the runner
detects them automatically. Prior source-resolution overflow counts are
historical evidence, not blockers attributed to the new 256 px default.

`--methods` selects comma-separated experiment IDs; default all ten. `--suites`
selects comma-separated IDs from the table. `LMMS_PYTHON` selects a different
Python executable; default the existing `.venv-lmms-eval/bin/python`.
`--attn sdpa` creates a separate protocol if FlashAttention 2 is unavailable.
Inspect logs and blockers before requesting a full matrix. An input fitting
context does not guarantee its vision encoder fits GPU memory; OOMs fail the
job without automatic downsampling.

Output defaults to `<experiment>/video_evaluation/video_resilient_v3_<policy_hash>/`
with separate `full/` and `smoke_N/` directories. The runner pins source code,
package versions, experiment configuration, source populations and policy.
Per-model provenance hashes include checkpoint weights. Changed provenance
rejects resume; use a fresh `--output` base after changes. Only one writer can
hold an output directory at a time. Failed attempts remain reviewable; rerunning
creates a new attempt and retains validated predictions from that pinned v3
run. Historical v2 partial predictions are not automatically imported.
Completed jobs are revalidated before skipping them.

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

## Historical source-resolution validation on 2026-10-17

The evidence below used the earlier **2 fps / source-resolution** policy,
not the new unified 256 px default. GPU smoke/preflight evidence was recorded
with v1 before the directory reorganization. Its original source bytes are preserved in the
[layout snapshot](../provenance/layout_20261017/manifest.json). The organized v2
code uses a fresh output namespace; layout validation uses CPU regression tests.

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

The context is 262,144 tokens, reserving 4,096 for output. These overflow counts
apply to the archived source-resolution policy. The user subsequently selected
256 px uniformly for all suites; its CPU projections fit context.
The earlier Video-MME audit lacked **253 of 900** referenced videos. After the
download completed, the source-resolution audit found 1,623/2,700 overflows and
zero first-frame/metadata probe errors; see the
[saved audit](../../experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/reports/videomme_input_audit_20261017.json).
All 900 extracted videos are now present.
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
[video_evaluation_validation.json](../../experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/reports/video_evaluation_validation.json).
