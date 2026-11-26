# Released-SFT QAT/GRACE with a 576-visual-token ceiling

Created 2026-10-05 at the user's request, before LLMC+ integration and the next supervisor update. This is an independent copy of the **recipe**, initialized afresh from the authors' released BF16 SFT checkpoint. Historical checkpoints, predictions, subsets and reports remain in the [98-token experiment](../released_bf16_init_32k_b128_p100352_t2048_seed42/README.md).

**Current revision:** the user raised `MODEL_MAX_LENGTH` to **4,096** before production preparation on 2026-10-05. This directory uses the `t4096` name; `recipe.env` and `experiment.json` describe the current 4,096-token run. Old 2,048-token memory evidence remains archived.

## What changes

| Setting | Previous experiment | This experiment |
| --- | --- | --- |
| Maximum processed image area | 100,352 pixels | **589,824 pixels** |
| Maximum merged visual tokens | 98 | **576** |
| Minimum image area | 4,096 pixels | 4,096 pixels |
| Total training sequence limit, including image/text/answer tokens | 2,048 | **4,096** |
| Subset target / training seed / subset seed | 32,768 / 42 / 2026 | Same |
| Global batch / per-device microbatch / accumulation / GPUs | 128 / 1 / 64 / 2 | Same |
| Optimizer updates / RCKA warmup | 256 / 25 | Same |
| Precision arms | W4G128 and W8G128 | Same |
| Initialization / teacher | Released 2B BF16 SFT / vanilla 8B | Same |
| LoRA / optimizer / CE, GDKD and RCKA settings | Existing matched recipe | Same |

For Qwen3-VL's 16-pixel patches and 2×2 spatial merging, `589824 / (16×2)² = 576`; a 768×768 processed square reaches the ceiling. Dynamic resizing preserves aspect ratio and does not force small images up to the ceiling. **576 is a maximum, not a fixed count per image.** Matching LLaVA's nominal token count does not equate its image representation with Qwen3-VL's, and neither 98 nor 576 alone establishes redundancy.

A [CPU preview of 256 historical-subset images](token_budget_preview.json) measured mean/median visual counts of **89.76/88 → 250.84/260**, with the maximum increasing **98 → 576**. At the new cap, 21/256 images had at least 400 tokens. This confirms additional room in this sample, while showing that the typical count is below 576. It is not a measurement of the as-yet-unprepared new full subset.

Both QAT and GRACE will share a newly prepared manifest. The maximum visual count and total sequence limit both increase relative to the old experiment. These changes can alter which examples or complete conversation turns are retained. `subset_comparison.json` records added/removed indices, retained-conversation changes, token-hash changes and order equality versus the previous experiment. Report these before interpreting a cross-resolution difference as an image-budget-only effect. Within this new experiment, all four arms use the same data and processor.

A subsequent [full audit of the old subset](../released_bf16_init_32k_b128_p100352_t2048_seed42/reports/token_lengths.md) found mean/maximum total lengths of **314.50 / 1,858**, and mean/maximum nonvisual lengths of **224.21 / 1,777**; none reached 2,048 and no retained conversations were trimmed. Even forcing every old image to 576 visual tokens would put only **4/32,768** retained conversations above 2,048. That audit motivated the original 2,048-token setup. The user subsequently selected 4,096 for additional headroom; inspect the newly prepared manifest rather than assuming exact identity.

The released SFT initialization has no GDKD/RCKA stage according to the authors' training README. QAT is CE-only adaptation; GRACE adds the unchanged GDKD/RCKA losses and controller. No token pruning is implemented in this experiment.

## Memory feasibility

Run the entire memory preflight with one command:

```bash
EXP=reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42
CUDA_VISIBLE_DEVICES=0,4 bash "$EXP/run_memory_checks.sh"
```

Choose two currently free A100 PCI indices. Each missing GPU check surveys ownership before launching; no workloads are reclaimed. The command reads `recipe.env` and runs GRACE W4/W8 then QAT W4/W8, with **576 visual / 4,096 total tokens** for the current settings. It exits nonzero on failure and does not launch production training or lock its full dataset recipe.

Generated/rebuilt outputs:

- `profile_memory.json`: current resolved production settings, diagnostic overrides, fingerprint and status.
- `memory_results.json`: per-arm measurements and overall `checking` / `failed` / `passed` status.
- [memory_report.md](memory_report.md): human-readable memory evidence and limitations.
- `memory_checks/<fingerprint>/<method>_w<bits>/attempt_*/`: isolated checkpoints, input envelope, GPU survey, measurements and validated completion evidence; the attempt log sits beside the attempt directory.

The fingerprint covers resolved settings, code, source model bytes, historical input manifest and relevant environment versions. Existing reports alone never count as a pass: only matching completed checks with intact hashed evidence and export artifacts are reused. Missing reports are regenerated from those checks; changed settings select a fresh fingerprint. A failed or incomplete attempt is preserved and a subsequent invocation retries it in a new directory. Older summaries are archived before replacing stale current reports. A concurrent second invocation is rejected by the matrix lock.

Each check performs three optimizer updates with diagnostic accumulation **2**, exercising accumulated gradients, and RCKA warmup **1**, exercising full RCKA on updates 2–3. The actual production accumulation remains **64**, warmup **25**, updates **256**. A synthetic long answer forces the requested sequence ceiling; checkpoint saving and merged/baked export are included. These **do not change the production recipe**, and diagnostic models must never be evaluated as trained results. The standalone `profile_memory.sh --method grace --bits 4` remains available for a single fresh check, but use `run_memory_checks.sh` for validated reuse and aggregate reports.

The current maximum-grid generator uses a square image and requires an exact square merged-token grid (576 = 24²). If a future image budget cannot produce that grid, it fails explicitly instead of silently testing fewer visual tokens. Use the actual-data `run_training.sh --profile` for other image-grid budgets.

Full training retains the existing 34-GiB reserved-memory guard and five-hour training-time pause limit. The memory guard checks at optimizer-update boundaries; it cannot prevent a transient OOM. A memory-screen pass is evidence of feasibility, not a guarantee for every training batch or a runtime estimate. A time-limit pause preserves a native checkpoint; resume that individual arm with `--resume`. Do not silently lower the image cap within this directory after preparation. If a lower cap proves necessary, create another named experiment and match both methods at that cap.

## Prepare and train

Run from the workspace root with the existing GRACE environment (the wrapper selects it by default):

```bash
EXP=reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42

# No GPU use: inspect all four commands, then validate/build the full shared subset.
bash "$EXP/run_training.sh" all --train --dry-run
bash "$EXP/run_training.sh" all --prepare-only

# Survey immediately before GPU work. Use two currently free A100 PCI indices.
GPU_GUARD_MINE=/raid/lvchengtao/Minh/Distillation gpuwho

# Optional additional profile on actual retained data, with the production batch.
# Three steps do not finish the production 25-step RCKA warmup;
# the synthetic screen separately checks its full-weight memory path.
CUDA_VISIBLE_DEVICES=0,4 PROFILE_STEPS=3 bash "$EXP/run_training.sh" all --profile

# Train all four arms sequentially: GRACE W4, QAT W4, GRACE W8, QAT W8.
CUDA_VISIBLE_DEVICES=0,4 bash "$EXP/run_training.sh" all --train
```

Indices `0,4` are examples, not a reservation. The wrapper sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`. It also clears old experiment/student/teacher overrides to avoid accidentally targeting the previous run. Reference model bytes are shared and verified; production outputs are new `w4g128/runs/{qat,grace}` and `w8g128/runs/{qat,grace}` directories.

To hold the selected GPUs while the launcher verifies weights and prepares the
subset on CPU, add `--reserve-gpus` to the same invocation:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash "$EXP/run_training.sh" all --train --reserve-gpus
```

The launcher first checks that both selected A100s have at least 36 GiB free.
A dummy process then allocates their available VRAM, leaving about 512 MiB per
GPU, and waits without a compute loop. It exits before the training GPU check
and launch. Errors, Ctrl-C, and SIGTERM release the holder; closing the parent
process also closes its lifetime pipe. This discourages competing launches but
does not provide scheduler-enforced exclusivity; a short handoff gap remains.
`--dry-run` never reserves GPUs. With `--prepare-only`, the reservation lasts
only until preparation finishes, so combine reservation and training in one
command when you want an immediate handoff.

Alternatively launch individual arms; this is also how to continue after a paused or completed arm:

```bash
CUDA_VISIBLE_DEVICES=0,4 bash "$EXP/run_training.sh" qat --bits 4 --train
CUDA_VISIBLE_DEVICES=0,4 bash "$EXP/run_training.sh" grace --bits 4 --train
# Resume only an unfinished arm with an existing native checkpoint:
CUDA_VISIBLE_DEVICES=0,4 bash "$EXP/run_training.sh" grace --bits 4 --train --resume
```

Repeat with `--bits 8` as needed. Completed exports are intentionally protected from overwrite. If `all --train` stops, inspect `run_summary.json`; launch/resume the unfinished arm individually rather than rerunning completed arms. Save completed per-bit runtime reports alongside the evaluation report. Do not reuse the old experiment's completion markers or resumable checkpoints.

## Evaluation model selection — 2026-11-01

The current image/video launchers use `evaluation.json`, which excludes the previous **local `sft_2b`** checkpoint and its comparison rows. The author-released **`released_bf16`** reference remains included. `experiment.json` remains the original training/provenance configuration; its historical SFT entry does not select a model in these evaluation launchers.

New default outputs are `local_development_layout2_no_local_sft_v1/`, `video_evaluation/no_local_sft_v1/`, and `video_evaluation/lmms_parallel_no_local_sft_v1/`. These keep the new model selection separate from existing pinned results. Training checkpoints and historical reports are retained. The independent `qat_int4_token_reduction_576_seed42` experiment already excludes local SFT and needs no selection change.

## Latest seven-suite evaluation

The local wrapper defaults to the four new LoRA exports and **`local_native_greedy256_v1`**: native saved processors, deterministic greedy decoding, 256 generated-token cap, corrected deterministic scoring. Suites: AI2D, HallusionBench, MMBench, MMStar, MMMU, ScienceQA and SEED. This is the current local development protocol; it is not a claim of paper-protocol reproduction.

```bash
# Preview before training; inspect/validate actual completed exports afterwards.
bash "$EXP/run_local_evaluation.sh" plan
bash "$EXP/run_local_evaluation.sh" check

GPU_GUARD_MINE=/raid/lvchengtao/Minh/Distillation gpuwho
# One free A100 per evaluation worker. Smoke and full outputs are separate.
bash "$EXP/run_local_evaluation.sh" run --gpu 0 --smoke 2
bash "$EXP/run_local_evaluation.sh" run --gpu 0
bash "$EXP/run_local_evaluation.sh" summarize
```

Use `--methods qat_w4,grace_w4` for a completed W4 pair, or `--methods all` to include all shared reference checkpoints. References keep their native processor budgets and therefore are not resolution-matched controls. Add the same method selection to `check`, `run` and `summarize`. Evaluation output is `local_development/local_native_greedy256_v1/{smoke_2,full}/`; `full/local_scores.md` and `local_scores.json` summarize the full round.

The MMBench validator now compares histogram keys in their JSON representation, fixing integer-versus-string key mismatches without changing scoring. The new run records the updated source hashes in fresh provenance. Historical evaluation artifacts have not been rewritten or marked complete by this fix.

## Measure inference image tokens on CPU

The current local evaluation inherits the saved **576 merged visual tokens per
image** ceiling. It does not force every image to use 576 tokens. The four
QAT/GRACE exports have identical processor configurations, so one audit covers
all four. The 4,096 training sequence limit is not an inference truncation limit;
the current local protocol caps generated responses at 256 tokens.

From the workspace root, run:

```bash
GRACE_ROOT=reproduction/GRACE
EXP="$GRACE_ROOT/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$GRACE_ROOT/.venv-lmms-eval/bin/python" \
  "$GRACE_ROOT/evaluation/audits/audit_image_visual_tokens.py" \
  --processor "$EXP/w4g128/runs/qat" \
  --output "$EXP/reports/inference_image_visual_tokens.json"
```

This loads no model weights and changes no processor settings. It reconstructs
the inference counts from the current datasets using the same image loaders and
selection rules as the seven evaluators, and the installed processor's patch-count
utility. By default, the first two image-bearing questions per suite are also
checked against actual `image_grid_thw` from CPU preprocessing. The reconstruction
is not telemetry recovered from the historical predictions. Existing output files
are protected; choose a new `--output` path for another audit.

The table/JSON reports mean tokens **per image occurrence** and **per question**,
min/max, percentiles and the fraction of images exactly at the ceiling. Repeated
images count each time they are evaluated. Text-only questions contribute zero to
the per-question mean; MMMU questions can contain multiple images, so their total
can exceed 576. The aggregate is weighted by questions/image occurrences, not an
unweighted average of suite means. Use `--limit 2` for a preview or
`--suites ai2d,mmmu` for selected suites. No generation or performance scores are
produced by this audit.

### Video has separate limits

The saved `video_processor` is independent of `image_processor`: it defaults to
**2 FPS, minimum 4 / maximum 8 sampled frames**, `temporal_patch_size=2`,
`patch_size=16`, `merge_size=2`, and size bounds **200,704 / 802,816**. In the
installed Transformers **5.9.0** video processor, these bounds control pixel
volume across the sampled clip (`padded_frames × resized_height × resized_width`),
not a 576-token budget per image or per frame. For an ordinary even-frame clip,
the nominal upper budget is **802816 / (2 × 32²) = 392 merged video tokens**,
often fewer after aspect-ratio/grid rounding. Temporal padding and minimum-grid
constraints mean this arithmetic should not replace measuring actual grids.

For videos the measured count is `sum(prod(video_grid_thw) / merge_size²)`;
`grid_t` already includes temporal patch grouping, so do not divide it by two
again. Video timestamps and surrounding text tokens are additional context,
outside this visual count. If frames are instead passed as independent images,
the image budget applies to each image and totals accumulate across frames.

There is no video task in this experiment's current seven-suite wrapper. Future
video counts depend on the chosen runner: the existing `evaluation/VideoMME/`
vLLM runner and vendored lmms-eval Qwen3-VL adapter set their own frame/pixel
budgets and can override native defaults. Record their resolved settings and
actual `video_grid_thw` before drawing token-reduction headroom conclusions;
simply reusing these model weights does not determine a video's token budget.

## Video evaluation (inference only)

`run_video_evaluation.sh` invokes `evaluation/runners/run_lmms_matrix.py --video`.
The default model order is **QAT W4 → QAT W8 → vanilla 2B → remaining models**;
`--methods` overrides the order. The default input policy is **2 fps / 720p** (720 short edge / 1280 long edge, downsample only,
rounded to patch grid). VideoMME long is always excluded: the matrix covers
TempCompass MC (1580), VideoMME short/medium (1800), and three Video-MMMU sets
(300 each), totaling 4280 questions per model.

VideoMME and Video-MMMU use pinned lmms-eval prompts and local scorers.
TempCompass MC uses a labelled deterministic local scoring fallback instead of
its API-judge fallback. API-only captioning is skipped. No API key is required.
Inputs that exceed context or GPU memory fail explicitly without reducing them.
Historical 256-pixel results are preserved in separate output directories.

```bash
EXP=reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42
bash "$EXP/run_video_evaluation.sh" plan
bash "$EXP/run_video_evaluation.sh" prepare --methods vanilla_2b --smoke 1
bash "$EXP/run_video_evaluation.sh" run --methods vanilla_2b --smoke 1 --gpu 2
bash "$EXP/run_video_evaluation.sh" run --gpu 2
# Separate reference/local matrices on two available physical GPUs:
bash "$EXP/run_video_evaluation_parallel.sh" plan 0 4
bash "$EXP/run_video_evaluation_parallel.sh" run 0 4
```

GPU indices are examples; select idle cards. The parallel launcher attempts
the three priority models sequentially first, then runs the remaining reference
and local models concurrently. It writes separate `references/` and `local/` matrices under
`video_evaluation/lmms_parallel_v1/`; override with `VIDEO_PARALLEL_OUTPUT`.
Reruns skip completed pairs after checking artifact hashes and retry incomplete
pairs from scratch in fresh attempts. Historical local partial predictions
cannot be imported into the new lmms protocol.

The runner saves exact visual/text token counts, EOS completion, frame geometry,
generation time and peak GPU allocation. The exports load as merged BF16;
packed low-bit inference requires a separate backend. See the
[video evaluation guide](../../evaluation/docs/README_video_evaluation.md)
for controls, scoring differences and preserved historical evidence.

## Supervisor report checklist

- Confirm all four exports completed 256 updates, and include the actual elapsed time, GPU-hours and peak memory.
- Report the resolved image/sequence limits and measured token distribution; keep the changed input budget visible next to old results.
- Summarize `subset_comparison.json`, including conversation trimming and replacements.
- Present all seven task scores, unresolved/truncated generation rates and the paired GRACE−QAT differences at each bit width.
- Compare against the previous experiment's **latest local greedy-256 evaluation**, not its older seven-suite or four-suite harness scores.
- Keep any cross-resolution improvement separate from a token-reduction claim: this experiment establishes the new unpruned controls.
- Preserve the known native-versus-merged/baked export caveat; packed low-bit deployment speed/memory is not established by these research exports.
