# Released-SFT QAT/GRACE with a 576-visual-token ceiling

**Completed and audited 2026-10-12:** all four production runs completed 256 updates and **28/28 local evaluation pairs** are validated. Read the [completed experiment report](experiment_report.md) for the comparison with the evaluation-only baseline, per-task/category results, data comparability and costs. New seven-suite means: **QAT W4 75.77, GRACE W4 74.56, QAT W8 77.37, GRACE W8 74.86**. These improve on the previous controls, but GRACE still trails QAT by **1.22/2.51 pp**. Production peak reserved memory was **9.55 GiB/GPU for QAT** and **27.13 GiB/GPU for GRACE**; synthetic stress-test values below describe a different input envelope. The preparation/training commands below are retained for reproducibility; completed outputs are protected from overwrite.


Created 2026-10-05 at the user's request, before LLMC+ integration and the next supervisor update. This is an independent copy of the **recipe**, initialized afresh from the authors' released BF16 SFT checkpoint. Historical checkpoints, predictions, subsets and reports remain in the [98-token experiment](../released_bf16_init_32k_b128_p100352_t2048_seed42/README.md).

**Current revision (2026-10-05):** the user raised `MODEL_MAX_LENGTH` to **4,096** before production preparation. The directory retains its original `t2048` name to preserve existing artifact paths; `recipe.env` and `experiment.json` describe the current 4,096-token run. Old 2,048-token memory evidence remains archived.

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

A [CPU preview of 256 historical-subset images](token_budget_preview.json) measured mean/median visual counts of **89.76/88 → 250.84/260**, with the maximum increasing **98 → 576**. At the new cap, 21/256 images had at least 400 tokens. This confirms additional room in this sample, while showing that the typical count is below 576. The completed [full-subset token audit](reports/training_token_audit.json) now measures a mean of **254.64 visual tokens**, median **260**, and maximum **576**.

All four completed QAT/GRACE arms share the newly prepared manifest. The maximum visual count and total sequence limit both increase relative to the old experiment. These changes can alter which examples or complete conversation turns are retained. `subset_comparison.json` records added/removed indices, retained-conversation changes, token-hash changes and order equality versus the previous experiment. Report these before interpreting a cross-resolution difference as an image-budget-only effect. Within this new experiment, all four arms use the same data and processor.

A subsequent [full audit of the old subset](../released_bf16_init_32k_b128_p100352_t2048_seed42/reports/token_lengths.md) found mean/maximum total lengths of **314.50 / 1,858**, and mean/maximum nonvisual lengths of **224.21 / 1,777**; none reached 2,048 and no retained conversations were trimmed. Even forcing every old image to 576 visual tokens would put only **4/32,768** retained conversations above 2,048. That audit motivated the original 2,048-token setup. The user subsequently selected 4,096 for additional headroom; inspect the newly prepared manifest rather than assuming exact identity.

The released SFT initialization has no GDKD/RCKA stage according to the authors' training README. QAT is CE-only adaptation; GRACE adds the unchanged GDKD/RCKA losses and controller. No token pruning is implemented in this experiment.

## Memory feasibility

Run the entire memory preflight with one command:

```bash
EXP=reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t2048_seed42
CUDA_VISIBLE_DEVICES=0,1 bash "$EXP/run_memory_checks.sh"
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
EXP=reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t2048_seed42

# No GPU use: inspect all four commands, then validate/build the full shared subset.
bash "$EXP/run_training.sh" all --train --dry-run
bash "$EXP/run_training.sh" all --prepare-only

# Survey immediately before GPU work. Use two currently free A100 PCI indices.
GPU_GUARD_MINE=/raid/lvchengtao/Minh/Distillation gpuwho

# Optional additional profile on actual retained data, with the production batch.
# Three steps do not finish the production 25-step RCKA warmup;
# the synthetic screen separately checks its full-weight memory path.
CUDA_VISIBLE_DEVICES=0,1 PROFILE_STEPS=3 bash "$EXP/run_training.sh" all --profile

# Train all four arms sequentially: GRACE W4, QAT W4, GRACE W8, QAT W8.
CUDA_VISIBLE_DEVICES=0,1 bash "$EXP/run_training.sh" all --train
```

Indices `0,1` are examples, not a reservation. The wrapper sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`. It also clears old experiment/student/teacher overrides to avoid accidentally targeting the previous run. Reference model bytes are shared and verified; production outputs are new `w4g128/runs/{qat,grace}` and `w8g128/runs/{qat,grace}` directories.

Alternatively launch individual arms; this is also how to continue after a paused or completed arm:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash "$EXP/run_training.sh" qat --bits 4 --train
CUDA_VISIBLE_DEVICES=0,1 bash "$EXP/run_training.sh" grace --bits 4 --train
# Resume only an unfinished arm with an existing native checkpoint:
CUDA_VISIBLE_DEVICES=0,1 bash "$EXP/run_training.sh" grace --bits 4 --train --resume
```

Repeat with `--bits 8` as needed. Completed exports are intentionally protected from overwrite. If `all --train` stops, inspect `run_summary.json`; launch/resume the unfinished arm individually rather than rerunning completed arms. Save completed per-bit runtime reports alongside the evaluation report. Do not reuse the old experiment's completion markers or resumable checkpoints.

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

## Supervisor report checklist

- Confirm all four exports completed 256 updates, and include the actual elapsed time, GPU-hours and peak memory.
- Report the resolved image/sequence limits and measured token distribution; keep the changed input budget visible next to old results.
- Summarize `subset_comparison.json`, including conversation trimming and replacements.
- Present all seven task scores, unresolved/truncated generation rates and the paired GRACE−QAT differences at each bit width.
- Compare against the previous experiment's **latest local greedy-256 evaluation**, not its older seven-suite or four-suite harness scores.
- Keep any cross-resolution improvement separate from a token-reduction claim: this experiment establishes the new unpruned controls.
- Preserve the known native-versus-merged/baked export caveat; packed low-bit deployment speed/memory is not established by these research exports.
