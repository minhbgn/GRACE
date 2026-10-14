# Constrained LoRA-GRACE from the released BF16 student

**Two-stage local re-evaluation:** [Commands and model lists](README_re_evaluation.md), using [run_re_evaluation.sh](run_re_evaluation.sh). Stage one evaluates the five published references; stage two evaluates the four LoRA checkpoints after the per-suite paper-gap gate passes. All new outputs go under `re_evaluation/`.

**Protocol recovery update (2026-09-25):** The [source audit and new evaluation workflows](../../evaluation/README_paper_protocol.md) recover all seven upstream script recipes and add a separate corrected standard harness protocol. Confirmed differences include checkpoint-native image budgets and MMStar's 64-token launcher limit; MMMU task loading and open-answer scoring bugs are fixed for the new harness protocol. CPU checks passed; no new GPU results or paper-tolerance claim are included here. Historical scores below are unchanged.

**lmms-eval follow-up:** The separate [pinned harness workflow](lmms_evaluation.md) and [launcher](run_lmms_evaluation.sh) cover all ten models and seven benchmarks. New outputs are isolated from the completed custom-protocol results. All 40 GPU evaluations across ScienceQA-image, AI2D, SEED-image and MMStar are complete; the workflow now includes the four-suite analysis and paper comparison. MMMU and the two judge-based suites remain pending.

Status: **all four full training runs and all 70 evaluations completed and validated; experiment report added 2026-09-23.** Plan created 2026-09-20; implementation added and consolidated into shared utilities on 2026-09-21. All required reference checkpoints are available locally. Separate stress-profile artifacts are not present in this experiment folder; feasibility is established by the completed full runs.

Read the **[completed experiment report](experiment_report.md)** for the full method comparison, training diagnostics, resource analysis, discussion and prioritized next steps. GRACE-LoRA averages **73.32% / 73.92% at W4 / W8**, versus **73.85% / 72.28%** for the locally evaluated full-method releases and **74.44% / 76.15%** for matched LoRA-QAT. Each GRACE run takes about **2.39 process hours on two GPUs**, with **26.07 GiB peak reserved memory per GPU**. The constrained stage is feasible, but the intended positive GRACE-versus-QAT mean gain is not achieved at either precision; large MMStar regressions require attention.

The objective was to initialize constrained W4G128 and W8G128 training from the author's released `ForeverBlue/Qwen3-VL-2B-GRACE-BF16`, then measure whether LoRA-GRACE improves over matched QAT-only training and how much quality it loses relative to the author's corresponding full-method releases. The acceptable quality decrease remains for supervisor discussion. The setup and reproducibility instructions below preserve the experiment's design; measured outcomes are in the report and linked result artifacts.

## Why this experiment

The [previous saved recipe](../constrained_32k_b128_p100352_t2048_seed42/comparison_recipe.json) records `models/Qwen3-VL-2B-Instruct` as the student initialization. Its SFT, QAT and GRACE runs were independent branches from that vanilla checkpoint; running `all --train` did not chain SFT into QAT or GRACE.

The released BF16 student initializes the W8/W4 Qwen3-VL runs. The authors explicitly identify [BF16 SFT](../../README.md#1-bf16-sft-baseline), launched by `finetune_qwen3vl_2b_bf16.slurm`, as the source of this checkpoint. Thus `GRACE-BF16` is the released **SFT baseline**, not a BF16-distilled student. Our branches independently start there and perform CE-only QAT or GRACE (CE + GDKD + RCKA with QAT). The broader model-card phrase “trained under the GRACE framework” names the framework and does not change this objective distinction.

The local reference GRACE SLURM script still defaults to vanilla `Qwen3-VL-2B-Instruct`, so merely using a reference launcher does not ensure the intended initialization. The new launcher must explicitly select and verify the released BF16 weights.

The [previous results](../constrained_32k_b128_p100352_t2048_seed42/evaluation/comparison.md) remain useful as the vanilla-initialized pilot: mean scores were 65.45 for vanilla 2B, 67.74 for local SFT, 65.89 for QAT and 64.96 for GRACE. Those numbers do not establish what happens with the released BF16 initialization. Preserve the previous experiment and label any cross-experiment comparison by initialization.

## Models and comparisons

Four fresh training runs and six evaluation-only references give **ten model rows and 70 merged results** across seven suites. The previous local SFT export is included as requested, in addition to the original nine-row proposal; no new SFT training is launched.

| Result ID | Checkpoint or initialization | Training in this experiment | Role |
| --- | --- | --- | --- |
| `vanilla_2b` | `Qwen/Qwen3-VL-2B-Instruct` | None | Original student baseline, BF16 |
| `vanilla_8b_teacher` | `Qwen/Qwen3-VL-8B-Instruct` | None; frozen during distillation | Teacher quality reference, BF16 |
| `sft_2b` | Previous `constrained_32k_b128_p100352_t2048_seed42/runs/sft` export | None | Local vanilla-initialized BF16 SFT reference |
| `released_bf16` | `ForeverBlue/Qwen3-VL-2B-GRACE-BF16` | None | Quality of the exact student initialization |
| `released_grace_w4` | `ForeverBlue/Qwen3-VL-2B-GRACE-W4G128` | None | Author's full-method W4 reference |
| `released_grace_w8` | `ForeverBlue/Qwen3-VL-2B-GRACE-W8G128` | None | Author's full-method W8 reference |
| `qat_w4` | Released BF16 | Constrained LoRA QAT-only, W4G128 | Matched W4 ablation |
| `grace_w4` | Same released BF16 | Constrained LoRA-GRACE, W4G128 | Proposed W4 method |
| `qat_w8` | Released BF16 | Constrained LoRA QAT-only, W8G128 | Matched W8 ablation |
| `grace_w8` | Same released BF16 | Constrained LoRA-GRACE, W8G128 | Proposed W8 method |

Within each bit width, QAT and GRACE must use identical initial weights, subset, order seed, trainable parameter policy, optimizer, input budgets, update count and export procedure. QAT uses CE with quantization; GRACE adds GDKD, RCKA and the adaptive IB controller with the frozen vanilla 8B teacher. This measures the additional benefit of distillation during the constrained stage, conditional on the released initialization.

Each of the four runs starts independently from BF16. Do not initialize W4 from the trained W8 model, use the previous local SFT export as a substitute, or reuse previous QAT/GRACE adapters or optimizer state. Another BF16 SFT stage is not part of this proposal.

Here, “QAT ablation” means the fresh matched local QAT runs above. Previous vanilla-initialized QAT is historical context, not the primary control. An author-released QAT-only checkpoint, if later identified with verifiable provenance, may be an additional reference; no such download is assumed here. The vanilla 8B teacher is a reference, not a guaranteed accuracy ceiling.

## Reference checkpoints acquired

| Source | Intended local directory under `reproduction/GRACE/models/` | Use |
| --- | --- | --- |
| [Released BF16](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-BF16) | `Qwen3-VL-2B-GRACE-BF16/` | Initialization and evaluation |
| [Released W4G128](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-W4G128) | `Qwen3-VL-2B-GRACE-W4G128/` | Evaluation only |
| [Released W8G128](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-W8G128) | `Qwen3-VL-2B-GRACE-W8G128/` | Evaluation only |

Reuse the existing vanilla 2B and 8B directories after validating their identity and completeness. For each checkpoint record repository ID, resolved immutable revision, local path, file hashes, config, tokenizer/processor and generation settings. Download model assets and applicable quantization metadata together. Capture the model cards used for provenance. Verify the released low-bit loading format before evaluation; use the research checkpoints above for the primary comparison, with a consistent Transformers evaluation path. Packed AWQ speed/memory benchmarking is a separate experiment.

## Completed constrained training recipe

The runs retained the [previous larger-input recipe](../../qwen-vl-finetune/scripts/recipes/constrained_32k_b128_p100352_t2048_seed42.env), changing the student initialization and adding explicit W8 support. The settings below are recorded in the completed W4/W8 recipes; measured costs and quality are in the [experiment report](experiment_report.md).

| Setting | Recorded value |
| --- | --- |
| Training data | Same eligible ShareGPT4V mix665k source and source quotas |
| Subset / seeds | 32,768 examples; training seed 42; subset-selection seed **2026** |
| GPUs / per-device batch / accumulation | 2 A100s / 1 / 64 |
| Effective batch / duration | 128 / one pass, 256 optimizer updates per run |
| Image limits | Minimum 4,096; maximum 100,352 pixels |
| Training sequence limit | 2,048 total tokens, including expanded visual tokens |
| Quantization | W4G128 and W8G128, in separate run groups |
| LoRA | Rank 32, alpha 64, dropout 0.05; language q/k/v/o projections |
| Trainable weights | LoRA adapters and BF16 visual merger/projector; quantization scales in LoRA-targeted layers |
| Frozen weights | Language base weights, embeddings, output head, vision tower; other eligible linears use frozen initialized quantization scales |
| Learning rates | Adapter 2e-4; projector 1e-5; quantization scales 2e-4 |
| Schedule | Cosine; 3% LR warmup; weight decay 0; max gradient norm 1 |
| GRACE losses | DKD temperature 2, alpha 1, beta 4; RCKA weight 3, layer -2, warmup 25 updates |
| IB controller | Tau 3, eta 0.003, initial beta 0.5, bounds [0.1, 1.0]; inherited EMA policy |
| Memory controls | BF16, gradient checkpointing, no training KV cache, ZeRO-2; inherited attention implementation |
| Save / budgets | Save every 50 updates, retain two native checkpoints plus final export; inherited 34 GiB peak-reserved and five-hour training-loop budgets per invocation |

Validate tokenizer, chat template and visual-token expansion against the released processor before preparing data. Prefer the previous exact example identities and retained conversations if they remain valid. Revalidate them under the released processor; if eligibility changes, prepare one new shared subset, record the differences and hash, and use it for all four runs. A changed subset would limit attribution of differences from the previous experiment to initialization alone.

Budgets are pause thresholds, not guarantees against OOM or total runtime estimates. Profile each bit width and method before committing to full runs. If a setting must change, update the matched pair and its recorded recipe rather than silently changing one method.

The existing LoRA implementation trains `Q(W)x + LoRA(x)` and exports `Q(W + delta_W)x`. Evaluate the final merged and requantized exports as the primary low-bit artifacts. Retain native checkpoints to diagnose export loss if necessary. BF16 storage of grid-constrained weights does not establish packed W4/W8 inference memory or throughput gains.

## Evaluation protocol

Evaluate all ten rows using the same local seven-suite protocol, dataset versions, question identities, scoring, prompt templates, decoding policy and image budgets. Use the same 4,096/100,352 pixel limits for every row, including teacher and released references. Record tokenizer/processor identity and explicitly verify compatible special tokens and chat templates. Specify processor selection directly; this experiment has no `runs/sft` processor to inherit.

The 2,048-token limit above is a **training** limit. Pin evaluation context handling and per-suite generation caps separately from the existing evaluator; do not silently truncate evaluation questions or change decoding to a checkpoint's own default. If evaluating reference models with their original larger image budgets later, report that as a separate protocol and table.

| Suite | Primary metric | Expected N under the previous local protocol |
| --- | --- | ---: |
| HallusionBench | aAcc | 1,129 |
| MMBench EN dev | Per-row accuracy | 4,329 |
| ScienceQA test | Full-test accuracy, including text-only items | 4,241 |
| AI2D test | Accuracy | 3,088 |
| MMMU validation | Local mixed-answer accuracy | 900 |
| SEED-Bench-IMG | Accuracy | 14,233 |
| MMStar | Accuracy | 1,500 |

Also retain HallusionBench qAcc/fAcc, ScienceQA image/text splits, categories, errors and parse failures. Count failed parses in the denominator. Validate merged prediction files, unique question IDs, matching gold labels and sample counts across all rows; rank shards alone do not establish completion. Investigate any change to the expected counts before comparison.

Use the unweighted mean of the seven primary scores as a proposed descriptive aggregate, with every per-suite score visible. It is not an official leaderboard metric. Reevaluate the author's W4/W8 checkpoints locally: differences from published headline numbers could reflect preprocessing or scoring, so published scores must not be substituted into the primary comparison. Reuse an existing vanilla result only if its full protocol and provenance match; otherwise rerun it.

For each bit width `b`, report percentage-point differences:

- **Distillation gain:** `mean(grace_b) - mean(qat_b)`.
- **Gap to full method:** `mean(released_grace_b) - mean(grace_b)`; positive means constrained training scores lower.
- **Gap to BF16 initialization:** `mean(released_bf16) - mean(grace_b)`.
- Context: differences from vanilla 2B and vanilla 8B, plus the corresponding per-suite differences for every comparison.

Training cost reporting covers these four additional constrained runs: completed updates, trainable parameter counts, measured time, GPU-hours and peak allocated/reserved memory with measurement scope. Report loading/export overhead separately where available. The released initialization inherits the author's prior training; do not describe the constrained-stage cost as end-to-end reproduction cost. The full-method gap also combines differences in data, optimization and parameter updates, so it cannot be attributed solely to LoRA.

## Success and supervisor discussion

The intended outcome is positive GRACE-versus-QAT gain at **each** bit width and an acceptable gap to the corresponding released full-method model under matched local evaluation. Report W4 and W8 independently, including any failed or mixed outcome.

The proposed first comparison is the seven-suite mean with per-suite regressions visible. The acceptable mean decrease, any per-suite regression limits, and whether “better than QAT” requires more than a positive mean difference remain **TBD with the supervisor**. Produce the complete evaluation table before that discussion; do not invent a tolerance or claim overall success until it is agreed. Record the date and that the acceptance rule was set after observing this pilot. Any later confirmatory run should use a frozen rule.

This single-seed experiment supports a pilot conclusion. Additional matched seeds or uncertainty analysis can follow if the supervisor needs a robustness claim. Optional later controls include BF16 continued LoRA-SFT or immediate quantization of the released BF16 model; they are outside the initial four-run scope.

## Running the implementation

Run from the workspace root. The launchers resolve paths relative to their own location.

```bash
export PYTHON=/home/air/miniconda3/envs/grace/bin/python
EXP=reproduction/GRACE/experiments/released_bf16_init_32k_b128_p100352_t2048_seed42

# Optional acquisition: download the three author releases at pinned revisions,
# validate local shards and write provenance/model cards. Network required.
"$PYTHON" reproduction/GRACE/qwen-vl-finetune/scripts/prepare_constrained_models.py \
  --config "$EXP/experiment.json" --download

# Inspect all four full-training commands; no GPU access or artifact writes.
bash "$EXP/run_training.sh" all --train --dry-run

# Validate/select data with the released processor, compare with the old subset,
# and pin both bit-width recipes. CPU only; may take substantial time.
bash "$EXP/run_training.sh" all --prepare-only

# Survey ownership, then select TWO free A100s in PCI order.
gpuwho
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1  # replace with two free GPU IDs

# Profiles are separate artifacts; each method/bit width starts from BF16.
# all runs GRACE before QAT at each bit width.
bash "$EXP/run_training.sh" all --profile

# Four independent full runs, 256 updates each.
bash "$EXP/run_training.sh" all --train

# Example: resume only an interrupted W8 GRACE run from its native checkpoint.
bash "$EXP/run_training.sh" grace --bits 8 --train --resume

# Inspect all 70 evaluation commands, then validate inputs on CPU.
bash "$EXP/run_local_evaluation.sh" --dry-run
bash "$EXP/run_local_evaluation.sh" --check

# References can be evaluated before training. This subset uses existing models.
bash "$EXP/run_local_evaluation.sh" --methods vanilla_2b vanilla_8b_teacher sft_2b released_bf16

# Evaluate all ten rows; complete suites are validated and skipped on reruns.
bash "$EXP/run_local_evaluation.sh"

# Rebuild the complete comparison or aggregate training cost without GPU use.
bash "$EXP/run_local_evaluation.sh" --summarize
"$PYTHON" reproduction/GRACE/qwen-vl-finetune/scripts/summarize_constrained_runtime.py \
  --config "$EXP/experiment.json" --kind runs
```

`run_training.sh {qat|grace|all} --bits {4|8|all}` defaults to both bit widths and
profile mode. `--train`, `--profile`, and `--prepare-only` are mutually exclusive;
`--dry-run` can accompany any mode. `PROFILE_STEPS` defaults to 10. Use
`EXPERIMENT_DIR` for a fresh output root, `BASE_MODEL_DIR` to relocate the **same
pinned** released BF16 checkpoint, and `TEACHER_MODEL_PATH` to relocate vanilla
8B. The inherited environment recipe pins batch/input/LoRA settings. Completed
outputs cannot be overwritten; resume only the specific unfinished method and
bit width. The launcher stops if a run pauses or fails to export.

The BF16 student is pinned to revision
`82f9a267340f65a3619be0e19daf6582d3998a5f`, with actual `model.safetensors` SHA256
`4b522cda73d4ebd792e97a89cbca51091287b65c87e705be53f56c035715bfd3`.
A renamed vanilla directory fails verification. Other checkpoints record local
HF download revisions when available and all asset hashes; the local directory
mapping is not independent remote verification of repository identity.
`prepare_constrained_models.py --config "$EXP/experiment.json" --download` resolves and persists immutable revisions before
downloading. It preserves cards and JSON/processor assets, and the preflight
rejects packed quantization backends or non-floating safetensors weights.
Existing vanilla models are reused. `--models released_bf16` selects individual
references for provenance recording.

The shared [matrix driver](../../qwen-vl-finetune/scripts/run_constrained_matrix.py)
uses `prepare_subset`, `comparison_recipe`, `build_command` and `launch_run` from
[run_constrained.py](../../qwen-vl-finetune/scripts/run_constrained.py). The old
launchers use these same preparation and execution functions. Recipe locking,
native checkpoint checks, optimizer/RNG resume and subprocess execution have one
implementation. The trainer remains `qwenvl.train.train_qwen_constrained`. It supplies `qat_bits=4` or `8`
explicitly; it does not depend on `QAT_BITS` or change the old launcher's defaults.
One CPU-validated root `subset.json` is shared by symlinks in `w4g128/` and
`w8g128/`. Deterministic preparation is repeated under the released processor;
`subset_comparison.json` records changed identities, ordering, retained turns and
token hashes relative to the previous larger-input experiment. Neither group
can silently substitute a different subset.

Evaluation uses each checkpoint's **own explicitly selected processor** after
checking vocabulary, special tokens, chat template and patch/merge compatibility.
All rows override image area to 4,096–100,352 pixels. Decoding is explicitly greedy
with one beam, repetition penalty 1, and checkpoint-specific generation defaults
cleared. Generation caps are 8 tokens for the six other suites and MMMU multiple
choice, and 32 for MMMU open answers. Evaluation does not apply the training
2,048-token truncation; it retains full questions within native model context.
Therefore old evaluation outputs are not copied into this experiment.

`NPROC` (default 2), `ATTN` (default `flash_attention_2`), `OUT_DIR`, `DATA_ROOT`
and the existing suite-specific dataset environment variables remain available.
`SFT_MODEL_DIR` relocates the previous completed 256-update SFT export. Use
`--methods` to select explicit result IDs. Every execution checks selected
checkpoints before allocating GPUs. The default full preflight requires all ten
models, including the two author low-bit references and four new completed runs.
It requires `EXPORT_COMPLETE`, 256 completed updates, the expected bit width,
merged adapters, nonempty quantized-module coverage and recorded BF16
initialization for local QAT/GRACE rows.

`evaluation/protocol.json` pins paths, preprocessing, decoding, evaluator hashes,
question-asset hashes and an external-image size/mtime inventory fingerprint.
Each row pins model/config/processor hashes in `provenance.json`. Changing the
protocol or a model requires a new output directory. Resume skips only merged
files that pass expected counts, unique IDs, score consistency and protocol
checks; invalid merged files and partial rank shards are retried. Suite failures
are collected while remaining suites run, and the process exits nonzero.
The full comparison additionally requires matching question identities and gold
labels across all ten rows. Gold labels are retained even after inference errors.

Once all 70 results validate, the wrapper writes `comparison.md`,
`comparison.csv`, and `deltas.csv`, including per-suite and mean W4/W8 gains/gaps,
HallusionBench qAcc/fAcc, ScienceQA modality splits, categories, errors and parse
failures. Acceptance thresholds remain for discussion after these measurements.

Per-bit runtime reports are written automatically after successful runs.
`summarize_constrained_runtime.py --config "$EXP/experiment.json" --kind runs`
combines the four completed runs under `reports/`; use `--kind profiles` for the
separate stress-profile report. Invocation timing
includes loading/export around `torchrun`; time outside recorded Trainer
intervals is reported separately, without claiming separate loading and export
measurements. GPU-hours use two selected GPUs. Memory covers logged training
boundaries, not total device occupancy or loading/export peaks.

## Shared code and experiment configuration

Before execution, this folder contained only `README.md`, `experiment.json` and the two thin bash
wrappers. It now also contains completed run/evaluation artifacts and the experiment report. Model mappings, the pinned student
revision/hash, bit widths and report comparisons are declared in
[experiment.json](experiment.json). Training hyperparameters continue to come
from the inherited shared environment recipe sourced by `run_training.sh`.
`EXPERIMENT_DIR` changes the artifact root without changing the config location.

```text
run_training.sh
  -> qwen-vl-finetune/scripts/run_constrained_matrix.py --config experiment.json
  -> run_constrained.prepare_subset / comparison_recipe / launch_run
  -> torch.distributed.run -m qwenvl.train.train_qwen_constrained

run_local_evaluation.sh
  -> evaluation/run_constrained_matrix_evals.py --config experiment.json
  -> evaluation/{suite}/eval_*.py
  -> evaluation/summarize_constrained_evals.py (matrix report functions)
```

Shared model/provenance checks live in
[qwen-vl-finetune/scripts/constrained_experiment.py](../../qwen-vl-finetune/scripts/constrained_experiment.py).
The acquisition utility is
[prepare_constrained_models.py](../../qwen-vl-finetune/scripts/prepare_constrained_models.py).
The existing [runtime summarizer](../../qwen-vl-finetune/scripts/summarize_constrained_runtime.py)
supports `--config` for matrix reports; its original `--experiment` interface is
unchanged. The existing [evaluation summarizer](../../evaluation/summarize_constrained_evals.py)
supports `--matrix --results-dir ... --write`, which the matrix runner also calls
automatically once all results validate. Regression tests live in
[tests/test_constrained_matrix.py](../../tests/test_constrained_matrix.py).

The previous experiment launchers retain their arguments, resolved defaults and
trainer commands. Existing experiment artifacts were not rewritten. Code hashes
remain strict: a run pinned before a launcher change requires its matching source
version to resume; this refactor does not silently migrate or bypass recipe pins.
There were no training/evaluation artifacts in this new experiment to migrate at implementation time.

## Implementation validation and completed results

CPU regression checks:

```bash
python3 -m unittest discover -s reproduction/GRACE/tests -p 'test_constrained_matrix.py' -v
```

The checks cover delegation to the shared launch function, fresh W4/W8 initialization and output routing, profile isolation,
all 70 evaluation commands, old launcher defaults, student weight rejection,
recipe drift, incomplete/duplicate/wrong-protocol results, complete summary
creation and mismatched gold labels. The available released BF16, vanilla 2B,
vanilla 8B and previous SFT processors were checked for compatibility. A real
CPU preflight passed for the released BF16 checkpoint and all seven local
dataset paths. Refactor validation compares all four training and 70 evaluation
dry-run commands with their pre-refactor output. No profiles, training or inference were run during implementation.

All four full runs completed 256 updates and exported successfully. All ten model rows
have seven validated merged results, and the score/runtime summaries are available.
The report audit also checked contiguous step logs, pinned recipes and training-source
hashes. Separate stress-profile artifacts were not found. Remaining research work is
to inspect native-versus-export behavior and parameter updates, test the loss/teacher
hypotheses, and discuss quality tolerances with the supervisor; see the
[report's next steps](experiment_report.md#7-recommended-next-steps-and-decision-criteria).

## Artifact layout

The completed artifact layout is below. Optional profile directories and the supervisor decision record are not present. Shared model weights remain under `reproduction/GRACE/models/`.

```text
released_bf16_init_32k_b128_p100352_t2048_seed42/
  README.md
  experiment_report.md             # completed analysis and next steps
  run_training.sh
  run_local_evaluation.sh
  experiment.json                 # model paths, pinned student identity, bit widths, comparisons
  provenance/                       # model revisions, hashes, environment, source cards
  subset.json                       # validated common manifest
  subset_comparison.json            # identities/retained turns versus previous experiment
  w4g128/
    comparison_recipe.json
    subset.json                     # identical validated manifest, with matching hash
    profiles/{qat,grace}/            # optional; not recorded here
    runs/{qat,grace}/                # native checkpoints and final exports
  w8g128/
    comparison_recipe.json
    subset.json                     # identical validated manifest, with matching hash
    profiles/{qat,grace}/            # optional; not recorded here
    runs/{qat,grace}/
  evaluation/
    protocol.json                   # explicit model/processor mapping and suite settings
    {result_id}/{suite}.json         # ten result IDs from the model matrix
    comparison.md
    comparison.csv
    deltas.csv
  reports/
    runtime_memory.md
    runtime_memory.json
    report_analysis.json            # derived statistics and result-file hashes
    supervisor_decisions.md         # pending supervisor discussion
```

Review checklist:

- [x] Implement released BF16 initialization and four-run W4/W8 QAT-versus-GRACE matrix.
- [x] Implement the inherited constrained recipe and matched evaluation protocol, including prior SFT.
- [x] Implement checkpoint acquisition, launchers and evaluation; validate CPU routing and preflight.
- [x] Complete four full runs and all 70 evaluations; validate final exports and results.
- [x] Report quality, resource costs, diagnostics and proposed next steps.
- [ ] Discuss measured quality gaps and acceptance criteria with the supervisor.

## Actual token-budget audit — 2026-10-05

A CPU audit of all 32,768 retained examples found mean/median/maximum total lengths of **314.50 / 234 / 1,858**, with mean/maximum text-plus-formatting counts of **224.21 / 1,777**. No retained sequence reaches 2,048 and no retained conversation was trimmed; one preparation candidate was rejected for exceeding the limit. See [the detailed token audit](reports/token_lengths.md), including the implications for the new 576-token visual ceiling.
