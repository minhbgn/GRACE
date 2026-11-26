# Pinned lmms-eval comparison

This workflow evaluates all **10 checkpoints × 7 benchmarks** from `experiment.json`
with the Qwen3-VL backend in lmms-eval **v0.7.1**, commit
`88b23e2bfa16a1edbc16e9e238ed82130b3a4f56`. It loads the merged Hugging Face
exports directly. No training or further quantization is needed.

The experiment's existing `evaluation/` results and report remain the custom
Transformers protocol. New outputs go to `lmms_evaluation/full/`; limited runs
go to separate `lmms_evaluation/smoke_N/` directories. These measure accuracy
with BF16 storage of the grid-projected weights; they do **not** measure the
memory, speed, or accuracy of a packed INT4/INT8 deployment backend.

## What “paper protocol” means here

The [GRACE README](https://github.com/ForeverBlue816/GRACE#evaluation) states that
it uses lmms-eval. However, its linked
[ScienceQA Slurm script](https://github.com/ForeverBlue816/GRACE/blob/main/evaluation/ScienceQA/eval_scienceqa.slurm)
invokes a custom `eval_scienceqa.py` through `torchrun`, reads `MODEL_PATH`
instead of the README's `MODEL`, and uses eight output tokens. The public recipe
does not establish the authors' exact harness revision, task variants, image
budgets, prompts, or judge settings.

Consequently, this is an **independent, pinned lmms-eval comparison**, not a
verified reproduction of the paper's exact evaluation. We preserve the pinned
harness's task prompts, model-specific prompt overrides, filters and scorers.
Results can determine whether changing the evaluation stack changes our scores;
they cannot by themselves identify which setting caused the published gap.

The first comparison uses the pinned backend's image defaults, **200,704 minimum
and 1,605,632 maximum pixels per image**, with greedy generation, zero-shot
prompts, batch size one and a single GPU. These are different from our old
4,096/100,352 limits and are **not recovered author settings**. Multiple-image
MMMU samples can use substantially more memory. An OOM is a failed run; do not
quietly shrink images to finish a row. Use a new protocol/output root for changes.
SDPA is explicitly selected for portability. The system prompt is
`You are a helpful assistant.`; interleaving is disabled, as in this backend's
default. See [lmms_protocol.json](lmms_protocol.json) for all explicit settings.

| Suite ID | Harness task / split | Primary metric | Output cap | Judge |
| --- | --- | --- | ---: | --- |
| `scienceqa` | `scienceqa_img` / test, ScienceQA-IMG | `exact_match,none` × 100 | 16 | No |
| `ai2d` | `ai2d` / test | `exact_match,flexible-extract` × 100 | 16 | No |
| `mmmu` | `mmmu_val` / validation | `mmmu_acc,none` × 100 | 128 | No |
| `mmstar` | `mmstar` / val | `average,none` × 100 | 128 | No |
| `seedbench` | `seedbench` / test | `seed_image,none` × 100 | 128 | No |
| `hallusionbench` | `hallusion_bench_image` / image | `aAcc,none`, already % | 128 | `gpt-4` |
| `mmbench` | `mmbench_en_dev` / dev, en | `gpt_eval_score,none`, already % | 1024 | `gpt-4o-2024-11-20` |

The 128-token caps for MMStar and SEED-Bench come from the backend fallback.
ScienceQA-image excludes text-only questions included in our old overall score.
HallusionBench-image has different coverage and judge scoring from our old
rule-based overall score. MMStar's `average` is a mean over L2-category accuracies.
SEED-Bench runs the harness task and selects its image metric; other emitted
metrics remain in the raw output. MMBench retains its rolling-question aggregation
and judge-assisted answer extraction. Means are only emitted for complete
seven-suite model rows, with all primary metrics normalized to percentages.

MMMU's standard scorer is local. Upstream nevertheless initializes an unused
judge while importing shared utilities. The wrapper disables that unused object
and verifies the selected function is `mmmu_process_results`; accidental access
to the disabled judge raises an error. It does not supply fake API credentials.

## Prepare and check without GPU inference

Run from the GRACE root:

```bash
cd /raid/lvchengtao/Minh/Distillation/reproduction/GRACE
# Already prepared on this machine; reproducible setup for another environment:
bash evaluation/prepare_lmms_eval.sh

EXP=experiments/released_bf16_init_32k_b128_p100352_t2048_seed42
bash "$EXP/run_lmms_evaluation.sh" plan
bash "$EXP/run_lmms_evaluation.sh" check
bash "$EXP/run_lmms_evaluation.sh" prepare
```

`plan` lists the 70 jobs. `check` imports the actual source backend, resolves all
seven YAMLs and checks the selected checkpoint headers/export completion.
`prepare` pins the protocol, package versions, runner/worker hashes and all seven
Hugging Face dataset commit IDs, including resolved repository redirects. It
fetches metadata, not dataset contents. The first actual evaluation downloads
its dataset and pins split lengths/fingerprints for subsequent models.

The isolated `.venv-lmms-eval` inherits the existing `grace` environment and
installs evaluation additions locally. The launcher explicitly puts the source
checkout on `PYTHONPATH`: the inherited wheel otherwise wins over the editable
install and lacks some extensionless task templates. The source must remain at
the clean pinned commit. Runtime package versions are locked per output root;
a later package update requires a fresh root rather than mixing results.

## Validation completed on this machine

CPU preflight resolved all seven task configurations and validated all ten local
checkpoint/export structures. Seven automated tests cover metric scales,
incomplete/duplicate samples, smoke/full separation, changed sample identities,
modified completed artifacts, and partial-matrix reporting. A separate synthetic
two-question test passed through the real harness's task loading, prompt
resolution, generation interface, scoring, sample logging and result validation.
No checkpoint inference or external judge requests were used for these checks.

```bash
.venv-lmms-eval/bin/python -m unittest discover -s evaluation/tests -p test_lmms_matrix.py -v
.venv-lmms-eval/bin/python evaluation/tests/cpu_lmms_integration.py
```

Actual model loading, GPU memory requirements, dataset downloads and judge
availability still need the staged execution below.

## Smoke test, then evaluate

Before launching inference, run `gpuwho`, check current ownership and memory,
and choose a free GPU in PCI-bus order. Do not reclaim another person's jobs.
Only one process
uses that GPU; all selected model/task pairs run sequentially.

(Running on GPU index 0)
```bash
gpuwho
bash "$EXP/run_lmms_evaluation.sh" run --gpu 0 \
  --methods released_grace_w4,released_grace_w8 \
  --suites scienceqa,ai2d,mmmu,mmstar,seedbench --smoke 8

# First obtain the full released-checkpoint comparison on non-judge suites:
bash "$EXP/run_lmms_evaluation.sh" run --gpu 0 \
  --methods released_grace_w4,released_grace_w8 \
  --suites scienceqa,ai2d,mmmu,mmstar,seedbench

# Then all ten models on the same five suites; verified completed jobs are skipped:
bash "$EXP/run_lmms_evaluation.sh" run --gpu 0 \
  --suites scienceqa,ai2d,mmmu,mmstar,seedbench
```

Limited samples test loading and generation; they are not benchmark estimates.
Use the five non-judge suites for the initial smoke test: HallusionBench's
figure/question group metrics can be undefined on an arbitrary first-N subset.

HallusionBench and MMBench require an explicitly configured judge endpoint and
credentials. The model IDs in the protocol are the harness defaults; their
availability and the authors' exact judges are unverified. Configure any changes
**before preparing a new output root**. Set `OPENAI_API_KEY` privately in the
environment (do not put it in scripts, JSON files, shell history or reports).
`--allow-judge` explicitly enables the external judge requests and their cost.

```bash
# With judge credentials already configured in this shell:
bash "$EXP/run_lmms_evaluation.sh" run --gpu GPU_INDEX \
  --suites hallusionbench,mmbench --allow-judge

# Equivalent whole-matrix command, resuming any completed jobs:
bash "$EXP/run_lmms_evaluation.sh" run --gpu GPU_INDEX --allow-judge
bash "$EXP/run_lmms_evaluation.sh" summarize
```

Each job saves a fresh attempt directory with `job.json`, `run.log`,
`results.json`, `samples.jsonl`, `dataset.json`, and `judge_audit.json`. Model
weights/configuration/tokenizer assets are hashed before evaluation. Dataset
fingerprints and document/target identities must agree across model rows.
A completion marker is written only after validating the expected task, metric
scale, requested limit, complete document coverage and judge audit. Resuming
verifies saved output hashes before skipping a job. Unfinished attempts are
retained; rerunning creates a new attempt, without generation-cache reuse.

The wrapper conservatively rejects an attempt with **any judge HTTP/response
failure**, even if a later retry succeeds, or MMBench's random-answer fallback.
Raw outputs remain available for diagnosis. This prevents an API outage from
silently becoming a reported model score. A successful HTTP response is not a
guarantee of judge quality: inspect the saved judge outputs as part of review.

`lmms_evaluation/full/comparison.{json,md}` contains the new matrix. Partial
matrices show missing cells and omit incomplete row means. Keep protocol labels
when comparing it with the old report: changed splits, image budgets, prompts,
and scorers prevent interpreting every difference as an accuracy improvement.

After this matrix is complete, compare the released W4/W8 results with the
published values and examine per-example outputs. If discrepancies remain,
run the proposed controlled 50,176/100,352/200,704-pixel crossover under a
separate protocol, followed by isolated generation/scoring changes. That
crossover is deliberately deferred until this harness comparison is complete.
