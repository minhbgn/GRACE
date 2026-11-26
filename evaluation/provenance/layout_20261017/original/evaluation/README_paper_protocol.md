# GRACE evaluation protocol recovery — 2026-09-25

**The public sources do not uniquely specify the recipe that produced the Qwen3
results. We can replay the supplied scripts exactly, and run a corrected standard
benchmark protocol separately. Neither has yet reproduced Table 6 within the
requested tolerances.** No new GPU results were produced during this change.

The reference is [arXiv v5, Table 6](https://arxiv.org/html/2601.22709v5#A1.T6).
Appendix D.4 describes greedy decoding and model-family templates, but does not
supply a complete Qwen3 recipe, task revisions, image budgets, judge identities
or raw predictions. The [repository README](https://github.com/ForeverBlue816/GRACE/tree/b73fa9b4d427aa64ed3ad274def641efb3145877#-evaluation)
says lmms-eval; its example launcher actually runs a custom Python evaluator.
All seven supplied launchers follow that custom path. Thus replacing these
scripts with harness defaults cannot be called recovery of a demonstrated author
recipe. A large score difference is a diagnostic signal, not proof of its cause.

The current upstream commit is `b73fa9b4d427aa64ed3ad274def641efb3145877`, also the
local checkout's base commit. Its seven Python evaluators and seven Slurm
launchers are archived byte-for-byte, with README, requirements and license, in
[provenance/upstream_20260925](provenance/upstream_20260925/manifest.json).
The replay runner verifies every archived hash before execution. The existing
locally modified evaluators and historical results remain available.

## Recovered script protocol

All scripts use BF16 Transformers inference, the checkpoint's own processor/chat
template, a user message containing images before question text, and
`do_sample=False`. They retain the checkpoint generation configuration, except
for explicit arguments to `generate`. They do not truncate to the training
context limit or explicitly override processor pixel bounds. The launchers select
FlashAttention 2. Our replay uses one GPU and one example at a time; the scripts'
data-parallel sharding is optional and does not change question scope.

| Suite | Published script scope and scoring | Output limit | Original local difference |
| --- | --- | ---: | --- |
| HallusionBench | 1,129 image + non-image rows; rule-based aAcc, also qAcc/fAcc | 8 | Common constrained pixels and reset generation config |
| MMBench | English dev 20231003, 4,329 rows; per-row accuracy | 8 | Same overrides; this is not rolling leaderboard scoring |
| ScienceQA | Full test, 4,241 rows; image/text breakdowns | 8 | Same overrides; image-only changes the scope |
| AI2D | 3,088 test rows; custom choice extraction | 8 | Same overrides |
| MMMU | 900 validation rows; custom MC + simplified open scorer | MC 8 / open 32 | Same overrides; upstream open scorer is not the official scorer |
| SEED | Image subset, 14,233 rows; question accuracy | 8 | Same overrides |
| MMStar | 1,500 rows; question accuracy, last-letter parser | **64** | Local runner forced 8 tokens, plus same overrides |

Read the archived code for the complete prompt strings. Some inherited choice
parsers scan letters within words; these are intentionally preserved **only in
script replay**. Repairing them silently would destroy the evidence about what
the supplied code does. The native HallusionBench/MMBench scripts make **no judge
calls**. Their scores must not be presented as the standard judged metrics.

Actual processor bounds resolved on this machine:

| Reference | Minimum pixels | Maximum pixels |
| --- | ---: | ---: |
| Vanilla 2B | 65,536 | 16,777,216 |
| Vanilla 8B | 65,536 | 16,777,216 |
| Released BF16 | 65,536 | 16,777,216 |
| Released W4 | 784 | 200,704 |
| Released W8 | 784 | 50,176 |

The original custom comparison forced **4,096–100,352** for every model. The
previous harness run instead forced **200,704–1,605,632**. Neither reproduces
checkpoint-native preprocessing. In particular, native W4/W8 processor bounds
differ; this is a confound for interpreting precision comparisons, not evidence
that these settings explain the paper gap. The replay records the resolved
processor and generation configuration for each model.

## Run the supplied scripts faithfully

From `reproduction/GRACE`:

```bash
.venv-lmms-eval/bin/python evaluation/run_upstream_matrix.py plan
.venv-lmms-eval/bin/python evaluation/run_upstream_matrix.py check

# Survey first; replace 0 with a currently free physical GPU in PCI order.
gpuwho
.venv-lmms-eval/bin/python evaluation/run_upstream_matrix.py run \
  --gpu 0 --methods vanilla_2b --smoke 8

# Five published reference models × all seven suites; no training or judge.
.venv-lmms-eval/bin/python evaluation/run_upstream_matrix.py run --gpu 0
.venv-lmms-eval/bin/python evaluation/run_upstream_matrix.py summarize
```

The default output root is the released-BF16 experiment's `upstream_evaluation/`.
Smoke and full results use separate subdirectories. `--methods`, `--suites`,
`--experiment`, and `--output` select a subset or another experiment. Native suite
ID for SEED is `seedbench_img`. To evaluate the trained rows later, explicitly
select those model IDs under the same protocol. No training resource limit is
carried into this evaluator. An OOM is recorded as a failure; the runner never
lowers resolution or drops examples to obtain a score. `--attn sdpa` is an
explicitly recorded portability deviation, not the exact Slurm setting.

The runner pins source, environment, model file hashes, dataset question-asset
hashes and external image inventory (path/size/mtime, not image-content hashes).
It verifies complete unique sample coverage, shared IDs/labels, errors, summary
arithmetic and saved artifact hashes. Invalid attempts remain visible and receive
no completion marker. Parsing failures remain in the denominator. New settings
or source changes require a new output root; historical results are not reused.

## Corrected standard benchmark comparison

For a shared standard protocol across vanilla, released and trained models, use
[run_paper_standard_evaluation.sh](../experiments/released_bf16_init_32k_b128_p100352_t2048_seed42/run_paper_standard_evaluation.sh).
This follows the README's stated harness family, but its unresolved choices are
**our declared protocol**, not authenticated author settings. It pins lmms-eval
v0.7.1 (`88b23e2bfa16a1edbc16e9e238ed82130b3a4f56`), SDPA, greedy decoding,
batch size one and common 200,704–1,605,632 pixel bounds. These bounds are harness
defaults and are independent of constrained training.

| Suite | Task / reported score | Output limit | Judge |
| --- | --- | ---: | --- |
| HallusionBench | `hallusion_bench_image`, aAcc, 951 rows | 128 | `gpt-4` |
| MMBench | `mmbench_en_dev`, rolling `gpt_eval_score` | 1,024 | `gpt-4o-2024-11-20`, fallback extraction |
| ScienceQA | `scienceqa_img`, 2,017 image questions | 16 | No |
| AI2D | `ai2d`, exact match with answer filter | 16 | No |
| MMMU | `mmmu_val`, 900 validation rows, official-style MC/open accuracy | 128 | No |
| SEED | `seedbench`, image score from 14,233 of 17,990 evaluated rows | 128 | No |
| MMStar | `mmstar`, mean over 18 L2 category accuracies | 128 | No |

Judge model IDs come from the pinned harness; the authors' judge IDs and current
API availability remain unverified. No credential is present in this session.
Judge errors and random MMBench fallbacks prevent completion certification.
They are not converted into acceptable benchmark scores.

Three MMMU fixes are implemented without modifying the pinned third-party tree:

1. Keep the no-judge guard active through the actual evaluation, including
   TaskManager's second YAML import. The selected exact scorer does not need
   an API key; any attempted judge access raises an error.
2. Use the task's standard MC/open prompt branching. Its Qwen3 override tells
   open questions to choose from empty options, so it is disabled in v2.
3. Preserve **all typed normalized open-answer candidates**. The pinned wrapper
   stringifies only the first candidate; `eval_open` then iterates characters,
   incorrectly rejecting correct numbers and ordinary multi-character words.
   The [official MMMU implementation](https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py)
   expects a candidate list. The adapter restores that input shape while retaining
   the pinned normalization, MC extraction and aggregation.

The prompt and scorer changes are enabled explicitly in
[paper_standard_protocol.json](../experiments/released_bf16_init_32k_b128_p100352_t2048_seed42/paper_standard_protocol.json).
The old protocol JSON and its completed four-suite results are unchanged.

```bash
# Run from reproduction/GRACE; wrapper defaults to the five reference models.
EXP=experiments/released_bf16_init_32k_b128_p100352_t2048_seed42
bash "$EXP/run_paper_standard_evaluation.sh" plan
bash "$EXP/run_paper_standard_evaluation.sh" check
bash "$EXP/run_paper_standard_evaluation.sh" prepare
bash "$EXP/run_paper_standard_evaluation.sh" run --gpu 0 \
  --methods vanilla_2b --suites mmmu --smoke 8
bash "$EXP/run_paper_standard_evaluation.sh" run --gpu 0 \
  --suites scienceqa,ai2d,mmmu,mmstar,seedbench

# Requires a securely configured OPENAI_API_KEY; enables paid external scoring.
bash "$EXP/run_paper_standard_evaluation.sh" run --gpu 0 \
  --suites hallusionbench,mmbench --allow-judge
bash "$EXP/run_paper_standard_evaluation.sh" summarize
```

Outputs go to `paper_standard_evaluation/`, isolated from both historical
protocols and native replay. Use a freshly prepared directory after this source
change; do not resume old manifests with new runner/worker hashes.

## Acceptance and what is still unresolved

[paper_acceptance.py](paper_acceptance.py) records the five Table 6 targets.
Both runners emit `paper_comparison.json`; native replay also emits Markdown.
For each full model/suite pair, an absolute gap of **≤3 pp** passes, **>3 to ≤5 pp**
is barely acceptable, and **>5 pp** fails. Actual judge-based harness suites use
**≤8 pp**. The native unjudged proxies use the non-judge bounds. Missing and smoke
results never pass. Do not average away opposing benchmark gaps. Numerical
acceptance and protocol fidelity are separate requirements.

CPU preflight passed for both seven-suite workflows and the five reference
checkpoints. Ten regression tests passed. The real MMMU harness integration test
loads both YAML passes without credentials and correctly scores synthetic MC
and open numeric answers, plus direct textual/alternative/wrong-answer checks;
the same correct numeric answer failed before the
scorer adapter. This verifies software behavior, **not model quality**. The GPU
smoke-test launch was declined, and live judging lacks credentials. Consequently
no new full-suite scores or tolerance claims are available.

To establish exact paper reproduction, the remaining author evidence is: the
Table 6 commands, harness commit/task names, dataset revisions/splits, any
processor overrides, model revisions, judge model/version/settings, and per-item
raw predictions. The public README/script conflict prevents inferring those
settings uniquely. Freeze an evidence-supported recipe for all five reference
models before running; do not change splits or prompts separately per model to
make the scores land near the targets.

Saved validation evidence: [CPU preflights, test log and source hashes](../experiments/released_bf16_init_32k_b128_p100352_t2048_seed42/reports/protocol_recovery/validation.json).
