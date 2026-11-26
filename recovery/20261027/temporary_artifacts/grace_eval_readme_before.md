# Evaluation-only reproduction: findings and progress

Status as of **2026-09-26**: the five reference models completed all seven benchmarks (**35/35 pairs**), but the original-script replay did **not** reproduce all published scores within our predefined tolerances. Several scoring issues have been identified; the remaining discrepancies are unresolved. A validated replacement protocol and fresh baselines are still pending.

This folder is a documentation hub for the evaluation-only reproduction effort. Existing predictions, logs, completion markers and diagnostic outputs remain in the original experiment directory; they have not been moved or replaced. Creating this summary does not run evaluations or change their scores.

See the [metric/protocol follow-up](metric_protocol_findings.md) for the five-model aAcc/qAcc/fAcc comparison, newly applied MMBench CircularEval, MMStar aggregation check, and the distinction between official MMMU rules and GRACE's separate GPT-assisted path. The [recomputed audit](metric_protocol_audit.json) preserves per-group MMBench evidence. Circular MMBench puts three models within ±3 pp and two within ±5 pp; it does not clear the full seven-suite gate.

## Objective and scope

The objective is to reproduce the GRACE paper's evaluation-only results before comparing our own LoRA-GRACE and LoRA-QAT training methods. Training may be resource-constrained, but evaluation differences must be explicit and controlled.

The five references are:

| Local model ID | Reference |
| --- | --- |
| `vanilla_2b` | Qwen3-VL-2B-Instruct |
| `vanilla_8b_teacher` | Qwen3-VL-8B-Instruct |
| `released_bf16` | Authors' released BF16/SFT checkpoint; **not** our local `sft_2b` checkpoint |
| `released_grace_w4` | Authors' released GRACE W4 checkpoint |
| `released_grace_w8` | Authors' released GRACE W8 checkpoint |

Suites: HallusionBench, MMBench, ScienceQA, AI2D, MMMU, SEEDBench image subset, and MMStar. Published targets are recorded in [paper_acceptance.py](../../evaluation/paper_acceptance.py), based on the [GRACE paper](https://arxiv.org/abs/2601.22709), Table 6 in the audited v5 version.

Acceptance was defined per model/benchmark pair: absolute difference ≤3 percentage points (pp) preferred; >3 to ≤5 pp barely acceptable for unjudged scoring; ≤8 pp for actual judge-based scoring. All seven archived GRACE scripts are unjudged, so the eight-point allowance does not apply to their results. Numerical agreement is separate from proof of protocol identity.

## Work completed

1. **Audited the initial evaluation mismatch.** Earlier constrained runs and an independent lmms-eval comparison used different image budgets, prompts, task scopes, token limits or scoring. Their numbers cannot be treated as interchangeable with the authors' supplied evaluators.
2. **Pinned and archived the original implementation.** Source is fixed at GRACE commit `b73fa9b4d427aa64ed3ad274def641efb3145877`. The archive contains the seven evaluators and launchers plus supporting repository files, with hashes. The README names lmms-eval, while the supplied launchers execute custom Python scripts; this does not uniquely identify the recipe behind the paper's scores.
3. **Built an original-script replay.** [run_upstream_matrix.py](../../evaluation/run_upstream_matrix.py) executes the archived, unmodified evaluators with native checkpoint processors/generation configuration and original launcher token limits, including MMStar 64 and MMMU MC 8/open 32. The local orchestration uses one GPU rather than the original multiprocess Slurm launch. This reproduces supplied evaluator behavior, not a proven exact copy of the authors' paper runs.
4. **Added a two-stage workflow.** Stage 1 evaluates the five references; stage 2 evaluates four local LoRA-GRACE/QAT exports only if the reference gate passes. Outputs are isolated under `re_evaluation`. Completion checks cover artifacts, question identity/coverage, inference errors and score arithmetic.
5. **Completed and audited stage 1.** All 35 pairs completed with zero recorded inference errors. The gate remains blocked by score differences. Completion means the runs passed integrity checks; it does not certify the answer parsers or paper reproduction.
6. **Diagnosed saved predictions.** The initial audit checked all 35 pairs. The focused follow-up checked all 20 pairs for HallusionBench, MMBench, MMMU and MMStar, calculated label-bias metrics, and applied alternative reference scoring offline. These diagnostics did not overwrite scores, relax the gate, launch new inference or call judges.
7. **Prepared a separate corrected harness path.** The pinned lmms-eval workflow includes explicit MMMU prompt/open-scoring fixes and CPU validation. Its implementation and settings are documented, but it is not an authenticated author protocol or a completed, accepted replacement seven-suite baseline. Its shared pixel defaults also differ from the currently requested native-processor policy.

See [evaluation differences from the original repository](../../evaluation/differences_from_org.md) for the source-level audit and entry-point distinctions. In particular, running the modified working-tree evaluators is different from running the archived scripts through `run_upstream_matrix.py`.

## Completed original-script results

Each cell is **local accuracy % (local minus paper, pp)**. These are the recorded stage-1 metrics, before diagnostic rescoring. MMBench here is per-rotation-row accuracy, not circular accuracy.

| Model | HallusionBench | MMBench | ScienceQA | AI2D | MMMU | SEED image | MMStar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla 2B | 62.44 (+11.04) | 83.81 (+5.41) | 83.19 (+1.79) | 76.62 (-0.28) | 41.22 (-12.18) | 76.79 (+5.59) | 46.87 (-11.43) |
| Vanilla 8B | 72.90 (+11.80) | 89.79 (+5.29) | 92.95 (+7.95) | 83.97 (-1.73) | 52.78 (-16.82) | 79.39 (+1.89) | 62.73 (-8.17) |
| Released BF16 / SFT | 67.05 (+0.15) | 94.66 (+8.26) | 86.18 (-0.02) | 81.12 (-0.18) | 72.00 (-0.10) | 76.28 (-0.42) | 68.00 (+0.70) |
| Released GRACE W4 | 64.57 (-0.83) | 92.63 (+8.03) | 84.08 (-0.22) | 79.40 (-0.10) | 61.56 (-8.94) | 73.86 (-1.24) | 66.20 (+0.40) |
| Released GRACE W8 | 59.17 (-6.93) | 92.49 (+6.99) | 82.48 (-2.82) | 75.94 (-4.46) | 58.11 (-13.19) | 70.08 (-5.82) | 63.67 (-2.83) |

Of 35 pairs, **17 pass**, **1 is barely acceptable**, and **17 fail**. The strict ±3 pp gate is blocked, and accepting the barely-acceptable band would not clear it. Released BF16 matches six of seven columns within 0.70 pp; MMBench is its exception. Vanilla discrepancies are larger and go in both directions, arguing against a single global explanation.

The [stage-1 gate](../released_bf16_init_32k_b128_p100352_t2048_seed42/re_evaluation/stage1_gate.md) contains all paper targets and pairwise statuses. Diagnostic values below are separate evidence, not revised official run scores.

## Findings from the four-suite investigation

### MMMU: original short cap and unreliable answer extraction

The original evaluator explicitly hardcodes **eight new tokens for multiple choice**. Its `--max_new_tokens 32` setting affects only open-ended questions. Removing the MC limit would be a documented protocol deviation, not restoration of an omitted setting.

The parser selects the first valid option character anywhere in uppercased text, including inside words. It can turn unfinished reasoning such as `To solve for the missing account balance,` into A. Better extraction cannot recover an answer that was never generated.

Of 847 MC questions, vanilla 2B/8B return only 49/118 bare-letter responses; 429/341 responses re-encode to at least the eight-token limit. Re-encoding is a truncation proxy because generated token IDs and finish reasons were not saved; not every capped response lacks an answer. The released BF16/W4/W8 models all return 847 bare-letter responses, so their MC differences are not explained by ambiguous parsing. An official-style open-answer helper recovers only two additional correct answers for each vanilla model, or 0.22 pp overall.

### MMBench: aggregation explains much of the gap

The archived evaluator averages **4,329 option-rotation rows**. Circular scoring instead requires every rotation of each of **1,164 base questions** to be correct. Applying the pinned harness's rule extraction and circular grouping to the same saved responses yields:

| Model | Paper % | Recorded row accuracy % | Reference circular diagnostic % |
| --- | ---: | ---: | ---: |
| Vanilla 2B | 78.40 | 83.81 | 76.80 |
| Vanilla 8B | 84.50 | 89.79 | 84.71–85.05 |
| Released BF16 | 86.40 | 94.66 | 91.15 |
| Released W4 | 84.60 | 92.63 | 88.66 |
| Released W8 | 85.50 | 92.49 | 88.40 |

The 8B range is a bound over unresolved extraction, not a confidence interval. Nine rows are unresolved, but only four groups could depend on a judge, limiting its possible overall effect to 0.34 pp. Every released response is directly parseable. This largely reconciles vanilla MMBench; it leaves BF16/W4 within five points but outside three. It does not prove that the paper used this exact dataset/prompt/scorer configuration.

### HallusionBench: label bias exists but does not explain vanilla gains

Gold labels are **645 No / 484 Yes**; always predicting No achieves 57.13%. Treating Yes as positive:

| Model | Accuracy % | Balanced accuracy % | Yes precision % | Yes recall % |
| --- | ---: | ---: | ---: | ---: |
| Vanilla 2B | 62.44 | 64.19 | 54.41 | 76.45 |
| Vanilla 8B | 72.90 | 71.92 | 69.69 | 65.08 |
| Released BF16 | 67.05 | 63.97 | 68.79 | 42.36 |
| Released W4 | 64.57 | 63.73 | 58.82 | 57.85 |
| Released W8 | 59.17 | 59.08 | 52.21 | 58.47 |

Balanced accuracy gives both classes equal weight. Vanilla 2B overpredicts the minority Yes label; balancing raises its score. For 8B, balancing lowers accuracy by only 0.98 pp. Neither explains the +11–12 pp paper gap. BF16 shows a stronger majority-No tendency and low Yes recall.

Both vanilla models output only canonical Yes/No strings, leaving little answer ambiguity for a correctness judge. Restricting to the 951 image rows raises their scores to 64.25/73.82%, farther from the paper; the remaining 178 rows are text-only. Question/figure consistency metrics are different from per-answer accuracy and cannot be substituted to force agreement. W4/W8 are below, not above, the published HallusionBench scores.

### MMStar: parser defects are real but insufficient

The original parser takes the last standalone option letter after uppercasing, then falls back to letters inside words. It can misread `C: A forest` as A. Fixing only clear leading-answer conflicts raises vanilla 2B from 46.87 to 47.40% and 8B from 62.73 to 62.80%, too little to close the gaps.

The pinned reference prefix parser instead produces 42.33/61.93% on the same responses: permissive extraction also creates accidental correct matches. These are rule disagreements, not a semantic adjudication of every response. Changing aggregation alone to the mean of 18 subcategory scores adds only 0.94/0.34 pp with the native parser. The 64-token re-encoding proxy is reached by 299/38 vanilla responses, warranting a controlled output-completeness diagnostic. All released responses are bare letters.

### Judge requirements

| Suite | Archived GRACE scoring | Selected reference scoring path |
| --- | --- | --- |
| HallusionBench | Rule-based Yes/No; no judge | GPT correctness/consistency assessment |
| MMBench | Choice extraction and row accuracy; no judge | Rule extraction, judge fallback if unresolved, circular scoring |
| MMMU | Rule-based MC/open; no judge | Rule-based MC/open scoring; judge not required |
| MMStar | Rule-based choices; no judge | Rule-based scoring; judge not required |

ScienceQA, AI2D and SEED-image also have judge-free scoring paths. A fixed reference answer does not imply a benchmark cannot use a judge: the judge may interpret whether free-form output matches that answer. No paid or live judging was performed in the saved-answer diagnosis.

## Other hypotheses and limits of the evidence

- **Nondeterminism:** archived generation explicitly uses `do_sample=False`. Numerical/software variation is possible, but run-to-run variability was not measured. It is not established as the cause of the large gaps.
- **Wrong downloaded checkpoints:** the recorded file-hash mappings for all five references match the earlier Hub-audited provenance. This weighs against a local model mix-up; the follow-up did not independently download or hash the full weights again. It cannot verify which private checkpoints produced the authors' table.
- **Undocumented settings or different evaluation code:** plausible, but unproven. Needed evidence includes exact Table 6 commands, evaluator revisions, dataset splits/revisions, processor overrides, checkpoint revisions, judge configuration and raw predictions. A discrepancy alone does not prove hidden settings, an author-side mistake or incorrect published results.

Per the user's instruction, **preserve checkpoint-native processor configurations and do not test a shared W4/W8 resolution**. The earlier shared-resolution proposal was withdrawn. Native maximum pixel budgets differ: 16,777,216 for vanilla/BF16, 200,704 for W4 and 50,176 for W8. These differences are recorded, not experimentally established causes. Future comparisons should disclose this per-checkpoint preprocessing policy rather than claim identical resolution across models.

## Agreed direction and remaining work

1. **Validate local scoring before treating it as the main evaluation.** Reconcile MMBench's circular metric and MMMU/MMStar extraction; document HallusionBench's population, metric and judge policy. Parser changes should follow answer content and benchmark definitions, not target paper numbers.
2. **Keep the original replay intact.** Any longer-generation test belongs to a separately named diagnostic protocol with a finite cap, the same changes across all five references, and native processors. No such follow-up inference was performed during this diagnosis.
3. **Freeze the validated protocol and establish fresh local baselines.** Then evaluate future LoRA-GRACE/LoRA-QAT methods under the same declared evaluation rules. These fresh seven-suite baselines are not yet complete. Moving to a local-comparison workflow requires an explicit documented workflow update; the existing paper-agreement gate has not been bypassed or relaxed.
4. **Keep all seven benchmarks in the report.** Do not discard benchmarks because their paper delta exceeds 10 pp. Mark unresolved or invalid comparisons clearly and avoid model-quality claims from known malformed-answer extraction. Do not label a reduced-suite mean as the paper's seven-suite average.
5. **Use validated local comparisons as the primary evidence, with paper scores separately labelled.** Preserve all reproduction discrepancies and avoid claiming that score agreement proves exact protocol identity.

Suggested reporting language:

> Running the publicly released GRACE evaluation scripts on the reference checkpoints did not reproduce several published scores within our predefined tolerances. We identified scoring and answer-extraction issues, but could not fully reconcile the remaining differences. The available materials do not establish whether these arise from undocumented settings, evaluator versions, or other protocol differences. Once the local protocol is validated and frozen, we will compare our methods against reference checkpoints evaluated under that same documented protocol, while retaining published scores as an external reference.

## Evidence and navigation

- [Original experiment report](../released_bf16_init_32k_b128_p100352_t2048_seed42/experiment_report.md) and [earlier harness evaluation](../released_bf16_init_32k_b128_p100352_t2048_seed42/lmms_evaluation.md).
- [Protocol recovery and runner documentation](../../evaluation/README_paper_protocol.md); historical status statements there predate the completed native replay.
- [Original-source archive and hashes](../../evaluation/provenance/upstream_20260925/manifest.json).
- [Two-stage workflow instructions](../released_bf16_init_32k_b128_p100352_t2048_seed42/README_re_evaluation.md) and [launcher](../released_bf16_init_32k_b128_p100352_t2048_seed42/run_re_evaluation.sh).
- [Completed native results and provenance](../released_bf16_init_32k_b128_p100352_t2048_seed42/re_evaluation/full/) and [stage-1 gate](../released_bf16_init_32k_b128_p100352_t2048_seed42/re_evaluation/stage1_gate.md).
- [Initial diagnosis](../released_bf16_init_32k_b128_p100352_t2048_seed42/re_evaluation/diagnosis/diagnosis_report.md) and [saved-answer analysis](../released_bf16_init_32k_b128_p100352_t2048_seed42/re_evaluation/diagnosis/saved_answer_analysis.json).
- [Focused four-suite findings](../released_bf16_init_32k_b128_p100352_t2048_seed42/re_evaluation/diagnosis/four_suite_followup.md), [analysis script](../released_bf16_init_32k_b128_p100352_t2048_seed42/re_evaluation/diagnosis/analyze_four_suites.py), and [machine-readable results](../released_bf16_init_32k_b128_p100352_t2048_seed42/re_evaluation/diagnosis/four_suite_analysis.json).
