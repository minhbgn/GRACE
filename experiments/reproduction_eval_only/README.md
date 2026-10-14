# Evaluation-only reproduction: findings and progress

Status as of **2026-09-26**: the original-script replay completed **35/35 reference pairs**, but strict paper reproduction remains unresolved. A separate pinned lmms-eval round completed **40/40 pairs across ten models and four non-judge suites**, including all five references. The agreed development direction is to use **validated local, deterministic scoring while GPT access is unavailable**, then switch the main evaluation workflow to **pinned lmms-eval with judges where the selected tasks require them** once API access is supplied. The corrected local scorer and native-processor launcher are now implemented and CPU-validated; fresh GPU baselines remain pending. This does not certify the historical proxy scores.

This folder is a documentation hub for the evaluation-only reproduction effort. Existing predictions, logs, completion markers and diagnostic outputs remain in the original experiment directory; they have not been moved or replaced. The new CPU-only scoring audit is stored separately; historical predictions and scores remain unchanged.

See the [metric/protocol follow-up](metric_protocol_findings.md) for the five-model aAcc/qAcc/fAcc comparison, newly applied MMBench CircularEval, MMStar aggregation check, and the distinction between official MMMU rules and GRACE's separate GPT-assisted path. The [recomputed audit](metric_protocol_audit.json) preserves per-group MMBench evidence. Circular MMBench puts three models within ±3 pp and two within ±5 pp; it does not clear the full seven-suite gate.

## Run the corrected local evaluation

Use [run_local_development.sh](run_local_development.sh); the full protocol, commands, metrics and limitations are in [README_local_development.md](../../evaluation/README_local_development.md). Default: `local_native_greedy256_v1`, native checkpoint processors, deterministic corrected scoring, and 256 new tokens for every suite/model. The historical paper-reproduction launcher is unchanged.

```bash
RUN=reproduction/GRACE/experiments/reproduction_eval_only/run_local_development.sh
bash "$RUN" check --methods all
bash "$RUN" run --methods all --smoke 2 --gpu 0
bash "$RUN" run --methods references --gpu 0
bash "$RUN" run --methods lora --gpu 0
bash "$RUN" summarize --methods all
```

Replace GPU `0` with an available GPU after checking ownership. All nine checkpoints passed CPU preflight. Regression tests and all seven worker entry points passed with fake CPU generation. The [35-pair saved-answer validation](local_scoring_validation.json) matches all five previously validated circular MMBench scores and records changed/unresolved responses. **No new GPU inference has been performed; fresh baselines remain pending.**

## Objective and scope

The objective is to establish reliable evaluation baselines for our LoRA-GRACE, LoRA-QAT, and token-reduction experiments, while continuing to investigate reproduction of the GRACE paper. **Strict paper-score agreement is no longer a prerequisite for development once the local evaluation protocol is validated.** Reference checkpoints and our methods must be compared under the same declared evaluation rules, with protocol differences explicit and controlled. The new local launcher validates artifacts independently of the paper-score gate and exposes artifact readiness separately from scoring ambiguity.

The five references are:

| Local model ID | Reference |
| --- | --- |
| `vanilla_2b` | Qwen3-VL-2B-Instruct |
| `vanilla_8b_teacher` | Qwen3-VL-8B-Instruct |
| `released_bf16` | Authors' released BF16/SFT checkpoint; **not** our local `sft_2b` checkpoint |
| `released_grace_w4` | Authors' released GRACE W4 checkpoint |
| `released_grace_w8` | Authors' released GRACE W8 checkpoint |

Suites: HallusionBench, MMBench, ScienceQA, AI2D, MMMU, SEEDBench image subset, and MMStar. Published targets are recorded in [paper_acceptance.py](../../evaluation/paper_acceptance.py), based on the [GRACE paper](https://arxiv.org/abs/2601.22709), Table 6 in the audited v5 version.

Acceptance was defined per model/benchmark pair: absolute difference ≤3 percentage points (pp) preferred; >3 to ≤5 pp barely acceptable for unjudged scoring; ≤8 pp for actual judge-based scoring. All seven archived GRACE scripts are unjudged, so the eight-point allowance does not apply to their results. Numerical agreement is separate from proof of protocol identity. These remain the historical **paper-reproduction thresholds**. Accepting the evidence as sufficient to continue development does not retroactively turn a +9.70 pp discrepancy into a reproduction pass or change the existing gate.

## Work completed

1. **Audited the initial evaluation mismatch.** Earlier constrained runs and an independent lmms-eval comparison used different image budgets, prompts, task scopes, token limits or scoring. Their numbers cannot be treated as interchangeable with the authors' supplied evaluators.
2. **Pinned and archived the original implementation.** Source is fixed at GRACE commit `b73fa9b4d427aa64ed3ad274def641efb3145877`. The archive contains the seven evaluators and launchers plus supporting repository files, with hashes. The README names lmms-eval, while the supplied launchers execute custom Python scripts; this does not uniquely identify the recipe behind the paper's scores.
3. **Built an original-script replay.** [run_upstream_matrix.py](../../evaluation/run_upstream_matrix.py) executes the archived, unmodified evaluators with native checkpoint processors/generation configuration and original launcher token limits, including MMStar 64 and MMMU MC 8/open 32. The local orchestration uses one GPU rather than the original multiprocess Slurm launch. This reproduces supplied evaluator behavior, not a proven exact copy of the authors' paper runs.
4. **Added the original two-stage reproduction workflow.** Stage 1 evaluates the five references; its existing stage-2 launcher evaluates four local LoRA-GRACE/QAT exports only if the reference gate passes. Outputs are isolated under `re_evaluation`. Completion checks cover artifacts, question identity/coverage, inference errors and score arithmetic.
5. **Completed and audited stage 1.** All 35 pairs completed with zero recorded inference errors. The gate remains blocked by score differences. Completion means the runs passed integrity checks; it does not certify the answer parsers or paper reproduction.
6. **Diagnosed saved predictions.** The initial audit checked all 35 pairs. The focused follow-up checked all 20 pairs for HallusionBench, MMBench, MMMU and MMStar, calculated label-bias metrics, and applied alternative reference scoring offline. These diagnostics did not overwrite scores, relax the gate, launch new inference or call judges.
7. **Prepared a separate corrected harness path.** The pinned lmms-eval workflow includes explicit MMMU prompt/open-scoring fixes and CPU validation. Its implementation and settings are documented, but it is not an authenticated author protocol or a completed, accepted replacement seven-suite baseline. Its shared pixel defaults also differ from the currently requested native-processor policy.
8. **Applied saved-answer CircularEval and compared metric definitions.** The [metric follow-up](metric_protocol_findings.md) and [CPU audit](metric_protocol_audit.json) establish circular MMBench scores within ±3 pp for three references and within ±5 pp for the other two. HallusionBench aAcc is the closest single metric across the five models; qAcc/fAcc do not reconcile the table. GRACE's MMStar script reports overall accuracy, equal to the mean of the six equally sized L1 categories; the harness's L2 mean is a different aggregation.
9. **Reviewed the completed non-judge harness evidence.** The earlier [lmms-eval report](../released_bf16_init_32k_b128_p100352_t2048_seed42/lmms_evaluation.md) contains four completed suites for all ten models. All three released checkpoints are within ±3 pp of the corresponding paper columns on those four suites. Vanilla discrepancies remain, including teacher ScienceQA at +9.70 pp. These results support continued development, without establishing a complete seven-suite reproduction.
10. **Agreed on local development now and harness evaluation after judge access.** The lack of GPT API access leaves its contribution to score differences unmeasured. Neither a large improvement nor a negligible effect is established. This was the decision preceding implementation below; the full lmms-eval transition remains pending.

11. **Implemented and CPU-validated corrected local scoring and the fresh-run path.** Circular MMBench reproduces the five-model offline audit exactly; MMMU/MMStar extraction, typed MMMU open matching, explicit HallusionBench population/metric, unresolved evidence and exact token/finish diagnostics are integrated. A separate resumable nine-model runner freezes native processors and a common finite generation policy, validates source coverage and rechecks saved scores without paper tolerances. Fresh GPU execution is left to the user.

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

## Completed non-judge lmms-eval evidence

The [completed four-suite report](../released_bf16_init_32k_b128_p100352_t2048_seed42/lmms_evaluation.md) covers **ScienceQA-image, AI2D, SEED-image, and MMStar** for ten models (40 jobs), including the five evaluation-only references (20 jobs). All 40 jobs have verified completion artifacts and zero judge calls. **MMMU, HallusionBench, and MMBench have no completed full-matrix scores in that historical harness round.**

For the five references, the following are **harness minus paper, in pp**, computed from the [unrounded harness scores](../released_bf16_init_32k_b128_p100352_t2048_seed42/reports/lmms_four_suite_scores.csv) and the Table 6 targets:

| Model | ScienceQA-image | AI2D | SEED-image | MMStar |
| --- | ---: | ---: | ---: | ---: |
| Vanilla 2B | +5.86 | -0.77 | +4.89 | -3.22 |
| Vanilla 8B | +9.70 | -1.70 | +1.06 | -6.35 |
| Released BF16 | +2.10 | +0.50 | -0.43 | +2.09 |
| Released W4 | +2.81 | +0.91 | +0.60 | +1.69 |
| Released W8 | +1.36 | -0.83 | -0.45 | -1.30 |

The largest absolute discrepancy is teacher ScienceQA at **+9.70 pp**. All 12 released-checkpoint cells meet the preferred ±3 pp band. Across the 20 reference cells, 15 meet ±3 pp, two are within >3 to ±5 pp, and three exceed ±5 pp. We accept this evidence as sufficient to support continued development after local bug fixes, while retaining the failed paper comparisons explicitly.

These are historical harness results with **200,704–1,605,632 pixel bounds**, ScienceQA's **2,017 image questions**, and MMStar's **mean over 18 L2 categories**. The native replay uses different preprocessing and ScienceQA scope (4,241 questions), and reports MMStar overall accuracy. The paper's exact recipe is not established. These deltas do not isolate any one setting's effect; the four-suite mean is not the seven-suite paper mean. Do not combine these four harness scores with three local scores into a purported single-protocol seven-suite result.

The earlier harness report's MMMU loading failure is historical: the later corrected harness path has CPU-validated task-loading/open-scoring fixes, but no completed fresh MMMU baseline is reported here. Its old shared-resolution defaults and proposed pixel crossover are also historical; the current instruction remains to preserve checkpoint-native processors.

## Findings from the four-suite investigation

### MMMU: original short cap and unreliable answer extraction

The original evaluator explicitly hardcodes **eight new tokens for multiple choice**. Its `--max_new_tokens 32` setting affects only open-ended questions. Removing the MC limit would be a documented protocol deviation, not restoration of an omitted setting.

The parser selects the first valid option character anywhere in uppercased text, including inside words. It can turn unfinished reasoning such as `To solve for the missing account balance,` into A. Better extraction cannot recover an answer that was never generated.

Of 847 MC questions, vanilla 2B/8B return only 49/118 bare-letter responses; 429/341 responses re-encode to at least the eight-token limit. Re-encoding is a truncation proxy because generated token IDs and finish reasons were not saved; not every capped response lacks an answer. The released BF16/W4/W8 models all return 847 bare-letter responses, so their MC differences are not explained by ambiguous parsing. An official-style open-answer helper recovers only two additional correct answers for each vanilla model, or 0.22 pp overall.

The MC extraction defect also affects complete answers: with four options, `The correct choice is A` becomes C and `The answer is B` becomes A; with five options, both become E from `The`. Correcting MC extraction can change scores in either direction. A bound based only on the 53 open questions does **not** bound the effect of rescoring all 900 questions.

The original docstring's GPT-judge claim does not identify the paper's actual scorer. The selected standard MMMU task supports deterministic MC/open scoring; GRACE also provides a separate `run_mmmu.py` path with rule extraction and GPT fallback for MC and reformulated open questions. **We have not measured GPT-assisted rescoring of the saved responses.** Its contribution to the remaining gap is unknown; it should not be assumed either to resolve the discrepancies or to be immaterial.

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

### MMStar: tested parser and aggregation changes leave gaps

The original parser takes the last standalone option letter after uppercasing, then falls back to letters inside words. It can misread `C: A forest` as A. Fixing only clear leading-answer conflicts raises vanilla 2B from 46.87 to 47.40% and 8B from 62.73 to 62.80%, too little to close the gaps.

The pinned reference prefix parser instead produces 42.33/61.93% on the same responses: permissive extraction also creates accidental correct matches. These are rule disagreements, not a semantic adjudication of every response. Changing aggregation alone to the mean of 18 subcategory scores adds only 0.94/0.34 pp with the native parser. The 64-token re-encoding proxy is reached by 299/38 vanilla responses, warranting a controlled output-completeness diagnostic. All released responses are bare letters.

### Judge requirements

| Suite | Archived GRACE scoring | Selected reference scoring path |
| --- | --- | --- |
| HallusionBench | Rule-based Yes/No; no judge | GPT correctness/consistency assessment |
| MMBench | Choice extraction and row accuracy; no judge | Rule extraction, judge fallback if unresolved, circular scoring |
| MMMU | Simplified rule-based MC/open; no judge | Selected standard task: rule-based MC/open; separate GRACE path offers GPT-assisted extraction |
| MMStar | Rule-based choices; no judge | Rule-based scoring; judge not required |

ScienceQA, AI2D and SEED-image also have judge-free scoring paths. A fixed reference answer does not imply a benchmark cannot use a judge: the judge may interpret whether free-form output matches that answer. No paid or live judging was performed in the saved-answer diagnosis. API access is currently unavailable. The planned switch to lmms-eval enables judging for the tasks that require it; it does not mean every suite uses GPT. Non-judge harness suites can already run without a key. The archived original README states that checkpoints are scored with lmms-eval, but does not specify a per-suite GPT recipe; the exact author configuration remains unverified.

## Other hypotheses and limits of the evidence

- **Nondeterminism:** archived generation explicitly uses `do_sample=False`. Numerical/software variation is possible, but run-to-run variability was not measured. It is not established as the cause of the large gaps.
- **Wrong downloaded checkpoints:** the recorded file-hash mappings for all five references match the earlier Hub-audited provenance. This weighs against a local model mix-up; the follow-up did not independently download or hash the full weights again. It cannot verify which private checkpoints produced the authors' table.
- **Undocumented settings or different evaluation code:** plausible, but unproven. Needed evidence includes exact Table 6 commands, evaluator revisions, dataset splits/revisions, processor overrides, checkpoint revisions, judge configuration and raw predictions. A discrepancy alone does not prove hidden settings, an author-side mistake or incorrect published results.

Per the user's instruction, **preserve checkpoint-native processor configurations and do not test a shared W4/W8 resolution**. The earlier shared-resolution proposal was withdrawn. Native maximum pixel budgets differ: 16,777,216 for vanilla/BF16, 200,704 for W4 and 50,176 for W8. These differences are recorded, not experimentally established causes. Future comparisons should disclose this per-checkpoint preprocessing policy rather than claim identical resolution across models.

## Agreed direction and remaining work

The workflow decision is **local development while GPT is unavailable, followed by a transition to pinned lmms-eval once judge access is supplied**. Strict paper reproduction remains a separate, unresolved objective. The transition must preserve protocol labels and compare both references and our methods under the selected workflow.

1. **Fix and validate local scoring before using it for development decisions.** Integrate the validated circular MMBench aggregation, repair MMMU/MMStar answer extraction, use documented MMMU open-answer matching, and name the HallusionBench population/metric. Record unresolved outputs and generation truncation. Deterministic scoring is useful only when it interprets answers reliably; the same parser bug can favor models that answer in different formats. **Implemented and CPU-validated:** see the [corrected protocol](../../evaluation/README_local_development.md) and [35-pair scoring audit](local_scoring_validation.json). Unresolved outputs remain explicit and count as wrong; this is deterministic local scoring, not semantic judging.
2. **Freeze the local protocol and establish fresh reference baselines.** Apply the same declared rules to all five references and our LoRA-GRACE/LoRA-QAT models. Preserve each checkpoint's native processor; do not introduce a shared W4/W8 resolution. If generation caps change to recover complete answers, use a separately named protocol with consistent finite limits and record token counts/finish reasons. **Protocol and launcher are ready:** `local_native_greedy256_v1` uses 256 new tokens for every suite/model, with an independently named short-cap variant available. All five references and four local LoRA exports passed CPU preflight. Fresh inference and baseline establishment remain pending for the user to run.
3. **Enable development through a separate readiness check.** Require valid artifacts, complete question coverage, no inference errors, validated scoring, and documented settings. Paper-score agreement is not a development prerequisite. The existing `run_re_evaluation.sh` stage-2 gate remains a reproduction gate and is still blocked; it is not modified or marked passed. The new local launcher now provides a separate artifact-validation path with complete source coverage, zero inference errors, provenance checks and deterministic rescoring; its report keeps unresolved/length-stop counts visible.
4. **Continue LoRA-GRACE/LoRA-QAT and token-reduction comparisons under the frozen local protocol.** Compare against locally evaluated references, rather than subtracting a paper score from a local method score. For token-reduction ablations, keep other evaluation settings fixed and report retained tokens, accuracy, latency, and memory. Keep all seven benchmarks visible, with missing or unresolved results explicitly labelled.
5. **Switch the main evaluation workflow to pinned lmms-eval when judge access is available.** Pin task variants, dataset revisions, prompts, generation settings, scorer/judge models, and endpoint configuration. Reconcile the historical harness's shared-pixel defaults with the current native-processor policy in a new protocol/output root before launching it. Validate the corrected MMMU path and judge access, then evaluate references and selected trained models under the same harness protocol. Retain local results as development history; do not relabel them as harness results. Non-judge tasks remain deterministic unless a different scorer is explicitly selected.
6. **Measure the judge effect and continue protocol recovery.** Where compatible, rescore the same saved responses first to isolate extraction/judging effects; rerun inference when the target protocol changes model inputs or generation. Compare changed decisions and model rankings. Ask the authors for the exact Table 6 commands, task/scorer revisions, splits, processor settings and judge configuration. GPT availability alone does not establish exact paper reproduction, and the remaining differences cannot currently be attributed to a single cause.

Suggested reporting language:

> We will use a validated, fixed, deterministic local evaluation protocol for LoRA-GRACE/LoRA-QAT and token-reduction development while GPT API access is unavailable. The corrected scorer and native-processor launcher have passed CPU validation; fresh local baselines are still pending GPU execution. The completed non-judge lmms-eval comparisons support continued development, although some reference-model scores remain outside the original reproduction tolerances. Once judge access is available, we will transition to a pinned lmms-eval workflow, with judges where required, and evaluate references and our methods under the same declared protocol. GPT-assisted rescoring has not yet been measured; its contribution and other causes of the published-score discrepancies remain unresolved. We do not claim strict reproduction of the paper's results.

## Evidence and navigation

- [Original experiment report](../released_bf16_init_32k_b128_p100352_t2048_seed42/experiment_report.md) and [earlier harness evaluation](../released_bf16_init_32k_b128_p100352_t2048_seed42/lmms_evaluation.md).
- [Protocol recovery and runner documentation](../../evaluation/README_paper_protocol.md); historical status statements there predate the completed native replay.
- [Original-source archive and hashes](../../evaluation/provenance/upstream_20260925/manifest.json).
- [Two-stage workflow instructions](../released_bf16_init_32k_b128_p100352_t2048_seed42/README_re_evaluation.md) and [launcher](../released_bf16_init_32k_b128_p100352_t2048_seed42/run_re_evaluation.sh).
- [Completed native results and provenance](../released_bf16_init_32k_b128_p100352_t2048_seed42/re_evaluation/full/) and [stage-1 gate](../released_bf16_init_32k_b128_p100352_t2048_seed42/re_evaluation/stage1_gate.md).
- [Initial diagnosis](../released_bf16_init_32k_b128_p100352_t2048_seed42/re_evaluation/diagnosis/diagnosis_report.md) and [saved-answer analysis](../released_bf16_init_32k_b128_p100352_t2048_seed42/re_evaluation/diagnosis/saved_answer_analysis.json).
- [Focused four-suite findings](../released_bf16_init_32k_b128_p100352_t2048_seed42/re_evaluation/diagnosis/four_suite_followup.md), [analysis script](../released_bf16_init_32k_b128_p100352_t2048_seed42/re_evaluation/diagnosis/analyze_four_suites.py), and [machine-readable results](../released_bf16_init_32k_b128_p100352_t2048_seed42/re_evaluation/diagnosis/four_suite_analysis.json).
