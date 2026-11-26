from pathlib import Path
p = Path('reproduction/GRACE/experiments/reproduction_eval_only/README.md')
s = p.read_text()
Path('/tmp/grace_eval_readme_before.md').write_text(s)
def replace(old, new):
    global s
    assert s.count(old) == 1, old[:120]
    s = s.replace(old, new, 1)
replace('Status as of **2026-09-26**: the five reference models completed all seven benchmarks (**35/35 pairs**), but the original-script replay did **not** reproduce all published scores within our predefined tolerances. Several scoring issues have been identified; the remaining discrepancies are unresolved. A validated replacement protocol and fresh baselines are still pending.',
'''Status as of **2026-09-26**: the original-script replay completed **35/35 reference pairs**, but strict paper reproduction remains unresolved. A separate pinned lmms-eval round completed **40/40 pairs across ten models and four non-judge suites**, including all five references. The agreed development direction is to use **validated local, deterministic scoring while GPT access is unavailable**, then switch the main evaluation workflow to **pinned lmms-eval with judges where the selected tasks require them** once API access is supplied. Known local parser fixes, validation, and fresh baselines remain pending; this decision does not certify the existing proxy scores.''')
replace('The objective is to reproduce the GRACE paper\'s evaluation-only results before comparing our own LoRA-GRACE and LoRA-QAT training methods. Training may be resource-constrained, but evaluation differences must be explicit and controlled.',
'''The objective is to establish reliable evaluation baselines for our LoRA-GRACE, LoRA-QAT, and token-reduction experiments, while continuing to investigate reproduction of the GRACE paper. **Strict paper-score agreement is no longer a prerequisite for development once the local evaluation protocol is validated.** Reference checkpoints and our methods must be compared under the same declared evaluation rules, with protocol differences explicit and controlled. This records the agreed workflow change; implementing a separate development-readiness check remains pending.''')
replace('Numerical agreement is separate from proof of protocol identity.',
'''Numerical agreement is separate from proof of protocol identity. These remain the historical **paper-reproduction thresholds**. Accepting the evidence as sufficient to continue development does not retroactively turn a +9.70 pp discrepancy into a reproduction pass or change the existing gate.''')
replace('4. **Added a two-stage workflow.** Stage 1 evaluates the five references; stage 2 evaluates four local LoRA-GRACE/QAT exports only if the reference gate passes.',
'''4. **Added the original two-stage reproduction workflow.** Stage 1 evaluates the five references; its existing stage-2 launcher evaluates four local LoRA-GRACE/QAT exports only if the reference gate passes.''')
needle='See [evaluation differences from the original repository]'
assert s.count(needle)==1
s=s.replace(needle,'''8. **Applied saved-answer CircularEval and compared metric definitions.** The [metric follow-up](metric_protocol_findings.md) and [CPU audit](metric_protocol_audit.json) establish circular MMBench scores within ±3 pp for three references and within ±5 pp for the other two. HallusionBench aAcc is the closest single metric across the five models; qAcc/fAcc do not reconcile the table. GRACE's MMStar script reports overall accuracy, equal to the mean of the six equally sized L1 categories; the harness's L2 mean is a different aggregation.
9. **Reviewed the completed non-judge harness evidence.** The earlier [lmms-eval report](../released_bf16_init_32k_b128_p100352_t2048_seed42/lmms_evaluation.md) contains four completed suites for all ten models. All three released checkpoints are within ±3 pp of the corresponding paper columns on those four suites. Vanilla discrepancies remain, including teacher ScienceQA at +9.70 pp. These results support continued development, without establishing a complete seven-suite reproduction.
10. **Agreed on local development now and harness evaluation after judge access.** The lack of GPT API access leaves its contribution to score differences unmeasured. Neither a large improvement nor a negligible effect is established. Known local scoring bugs must be fixed before freezing new baselines; the full lmms-eval transition and a separate development-readiness check are planned, not completed.

'''+needle,1)
needle='## Findings from the four-suite investigation\n'
insert='''## Completed non-judge lmms-eval evidence

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

'''
replace(needle,insert+needle)
needle='### MMBench: aggregation explains much of the gap\n'
insert='''The MC extraction defect also affects complete answers: with four options, `The correct choice is A` becomes C and `The answer is B` becomes A; with five options, both become E from `The`. Correcting MC extraction can change scores in either direction. A bound based only on the 53 open questions does **not** bound the effect of rescoring all 900 questions.

The original docstring's GPT-judge claim does not identify the paper's actual scorer. The selected standard MMMU task supports deterministic MC/open scoring; GRACE also provides a separate `run_mmmu.py` path with rule extraction and GPT fallback for MC and reformulated open questions. **We have not measured GPT-assisted rescoring of the saved responses.** Its contribution to the remaining gap is unknown; it should not be assumed either to resolve the discrepancies or to be immaterial.

'''
replace(needle,insert+needle)
replace('### MMStar: parser defects are real but insufficient','### MMStar: tested parser and aggregation changes leave gaps')
replace('| MMMU | Rule-based MC/open; no judge | Rule-based MC/open scoring; judge not required |', '| MMMU | Simplified rule-based MC/open; no judge | Selected standard task: rule-based MC/open; separate GRACE path offers GPT-assisted extraction |')
needle='No paid or live judging was performed in the saved-answer diagnosis.'
replace(needle,needle+''' API access is currently unavailable. The planned switch to lmms-eval enables judging for the tasks that require it; it does not mean every suite uses GPT. Non-judge harness suites can already run without a key. The archived original README states that checkpoints are scored with lmms-eval, but does not specify a per-suite GPT recipe; the exact author configuration remains unverified.''')
start=s.index('## Agreed direction and remaining work\n')
end=s.index('## Evidence and navigation\n',start)
s=s[:start]+'''## Agreed direction and remaining work

The workflow decision is **local development while GPT is unavailable, followed by a transition to pinned lmms-eval once judge access is supplied**. Strict paper reproduction remains a separate, unresolved objective. The transition must preserve protocol labels and compare both references and our methods under the selected workflow.

1. **Fix and validate local scoring before using it for development decisions.** Integrate the validated circular MMBench aggregation, repair MMMU/MMStar answer extraction, use documented MMMU open-answer matching, and name the HallusionBench population/metric. Record unresolved outputs and generation truncation. Deterministic scoring is useful only when it interprets answers reliably; the same parser bug can favor models that answer in different formats. The offline MMBench correction is complete, but the unified corrected local evaluator is not yet ready.
2. **Freeze the local protocol and establish fresh reference baselines.** Apply the same declared rules to all five references and our LoRA-GRACE/LoRA-QAT models. Preserve each checkpoint's native processor; do not introduce a shared W4/W8 resolution. If generation caps change to recover complete answers, use a separately named protocol with consistent finite limits and record token counts/finish reasons. Any required new inference remains pending.
3. **Enable development through a separate readiness check.** Require valid artifacts, complete question coverage, no inference errors, validated scoring, and documented settings. Paper-score agreement is not a development prerequisite. The existing `run_re_evaluation.sh` stage-2 gate remains a reproduction gate and is still blocked; this documentation update does not modify the launcher or mark it passed. Implement a separate development path before using that workflow for new comparisons.
4. **Continue LoRA-GRACE/LoRA-QAT and token-reduction comparisons under the frozen local protocol.** Compare against locally evaluated references, rather than subtracting a paper score from a local method score. For token-reduction ablations, keep other evaluation settings fixed and report retained tokens, accuracy, latency, and memory. Keep all seven benchmarks visible, with missing or unresolved results explicitly labelled.
5. **Switch the main evaluation workflow to pinned lmms-eval when judge access is available.** Pin task variants, dataset revisions, prompts, generation settings, scorer/judge models, and endpoint configuration. Reconcile the historical harness's shared-pixel defaults with the current native-processor policy in a new protocol/output root before launching it. Validate the corrected MMMU path and judge access, then evaluate references and selected trained models under the same harness protocol. Retain local results as development history; do not relabel them as harness results. Non-judge tasks remain deterministic unless a different scorer is explicitly selected.
6. **Measure the judge effect and continue protocol recovery.** Where compatible, rescore the same saved responses first to isolate extraction/judging effects; rerun inference when the target protocol changes model inputs or generation. Compare changed decisions and model rankings. Ask the authors for the exact Table 6 commands, task/scorer revisions, splits, processor settings and judge configuration. GPT availability alone does not establish exact paper reproduction, and the remaining differences cannot currently be attributed to a single cause.

Suggested reporting language:

> We will use a validated, fixed, deterministic local evaluation protocol for LoRA-GRACE/LoRA-QAT and token-reduction development while GPT API access is unavailable. Known scoring fixes and fresh local baselines are still pending. The completed non-judge lmms-eval comparisons support continued development, although some reference-model scores remain outside the original reproduction tolerances. Once judge access is available, we will transition to a pinned lmms-eval workflow, with judges where required, and evaluate references and our methods under the same declared protocol. GPT-assisted rescoring has not yet been measured; its contribution and other causes of the published-score discrepancies remain unresolved. We do not claim strict reproduction of the paper's results.

'''+s[end:]
p.write_text(s)
