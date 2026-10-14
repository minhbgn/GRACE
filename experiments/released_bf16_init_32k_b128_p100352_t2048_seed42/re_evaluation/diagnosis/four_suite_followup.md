# Four-suite follow-up — 2026-09-26

Recommendation: retain all suites in the report, distinguish archived GRACE-script scores from reference-scoring diagnostics, and do not claim paper-protocol equivalence for unresolved cells. Do not select benchmarks by whether their scores agree with the paper. Prioritize MMBench metric reconciliation and MMMU/MMStar response-format diagnosis. A judge alone is unlikely to explain the observed pattern.

This follow-up honors the request to preserve the original repository's native processor configurations. **The earlier recommendation to test a shared W4/W8 resolution is withdrawn.** No resolution override, longer-generation run, GPU inference or judge request was performed. Historical benchmark results, acceptance thresholds and the stage-one gate are unchanged.

Evidence: [reproducible script](analyze_four_suites.py), [detailed output](four_suite_analysis.json), and [earlier token/response audit](saved_answer_analysis.json). The new script verified completion artifacts and original result hashes for all 20 model/suite pairs. It reuses the pinned lmms-eval scoring helpers without calling their API paths. Reproduce from the GRACE root:

```bash
.venv-lmms-eval/bin/python experiments/released_bf16_init_32k_b128_p100352_t2048_seed42/re_evaluation/diagnosis/analyze_four_suites.py
```

## MMMU: eight tokens are explicitly upstream

The archived upstream `evaluation/mmmu/eval_mmmu.py`, commit `b73fa9b4d427aa64ed3ad274def641efb3145877`, hardcodes `max_toks = 8` at line 261 for multiple-choice questions. The argument definition at lines 209–210 explicitly says multiple-choice forces eight. The Slurm launcher passes `--max_new_tokens 32`; that controls open-ended questions only. This is not a cap introduced by our constrained training or local wrapper.

Consequently, removing it would be a documented deviation from the released evaluator. Preserve eight for the archived replay. A separately named diagnostic protocol could use a larger finite generation budget, identical across models, with actual token IDs and finish reasons logged. It must not silently replace the faithful replay or be tuned to achieve a target paper score.

The upstream parser uppercases the response and picks the first valid option character anywhere, including inside a word. For example, `To solve for the missing account balance,` becomes A even though no answer was stated. A valid extractor should recognize explicit answer markers, standalone or parenthesized choices and unambiguous option text; conflicting or missing answers should remain unresolved instead of scanning inside words. Parsing cannot recover an answer cut off before it was generated.

| MMMU evidence | Vanilla 2B | Vanilla 8B | BF16 | W4 | W8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MC questions | 847 | 847 | 847 | 847 | 847 |
| Bare-letter MC responses | 49 | 118 | 847 | 847 | 847 |
| Re-encoded MC responses at least eight tokens | 429 | 341 | 0 | 0 | 0 |
| MC correct | 365 | 469 | 632 | 543 | 512 |
| Open correct, out of 53 | 6 | 6 | 16 | 11 | 11 |

Re-encoding is a truncation proxy, not an exact finish-reason measurement; some capped responses already state a choice. The pinned official-style open-answer helper recovers only two answers for each vanilla model, or 0.22 pp overall. It changes none of the released-model open scores.

The W4/W8 MC gaps are actual differences in saved selected options, not ambiguous parsing: against BF16, W4 loses 152 previously correct questions and gains 63 (net −89); W8 loses 164 and gains 44 (net −120). All models used the same MC IDs and gold labels. Neither a judge nor longer generation can repair already completed bare-letter responses by rescoring them. Preserve native processors as requested; residual causes remain unproven without the authors' exact Table 6 run records.

MMMU's [reference scorer](https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py) supports rule-based MC and open scoring; no judge is required. Its unparseable MC fallback uses random guessing, so adopting it requires explicitly recording that policy and its seed. No random guessing was used in this diagnosis.

## HallusionBench: label imbalance does not explain the vanilla paper gap

Gold labels are **645 No / 484 Yes**, or **57.13% / 42.87%**. Always predicting No scores 57.13%. This baseline is useful, but cannot itself explain the difference between our scores and the paper's scores without knowing the paper's predictions and metric.

Positive class below is Yes. Balanced accuracy averages Yes recall and No recall, weighting both equally. Invalid predictions count as errors and remain separate from No.

| Model | Accuracy % | Balanced accuracy % | Yes precision % | Yes recall % | No recall % | Macro F1 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla 2B | 62.44 | 64.19 | 54.41 | 76.45 | 51.94 | 62.41 |
| Vanilla 8B | 72.90 | 71.92 | 69.69 | 65.08 | 78.76 | 72.08 |
| Released BF16 | 67.05 | 63.97 | 68.79 | 42.36 | 85.58 | 63.61 |
| Released W4 | 64.57 | 63.73 | 58.82 | 57.85 | 69.61 | 63.76 |
| Released W8 | 59.17 | 59.08 | 52.21 | 58.47 | 59.69 | 59.04 |

Vanilla 2B has a **Yes prediction bias**: 680/1129 outputs are Yes, despite Yes being the minority gold label. Its confusion counts are TP370, FP310, TN335, FN114. Equal class weighting increases its accuracy by 1.75 pp. Vanilla 8B has TP315, FP137, TN508, FN169; equal weighting lowers accuracy by only 0.98 pp. Neither accounts for their +11.04/+11.80 pp paper differences. BF16 displays the clearer majority-No tendency: 831 No predictions and only 42.36% Yes recall. Thus response bias exists, but the proposed explanation is insufficient for the vanilla discrepancy.

The upward direction is not universal: BF16 differs by +0.15 pp, W4 by −0.83 pp and W8 by −6.93 pp. W8 has eight invalid responses, retained as errors. Both vanilla models output only canonical Yes/No strings, leaving no ambiguous prose for a correctness judge to interpret.

Image-only accuracy on the 951 image rows is 64.25/73.82% for vanilla 2B/8B, farther above the paper than the full-set scores. The remaining 178 rows are text-only. The local question-group/figure-group all-correct metrics are 27.91/35.84% for 2B and 43.52/51.16% for 8B; these measure different units and must not be substituted for per-answer accuracy or described as precision/recall.

The [official HallusionBench evaluator](https://github.com/tianyi-lab/HallusionBench/blob/main/evaluation.py) uses GPT correctness and consistency assessment. The archived GRACE evaluator instead matches Yes/No labels. A reference-judge audit can establish agreement with that procedure, but there is no evidence that it would remove the vanilla +11–12 pp discrepancy. The relevant unknowns include exact split, prompt and reported metric.

## MMBench: circular aggregation explains much of the discrepancy

The [benchmark's official repository](https://github.com/open-compass/MMBench) describes CircularEval. Our archived GRACE replay averages 4,329 rotation rows; circular scoring requires all rotations of each of 1,164 base questions to be correct.

This follow-up additionally applies the pinned harness's `MMBench_Evaluator.prefetch_answer` to saved raw answers and original option text. Unlike the earlier circular diagnostic, it no longer inherits GRACE's extracted choices. Unresolved rows remain unresolved, producing lower/upper bounds over possible judge outcomes rather than guessed answers.

| Model | Paper % | GRACE row accuracy % | Circular, GRACE choices % | Circular, reference extraction % | Reference minus paper pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vanilla 2B | 78.40 | 83.81 | 77.75 | 76.80 | −1.60 |
| Vanilla 8B | 84.50 | 89.79 | 85.82 | 84.71–85.05 | +0.21–+0.55 |
| Released BF16 | 86.40 | 94.66 | 91.15 | 91.15 | +4.75 |
| Released W4 | 84.60 | 92.63 | 88.66 | 88.66 | +4.06 |
| Released W8 | 85.50 | 92.49 | 88.40 | 88.40 | +2.90 |

Only vanilla 8B has unresolved answers: nine rows, with four groups whose final score could depend on a judge. Other groups containing unresolved answers are already disqualified by an explicitly wrong rotation. The maximum judge-dependent overall difference is 4/1164 = 0.34 pp. All released answers are bare letters and need no semantic judge fallback. Remaining BF16/W4 discrepancies are within the previously defined barely-acceptable five-point band, but outside the preferred three-point band.

These are same-response reference-scoring diagnostics, not proof that the authors used that exact split/prompt/version. Retain MMBench; label row accuracy versus circular accuracy explicitly and do not mix them in the same comparison column.

## MMStar: reference parsing does not close the vanilla gap

The archived parser selects the last standalone A–D after uppercasing; if none exists, it scans backward inside words. This can misread `C: A forest` as A. Correcting only clear leading-answer conflicts raises 2B from 46.87 to 47.40%, and 8B from 62.73 to 62.80%. That is too small to explain their paper gaps.

A broader same-response audit using the pinned harness's MMStar `exact_match` rule produces:

| Model | GRACE parser, row mean % | Reference parser, row mean % | Reference parser, 18-category mean % |
| --- | ---: | ---: | ---: |
| Vanilla 2B | 46.87 | 42.33 | 42.53 |
| Vanilla 8B | 62.73 | 61.93 | 62.29 |
| Released BF16 | 68.00 | 68.00 | 68.18 |
| Released W4 | 66.20 | 66.20 | 66.86 |
| Released W8 | 63.67 | 63.67 | 64.25 |

For 2B, 83 previously correct rows become wrong and 15 become correct under this parser; for 8B the counts are 13 and one. This demonstrates parser dependence, including accidental matches in GRACE's permissive extraction. It does not establish that every rule disagreement has been semantically adjudicated. The reference helper itself is a simple prefix rule, not an oracle.

MMStar's six broad categories each have 250 rows, but its 18 subcategories are not equally sized. Holding the native parser fixed, switching only to the pinned harness's subcategory mean changes 2B/8B by +0.94/+0.34 pp, again insufficient. All three released models return bare letters; reference parsing leaves their row accuracies unchanged. The existing token audit found 299/38 responses at the 64-token re-encoding proxy for 2B/8B, including 295/36 reasoning prefixes. A fixed diagnostic subset with complete outputs is more useful than adding a judge. MMStar has a judge-free scoring path.

## Reporting and next steps

1. Keep all four suites visible, including mismatches and failed protocol checks. Do not discard a suite because its delta exceeds a threshold. Keep archived GRACE-script reproduction results intact.
2. Report MMBench circular diagnostics separately now. Only four vanilla-8B groups potentially need semantic judging; releases do not. Do not silently rewrite the stage-one gate using a new metric.
3. For MMMU/MMStar, retain the native caps for repository replay. If pursuing causal diagnosis, use a separately labelled finite larger-budget protocol on a fixed representative subset, preserving native resolutions and applying identical changes to all five models. Evaluate extraction without consulting target paper scores. The present request has not triggered such a run.
4. Keep HallusionBench's label balance, per-class metrics, image/text coverage and exact metric definition in the report. A small official-judge agreement audit may be useful, but cannot be assumed to reconcile the scores.
5. Withhold claims of faithful paper-level equivalence for unresolved comparisons. If an evaluator is known to extract unfinished text as an answer, those cells should not support a model-quality headline until validated. Continue to show them as diagnostic/repository-replay evidence. Once a local scoring protocol is validated and frozen, use it for controlled comparisons and retain published paper numbers as separately labelled references. Do not call a reduced-suite average the paper's seven-suite average.

The key distinction is public-code fidelity versus verified fidelity to the authors' actual Table 6 runs. We can establish the former. The current public materials and local score agreement do not establish the latter; exact author commands, predictions and evaluator revisions could resolve remaining uncertainty.
