# Metric and protocol follow-up — 2026-09-26

The saved answers support a concrete MMBench scoring correction, but do not establish full reproduction. Tolerances remain ≤3 percentage points (pp) preferred, >3 to ≤5 pp barely acceptable for these unjudged results. No GPU inference, resolution changes, judge calls, or gate changes were made.

Numbers below were recomputed from hash-verified native replay artifacts by [rescore_saved_protocols.py](../../evaluation/rescore_saved_protocols.py). [The JSON audit](metric_protocol_audit.json) contains input/scorer/dataset hashes, per-group MMBench extraction evidence, metric comparisons, and MMMU upper bounds. Targets are [GRACE v5 Table 6](https://arxiv.org/html/2601.22709v5#A1.T6).

## 1. HallusionBench: aAcc is closest across the five models

Definitions in the supplied GRACE evaluator:

- **aAcc:** fraction of individual answers that are correct, over 1,129 image and text-only rows.
- **qAcc:** fraction of 455 related-question groups for which every answer is correct.
- **fAcc:** fraction of 346 figure groups for which every answer is correct; text-only rows excluded.

| Model | Paper | aAcc | qAcc | fAcc | aAcc − paper | aAcc acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Vanilla 2B | 51.40 | 62.44 | 27.91 | 35.84 | +11.04 | Fail |
| Vanilla 8B | 61.10 | 72.90 | 43.52 | 51.16 | +11.80 | Fail |
| Released BF16 | 66.90 | 67.05 | 34.73 | 37.28 | +0.15 | Pass |
| Released W4 | 65.40 | 64.57 | 31.65 | 36.13 | −0.83 | Pass |
| Released W8 | 66.10 | 59.17 | 23.08 | 31.79 | −6.93 | Fail |

Mean absolute discrepancy over five models is **6.15 pp for aAcc**, **30.00 pp for qAcc**, and **23.74 pp for fAcc**. Thus aAcc is the best single metric across the table. For vanilla 8B individually, fAcc happens to be closer, but still misses by 9.94 pp. No qAcc/fAcc result passes even the five-point allowance. Selecting different metrics for different models would not be a valid comparison.

This rules out a simple aAcc/qAcc/fAcc swap as the explanation for the observed discrepancies. It does **not** prove which metric, population, or judge produced the paper's HallB column. The [official HallusionBench pipeline](https://github.com/tianyi-lab/HallusionBench/blob/main/evaluation.py) assesses free-form answers with a judge and has special uncertainty handling in its utilities. The values here retain GRACE's simpler Yes/No correctness rules. Image-only versus image-plus-text populations must also be named explicitly.

## 2. MMBench: CircularEval applied to saved predictions

[MMBench defines CircularEval](https://github.com/open-compass/MMBench#evaluation) as requiring success across option rotations of each base question. A per-row average gives partial credit for inconsistent rotations and is a different metric.

The new offline scorer groups by `index % 1_000_000`, checks exact coverage and gold labels against the pinned TSV, uses the pinned lmms-eval MMBench rule extractor, and requires every provided rotation in a group to be correct. It scores **1,164 base questions**, using **4,329 saved rows per model**.

| Model | Paper | Original row accuracy | Circular accuracy | Circular − paper | Acceptance |
| --- | ---: | ---: | ---: | ---: | --- |
| Vanilla 2B | 78.40 | 83.81 | 76.80 | −1.60 | Pass |
| Vanilla 8B | 84.50 | 89.79 | 84.71–85.05 | +0.21 to +0.55 | Pass throughout bound |
| Released BF16 | 86.40 | 94.66 | 91.15 | +4.75 | Barely acceptable |
| Released W4 | 84.60 | 92.63 | 88.66 | +4.06 | Barely acceptable |
| Released W8 | 85.50 | 92.49 | 88.40 | +2.90 | Pass |

The conservative exact-match score for vanilla 8B is **84.71%**, counting unresolved extraction as incorrect. Nine rows are unresolved, but only four base-question outcomes can change after adjudication. Therefore 84.71–85.05% is a bound on possible judge completion, not a confidence interval or a completed judged score. Other models have no unresolved rows. No random fallback is used.

Dataset detail: the TSV supplies 926 four-row groups, 149 three-row groups and 89 two-row groups. Of the three-row groups, 52 have four populated option columns. The scorer follows the provided TSV groups, as the pinned reference evaluator does; it does not invent additional rotations. This is recorded in the audit and should be checked when comparing dataset revisions with the authors.

Circular scoring therefore brings **all five models inside five points**, with **three inside three points**. BF16 and W4 still fail the preferred threshold. The archived row scores and stage-1 gate remain historical replay outputs; the corrected values are a separate scoring result, not a silent rewrite of the original protocol. Future validated comparison reports should use this declared circular metric.

## 3. MMStar: supplied code reports overall accuracy

The [pinned GRACE MMStar evaluator](https://github.com/ForeverBlue816/GRACE/blob/b73fa9b4d427aa64ed3ad274def641efb3145877/evaluation/MMStar/eval_mmstar.py) computes `overall_acc = correct / n`. Its `by_l1` and `by_l2` accumulators produce diagnostic category breakdowns; they do not feed a special transformation of the reported overall score.

This supports overall accuracy as the intended metric of the supplied code. It does not prove that the paper's private run used that exact code. Table 6 and Appendix D.4 do not specify a Qwen3-specific aggregation recipe.

The [MMStar dataset](https://github.com/MMStar-Benchmark/MMStar) has six L1 categories with exactly 250 questions each. Consequently, the unweighted mean of the six L1 accuracies **equals** overall accuracy. Its 18 L2 categories have unequal sizes, so their unweighted mean differs. A sample-count-weighted average of either category level always recovers overall accuracy.

| Model | Paper | Overall = mean L1 | Unweighted mean L2 |
| --- | ---: | ---: | ---: |
| Vanilla 2B | 58.30 | 46.87 | 47.80 |
| Vanilla 8B | 70.90 | 62.73 | 63.08 |
| Released BF16 | 67.30 | 68.00 | 68.18 |
| Released W4 | 65.80 | 66.20 | 66.86 |
| Released W8 | 66.50 | 63.67 | 64.25 |

The pinned lmms-eval `average` metric takes the mean of 18 L2 scores. That is a harness aggregation difference, not evidence GRACE used it. Holding predictions and parsing fixed, the aggregation change adds only 0.18–0.94 pp and cannot explain the vanilla gaps. MMStar also defines multimodal-gain and leakage diagnostics; there is no evidence that GRACE's MStar accuracy column substitutes those metrics.

## 4. MMMU: the docstring conflates two evaluation paths

The quoted warning is present in the original GRACE evaluator. Its claim of an obligatory “official GPT-judge eval” is not supported by the [official MMMU evaluation code](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/eval_utils.py): `parse_open_response` extracts answer candidates and numbers; `eval_open` checks normalized text/numeric answers, without an LLM judge. [MMMU's official repository](https://github.com/MMMU-Benchmark/MMMU#evaluation) directs users to its evaluation pipeline.

Three distinct implementations must be kept separate:

1. **GRACE `eval_mmmu.py`:** simplified MC/open rules, including the unreliable character-scanning MC parser. This is the completed native replay.
2. **Official MMMU rules:** richer candidate extraction, normalization and matching. A judge is not required for this path; “rule-based” does not mean every homemade scorer is equivalent.
3. **GRACE `run_mmmu.py` plus `eval_utils.py`:** a separate `MMMU_DEV_VAL.tsv` workflow with rule extraction and GPT fallback. `dataset_utils.MMMU_preproc` reformulates open questions with A = gold answer and B = `Other Answers`; `run_evaluation` maps their gold label to A. GPT, when needed, matches the saved response to those options. It is not the same open-answer scorer as the official MMMU rules. It also contains a random-choice fallback after repeated extraction failures, which must be disabled in a validated reproduction.

The earlier statement that MMMU “does not need GPT” should have explicitly referred to path 2. It did not establish that path 1 was adequate, or that the paper used path 2 rather than path 3. Exact Table 6 judge usage remains unverified.

The open-score concern is valid, but its possible effect on these saved outputs is bounded. There are **847 MC and 53 open questions**. Even changing every currently incorrect open answer to correct, while retaining the recorded MC scores, gives:

| Model | Current MMMU | Ceiling if every open answer were correct | Paper |
| --- | ---: | ---: | ---: |
| Vanilla 2B | 41.22 | 46.44 | 53.40 |
| Vanilla 8B | 52.78 | 58.00 | 69.60 |
| Released BF16 | 72.00 | 76.11 | 72.10 |
| Released W4 | 61.56 | 66.22 | 70.50 |
| Released W8 | 58.11 | 62.78 | 71.30 |

This is an intentionally optimistic mathematical ceiling, not a predicted judge result. Even at this ceiling, vanilla 2B/8B and W8 remain more than five points below the paper; W4 remains outside three points. Changing open scoring alone cannot recover the published results. If a judge also changes MC extraction, that is a separate effect, outside this bound.

The prior official-style open-helper diagnostic recovered **two answers per vanilla model (+0.22 pp overall)** and none for BF16/W4/W8. For vanilla models, the eight-token MC cap and malformed answer extraction remain stronger leads. The released models already output bare option letters on all MC rows, so that explanation does not account for their W4/W8 gaps.

## What to do next

1. **Use circular MMBench scoring in the validated comparison protocol.** The saved outputs are sufficient for the results above. Resolve the four undecided 8B groups only if a fully judge-completed scalar is needed; their uncertainty cannot change the acceptance conclusion.
2. **Fix MMMU/MMStar answer handling, then run a controlled output-length diagnostic.** Use the same declared prompt/parser and finite generation budget across all five reference models, with each checkpoint's native processor. Save generated token counts and finish reasons. Apply official MMMU open-answer rules. Keep this as a separately named protocol because longer generation changes the original recipe.
3. **Treat GPT scoring as a separate comparison if the authors confirm it was used.** First audit the 53 saved open responses per model and compare deterministic scoring with semantic adjudication. Do not expect an open-only judge change to repair the large deficits.
4. **Request the exact Table 6 recipe from the authors:** HallB metric and population; MMBench split/revision and circular scorer; MMStar aggregation; MMMU entry point, split and judge configuration; checkpoint revisions and generation settings. The evidence does not yet justify calling the published table wrong. No message has been sent.
5. **Freeze the validated seven-suite protocol and establish fresh reference baselines before comparing our trained methods.** Continue reporting paper discrepancies separately. MMBench is substantially reconciled; HallusionBench, vanilla MMStar, and several MMMU scores remain unresolved.

Reproduce the CPU audit from the workspace root:

```bash
reproduction/GRACE/.venv-lmms-eval/bin/python reproduction/GRACE/evaluation/rescore_saved_protocols.py \
  --full-root reproduction/GRACE/experiments/released_bf16_init_32k_b128_p100352_t2048_seed42/re_evaluation/full \
  --output reproduction/GRACE/experiments/reproduction_eval_only/metric_protocol_audit.json
```
