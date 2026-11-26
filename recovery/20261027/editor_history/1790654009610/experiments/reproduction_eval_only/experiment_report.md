# Local evaluation experiment report: LoRA-GRACE and LoRA-QAT

Prepared **2026-09-29** from the full runs dated **2026-09-26–27 UTC**. Protocol: **`local_native_greedy256_v1`**. Scope: nine checkpoints × seven benchmarks; smoke runs are excluded.

This report follows the [agreed direction in the README](README.md#agreed-direction-and-remaining-work). All comparisons below use fresh local results from this protocol. The README's statements that fresh inference is pending predate these runs. Paper scores and historical lmms-eval results are not mixed into these tables.

## Main findings for discussion

**LoRA-GRACE W4 is 1.44 percentage points (pp) below released GRACE W4 on the equal-weight seven-benchmark mean; LoRA-GRACE W8 is 2.22 pp above released GRACE W8. Neither mean describes uniform agreement.** The largest negative differences are on MMStar: **−10.80 pp at W4 and −10.47 pp at W8**. W8 gains elsewhere offset its MMStar loss.

**LoRA-QAT outperforms LoRA-GRACE on the seven-benchmark mean at both precisions:** by **1.16 pp at W4** and **2.20 pp at W8**. The present checkpoints therefore do not demonstrate an overall benefit from adding the LoRA-GRACE training recipe. LoRA-QAT W8 is the strongest evaluated low-bit checkpoint on this mean.

The results support using the local setup to investigate and compare development changes, provided the MMStar regression remains visible and the MMBench validation issue below is repaired before declaring the runner fully ready. They do **not** establish that LoRA-GRACE is an interchangeable substitute for released GRACE, or that the paper has been reproduced. The acceptable task-specific loss remains a decision for the supervisor; no new acceptance threshold is assumed here.

## Evidence status and the MMBench completion issue

There are **63/63 saved full result files**, covering **264,780 prediction rows** (29,420 per model, including MMBench rotation rows), with **zero recorded inference errors** and no unknown generation telemetry. However, the existing [combined runner summary](local_development/local_native_greedy256_v1/full/local_scores.json) reports **`artifact_ready: false`**: **54/63 pairs have verified completion markers**, while all nine MMBench pairs lack `validated.json` and `COMPLETE.json`.

A CPU-only audit for this report reproduced the cause. In `validate()` in [run_local_development.py](../../evaluation/run_local_development.py), the saved circular summary is compared directly with the recomputed Python dictionary. `circular.rotation_count_histogram` has JSON string keys (`"4"`, `"3"`, `"2"`) on disk but integer keys (`4`, `3`, `2`) when recomputed. This alone triggers `Summary disagrees with corrected scoring/diagnostics` for every full MMBench run.

For the report audit, only those histogram keys were converted back to integers **in an in-memory copy**. All nine then passed the existing validator, including source IDs/gold labels, coverage, native processor settings, generation policy/token evidence, deterministic answer rescoring, circular groups and arithmetic. The original and recomputed summaries also match after JSON normalization. All other 54 pairs passed the unmodified validator and completion/provenance hash checks. Evaluator source hashes and the experiment-configuration hash match the frozen protocol.

Consequently, **MMBench scores marked † below are saved scores independently verified by this CPU audit, but remain unmarked as complete by the original runner**. All seven-suite means include those audited scores. This report changes neither predictions nor scores, source code, completion markers or the original readiness flag. The audit did not rehash the checkpoint weight files; it checked the saved provenance records against completion hashes where available. The [machine-readable audit](experiment_report_audit.json) records each selected result path, SHA-256, validation status, score and diagnostic counts.

## Comparison policy and naming

Scores are accuracy percentages; differences are **left minus right in pp**, calculated from unrounded primary scores. “Mean” is the **unweighted arithmetic mean of the seven benchmark percentages**, not accuracy pooled across examples and not a paper-defined aggregate. Each benchmark gets equal weight, so SEED's larger sample count does not dominate the mean. Higher is better.

| Report name | Folder ID |
| --- | ---: |
| Vanilla 2B | `vanilla_2b` |
| Vanilla 8B teacher | `vanilla_8b_teacher` |
| Released BF16 / SFT | `released_bf16` |
| Released GRACE W4 | `released_grace_w4` |
| Released GRACE W8 | `released_grace_w8` |
| LoRA-QAT W4 | `qat_w4` |
| LoRA-GRACE W4 | `grace_w4` |
| LoRA-QAT W8 | `qat_w8` |
| LoRA-GRACE W8 | `grace_w8` |

The local training [experiment configuration](../released_bf16_init_32k_b128_p100352_t2048_seed42/experiment.json) identifies released BF16 as the initialization for all four LoRA exports, and vanilla 8B as the LoRA-GRACE teacher. Their saved launch recipes specify a 32,768-example subset, 256 steps, global batch 128, seed 42, group size 128, and attention-projection LoRA with rank 32 / alpha 64; all four training summaries record completion at step 256. Thus released-versus-local differences include differences in training recipe and checkpoint, in addition to the native image budgets. The previous local `sft_2b` model has no run in this nine-model matrix and is not substituted for released BF16.

The [frozen protocol](local_development/local_native_greedy256_v1/full/protocol.json) uses greedy decoding, one beam, repetition penalty 1, BF16 loading, FlashAttention 2, and **256 new tokens for every model and suite**, including both MMMU question types. Scoring is `deterministic_local_v1`, with no judge calls. Unresolved outputs count as wrong; length-stopped outputs are scored and counted explicitly. Full details are in the [local evaluation documentation](../../evaluation/README_local_development.md).

| Benchmark | Evaluated population | Primary metric |
| --- | ---: | ---: |
| HallusionBench | 1,129: 951 image + 178 text | Rule-based per-answer aAcc |
| MMBench EN dev 20231003 | 4,329 rotations / 1,164 base questions | Circular accuracy: all supplied rotations correct |
| ScienceQA | 4,241 test questions, image + text | Micro accuracy |
| AI2D | 3,088 test questions | Micro accuracy |
| MMMU | 900 validation: 847 MC + 53 open | Combined MC/open micro accuracy |
| SEED image | 14,233 image questions | Micro accuracy |
| MMStar | 1,500 questions | Overall accuracy; six equal-sized L1 categories |

Every checkpoint uses its own saved processor. The native-processor policy is shared, while the actual image budgets differ:

| Checkpoints | Minimum pixels | Maximum pixels |
| --- | ---: | ---: |
| Vanilla 2B / 8B and released BF16 | 65,536 | 16,777,216 |
| Released GRACE W4 | 784 | 200,704 |
| Released GRACE W8 | 784 | 50,176 |
| All LoRA-GRACE / LoRA-QAT exports | 4,096 | 100,352 |

Local exports therefore have half the released W4 maximum pixel budget and twice the released W8 maximum. These bounds are not measurements of actual visual-token counts. Released-versus-LoRA comparisons cannot isolate the effect of the training method from preprocessing differences. LoRA-GRACE versus LoRA-QAT at the same precision is the more controlled development comparison: initialization, recorded training budget and processor bounds match. Native processors were preserved throughout this analysis.

## Decision proposed for discussion

1. **Use these results as a documented local development baseline**, with all seven suites visible and released BF16 plus matching-precision LoRA-QAT retained as controls. The independent audit supports score arithmetic and coverage; the original runner readiness flag still needs the MMBench repair.
2. **Repair the MMBench validation bookkeeping before declaring all 63 pairs formally ready.** Normalize the histogram key representation at the appropriate boundary, then revalidate/adopt the existing saved outputs through a documented recovery path. The diagnosed failure does not require new model inference. Preserve the frozen protocol history when changing validator source hashes.
3. **Treat MMStar as an unresolved method-quality issue.** An overall mean within a few pp is insufficient evidence of acceptable agreement when a suite loses more than 10 pp. If the supervisor accepts that tradeoff for exploratory work, state it explicitly and continue tracking the regression; otherwise, investigate it before using LoRA-GRACE as the main baseline.
4. **Keep the planned transition to pinned lmms-eval when judge access is available**, with references and local methods evaluated under the same declared harness setup. The present evidence does not resolve the outstanding paper-reproduction questions.

Suggested wording for the meeting:

> Under our frozen local protocol, LoRA-GRACE differs from released GRACE by −1.44 pp at W4 and +2.22 pp at W8 on the seven-suite mean. W4 is close on most suites and W8 improves several suites, but both lose about 10.5–10.8 pp on MMStar. LoRA-QAT also exceeds LoRA-GRACE overall at both precisions. The saved results pass a CPU audit, with a JSON histogram-key issue explaining why MMBench lacks completion markers. We propose retaining this setup for controlled local development after that bookkeeping repair, while explicitly investigating MMStar and avoiding a claim of equivalent performance or strict paper reproduction.

## Source artifacts

- [Agreed direction and historical context](README.md) and [local scoring/processor documentation](../../evaluation/README_local_development.md).
- [Frozen full-run protocol](local_development/local_native_greedy256_v1/full/protocol.json), [original runner summary](local_development/local_native_greedy256_v1/full/local_scores.md), and [full result tree](local_development/local_native_greedy256_v1/full/).
- [Report audit](experiment_report_audit.json): all 63 result paths and hashes, raw-precision scores, validation evidence, diagnostics and paired MMStar counts. Values come from `summary.primary_score`; MMBench's `overall_acc` is row accuracy and is not used as its primary result.
- [Training matrix and model mapping](../released_bf16_init_32k_b128_p100352_t2048_seed42/experiment.json). Training settings and completion are in each local export's `launch_recipe.json` and `run_summary.json` under `w4g128/runs/{qat,grace}` and `w8g128/runs/{qat,grace}` in that experiment.

All deltas and means were computed before rounding. The only persistent additions from this reporting task are this report and its audit JSON; source evaluations and their completion status are preserved.
