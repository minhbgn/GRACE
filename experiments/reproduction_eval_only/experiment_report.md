# Local evaluation experiment report: LoRA-GRACE and LoRA-QAT

Prepared **2026-09-29** from the full runs dated **2026-09-26–27 UTC**. Protocol: **`local_native_greedy256_v1`**. Scope: nine checkpoints × seven benchmarks; smoke runs are excluded.

This report follows the [agreed direction in the README](README.md#agreed-direction-and-remaining-work). All comparisons below use fresh local results from this protocol. The README's statements that fresh inference is pending predate these runs. Paper scores and historical lmms-eval results are not mixed into these tables.

## Configuration and differences from the original setup

There are two relevant references: the original paper used LLaVA-1.5 and Qwen2-VL, while the released repository reports Qwen3-VL results. Our experiment uses Qwen3-VL, so the repository's full-training launchers are the closer configuration reference. They describe defaults, rather than a verified record of every setting used to produce the published checkpoints.

| Setting | Our constrained experiment | Repository full-training reference |
| --- | --- | --- |
| Hardware | 2 × A100 40GB | Launchers specify 16 × A100 80GB |
| Training data | 32,768 shared, validated, single-image examples | ShareGPT4V mix665k, approximately 665,000 examples |
| Training duration | One pass; exactly 512 optimizer updates per method | One epoch by default, over the full mix665k dataset |
| Model updates | Small LoRA adapters in attention q/k/v/o projections, plus the visual merger/projector | Full language-model and merger/projector fine-tuning |
| Frozen components | Base language-model weights, embeddings, output head, vision encoder | Vision encoder frozen; language model trained |
| LoRA settings | Rank 32, alpha 64, dropout 0.05 | Full-training launchers do not enable LoRA |
| Examples per GPU per forward/backward pass | 1 for every method | GRACE: 2; SFT/QAT: 4 |
| Gradient accumulation | 32 passes before an optimizer update | GRACE: 16; SFT/QAT: 4 |
| Effective batch size | 2 × 1 × 32 = **64 examples/update** | GRACE: **512**; SFT/QAT: **256** |
| Image budget | Maximum **50,176 pixels**, approximately a 224 × 224 square image; minimum 4,096 | GRACE launcher: maximum **200,704 pixels**, approximately 448 × 448; QAT launcher already uses 50,176 |
| Input length | Strict limit of **1,024 tokens**, including image tokens | GRACE/QAT declare 8,192; the SFT launcher declares 32,768 |
| Learning rates | LoRA: 2e-4; projector: 1e-5; quantization scales: 2e-4 | GRACE/QAT model LR: 1e-5; SFT: 2e-5; scale LR: 2e-4 |
| Schedule | Cosine decay, 3% LR warmup, zero weight decay | Cosine and 3% warmup; GRACE weight decay 1e-3, SFT zero |
| Learned quantization scales | Only LoRA-targeted attention layers; other eligible language-layer scales remain fixed | Full-weight QAT allows broader joint weight/scale adaptation |
| Projector precision | Trainable BF16 projector in all methods | Repository overview describes quantizing the language model and MLP projector |
| GRACE loss settings | Temperature 2; DKD weights 1/4; RCKA weight 3; penultimate layer; RCKA warmup 50 updates | Same GRACE launcher values except RCKA warmup defaults to 400 updates |
| Adaptive distillation controller | Target 3; step size .003; initial weight .5; bounds [.1, 1]; EMA .99 | Same values in the local GRACE launcher |
| Seeds | Training seed **42**; subset-selection seed **2026** | Our subset and initialization are fixed across methods |

## Main findings for discussion

**LoRA-GRACE W4 is 1.44 percentage points (pp) below released GRACE W4 on the equal-weight seven-benchmark mean; LoRA-GRACE W8 is 2.22 pp above released GRACE W8. Neither mean describes uniform agreement.** The largest negative differences are on MMStar: **−10.80 pp at W4 and −10.47 pp at W8**. W8 gains elsewhere offset its MMStar loss.

**LoRA-QAT outperforms LoRA-GRACE on the seven-benchmark mean at both precisions:** by **1.16 pp at W4** and **2.20 pp at W8**. The present checkpoints therefore do not demonstrate an overall benefit from adding the LoRA-GRACE training recipe. LoRA-QAT W8 is the strongest evaluated low-bit checkpoint on this mean.

The results support using the local setup to investigate and compare development changes, provided the MMStar regression remains visible and the MMBench validation issue below is repaired before declaring the runner fully ready. They do **not** establish that LoRA-GRACE is an interchangeable substitute for released GRACE, or that the paper has been reproduced. The acceptable task-specific loss remains a decision for the supervisor; no new acceptance threshold is assumed here.


## Full local results

All cells are percentages, rounded to two decimals. **† MMBench and the seven-suite means have the audit/completion distinction described above.**

| Model | Hallusion | MMBench† | ScienceQA | AI2D | MMMU | SEED image | MMStar | Mean† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Vanilla 2B](local_development/local_native_greedy256_v1/full/vanilla_2b/) | 62.44 | 76.80 | 83.26 | 76.59 | 39.67 | 76.79 | 47.60 | 66.16 |
| [Vanilla 8B teacher](local_development/local_native_greedy256_v1/full/vanilla_8b_teacher/) | 72.90 | 84.71 | 92.74 | 83.97 | 53.22 | 79.36 | 62.73 | 75.66 |
| [Released BF16 / SFT](local_development/local_native_greedy256_v1/full/released_bf16/) | 67.05 | 91.15 | 86.18 | 81.12 | 72.00 | 76.28 | 68.00 | 77.40 |
| [Released GRACE W4](local_development/local_native_greedy256_v1/full/released_grace_w4/) | 64.57 | 88.66 | 84.08 | 79.40 | 61.56 | 73.86 | 66.20 | 74.05 |
| [Released GRACE W8](local_development/local_native_greedy256_v1/full/released_grace_w8/) | 59.17 | 88.40 | 82.48 | 75.94 | 58.11 | 70.08 | 63.67 | 71.12 |
| [LoRA-QAT W4](local_development/local_native_greedy256_v1/full/qat_w4/) | 63.86 | 87.63 | 82.88 | 79.37 | 66.33 | 72.44 | 63.87 | 73.77 |
| [LoRA-GRACE W4](local_development/local_native_greedy256_v1/full/grace_w4/) | 64.92 | 87.46 | 83.73 | 79.05 | 64.67 | 73.04 | 55.40 | 72.61 |
| [LoRA-QAT W8](local_development/local_native_greedy256_v1/full/qat_w8/) | 64.92 | 89.18 | 84.89 | 80.76 | 70.67 | 72.63 | 65.73 | 75.54 |
| [LoRA-GRACE W8](local_development/local_native_greedy256_v1/full/grace_w8/) | 66.43 | 88.40 | 85.71 | 79.44 | 67.44 | 72.77 | 53.20 | 73.34 |

Released BF16 has the highest seven-suite mean (**77.40%**), followed by vanilla 8B (**75.66%**) and LoRA-QAT W8 (**75.54%**). All four LoRA checkpoints exceed vanilla 2B's mean but remain below released BF16. These rankings describe this local protocol; the vanilla models' extraction and length-stop diagnostics below limit interpretations about general model capability.

## LoRA-GRACE versus released GRACE

The principal comparison is the matching bit width. Values below are **LoRA-GRACE minus released GRACE**, in pp.

| Comparison | Hallusion | MMBench† | ScienceQA | AI2D | MMMU | SEED image | MMStar | Mean† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA-GRACE W4 − Released GRACE W4 | +0.35 | -1.20 | -0.35 | -0.36 | +3.11 | -0.82 | -10.80 | -1.44 |
| LoRA-GRACE W8 − Released GRACE W8 | +7.26 | +0.00 | +3.23 | +3.50 | +9.33 | +2.69 | -10.47 | +2.22 |

| Comparison | Higher / tied / lower suites | Mean absolute gap (pp)† | Largest deficit | Largest gain |
| --- | ---: | ---: | ---: | ---: |
| W4 | 2 / 0 / 5 | 2.43 | MMStar: −10.80 pp | MMMU: +3.11 pp |
| W8 | 5 / 1 / 1 | 5.21 | MMStar: −10.47 pp | MMMU: +9.33 pp |

At **W4**, five suites are within **1.21 pp** of released GRACE; MMMU improves by **3.11 pp** and MMStar drops by **10.80 pp**. The mean difference over the other six suites is **+0.12 pp**. This is a useful sensitivity calculation showing where the overall deficit originates; MMStar remains part of the primary seven-suite result.

At **W8**, LoRA-GRACE improves HallusionBench by **7.26 pp** and MMMU by **9.33 pp**, ties circular MMBench, and improves ScienceQA, AI2D and SEED. It still loses **10.47 pp** on MMStar. The signed mean gain of **2.22 pp** masks a much larger **5.21 pp mean absolute difference** across suites, so it should not be interpreted as close per-benchmark agreement. Native image budgets differ, and the present experiment does not determine how much of these gains comes from the training recipe versus preprocessing.

The README's ±3/±5 pp bands were defined for historical paper reproduction. They are not an agreed tolerance for LoRA-versus-released equivalence and are not used to assign pass/fail here.

## LoRA-QAT and other reference comparisons

Values are **left minus right**, in pp.

| Comparison | Hallusion | MMBench† | ScienceQA | AI2D | MMMU | SEED image | MMStar | Mean† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA-GRACE W4 − LoRA-QAT W4 | +1.06 | -0.17 | +0.85 | -0.32 | -1.67 | +0.60 | -8.47 | -1.16 |
| LoRA-GRACE W8 − LoRA-QAT W8 | +1.51 | -0.77 | +0.83 | -1.33 | -3.22 | +0.15 | -12.53 | -2.20 |
| LoRA-QAT W4 − Released GRACE W4 | -0.71 | -1.03 | -1.20 | -0.03 | +4.78 | -1.43 | -2.33 | -0.28 |
| LoRA-QAT W8 − Released GRACE W8 | +5.76 | +0.77 | +2.41 | +4.83 | +12.56 | +2.54 | +2.07 | +4.42 |

LoRA-GRACE beats LoRA-QAT on HallusionBench, ScienceQA and SEED at both precisions, but loses on the other four suites. At W4 the mean difference excluding MMStar is only **+0.06 pp**; at W8 it is **−0.47 pp**. The large MMStar losses drive much of the overall disadvantage, with MMMU also lower by **1.67 / 3.22 pp** at W4 / W8.

LoRA-QAT W4 is **0.28 pp** below released GRACE W4 on the mean, with every suite within **4.78 pp** and only MMMU beyond 3 pp. LoRA-QAT W8 exceeds released GRACE W8 on **all seven suites**, averaging **+4.42 pp**, subject to the native-processor confound. For continued development, LoRA-QAT is therefore a necessary baseline; comparison only with released GRACE W8 would obscure the fact that the local LoRA-GRACE checkpoint underperforms local LoRA-QAT.

The following gives the seven-suite mean differences against the remaining references:

| Local method | vs released BF16† | vs vanilla 2B† | vs vanilla 8B teacher† |
| --- | ---: | ---: | ---: |
| LoRA-QAT W4 | -3.63 | +7.60 | -1.89 |
| LoRA-GRACE W4 | -4.79 | +6.44 | -3.05 |
| LoRA-QAT W8 | -1.86 | +9.37 | -0.12 |
| LoRA-GRACE W8 | -4.06 | +7.18 | -2.32 |

Within the local methods, moving W4 → W8 raises LoRA-QAT's mean by **1.77 pp** and LoRA-GRACE's mean by **0.73 pp**. LoRA-GRACE W8 nevertheless scores **2.20 pp lower on MMStar** than LoRA-GRACE W4. Released GRACE W8 also scores below released W4 under this protocol; different native processors prevent attributing that ordering to precision alone.

## MMStar: the main regression to investigate

Each category contains 250 questions. The category accuracies below average exactly to the overall MMStar score.

| Category | Released GRACE W4 | LoRA-QAT W4 | LoRA-GRACE W4 | Released GRACE W8 | LoRA-QAT W8 | LoRA-GRACE W8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse perception | 77.60 | 80.00 | 74.40 | 79.60 | 79.60 | 72.80 |
| fine-grained perception | 59.20 | 58.80 | 52.00 | 52.00 | 59.60 | 49.60 |
| instance reasoning | 78.40 | 78.40 | 70.00 | 75.20 | 78.80 | 64.40 |
| logical reasoning | 65.60 | 60.40 | 54.00 | 63.60 | 63.60 | 50.80 |
| math | 58.80 | 56.40 | 36.80 | 58.00 | 58.40 | 35.20 |
| science & technology | 57.60 | 49.20 | 45.20 | 53.60 | 54.40 | 46.40 |

LoRA-GRACE is below both its released-GRACE and LoRA-QAT counterparts in **all six categories** at both precisions. The largest loss is math: versus released GRACE, **−22.00 pp at W4 and −22.80 pp at W8**; versus LoRA-QAT, **−19.60 and −23.20 pp**. Thus the regression is broad, with a particularly large math component.

All **1,500/1,500 responses for each released and LoRA checkpoint are bare A/B/C/D letters**, with zero unresolved outputs, inference errors or length stops. These MMStar deficits are differences in selected answers under the local protocol; missing answer extraction or the 256-token cap does not explain them. This does not establish a training cause or rule out other implementation issues.

A paired check on the same 1,500 questions confirms the net losses:

| Left − right | Only left correct | Only right correct | Net correct answers |
| --- | ---: | ---: | ---: |
| LoRA-GRACE W4 − Released GRACE W4 | 124 | 286 | -162 |
| LoRA-GRACE W8 − Released GRACE W8 | 164 | 321 | -157 |
| LoRA-GRACE W4 − LoRA-QAT W4 | 84 | 211 | -127 |
| LoRA-GRACE W8 − LoRA-QAT W8 | 61 | 249 | -188 |

A useful next analysis is to inspect these disagreement sets, especially math, against the LoRA-QAT controls and saved training/export configuration. The aggregate results alone cannot attribute the regression to distillation losses, LoRA capacity, training duration or another specific cause.

## Output-quality diagnostics and interpretation limits

Counts below cover all seven suites, including audited MMBench. Unresolved and length-stop counts can overlap; they should not be added together as distinct failed examples.

| Model | Unresolved outputs | Length stops |
| --- | ---: | ---: |
| Vanilla 2B | 327 | 411 |
| Vanilla 8B teacher | 75 | 66 |
| Released BF16 / SFT | 0 | 0 |
| Released GRACE W4 | 0 | 0 |
| Released GRACE W8 | 8 | 1 |
| LoRA-QAT W4 | 0 | 0 |
| LoRA-GRACE W4 | 0 | 0 |
| LoRA-QAT W8 | 6 | 5 |
| LoRA-GRACE W8 | 0 | 0 |

The largest diagnostic burdens are in vanilla 2B: **133/900 unresolved MMMU answers (14.78%)**, with **175 length stops (19.44%)**; and **181/1,500 unresolved MMStar answers (12.07%)**, with **232 length stops (15.47%)**. Vanilla 8B has 22 unresolved / 24 length stops on MMMU and 22 unresolved / 27 length stops on MMStar. Even at 256 tokens, the frozen protocol does not obtain complete, extractable answers uniformly across model families. LoRA gains over vanilla should be understood as gains under this evaluation and response-format policy, not purely as improved reasoning.

Released GRACE W8 has eight unresolved HallusionBench outputs and one MMMU length stop. Even if all eight unresolved HallusionBench outputs were recovered as correct, its Hallusion score would increase by only **0.71 pp**, leaving most of LoRA-GRACE W8's **7.26 pp** advantage. LoRA-QAT W8 has six unresolved ScienceQA outputs and five length stops; resolving all six as correct could raise its ScienceQA score by at most **0.14 pp**. These are optimistic unresolved-only bounds, not measured judge effects.

Vanilla 8B's six unresolved MMBench rows affect only three potentially recoverable circular groups: its stored lower/possible-resolution upper bounds are **84.71–84.97%**. All other models' MMBench outputs are resolved. A circular group with a definite wrong rotation stays incorrect even if another rotation is later resolved.

This is one evaluated checkpoint per local training configuration, with no repeated-training or repeated-inference variability estimate. Small differences should not be described as statistically established improvements. No GPT/semantic adjudication has been measured, and no latency, deployed low-bit memory or retained-token measurements are supplied by this accuracy report. Native image budgets, prompts, populations and aggregation differ from some historical harness/paper settings, so these scores cannot certify that workflow's rankings or strict paper agreement.

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

The local training [experiment configuration](../released_bf16_init_32k_b128_p100352_t2048_seed42/experiment.json) identifies released BF16 as the initialization for all four LoRA exports, and vanilla 8B as the LoRA-GRACE teacher. The [authors' training README](../../README.md#1-bf16-sft-baseline) explicitly identifies that released BF16 checkpoint as an SFT baseline. Accordingly, these branches compare SFT → CE-only QAT (no distillation) with SFT → GRACE (QAT + GDKD/RCKA). Their saved launch recipes specify a 32,768-example subset, 256 steps, global batch 128, seed 42, group size 128, and attention-projection LoRA with rank 32 / alpha 64; all four training summaries record completion at step 256. Thus released-versus-local differences include differences in training recipe and checkpoint, in addition to the native image budgets. The previous local `sft_2b` model has no run in this nine-model matrix and is not substituted for released BF16.

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
