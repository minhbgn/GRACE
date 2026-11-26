# Experiment report: constrained_32k_b128_p100352_t2048_seed42

Follow-up: [evaluation-script and upstream-code audit, with research-direction recommendations](implementation_and_next_steps.md) clarifies the inference token budgets and the role of released GRACE-BF16 checkpoints.

**The new configuration improves absolute accuracy, but does not establish a benefit from GRACE over the matched training controls.** GRACE improves all seven primary scores and raises its mean from **63.39% to 64.96% (+1.57 percentage points)**. However, vanilla improves by +1.72 pp, SFT by +1.92 pp and QAT by +1.90 pp. GRACE's deficit to QAT grows from **0.59 to 0.93 pp**, and its deficit to SFT grows from **2.42 to 2.77 pp**. The mean ordering remains **SFT > QAT > Vanilla > GRACE**.

For the supervisor, present both experiments as a controlled comparison of methods within each configuration and a sensitivity study across configurations. The useful finding is that the larger input/batch recipe raises absolute performance without resolving the disadvantage of the current teacher-guided objective. The immediate priorities are to separate evaluation-resolution effects from training effects, check native versus exported model behavior, and then isolate the distillation losses. Increasing scale again would leave the main uncertainty unresolved.

This report uses completed local artifacts. It adds no training or inference runs. Scores and differences are computed from unrounded merged-result JSON values; displayed values are rounded to two decimals. “Mean” always means the unweighted mean of seven primary suite scores, not an official benchmark aggregate.

## 1. Experimental setup and what actually changed

The student is Qwen3-VL-2B-Instruct. GRACE additionally uses a frozen Qwen3-VL-8B-Instruct teacher. Vanilla is the original 2B model without experiment training. SFT trains with answer cross-entropy (CE); QAT adds simulated W4G128 weight quantization; GRACE adds confidence-weighted prediction distillation (GDKD) and visual relationship alignment (RCKA) to QAT.

All trained methods update attention q/k/v/o LoRA adapters and the BF16 visual merger. QAT/GRACE also expose quantization scales in the adapted attention layers to optimization. Base language weights and the vision encoder remain frozen. The saved inventories contain 38,019,072 trainable parameters for SFT and 40,771,584 for QAT/GRACE; inventories establish eligibility to train, not successful updates of every parameter.

| Setting | Previous: `constrained_32k_seed42` | Current experiment |
| --- | ---: | ---: |
| Hardware | 2 × A100 40GB | Same |
| Examples / passes | 32,768 / one | Same |
| Training / subset seeds | 42 / 2026 | Same |
| Per-GPU micro-batch | 1 | 1 |
| Gradient accumulation | 32 | 64 |
| Effective batch | 64 | 128 |
| Optimizer updates | 512 | 256 |
| Maximum image area, training and evaluation | 50,176 pixels | 100,352 pixels |
| Approximate upper bound on merged tokens per image | 49 | 98 |
| Strict training input-token limit | 1,024 | 2,048 |
| RCKA warmup | 50 updates | 25 updates |
| LoRA rank / alpha / dropout | 32 / 64 / 0.05 | Same |
| LoRA / merger / QAT-scale learning rates | 2e-4 / 1e-5 / 2e-4 | Same |
| LR schedule / warmup / weight decay | Cosine / 3% / zero | Same |
| GDKD temperature / component weights | 2 / 1 and 4 | Same |
| RCKA weight / layer | 3 / penultimate | Same |
| Controller target / step size / initial beta / bounds | 3 / .003 / .5 / [.1, 1] | Same |

The image limit is area, not an edge length. Actual visual-token counts depend on image shape. The larger batch is implemented through accumulation, so it neither holds 128 examples' activations simultaneously nor enlarges the per-forward image used for RCKA. It halves optimizer and controller updates per pass. RCKA warmup covers approximately the same 3,200 examples in both recipes; controller step size and EMA are still defined per update, so their dynamics per example are not identical.

### Data comparability: stronger than the README's prospective warning

The README correctly warns that regenerating the subset *can* change selection. The completed manifests show that it did not do so here:

- All **32,768 source annotation indices match, in the same manifest order**, with identical image-source quotas.
- **32,700 conversations are identical; 68 differ (0.21%)**, with 670 additional retained question/answer pairs overall: 174,260 → 174,930 (+0.38%).
- Mean recorded input length rises from **267.20 to 314.50 tokens**; median rises from 187 to 234. The maximum rises from 1,024 to 1,858; only **90 current records (0.27%)** exceed 1,024 tokens.
- Thus, this is not a comparison of different selected source examples. Image preprocessing changes broadly, while additional retained conversation content is concentrated in a small fraction of records. That fraction does not by itself establish how much the longer limit affects learning.

The saved recipes record identical library versions and hashes for the data, training, loss and quantization modules; the launcher hash differs. All six method runs match their experiment's pinned recipe, finish the expected updates, and have `EXPORT_COMPLETE`. The only saved tokenizer-configuration difference between experiments is `model_max_length`; tokenizer vocabulary and generation configuration match. The image-processor difference is its maximum image area.

The current evaluation scripts do not explicitly request input truncation at the saved tokenizer limit. Therefore, the 2,048-token setting is a verified training limit, not evidence that evaluation truncates every input at 2,048. The larger evaluation image budget is explicitly recorded in all 28 current result files.

## 2. Current results and their meaning

| Primary benchmark | N | Vanilla (%) | SFT (%) | QAT (%) | GRACE (%) | GRACE−QAT (pp) | GRACE−SFT (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HallusionBench aAcc | 1,129 | 59.88 | 64.13 | 56.95 | 60.05 | +3.10 | −4.07 |
| MMBench EN dev, per-row | 4,329 | 83.37 | 83.85 | 82.88 | 81.89 | −0.99 | −1.96 |
| ScienceQA, full test | 4,241 | 82.95 | 82.72 | 81.54 | 80.45 | −1.08 | −2.26 |
| AI2D | 3,088 | 74.81 | 74.94 | 72.93 | 72.41 | −0.52 | −2.53 |
| MMMU validation | 900 | 40.00 | 43.00 | 43.78 | 39.89 | −3.89 | −3.11 |
| SEED-Bench-IMG | 14,233 | 72.46 | 73.06 | 73.03 | 72.59 | −0.44 | −0.47 |
| MMStar | 1,500 | 44.67 | 52.47 | 50.13 | 47.47 | −2.67 | −5.00 |
| **Unweighted mean** | — | **65.45** | **67.74** | **65.89** | **64.96** | **−0.93** | **−2.77** |

**SFT is the strongest general quality reference.** It has the highest score on five suites; vanilla leads ScienceQA and QAT leads MMMU. SFT improves six of seven scores over vanilla and its mean by +2.29 pp. The earlier SFT run improved all seven, so the new recipe is not uniformly beneficial relative to the starting checkpoint.

**QAT is the stronger current W4G128 candidate on the aggregate.** It exceeds GRACE on six suites and trails it only on HallusionBench. Its +0.45 pp mean advantage over vanilla is small and should not be interpreted as an established general benefit of quantization. Its deficit to SFT is 1.85 pp, almost unchanged from the previous 1.83 pp. A no-training W4 control is absent, so this is the combined outcome of quantization, adaptation and export rather than an isolated quantization penalty.

**GRACE has a task-specific HallusionBench benefit over QAT, but no broad advantage.** It trails SFT on every primary score. Against vanilla, it gains +0.18 pp on HallusionBench, +0.13 pp on SEED and +2.80 pp on MMStar, while losing −1.48 pp on MMBench, −2.50 pp on ScienceQA, −2.40 pp on AI2D and −0.11 pp on MMMU. The small HallusionBench/SEED gains over vanilla amount to only 2/19 extra correct rows. The overall mean is −0.48 pp below vanilla.

### HallusionBench: a persistent QAT-relative benefit, with an important reversal

| Metric | Vanilla (%) | SFT (%) | QAT (%) | GRACE (%) | GRACE−QAT (pp) | GRACE−SFT (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qAcc, 455 question groups | 26.37 | 31.87 | 24.18 | 26.37 | +2.20 | −5.49 |
| fAcc, 346 figure groups | 31.79 | 35.55 | 29.77 | 34.68 | +4.91 | −0.87 |

These metrics require all answers in a group to be correct. GRACE retains better grouped consistency than QAT. However, the previous report's observation that GRACE beats SFT on both grouped metrics **does not persist**: SFT now leads aAcc, qAcc and fAcc. GRACE's fAcc rises only 34.10 → 34.68 (+0.58 pp, two additional correct groups), while SFT rises 30.92 → 35.55. This narrows the case for GRACE to a benefit over the quantized control on this benchmark, rather than the best hallucination-related performance overall.

Within current HallusionBench, GRACE beats QAT on image rows by +3.68 pp, but ties it on text rows at 47.19%, below vanilla's 52.81% and SFT's 57.30%. This pattern motivates investigating the visual objective; it does not identify RCKA as the cause without a component ablation.

### Diagnostic slices

**ScienceQA separates visual improvement from text regression.** Current image/text scores are 82.15/78.91 for GRACE, 82.50/80.67 for QAT and 84.78/80.85 for SFT. Relative to the old GRACE run, image accuracy rises **+2.48 pp (50 more correct)** while text accuracy falls **−1.12 pp (25 fewer correct)**, leaving just 25 additional correct answers overall. Vanilla improves +4.31 pp on image questions and has identical predictions on all 2,224 text questions across experiments. This supports a meaningful visual-preprocessing contribution to the absolute gains. GRACE's text regression shows that missing image detail cannot explain all remaining weaknesses.

**MMStar's gain over vanilla does not establish successful teacher transfer.** GRACE's +2.80 pp overall advantage over vanilla is unchanged from the old run; QAT and SFT now gain +5.47 and +7.80 pp. GRACE's math score is 43.60 versus vanilla's 28.00, but QAT/SFT score 46.80/47.60. On science & technology, GRACE remains weakest at 28.40 versus 32.40/38.80/35.20 for vanilla/SFT/QAT. The useful improvement is not unique to teacher guidance.

**SEED exposes modest average gains and specific residual deficits.** GRACE's text-understanding score rises 55.29 → 65.88, but this is only nine additional correct answers out of 85. On the larger counting category (2,447 rows), GRACE scores 63.75 versus QAT's 65.43 and SFT's 66.82. Treat the small OCR-related slice as a hypothesis for follow-up, not the principal success claim.

**MMMU is still a concern, despite a small absolute improvement.** GRACE gains only seven correct answers over its old run and remains 35 behind QAT. On 53 open questions it gets 2 correct, unchanged from the previous run; QAT and SFT get 4. Most of its current QAT deficit is therefore on multiple-choice questions (33 of the 35 net lost answers), so the open-answer scorer cannot explain away that deficit.

## 3. Comparison with the previous experiment

### Absolute changes

Every entry below is current minus previous accuracy in percentage points.

| Benchmark | Vanilla Δ | SFT Δ | QAT Δ | GRACE Δ |
| --- | ---: | ---: | ---: | ---: |
| HallusionBench aAcc | +2.57 | +4.69 | +1.33 | +2.13 |
| MMBench | +1.11 | +0.90 | +1.34 | +1.11 |
| ScienceQA | +2.05 | +1.37 | +0.83 | +0.59 |
| AI2D | +2.62 | +2.36 | +2.98 | +2.53 |
| MMMU | −0.67 | −1.33 | +1.89 | +0.78 |
| SEED-Bench-IMG | +2.24 | +2.11 | +2.29 | +1.73 |
| MMStar | +2.13 | +3.33 | +2.67 | +2.13 |
| **Mean** | **+1.72** | **+1.92** | **+1.90** | **+1.57** |

GRACE and QAT improve all seven scores; vanilla and SFT improve six. The new GRACE export is a better measured model-plus-processor system than the old GRACE export. This does not yet show that its learned weights are better when evaluated at the same resolution.

### Relative changes: did the recipe improve GRACE's advantage?

| Comparison of means | Previous (pp) | Current (pp) | Change in gap (pp) |
| --- | ---: | ---: | ---: |
| SFT−Vanilla | +2.09 | +2.29 | +0.20 |
| QAT−Vanilla | +0.27 | +0.45 | +0.18 |
| GRACE−Vanilla | −0.33 | −0.48 | −0.15 |
| GRACE−QAT | −0.59 | −0.93 | −0.33 |
| GRACE−SFT | −2.42 | −2.77 | −0.35 |
| QAT−SFT | −1.83 | −1.85 | −0.02 |

The relative evidence does not favor the new recipe as a solution to GRACE's underperformance. Its gap to QAT worsens on **six of seven suites**; HallusionBench is the exception, with the advantage expanding +2.30 → +3.10 pp. GRACE previously beat QAT on HallusionBench and SEED; it now wins only HallusionBench. Its deficit to SFT grows on HallusionBench, ScienceQA, SEED and MMStar, while narrowing on MMBench, AI2D and MMMU.

The simple baseline-adjusted change is:

`(current GRACE − previous GRACE) − (current vanilla − previous vanilla) = 1.57 − 1.72 = −0.15 pp`.

This is a descriptive difference of differences, **not a causal estimate of training quality**. Trained models can respond differently to resolution; batch size, update count, retained content and preprocessing also change together. The vanilla improvement establishes that training is not required for much of the observed direction of improvement, but does not tell us what fraction of GRACE's gain came from preprocessing. Evaluate the same checkpoints under a common processor to answer that question.

### What the paired predictions show

Rows were aligned by benchmark-specific identity, not file order. A “win” means GRACE is correct and the comparison model is wrong; a “loss” means the reverse. Rows on which both are correct or both wrong are omitted.

| Suite | Current GRACE vs current QAT: wins / losses | Current vs previous GRACE: wins / losses |
| --- | ---: | ---: |
| HallusionBench | 87 / 52 | 98 / 74 |
| MMBench | 92 / 135 | 144 / 96 |
| ScienceQA | 168 / 214 | 158 / 133 |
| AI2D | 137 / 153 | 206 / 128 |
| MMMU | 69 / 104 | 42 / 35 |
| SEED-Bench-IMG | 509 / 572 | 914 / 668 |
| MMStar | 106 / 146 | 113 / 81 |

These counts show that improved totals coexist with many regressions on individual questions. For example, the +0.78 pp MMMU gain is 42 fixes offset by 35 new errors. They also identify useful disagreement pools for qualitative inspection. They are not significance tests: HallusionBench groups, repeated MMBench variants and questions sharing images can violate independent-row assumptions. No claim of statistically significant improvement or degradation is made here; one training seed cannot establish repeatability. A future uncertainty analysis should pair predictions and resample at the appropriate question/image group, and additional training seeds should measure training variation separately.

## 4. Training diagnostics and plausible explanations

### The controller saturation problem persists

| GRACE diagnostic | Previous | Current |
| --- | ---: | ---: |
| First logged update at beta = 1 | 7 | 8 |
| Updates logged at beta = 1 | 506 / 512 (98.83%) | 249 / 256 (97.27%) |
| GDKD warmup | None | None |
| Tail-window size | Last 50 updates | Last 25 updates |
| Tail mean logged CE | 1.297 | 1.221 |
| Tail mean logged GDKD | 7.588 | 8.119 |
| Tail mean logged RCKA | 0.310 | 0.438 |
| Tail weighted RCKA contribution, weight 3 | 0.929 | 1.313 |

The tail windows cover the same number of examples (3,200), but their measurements are **last-micro-batch snapshots at optimizer boundaries**, not averages over those examples. They therefore do not justify a precise cross-run loss comparison. Both tails have full distillation warmup scales and beta = 1.

The controller increases beta when its smoothed GDKD exceeds target 3. It reaches its maximum quickly in both runs and stays there. Thus, the adaptive weight spends little of the run adapting within its range. The distillation terms are numerically large relative to CE in these snapshots, which makes loss balance a concrete next hypothesis. Loss magnitude alone does not establish gradient dominance, gradient conflict or causation. Lowering the controller target would push beta upward under this implementation; it is not a way to weaken distillation.

The higher logged RCKA value is also not proof of worse representations: the visual-token count and sampled examples differ. Before attributing the result to one component, measure accumulation-averaged losses and separate CE/GDKD/RCKA gradient norms and directions on fixed diagnostic batches.

### Some implementation questions remain open

The current [10-step profile diagnostics](profiles/grace/profiling/learning_diagnostics_summary.json) confirm a frozen teacher, backward connectivity of both distillation terms, gradients in LoRA/scales/merger, and changed sampled LoRA parameters. However, `qat_scale_parameter_changed` is false and the sampled attention scale and merger norm show zero change, just as the earlier short profile raised a concern about those categories. A sampled unchanged tensor does not establish that all scales or the whole merger failed to learn. Conversely, gradients and a trainable inventory do not prove useful parameter updates. The full run has no saved learning-diagnostics summary, so verify optimizer membership, FP32/master-state behavior and full-tensor changes before asserting that learned scale adaptation worked as intended.

The current [quantized LoRA implementation](../../qwen-vl-finetune/qwenvl/train/qat_modules.py) computes `Q(W) + ΔW` during native evaluation, then merges and bakes weights to produce `Q(W + ΔW)` for export. These generally differ. Both QAT and GRACE use this path, but the export penalty need not be equal. Native-versus-export benchmark behavior has not been measured in these results. This is a measurable uncertainty, not an established explanation of the GRACE deficit.

### What the evidence supports, and what it does not

| Explanation | Evidence from these runs | Remaining uncertainty |
| --- | --- | --- |
| More visual information helps absolute scores | Vanilla improves six suites; ScienceQA image improves while its text predictions are unchanged | Checkpoint-specific resolution effects need crossed evaluation |
| Current teacher-loss balance is unhelpful | GRACE trails QAT on six suites; beta is capped for most updates in both runs | Need component/strength ablations and gradient measurements |
| Fewer updates limit GRACE adaptation | Batch doubles and update count halves without changing LR | Batch, inputs and retained content changed together; no isolated batch result |
| Restricted LoRA capacity or limited data constrains transfer | Both experiments use the same small subset and attention-only adapters plus merger | SFT/QAT still improve; these limits alone do not explain GRACE's ranking |
| Export or scale optimization contributes to the gap | Non-equivalent native/export functions; sampled scales unchanged in profile | Need direct native/export and optimizer-state checks |

The experiment weakens the explanation that the old outcome was simply caused by too little image detail. Larger inputs improve the system's scores but leave the relative GRACE problem intact. It does not rule out further visual-budget benefits, nor does it establish a general failure of GRACE outside this constrained LoRA/W4 recipe.

## 5. Runtime, memory and practical trade-offs

| Method | Previous training time (h) | Current training time (h) | Time change | Previous peak reserved (GiB) | Current peak reserved (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT | 1.45 | 1.77 | +21.95% | 6.59 | 8.46 |
| QAT | 1.90 | 2.19 | +15.02% | 6.64 | 8.47 |
| GRACE | 2.37 | 2.36 | −0.50% | 24.13 | 26.07 |

These are completed **actual-data training-loop** summaries, not stress-profile projections, total end-to-end runtimes or total GPU occupancy. They exclude claims about evaluation cost, deployment speed, and loading/export peaks. Different scheduling and checkpoint overhead, and uncontrolled machine conditions, also limit causal throughput claims. In particular, GRACE's roughly unchanged runtime is not evidence that doubling the batch halves training cost.

The new recipe is feasible on the available hardware in the recorded runs. GRACE uses about 17.59 GiB more peak reserved memory than QAT and about 7.8% more recorded training time, while scoring lower on the mean. SFT remains the quality reference; QAT is the more attractive current quantized comparison unless HallusionBench behavior is the primary requirement.

W4G128 exports contain rounded/dequantized weights stored in BF16 tensors. These experiments do **not** establish packed INT4 memory savings, faster inference or a deployment advantage over SFT. Evaluate those separately if compression efficiency is part of the supervisor-facing claim.

## 6. Recommended next steps and decision criteria

### First: use existing checkpoints to resolve the largest confound

Evaluate both old and current checkpoints at **one fixed evaluation processor configuration**, keeping prompts, parsing, decoding and question identities fixed. The fastest useful addition is old QAT and old GRACE at 100,352 pixels, compared with the already-completed current runs at that budget. Include old SFT at the same budget for the full control set. A complete crossover adds current checkpoints at 50,176 pixels as well.

This separates changes in weights from changes in inference inputs: if the old GRACE checkpoint gains nearly as much merely by changing its processor, little of the present absolute improvement can be credited to the new training recipe. If the current checkpoint still wins at fixed preprocessing, it demonstrates a training-recipe benefit, although batch size and training inputs remain confounded. Explicitly set all processor options rather than relying on each checkpoint's saved defaults.

On a fixed development set, compare reconstructed native QAT/GRACE checkpoints with their final exports, using the matching wrapper and saved scales. Check logits, CE and predictions; inspect missing/unexpected state keys. Check full scale/merger tensors and optimizer state as well. A meaningful export loss should be addressed before spending more compute on loss tuning. Add a no-training W4 export with the same quantization scope and evaluation protocol to establish how much QAT training recovers from rounding alone.

### Then: isolate teacher guidance under one pinned recipe

If those checks pass, keep the current data, batch, input limits, LR schedule and evaluation processor fixed for a small objective ablation. The following is a proposed design, not a claim that these weights are optimal:

| Arm | Objective | Question answered |
| --- | --- | --- |
| A: existing QAT | CE | Matched zero-distillation reference |
| B: GDKD only | CE + current adaptive beta × GDKD; RCKA = 0 | Does prediction distillation help on its own? |
| C: RCKA only | CE + 3 × warmed-up RCKA; GDKD = 0 | Does visual alignment help on its own? |
| D: existing GRACE | CE + current adaptive beta × GDKD + 3 × warmed-up RCKA | Existing combined reference |

Use the existing A/D runs only if model construction, data order, seeds, quantization and export remain identical. If a diagnostic fix changes the implementation, rerun the affected controls. Zero-weight disabled components should not inadvertently change initialization or RNG consumption relative to the intended control.

After the component result, test **one strength change**: for example, fixed GDKD coefficient 0.25 instead of the saturated controller, retaining the chosen other settings. Treat 0.25 as a diagnostic probe. Test a longer GDKD warmup separately; changing strength and warmup simultaneously would again obscure the cause. Prefer an independent development set for choosing weights and schedules, then evaluate the selected configuration on all seven suites. Repeatedly selecting against these already-inspected benchmark results would turn them into tuning data.

If the immediate research question is instead whether batch 128 helped, add **batch 64 at the current image/token limits**, with the exact same retained examples. One pass then has 512 updates. This isolates the batch/update-schedule package at fixed data exposure; it still does not separate batch size from update count. A 512-update batch-128 run uses two passes and must be labeled as increased training exposure, not a pure batch control.

### Scale only after a useful effect survives these controls

The main GRACE success criterion should be a positive mean **GRACE−QAT** under matched preprocessing, supported by per-suite behavior and additional seeds. Retain HallusionBench aAcc/qAcc/fAcc as explicit secondary outcomes, while monitoring ScienceQA text and MMStar science/technology for regressions. A HallusionBench-only benefit can still be reported as a trade-off if the application values it; it should not be described as general accuracy improvement.

Once a configuration looks useful, repeat it and its QAT control with at least two additional training seeds, initially holding the selected data fixed. Report seed variation separately from benchmark-sampling uncertainty. Larger data, broader LoRA targets, longer training or staged BF16 teacher-guided initialization are later investigations; choose one at a time based on the remaining bottleneck.

## 7. What to report to the supervisor

Report **both experiments**, with the new experiment as the current result and the earlier experiment as context. For a quality demonstration, the new SFT export is the leading measured candidate. For the constrained W4 study, report new QAT alongside GRACE; omitting QAT would hide the central negative result. If the purpose is specifically to study GRACE, the negative result and its diagnostics are useful research progress, even though they are not a successful accuracy reproduction.

> We completed two matched SFT/QAT/GRACE comparisons using a Qwen3-VL-2B student, an 8B teacher for GRACE, 32,768 examples and two A100 40GB GPUs. Increasing effective batch from 64 to 128, image area from 50,176 to 100,352 pixels and the training token limit from 1,024 to 2,048 improves GRACE on all seven benchmarks: its mean rises from 63.39% to 64.96%. However, the current vanilla, SFT and QAT means are 65.45%, 67.74% and 65.89%. GRACE's deficit to QAT grows from 0.59 to 0.93 percentage points, so the higher absolute score does not establish better teacher-guided training. Vanilla itself gains 1.72 points with the changed evaluation inputs, and both training manifests select the same source examples. GRACE retains a HallusionBench advantage over QAT, but no longer beats SFT on grouped HallusionBench metrics. Its distillation weight remains capped for 97% of updates. Our next steps are fixed-processor checkpoint comparisons, native/export and scale-update checks, and controlled loss-component ablations before increasing training scale. These are single-seed local evaluations, not official leaderboard or packed-INT4 deployment results.

## 8. Evidence, validation and limits

The audit reloaded **56 merged result files**, verified row counts, correct totals, primary accuracies, unique identities and gold-label equality within and across experiments, and regenerated both comparison reports in memory with exact text matches. All six training summaries report completion without a budget stop; trainer states and contiguous step logs match 512/256 expected updates. No inference errors are recorded. Current reported parse failures total 4 for vanilla, 0 for SFT, 4 for QAT and 7 for GRACE; MMMU has no separate parse-failure count. Even correcting all seven reported GRACE parse failures would add only about **0.025 pp** to the seven-suite mean, far short of the 0.93 pp QAT deficit. This bound does not audit every parser decision or MMMU open-answer scoring.

The comparisons use local protocols: MMBench is per-row, ScienceQA includes the full test set, and MMMU uses the local open-answer scorer. Saved question identities and labels establish matching evaluation content, not byte-for-byte provenance of every historical inference environment or prompt. Old trained-result files do not record image-budget metadata; their saved processors record 50,176 pixels, and old vanilla result metadata explicitly confirms that budget. No new significance test, contamination audit or official leaderboard validation was performed. The small subset of category examples discussed above is exploratory.

Primary local evidence:

- [Current setup](README.md), [pinned recipe](comparison_recipe.json), and [training manifest](subset.json).
- [Current complete evaluation table](evaluation/comparison.md) and [previous complete evaluation table](../constrained_32k_seed42/evaluation/comparison.md).
- [Previous experiment report](../constrained_32k_seed42/experiment_report.md), including its detailed implementation-audit appendix. This report does not revalidate its external paper/release comparisons.
- Current run summaries: [SFT](runs/sft/run_summary.json), [QAT](runs/qat/run_summary.json), [GRACE](runs/grace/run_summary.json); [GRACE update log](runs/grace/step_metrics.jsonl).
- Previous summaries: [SFT](../constrained_32k_seed42/runs/sft/run_summary.json), [QAT](../constrained_32k_seed42/runs/qat/run_summary.json), [GRACE](../constrained_32k_seed42/runs/grace/run_summary.json); [previous GRACE log](../constrained_32k_seed42/runs/grace/step_metrics.jsonl).
- [Current profile diagnostics](profiles/grace/profiling/learning_diagnostics_summary.json), [evaluation validator](../../evaluation/summarize_constrained_evals.py), [GRACE objective/controller integration](../../qwen-vl-finetune/qwenvl/train/train_qwen_grace.py), and [quantization/LoRA/export implementation](../../qwen-vl-finetune/qwenvl/train/qat_modules.py).

To revalidate either seven-suite comparison without launching inference or changing saved results, run from the workspace root:

```bash
python3 reproduction/GRACE/evaluation/summarize_constrained_evals.py \
  --results-dir reproduction/GRACE/experiments/constrained_32k_b128_p100352_t2048_seed42/evaluation

python3 reproduction/GRACE/evaluation/summarize_constrained_evals.py \
  --results-dir reproduction/GRACE/experiments/constrained_32k_seed42/evaluation
```
