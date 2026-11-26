# Constrained LoRA-GRACE from the released BF16 student

Status: **proposal for review; implementation and GPU runs deferred to the next session when the server is available.** Created 2026-09-20. This folder currently contains only this plan. Checkpoint downloads, launchers, profiles, training and evaluation are pending.

The objective is to initialize constrained W4G128 and W8G128 training from the author's released `ForeverBlue/Qwen3-VL-2B-GRACE-BF16`, then measure whether LoRA-GRACE improves over matched QAT-only training and how much quality it loses relative to the author's corresponding full-method releases. Obtain the evaluation results first; discuss the acceptable quality decrease with the supervisor before declaring overall success.

## Why this experiment

The [previous saved recipe](../constrained_32k_b128_p100352_t2048_seed42/comparison_recipe.json) records `models/Qwen3-VL-2B-Instruct` as the student initialization. Its SFT, QAT and GRACE runs were independent branches from that vanilla checkpoint; running `all --train` did not chain SFT into QAT or GRACE.

The [repository model zoo](../../README.md#model-zoo) and [released BF16 model card](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-BF16#model-zoo) identify the released BF16 student as the initialization for the W8/W4 Qwen3-VL runs. The repository training section describes BF16 SFT as the source of the release, while the model card uses the broader description “trained under the GRACE framework.” This plan pins the actual released checkpoint; it does not depend on reconstructing its complete training history. Save that provenance during implementation.

The local reference GRACE SLURM script still defaults to vanilla `Qwen3-VL-2B-Instruct`, so merely using a reference launcher does not ensure the intended initialization. The new launcher must explicitly select and verify the released BF16 weights.

The [previous results](../constrained_32k_b128_p100352_t2048_seed42/evaluation/comparison.md) remain useful as the vanilla-initialized pilot: mean scores were 65.45 for vanilla 2B, 67.74 for local SFT, 65.89 for QAT and 64.96 for GRACE. Those numbers do not establish what happens with the released BF16 initialization. Preserve the previous experiment and label any cross-experiment comparison by initialization.

## Models and comparisons

Four fresh training runs and five evaluation-only references are proposed, giving nine model rows and 63 merged results across seven suites.

| Result ID | Checkpoint or initialization | Training in this experiment | Role |
| --- | --- | --- | --- |
| `vanilla_2b` | `Qwen/Qwen3-VL-2B-Instruct` | None | Original student baseline, BF16 |
| `vanilla_8b_teacher` | `Qwen/Qwen3-VL-8B-Instruct` | None; frozen during distillation | Teacher quality reference, BF16 |
| `released_bf16` | `ForeverBlue/Qwen3-VL-2B-GRACE-BF16` | None | Quality of the exact student initialization |
| `released_grace_w4` | `ForeverBlue/Qwen3-VL-2B-GRACE-W4G128` | None | Author's full-method W4 reference |
| `released_grace_w8` | `ForeverBlue/Qwen3-VL-2B-GRACE-W8G128` | None | Author's full-method W8 reference |
| `qat_w4` | Released BF16 | Constrained LoRA QAT-only, W4G128 | Matched W4 ablation |
| `grace_w4` | Same released BF16 | Constrained LoRA-GRACE, W4G128 | Proposed W4 method |
| `qat_w8` | Released BF16 | Constrained LoRA QAT-only, W8G128 | Matched W8 ablation |
| `grace_w8` | Same released BF16 | Constrained LoRA-GRACE, W8G128 | Proposed W8 method |

Within each bit width, QAT and GRACE must use identical initial weights, subset, order seed, trainable parameter policy, optimizer, input budgets, update count and export procedure. QAT uses CE with quantization; GRACE adds GDKD, RCKA and the adaptive IB controller with the frozen vanilla 8B teacher. This measures the additional benefit of distillation during the constrained stage, conditional on the released initialization.

Each of the four runs starts independently from BF16. Do not initialize W4 from the trained W8 model, use the previous local SFT export as a substitute, or reuse previous QAT/GRACE adapters or optimizer state. Another BF16 SFT stage is not part of this proposal.

Here, “QAT ablation” means the fresh matched local QAT runs above. Previous vanilla-initialized QAT is historical context, not the primary control. An author-released QAT-only checkpoint, if later identified with verifiable provenance, may be an additional reference; no such download is assumed here. The vanilla 8B teacher is a reference, not a guaranteed accuracy ceiling.

## Checkpoints to acquire in the implementation session

| Source | Intended local directory under `reproduction/GRACE/models/` | Use |
| --- | --- | --- |
| [Released BF16](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-BF16) | `Qwen3-VL-2B-GRACE-BF16/` | Initialization and evaluation |
| [Released W4G128](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-W4G128) | `Qwen3-VL-2B-GRACE-W4G128/` | Evaluation only |
| [Released W8G128](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-W8G128) | `Qwen3-VL-2B-GRACE-W8G128/` | Evaluation only |

Reuse the existing vanilla 2B and 8B directories after validating their identity and completeness. For each checkpoint record repository ID, resolved immutable revision, local path, file hashes, config, tokenizer/processor and generation settings. Download model assets and applicable quantization metadata together. Capture the model cards used for provenance. Verify the released low-bit loading format before evaluation; use the research checkpoints above for the primary comparison, with a consistent Transformers evaluation path. Packed AWQ speed/memory benchmarking is a separate experiment.

## Proposed constrained training recipe

Retain the [previous larger-input recipe](../../qwen-vl-finetune/scripts/recipes/constrained_32k_b128_p100352_t2048_seed42.env), changing the student initialization and adding explicit W8 support. These are proposed settings to freeze during implementation, not completed measurements.

| Setting | Planned value |
| --- | --- |
| Training data | Same eligible ShareGPT4V mix665k source and source quotas |
| Subset / seeds | 32,768 examples; training seed 42; subset-selection seed **2026** |
| GPUs / per-device batch / accumulation | 2 A100s / 1 / 64 |
| Effective batch / duration | 128 / one pass, 256 optimizer updates per run |
| Image limits | Minimum 4,096; maximum 100,352 pixels |
| Training sequence limit | 2,048 total tokens, including expanded visual tokens |
| Quantization | W4G128 and W8G128, in separate run groups |
| LoRA | Rank 32, alpha 64, dropout 0.05; language q/k/v/o projections |
| Trainable weights | LoRA adapters and BF16 visual merger/projector; quantization scales in LoRA-targeted layers |
| Frozen weights | Language base weights, embeddings, output head, vision tower; other eligible linears use frozen initialized quantization scales |
| Learning rates | Adapter 2e-4; projector 1e-5; quantization scales 2e-4 |
| Schedule | Cosine; 3% LR warmup; weight decay 0; max gradient norm 1 |
| GRACE losses | DKD temperature 2, alpha 1, beta 4; RCKA weight 3, layer -2, warmup 25 updates |
| IB controller | Tau 3, eta 0.003, initial beta 0.5, bounds [0.1, 1.0]; inherited EMA policy |
| Memory controls | BF16, gradient checkpointing, no training KV cache, ZeRO-2; inherited attention implementation |
| Save / budgets | Save every 50 updates, retain two native checkpoints plus final export; inherited 34 GiB peak-reserved and five-hour training-loop budgets per invocation |

Validate tokenizer, chat template and visual-token expansion against the released processor before preparing data. Prefer the previous exact example identities and retained conversations if they remain valid. Revalidate them under the released processor; if eligibility changes, prepare one new shared subset, record the differences and hash, and use it for all four runs. A changed subset would limit attribution of differences from the previous experiment to initialization alone.

Budgets are pause thresholds, not guarantees against OOM or total runtime estimates. Profile each bit width and method before committing to full runs. If a setting must change, update the matched pair and its recorded recipe rather than silently changing one method.

The existing LoRA implementation trains `Q(W)x + LoRA(x)` and exports `Q(W + delta_W)x`. Evaluate the final merged and requantized exports as the primary low-bit artifacts. Retain native checkpoints to diagnose export loss if necessary. BF16 storage of grid-constrained weights does not establish packed W4/W8 inference memory or throughput gains.

## Evaluation protocol

Evaluate all nine rows using the same local seven-suite protocol, dataset versions, question identities, scoring, prompt templates, decoding policy and image budgets. Use the same 4,096/100,352 pixel limits for every row, including teacher and released references. Record tokenizer/processor identity and explicitly verify compatible special tokens and chat templates. Specify processor selection directly; this experiment has no `runs/sft` processor to inherit.

The 2,048-token limit above is a **training** limit. Pin evaluation context handling and per-suite generation caps separately from the existing evaluator; do not silently truncate evaluation questions or change decoding to a checkpoint's own default. If evaluating reference models with their original larger image budgets later, report that as a separate protocol and table.

| Suite | Primary metric | Expected N under the previous local protocol |
| --- | --- | ---: |
| HallusionBench | aAcc | 1,129 |
| MMBench EN dev | Per-row accuracy | 4,329 |
| ScienceQA test | Full-test accuracy, including text-only items | 4,241 |
| AI2D test | Accuracy | 3,088 |
| MMMU validation | Local mixed-answer accuracy | 900 |
| SEED-Bench-IMG | Accuracy | 14,233 |
| MMStar | Accuracy | 1,500 |

Also retain HallusionBench qAcc/fAcc, ScienceQA image/text splits, categories, errors and parse failures. Count failed parses in the denominator. Validate merged prediction files, unique question IDs, matching gold labels and sample counts across all rows; rank shards alone do not establish completion. Investigate any change to the expected counts before comparison.

Use the unweighted mean of the seven primary scores as a proposed descriptive aggregate, with every per-suite score visible. It is not an official leaderboard metric. Reevaluate the author's W4/W8 checkpoints locally: differences from published headline numbers could reflect preprocessing or scoring, so published scores must not be substituted into the primary comparison. Reuse an existing vanilla result only if its full protocol and provenance match; otherwise rerun it.

For each bit width `b`, report percentage-point differences:

- **Distillation gain:** `mean(grace_b) - mean(qat_b)`.
- **Gap to full method:** `mean(released_grace_b) - mean(grace_b)`; positive means constrained training scores lower.
- **Gap to BF16 initialization:** `mean(released_bf16) - mean(grace_b)`.
- Context: differences from vanilla 2B and vanilla 8B, plus the corresponding per-suite differences for every comparison.

Training cost reporting covers these four additional constrained runs: completed updates, trainable parameter counts, measured time, GPU-hours and peak allocated/reserved memory with measurement scope. Report loading/export overhead separately where available. The released initialization inherits the author's prior training; do not describe the constrained-stage cost as end-to-end reproduction cost. The full-method gap also combines differences in data, optimization and parameter updates, so it cannot be attributed solely to LoRA.

## Success and supervisor discussion

The intended outcome is positive GRACE-versus-QAT gain at **each** bit width and an acceptable gap to the corresponding released full-method model under matched local evaluation. Report W4 and W8 independently, including any failed or mixed outcome.

The proposed first comparison is the seven-suite mean with per-suite regressions visible. The acceptable mean decrease, any per-suite regression limits, and whether “better than QAT” requires more than a positive mean difference remain **TBD with the supervisor**. Produce the complete evaluation table before that discussion; do not invent a tolerance or claim overall success until it is agreed. Record the date and that the acceptance rule was set after observing this pilot. Any later confirmatory run should use a frozen rule.

This single-seed experiment supports a pilot conclusion. Additional matched seeds or uncertainty analysis can follow if the supervisor needs a robustness claim. Optional later controls include BF16 continued LoRA-SFT or immediate quantization of the released BF16 model; they are outside the initial four-run scope.

## Implementation work for the next session

1. Review this plan; resolve checkpoint revisions and download the three author releases. Record provenance for all five reference models.
2. Add a dedicated recipe/launcher that explicitly initializes from released BF16 and supports separate W4/W8 experiment groups. The current [constrained launcher](../../qwen-vl-finetune/scripts/run_constrained.py) uses `BASE_MODEL_DIR` for the student and **hardcodes `qat_bits=4`**; setting `QAT_BITS=8` alone currently does not select W8. Validate both bit widths through initialization, training, recipe pinning and final export.
3. Prepare and hash the shared subset after processor validation. Pin the recipe separately for each bit width because the existing comparison-recipe guard includes quantization bits. Preserve existing outputs and defaults for historical experiments.
4. Extend evaluation orchestration and summary generation to the nine explicit result IDs. The current [experiment wrapper](../constrained_32k_b128_p100352_t2048_seed42/run_local_evaluation.sh) pins `baseline sft qat grace`, while its shared launcher assumes an SFT processor; neither can be copied unchanged for this matrix. Add explicit model/processor mappings, resumability and complete-result validation.
5. Validate dry-run commands, source checkpoint identity, fresh initialization versus resume, W4/W8 settings, output isolation, subset equality, quantized-module coverage and export metadata. Add focused checks for the new initialization and bit-width routing.
6. When GPUs are available, inspect ownership, select free cards and run short profiles for all four configurations, beginning with GRACE for memory feasibility. Keep profile artifacts separate from accuracy artifacts. Check finite losses, teacher freezing, trainable parameters and successful export. Do not claim memory improvements before measuring them.
7. Evaluate the five reference models and train/evaluate the four final constrained models. All full runs start fresh after profiling; resume only native checkpoints from the same run and pinned recipe. Require 256 completed updates and a validated final export before accepting a training row.
8. Generate the validated 63-result comparison, per-bit-width deltas and constrained-stage runtime/memory report. Review the observed gaps with the supervisor and record the agreed acceptance criteria and follow-up decisions.

No runnable commands for this new experiment are supplied yet: the required W8 routing and evaluation matrix are implementation work for the next session. The old `all --train` command still selects the previous three-method experiment.

## Planned artifact layout

Only `README.md` exists at plan creation. The following is the intended implementation layout; shared model weights remain under `reproduction/GRACE/models/`.

```text
released_bf16_init_32k_b128_p100352_t2048_seed42/
  README.md
  provenance/                       # model revisions, hashes, environment, source cards
  subset.json                       # validated common manifest
  subset_comparison.json            # identities/retained turns versus previous experiment
  w4g128/
    comparison_recipe.json
    subset.json                     # identical validated manifest, with matching hash
    profiles/{qat,grace}/
    runs/{qat,grace}/                # native checkpoints and final exports
  w8g128/
    comparison_recipe.json
    subset.json                     # identical validated manifest, with matching hash
    profiles/{qat,grace}/
    runs/{qat,grace}/
  evaluation/
    protocol.json                   # explicit model/processor mapping and suite settings
    {result_id}/{suite}.json         # nine result IDs from the model matrix
    comparison.md
    comparison.csv
  reports/
    runtime_memory.md
    supervisor_decisions.md
```

Review checklist:

- [ ] Confirm the released BF16 initialization and four-run W4/W8 QAT-versus-GRACE matrix.
- [ ] Confirm the inherited constrained recipe and matched evaluation protocol.
- [ ] Implement checkpoint acquisition, launchers and evaluation in the next session.
- [ ] Run profiles, full training and evaluation when the server is available.
- [ ] Discuss measured quality gaps and acceptance criteria with the supervisor.
