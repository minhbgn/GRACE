# Larger visual-budget QAT/GRACE experiment — report scaffold

Prepared 2026-10-05. **Production training and evaluation have not been run.** This file records the setup and the report structure; it contains no new benchmark results.

## Question and controls

Does increasing the image ceiling from 98 to 576 merged visual tokens change the relative behavior of CE-only LoRA-QAT and LoRA-GRACE under the existing small adaptation budget? These unpruned controls also provide more visual-token headroom for subsequent LLMC+ work.

Both methods initialize from the authors' released BF16 SFT checkpoint. The new maximum is 589,824 pixels, with minimum 4,096 and a **4,096** total-token sequence limit (raised by the user before production preparation). Target: 32,768 examples, global batch 128 over two A100 40GB GPUs, 256 updates, training seed 42 and subset seed 2026; W4G128 and W8G128. The LoRA, optimizer, losses and controller remain those of the previous experiment. See the [run guide](README.md), [configuration](experiment.json), [recipe](recipe.env), and [memory checks](memory_report.md).

## Input-budget evidence

The [256-image CPU preview](token_budget_preview.json) increased mean visual tokens from 89.76 to 250.84, median from 88 to 260, and maximum from 98 to 576. This sample comes from the historical shuffled subset. Report the new full manifest's actual distribution after preparation.

Pending: `subset_comparison.json` counts of added/removed examples, changed retained conversations and order equality. Both the image budget and total sequence limit differ from the previous experiment, potentially changing retained text and data eligibility; this is not an isolated image-resolution intervention.

## Current 4,096-token memory preflight — passed

The user raised the sequence limit before production preparation. All four three-update checks passed at 576 visual / 4,096 total tokens, including 3,506 supervised tokens, accumulated gradients, full distillation losses for GRACE and completed exports. Peak reserved memory per GPU: **GRACE W4/W8 31.89 GiB; QAT W4/W8 13.08 GiB**, below the unchanged 34-GiB guard. Diagnostic accumulation=2 and RCKA warmup=1; production retains accumulation=64 and warmup=25. These are synthetic memory checks, not production training or quality results. See [the generated memory report](memory_report.md).

## Training completion and cost — pending

| Arm | Completed updates | Elapsed hours | GPU-hours | Peak reserved GiB/GPU |
| --- | --- | --- | --- | --- |
| QAT W4 | Pending | — | — | — |
| GRACE W4 | Pending | — | — | — |
| QAT W8 | Pending | — | — | — |
| GRACE W8 | Pending | — | — | — |

Use production `run_summary.json`, `invocations.jsonl` and runtime reports. Exclude synthetic memory-check checkpoints from quality results; state whether profiling/preparation/evaluation costs are included in each reported total.

## Latest seven-suite results — pending

Protocol: `local_native_greedy256_v1`, native saved processors, deterministic scoring and 256 generated tokens. Summaries will be written to `local_development/local_native_greedy256_v1/full/local_scores.{json,md}`. Require validated completion for all selected model–suite pairs. Report per-task scores and GRACE−QAT differences for each bit width, alongside unresolved and truncated outputs.

For historical comparison use the [previous latest local evaluation](../reproduction_eval_only/experiment_report.md), not the older training report's evaluation table. Reference checkpoint processors may have different native budgets; label those comparisons accordingly.

## Interpretation — pending

Discuss whether the QAT-versus-GRACE ranking changes, which tasks benefit or regress, and observed compute costs. Keep the known native-versus-baked-export caveat and single-seed limitation visible. This experiment implements no token reduction and supplies no evidence yet of pruning benefit, packed integer acceleration or exact paper reproduction.
