# Pinned lmms-eval comparison

**Protocol recovery update (2026-09-25):** The [source audit and new evaluation workflows](../../evaluation/README_paper_protocol.md) recover all seven upstream script recipes and add a separate corrected standard harness protocol. Confirmed differences include checkpoint-native image budgets and MMStar's 64-token launcher limit; MMMU task loading and open-answer scoring bugs are fixed for the new harness protocol. CPU checks passed; no new GPU results or paper-tolerance claim are included here. Historical scores below are unchanged.

## Completed four-suite round — 2026-09-25

**All 40 requested model/benchmark evaluations are complete and verified.** This
round covers ScienceQA-image, AI2D, SEED-Bench-image and MMStar for all ten models.
MMMU, HallusionBench and MMBench remain unevaluated in the full harness matrix.
The completed four-suite result supports the earlier finding that **LoRA-QAT is
the stronger average-quality baseline under our constrained training recipe**.
LoRA-GRACE remains competitive with the released full-method checkpoints, but
its added distillation stage does not improve the matched QAT mean.

This section reports **Mean-4**, the unweighted arithmetic mean of these four
percentage scores. It is a descriptive summary of this completed subset, not
the paper's seven-benchmark average and not a replacement for the missing three
results. The original [seven-suite status table](lmms_evaluation/full/comparison.md)
correctly retains empty full-average cells. No missing score is treated as zero.

### Coverage and verification

| Reported benchmark | Evaluated rows per model | Rows contributing to this score | Aggregation |
| --- | ---: | ---: | --- |
| ScienceQA-image | 2,017 | 2,017 | Image-only test accuracy |
| AI2D | 3,088 | 3,088 | Accuracy with harness answer extraction |
| SEED-Bench-image | 17,990 | 14,233 | Image-question accuracy; 3,757 video questions excluded from this column |
| MMStar | 1,500 | 1,500 | Equal mean of 18 L2-category accuracies |

All 40 completion markers and their saved artifact hashes were checked. Scores
were independently recomputed from the sample-level metric records, including
SEED's image-only denominator and MMStar's category weighting; all matched the
reported primary metrics. Every job has `limit=null`, complete unique document
coverage and a zero-call, zero-error judge audit. Dataset fingerprints,
saved document contents and targets agree across the ten models. This verifies
the saved scoring and comparison, without rerunning model inference or checking
every model weight again. The common recorded protocol uses SDPA and
200,704/1,605,632 minimum/maximum image pixels.

Evidence and reproducibility: [unrounded scores (CSV)](reports/lmms_four_suite_scores.csv),
[analysis, paired statistics and source hashes (JSON)](reports/lmms_four_suite_analysis.json),
[generated tables](reports/lmms_four_suite_tables.md), and the
[CPU-only compilation script](reports/compile_lmms_four_suite_report.py).
Run the script with `.venv-lmms-eval/bin/python`; it reads the existing outputs
and rewrites only these derived report artifacts.

### Follow-up integrity audit (2026-09-25)

The [checkpoint, dataset and evaluation audit](reports/evaluation_integrity_audit.md)
found no download, path, dataset, averaging or answer-parsing error explaining
the local low-bit models' advantages. All ten checkpoints match their evaluation
manifests; all 59 files in the five downloaded references match official Hub
commits, and all 278 cached benchmark Parquets match Hub checksums. The recorded
paths/configs/prompts agree, a CPU processor probe is identical across models,
and sampled W4 groups confirm grid-projected values in both local and released
exports. Independent explicit-choice rescoring reproduces every sample's
correctness for the released and local low-bit models.

Three small parser-sensitive cases affect only vanilla references and do not
change the low-bit-model comparison. Full details, limitations and reproducible
CPU audit scripts are linked in the audit. These checks strengthen the internal
four-suite result; they do not establish exact paper-protocol reproduction or
universal superiority to full GRACE. The known MMMU runtime bug remains pending.

### Results for all ten models

All entries are percentages; differences below are percentage points (pp).
`qat_*` and `grace_*` are the constrained LoRA runs initialized independently
from `released_bf16`. The `released_grace_*` rows are the authors' full-method
research checkpoints. Local SFT is the previous vanilla-initialized run.

| Model | SQA-IMG | AI2D | SEED-IMG | MMStar | Mean-4 |
| --- | --- | --- | --- | --- | --- |
| vanilla_2b | 87.26 | 76.13 | 76.09 | 55.08 | 73.64 |
| vanilla_8b_teacher | 94.70 | 84.00 | 78.56 | 64.55 | 80.45 |
| sft_2b | 86.76 | 77.53 | 76.68 | 56.10 | 74.27 |
| released_bf16 | 88.30 | 81.80 | 76.27 | 69.39 | 78.94 |
| released_grace_w4 | 87.11 | 80.41 | 75.70 | 67.49 | 77.68 |
| released_grace_w8 | 86.66 | 79.57 | 75.45 | 65.20 | 76.72 |
| qat_w4 | 86.12 | 80.63 | 76.39 | 68.29 | 77.86 |
| grace_w4 | 86.71 | 79.57 | 77.09 | 63.98 | 76.84 |
| qat_w8 | 87.70 | 82.32 | 76.73 | 70.66 | 79.35 |
| grace_w8 | 87.95 | 81.02 | 77.14 | 64.99 | 77.78 |

The 8B teacher has the highest Mean-4 at **80.45**. Among 2B models, **LoRA-QAT W8
is highest at 79.35**, followed by the released BF16 initialization at 78.94.
QAT W4 scores 77.86 and GRACE W4 scores 76.84. At W8, GRACE scores 77.78.
These are fixed-run rankings; small differences do not establish reproducible
training improvements without additional seeds.

The released BF16 student's +5.30 pp over vanilla 2B is a substantial inherited
advantage. The local low-bit runs start from that stronger checkpoint. Their
performance cannot be attributed entirely to our short LoRA training stage,
nor does this round establish that the authors' earlier full training can be
reproduced at our total compute cost.

### Comparison with the paper's claimed results

The reference is **Table 6 in arXiv v5**, which contains the Qwen3-VL rows, rather
than the LLaVA or Qwen2-VL tables. Its values agree with the repository's model
zoo. The table below extracts the same four named columns and recomputes their
mean; the paper does not publish a separate Mean-4. The three-decimal reference
means are arithmetic on the paper's one-decimal entries, not extra measurement
precision. [Paper, Table 6](https://arxiv.org/html/2601.22709v5#A1.T6),
[repository results](https://github.com/ForeverBlue816/GRACE#-results).

| Model | SQA paper → ours | AI2D paper → ours | SEED paper → ours | MMStar paper → ours | Paper Mean-4 | Our Mean-4 | Δ Mean-4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vanilla_8b_teacher | 85.0 → 94.70 | 85.7 → 84.00 | 77.5 → 78.56 | 70.9 → 64.55 | 79.775 | 80.45 | +0.68 |
| vanilla_2b | 81.4 → 87.26 | 76.9 → 76.13 | 71.2 → 76.09 | 58.3 → 55.08 | 71.950 | 73.64 | +1.69 |
| released_bf16 | 86.2 → 88.30 | 81.3 → 81.80 | 76.7 → 76.27 | 67.3 → 69.39 | 77.875 | 78.94 | +1.07 |
| released_grace_w8 | 85.3 → 86.66 | 80.4 → 79.57 | 75.9 → 75.45 | 66.5 → 65.20 | 77.025 | 76.72 | -0.31 |
| released_grace_w4 | 84.3 → 87.11 | 79.5 → 80.41 | 75.1 → 75.70 | 65.8 → 67.49 | 76.175 | 77.68 | +1.50 |

**Released W4 now exceeds the published value on all four tested benchmarks:**
ScienceQA +2.81, AI2D +0.91, SEED +0.60 and MMStar +1.69 pp. Its Mean-4 is
**+1.50 pp** above the corresponding published columns. Released W8 is higher
on ScienceQA (+1.36) but lower on AI2D (−0.83), SEED (−0.45) and MMStar (−1.30),
leaving its Mean-4 **0.31 pp below** the corresponding reference.

This is encouraging for the released checkpoints' evaluated capabilities,
but it does **not** resolve the original seven-suite discrepancies of
73.85 versus 75.0 for W4 and 72.28 versus 75.9 for W8. In particular, MMMU was
the largest negative discrepancy in the earlier custom evaluation: −9.28 pp
for W4 and −14.41 pp for W8 relative to the paper. It has not been retested
successfully here. Neither the four-suite averages nor the old three remaining
scores should be combined into a hybrid seven-suite claim.

There are still clear protocol-dependent discrepancies. The teacher's
ScienceQA score is +9.70 pp above the published column while its MMStar score
is −6.35 pp below it; these partly cancel in its Mean-4. The ScienceQA image-only
scope, exact prompts, image budgets, dataset revisions and aggregation settings
must be confirmed before treating matching column names as identical tests.
Closeness of an average is insufficient evidence of exact reproduction.

**The unexpected released-checkpoint ordering persists:** W4 beats W8 on all
four suites, by **0.96 pp in Mean-4**. The corresponding published four-column
mean favors W8 by 0.85 pp. This discrepancy survives the change to lmms-eval
and a common explicit image budget. Native W4/W8 processor defaults alone
therefore cannot explain the ordering in this round. Checkpoint identity,
export/quantization behavior and the authors' exact evaluation configuration
remain possible causes; this result does not establish which is responsible.

Table 6 contains no matched Qwen3-VL QAT-only row. Consequently, our direct
GRACE-versus-QAT inference comes from our own matched LoRA runs, not a claimed
Qwen3 QAT comparison in that table. [Paper, Table 6](https://arxiv.org/html/2601.22709v5#A1.T6).

### LoRA-GRACE versus matched QAT and full-method releases

| Contrast (left − right) | scienceqa | ai2d | seedbench | mmstar | Δ Mean-4 |
| --- | --- | --- | --- | --- | --- |
| grace_w4_minus_qat_w4 | +0.59 | -1.07 | +0.70 | -4.31 | -1.02 |
| grace_w8_minus_qat_w8 | +0.25 | -1.30 | +0.41 | -5.67 | -1.58 |
| released_grace_w4_minus_released_grace_w8 | +0.45 | +0.84 | +0.25 | +2.30 | +0.96 |
| grace_w4_minus_released_grace_w4 | -0.40 | -0.84 | +1.39 | -3.51 | -0.84 |
| grace_w8_minus_released_grace_w8 | +1.29 | +1.46 | +1.69 | -0.21 | +1.06 |

At both precisions, LoRA-GRACE improves ScienceQA and SEED relative to matched
QAT, but loses on AI2D and MMStar. The net Mean-4 difference is **−1.02 pp at W4**
and **−1.58 pp at W8**. MMStar contributes −1.08 and −1.42 pp respectively to
those four-suite mean differences. Averaging only the other three benchmarks
gives +0.08 pp at W4 and −0.21 pp at W8: the main deficit is concentrated in
MMStar, while the remaining tasks show a small trade-off rather than a large
distillation gain.

Against the corresponding full-method release, LoRA-GRACE is **0.84 pp lower
at W4** and **1.06 pp higher at W8** on Mean-4. W4's SEED improvement offsets
losses on the other three tasks; W8 improves three tasks and is almost tied
on MMStar. This demonstrates useful constrained adaptation of the released
BF16 initialization on the measured subset. It does not establish overall
superiority to full GRACE, especially with MMMU and the judge-based suites
missing and released W8's unexpected ordering unresolved.

For development, QAT is at least as compelling: QAT W4 is **0.18 pp above** its
release's Mean-4, and QAT W8 is **2.63 pp above** its release. Compared with the
BF16 initialization, QAT W4 loses 1.09 pp while QAT W8 gains 0.41 pp; GRACE W4
and W8 lose 2.11 and 1.16 pp. The +0.41 pp QAT W8 result is an observed score
difference, not evidence that quantization inherently improves the model.

The [training report](experiment_report.md) records roughly **1.93 hours and
8.47 GiB peak reserved memory per GPU for QAT**, versus **2.39 hours and
26.07 GiB for GRACE**, both using two GPUs. With higher Mean-4, less training
memory and shorter runtime, QAT remains the stronger measured starting point
for our budget. These are local-stage training costs, not end-to-end costs
including the released initialization, and not packed inference measurements.

### How strong are the task-level differences?

For GRACE versus QAT, predictions were paired by the same saved documents and
targets. The following 95% intervals use 20,000 paired bootstrap replicates
(seed 42); MMStar is resampled within each of its 18 L2 categories to preserve
the reported macro metric. Win/loss counts are unweighted example counts, so
MMStar's net count does not directly equal its macro-score difference.

| Bits | Benchmark | GRACE-only correct / QAT-only correct | Δ pp | Paired 95% interval |
| --- | --- | --- | --- | --- |
| w4 | scienceqa | 69 / 57 | +0.59 | [-0.50, +1.69] |
| w4 | ai2d | 114 / 147 | -1.07 | [-2.07, -0.03] |
| w4 | seedbench | 495 / 395 | +0.70 | [+0.29, +1.12] |
| w4 | mmstar | 60 / 139 | -4.31 | [-6.22, -2.39] |
| w8 | scienceqa | 68 / 63 | +0.25 | [-0.84, +1.39] |
| w8 | ai2d | 120 / 160 | -1.30 | [-2.36, -0.23] |
| w8 | seedbench | 582 / 524 | +0.41 | [-0.05, +0.86] |
| w8 | mmstar | 64 / 155 | -5.67 | [-7.72, -3.59] |

The MMStar deficits are substantial relative to this sample uncertainty at
both bit widths. W4's SEED improvement is also clearer than the ScienceQA
gains. Both ScienceQA intervals and W8's SEED interval include zero. AI2D
leans against GRACE, with W4's interval especially close to zero. These are
descriptive per-benchmark intervals without multiple-comparison correction;
they do not measure training-seed variability or uncertainty in the paper's
results. No confidence interval for the four-suite mean is asserted.

### MMStar diagnosis

All six broad MMStar categories decline under GRACE relative to matched QAT:

| MMStar category | W4 GRACE − QAT | W8 GRACE − QAT | Teacher − BF16 init |
| --- | --- | --- | --- |
| coarse perception | -6.63 | -4.69 | -4.42 |
| fine-grained perception | -2.31 | -4.77 | -0.67 |
| instance reasoning | -3.68 | -8.58 | -7.64 |
| logical reasoning | -3.31 | -2.78 | +1.03 |
| math | -4.24 | -5.30 | -6.72 |
| science & technology | -5.68 | -7.91 | -10.63 |

The largest W8 losses occur in instance reasoning (−8.58 pp) and science and
technology (−7.91 pp). The teacher is itself below the released BF16 student
in five of six categories and by **4.84 pp overall** on this benchmark. This is
consistent with the earlier hypothesis that guidance from a teacher weaker
on some student capabilities can cause negative transfer. It is not a causal
test: confidence on the training distribution, RCKA, optimization and the
loss controller can all contribute. The previously observed early saturation
of the distillation controller remains a reason to test weaker or delayed
supervision, rather than assuming additional teacher matching will help.

The result is not merely an artifact of MMStar's macro weighting. Ordinary
question-level accuracy is **62.80 versus 68.07** for GRACE/QAT W4 and **64.13
versus 70.20** for GRACE/QAT W8: deficits of 5.27 and 6.07 pp also remain under
micro averaging. These diagnostic micro scores are not substituted into the
primary harness result.

### What changed relative to the old custom evaluator?

To avoid counting the ScienceQA scope change as an improvement, this table
uses the **old ScienceQA image subset** (2,017 questions), not the old overall
ScienceQA score (4,241 questions). SEED likewise uses image questions in both
columns. This aligns the broad coverage; it does not make the protocols
identical. In particular, old MMStar uses question-level accuracy while the
new primary score uses L2-category macro accuracy.

| Model | Old SQA-IMG | New SQA-IMG | Old Mean-4 | New Mean-4 | Δ Mean-4 |
| --- | --- | --- | --- | --- | --- |
| vanilla_2b | 85.62 | 87.26 | 69.39 | 73.64 | +4.25 |
| vanilla_8b_teacher | 93.36 | 94.70 | 77.68 | 80.45 | +2.77 |
| sft_2b | 84.78 | 86.76 | 71.31 | 74.27 | +2.96 |
| released_bf16 | 86.07 | 88.30 | 76.10 | 78.94 | +2.84 |
| released_grace_w4 | 85.52 | 87.11 | 75.31 | 77.68 | +2.36 |
| released_grace_w8 | 84.88 | 86.66 | 74.65 | 76.72 | +2.07 |
| qat_w4 | 84.28 | 86.12 | 74.99 | 77.86 | +2.87 |
| grace_w4 | 85.47 | 86.71 | 73.24 | 76.84 | +3.60 |
| qat_w8 | 86.22 | 87.70 | 76.34 | 79.35 | +3.02 |
| grace_w8 | 86.61 | 87.95 | 73.01 | 77.78 | +4.77 |

Every model's four-suite mean rises, by **2.07–4.77 pp**. GRACE's MMStar scores
rise especially strongly: +8.58 pp at W4 and +11.79 pp at W8 in the displayed
primary metrics. With micro accuracy on both sides, the increases are still
+7.40 and +10.93 pp. The earlier very low MMStar scores were therefore sensitive
to the evaluation setup, but the GRACE-versus-QAT regression survives it.

The change simultaneously increased allowed image resolution, changed output
budgets and prompts, used harness answer extraction and scoring, and changed
some task scopes/weighting. This comparison cannot allocate the improvement
to any single cause. Larger image budgets plausibly recover visual detail lost
under the training-sized cap, but that explanation needs a controlled test.
A parser change or the fact that the implementation is called lmms-eval is not,
on its own, evidence that the model has improved.

### MMMU status and next steps

MMMU is omitted as requested. The saved
[W4 smoke-test log](lmms_evaluation/smoke_8/released_grace_w4/mmmu/attempt_20260923T153703_467649Z/run.log)
confirms `OpenAIError: Missing credentials` during task loading. The standard
`mmmu_val` scorer is local; its shared utilities initialize an unused judge.
Our temporary guard covered the initial `task_config()` import, but
`TaskManager.load_config()` subsequently reloaded the registered task YAML
after that guard had been restored. The earlier CPU preflight did not exercise
this second import and therefore did not establish credential-free runtime
readiness. This is an integration bug to fix before a future MMMU run, not a
requirement to pay for judging the selected standard MMMU metric. The current
analysis makes no changes to the pinned evaluator or its recorded results.

1. **Use LoRA-QAT as the main token-reduction baseline**, with W8 as the higher
   quality reference and W4 as the stronger weight-compression candidate.
   Keep GRACE as a matched ablation or a task-specific option. The current
   results establish a feasible starting point, not that token reduction will
   preserve the same quality.
2. **Define the deployment image/token budget before optimizing token reduction.**
   These evaluations allow up to 1,605,632 pixels per image, whereas training
   used 100,352. Compare no reduction, token reduction alone, QAT alone, and
   joint token reduction plus QAT under the same evaluation budget. Record actual
   retained visual tokens, accuracy, prefill latency, peak memory and KV-cache
   use; changing resolution is not equivalent to pruning/merging tokens.
3. **Investigate GRACE with small controlled loss ablations:** CE-only QAT,
   weaker/fixed/delayed GDKD, RCKA separately, then their combination. Prioritize
   MMStar and retain ScienceQA/SEED to detect lost benefits. If student and
   teacher retain different tokens, define correspondence or pooling for RCKA
   rather than applying the original token-pair loss blindly.
4. **Complete or explicitly retain the reduced evaluation scope.** A later
   MMMU fix should cover the full task-loading path and use a separately
   versioned protocol/output root, preserving this round. HallusionBench and
   MMBench still need the intended judge configuration. Do not report a full
   paper-average reproduction until those tasks are comparable and complete.
5. **If pursuing the remaining release discrepancy, run the controlled pixel
   crossover next**, first on released W4/W8 and MMStar/AI2D: 50,176, 100,352 and
   200,704 maximum pixels, with a common minimum no greater than 50,176 and
   all other settings fixed. The current 200,704 minimum cannot be retained
   for lower maximum budgets. Compare these separately with the present
   high-budget run, then isolate generation and scoring settings. Request
   the authors' exact commands and raw predictions before assigning a cause.

For an extreme-compression claim, the final candidates must also be exported
to the intended packed low-bit backend and re-evaluated there. These runs use
BF16-stored quantization-grid weights; they establish accuracy feasibility,
not INT4 memory/throughput savings or successful joint token compression.

## Evaluation workflow and protocol reference

The full workflow supports **10 checkpoints × 7 benchmarks** from `experiment.json`
with the Qwen3-VL backend in lmms-eval **v0.7.1**, commit
`88b23e2bfa16a1edbc16e9e238ed82130b3a4f56`. It loads the merged Hugging Face
exports directly. No training or further quantization is needed.

The experiment's existing `evaluation/` results and report remain the custom
Transformers protocol. New outputs go to `lmms_evaluation/full/`; limited runs
go to separate `lmms_evaluation/smoke_N/` directories. These measure accuracy
with BF16 storage of the grid-projected weights; they do **not** measure the
memory, speed, or accuracy of a packed INT4/INT8 deployment backend.

## What “paper protocol” means here

The [GRACE README](https://github.com/ForeverBlue816/GRACE#evaluation) states that
it uses lmms-eval. However, its linked
[ScienceQA Slurm script](https://github.com/ForeverBlue816/GRACE/blob/main/evaluation/ScienceQA/eval_scienceqa.slurm)
invokes a custom `eval_scienceqa.py` through `torchrun`, reads `MODEL_PATH`
instead of the README's `MODEL`, and uses eight output tokens. The public recipe
does not establish the authors' exact harness revision, task variants, image
budgets, prompts, or judge settings.

Consequently, this is an **independent, pinned lmms-eval comparison**, not a
verified reproduction of the paper's exact evaluation. We preserve the pinned
harness's task prompts, model-specific prompt overrides, filters and scorers.
Results can determine whether changing the evaluation stack changes our scores;
they cannot by themselves identify which setting caused the published gap.

The first comparison uses the pinned backend's image defaults, **200,704 minimum
and 1,605,632 maximum pixels per image**, with greedy generation, zero-shot
prompts, batch size one and a single GPU. These are different from our old
4,096/100,352 limits and are **not recovered author settings**. Multiple-image
MMMU samples can use substantially more memory. An OOM is a failed run; do not
quietly shrink images to finish a row. Use a new protocol/output root for changes.
SDPA is explicitly selected for portability. The system prompt is
`You are a helpful assistant.`; interleaving is disabled, as in this backend's
default. See [lmms_protocol.json](lmms_protocol.json) for all explicit settings.

| Suite ID | Harness task / split | Primary metric | Output cap | Judge |
| --- | --- | --- | ---: | --- |
| `scienceqa` | `scienceqa_img` / test, ScienceQA-IMG | `exact_match,none` × 100 | 16 | No |
| `ai2d` | `ai2d` / test | `exact_match,flexible-extract` × 100 | 16 | No |
| `mmmu` | `mmmu_val` / validation | `mmmu_acc,none` × 100 | 128 | No |
| `mmstar` | `mmstar` / val | `average,none` × 100 | 128 | No |
| `seedbench` | `seedbench` / test | `seed_image,none` × 100 | 128 | No |
| `hallusionbench` | `hallusion_bench_image` / image | `aAcc,none`, already % | 128 | `gpt-4` |
| `mmbench` | `mmbench_en_dev` / dev, en | `gpt_eval_score,none`, already % | 1024 | `gpt-4o-2024-11-20` |

The 128-token caps for MMStar and SEED-Bench come from the backend fallback.
ScienceQA-image excludes text-only questions included in our old overall score.
HallusionBench-image has different coverage and judge scoring from our old
rule-based overall score. MMStar's `average` is a mean over L2-category accuracies.
SEED-Bench runs the harness task and selects its image metric; other emitted
metrics remain in the raw output. MMBench retains its rolling-question aggregation
and judge-assisted answer extraction. Means are only emitted for complete
seven-suite model rows, with all primary metrics normalized to percentages.

MMMU's standard scorer is local, but the current runtime still triggers unused
judge initialization on a second task import. The initial wrapper guard proved
insufficient; see the failure analysis above. MMMU is excluded from this round.
HallusionBench and MMBench remain judge-based tasks.

## Prepare and check without GPU inference

Run from the GRACE root:

```bash
cd /raid/lvchengtao/Minh/Distillation/reproduction/GRACE
# Already prepared on this machine; reproducible setup for another environment:
bash evaluation/prepare_lmms_eval.sh

EXP=experiments/released_bf16_init_32k_b128_p100352_t2048_seed42
bash "$EXP/run_lmms_evaluation.sh" plan
bash "$EXP/run_lmms_evaluation.sh" check
bash "$EXP/run_lmms_evaluation.sh" prepare
```

`plan` lists the 70 jobs. `check` imports the actual source backend, resolves all
seven YAMLs and checks the selected checkpoint headers/export completion.
`prepare` pins the protocol, package versions, runner/worker hashes and all seven
Hugging Face dataset commit IDs, including resolved repository redirects. It
fetches metadata, not dataset contents. The first actual evaluation downloads
its dataset and pins split lengths/fingerprints for subsequent models.

The isolated `.venv-lmms-eval` inherits the existing `grace` environment and
installs evaluation additions locally. The launcher explicitly puts the source
checkout on `PYTHONPATH`: the inherited wheel otherwise wins over the editable
install and lacks some extensionless task templates. The source must remain at
the clean pinned commit. Runtime package versions are locked per output root;
a later package update requires a fresh root rather than mixing results.

## Initial CPU validation (before the completed GPU round)

CPU preflight resolved all seven task configurations and validated all ten local
checkpoint/export structures. Seven automated tests cover metric scales,
incomplete/duplicate samples, smoke/full separation, changed sample identities,
modified completed artifacts, and partial-matrix reporting. A separate synthetic
two-question test passed through the real harness's task loading, prompt
resolution, generation interface, scoring, sample logging and result validation.
No checkpoint inference or external judge requests were used for these checks.

```bash
.venv-lmms-eval/bin/python -m unittest discover -s evaluation/tests -p test_lmms_matrix.py -v
.venv-lmms-eval/bin/python evaluation/tests/cpu_lmms_integration.py
```

GPU inference and dataset loading have since completed for the four suites
reported above. MMMU failed during loading; judge availability for the two
judge-based suites remains untested.

## Smoke test, then evaluate

Before launching inference, run `gpuwho`, check current ownership and memory,
and choose a free GPU in PCI-bus order. Do not reclaim another person's jobs.
Only one process
uses that GPU; all selected model/task pairs run sequentially.

(Running on GPU index 0)
```bash
gpuwho
bash "$EXP/run_lmms_evaluation.sh" run --gpu 0 \
  --methods released_grace_w4,released_grace_w8 \
  --suites scienceqa,ai2d,mmstar,seedbench --smoke 8

# First obtain the full released-checkpoint comparison on non-judge suites:
bash "$EXP/run_lmms_evaluation.sh" run --gpu 0 \
  --methods released_grace_w4,released_grace_w8 \
  --suites scienceqa,ai2d,mmstar,seedbench

# Then all ten models on the same four suites; verified completed jobs are skipped:
bash "$EXP/run_lmms_evaluation.sh" run --gpu 0 \
  --suites scienceqa,ai2d,mmstar,seedbench
```

Limited samples test loading and generation; they are not benchmark estimates.
Use the four completed non-judge suites for this round: HallusionBench's
figure/question group metrics can be undefined on an arbitrary first-N subset.

HallusionBench and MMBench require an explicitly configured judge endpoint and
credentials. The model IDs in the protocol are the harness defaults; their
availability and the authors' exact judges are unverified. Configure any changes
**before preparing a new output root**. Set `OPENAI_API_KEY` privately in the
environment (do not put it in scripts, JSON files, shell history or reports).
`--allow-judge` explicitly enables the external judge requests and their cost.

```bash
# With judge credentials already configured in this shell:
bash "$EXP/run_lmms_evaluation.sh" run --gpu GPU_INDEX \
  --suites hallusionbench,mmbench --allow-judge

# Equivalent whole-matrix command, resuming any completed jobs:
bash "$EXP/run_lmms_evaluation.sh" run --gpu GPU_INDEX --allow-judge
bash "$EXP/run_lmms_evaluation.sh" summarize
```

Each job saves a fresh attempt directory with `job.json`, `run.log`,
`results.json`, `samples.jsonl`, `dataset.json`, and `judge_audit.json`. Model
weights/configuration/tokenizer assets are hashed before evaluation. Dataset
fingerprints and document/target identities must agree across model rows.
A completion marker is written only after validating the expected task, metric
scale, requested limit, complete document coverage and judge audit. Resuming
verifies saved output hashes before skipping a job. Unfinished attempts are
retained; rerunning creates a new attempt, without generation-cache reuse.

The wrapper conservatively rejects an attempt with **any judge HTTP/response
failure**, even if a later retry succeeds, or MMBench's random-answer fallback.
Raw outputs remain available for diagnosis. This prevents an API outage from
silently becoming a reported model score. A successful HTTP response is not a
guarantee of judge quality: inspect the saved judge outputs as part of review.

`lmms_evaluation/full/comparison.{json,md}` contains the new matrix. Partial
matrices show missing cells and omit incomplete row means. Keep protocol labels
when comparing it with the old report: changed splits, image budgets, prompts,
and scorers prevent interpreting every difference as an accuracy improvement.

The completed four-suite comparison and follow-up priorities are reported
above. Additional benchmarks or a controlled pixel-budget crossover belong
to a separately recorded follow-up; no additional evaluations were launched
while compiling this report.
