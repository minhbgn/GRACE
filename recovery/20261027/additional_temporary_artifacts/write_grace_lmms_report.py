from pathlib import Path
import json
p=Path('reproduction/GRACE/experiments/released_bf16_init_32k_b128_p100352_t2048_seed42/lmms_evaluation.md')
exp=p.parent
x=json.loads((exp/'reports/lmms_four_suite_analysis.json').read_text())
tables={}
for part in (exp/'reports/lmms_four_suite_tables.md').read_text().split('## ')[1:]:
    title,content=part.split('\n\n',1);tables[title]=content.strip()
report='''## Completed four-suite round — 2026-09-25

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

Evidence and reproducibility: [full-precision scores (CSV)](reports/lmms_four_suite_scores.csv),
[analysis, paired statistics and source hashes (JSON)](reports/lmms_four_suite_analysis.json),
[generated tables](reports/lmms_four_suite_tables.md), and the
[CPU-only compilation script](reports/compile_lmms_four_suite_report.py).
Run the script with `.venv-lmms-eval/bin/python`; it reads the existing outputs
and rewrites only these derived report artifacts.

### Results for all ten models

All entries are percentages; differences below are percentage points (pp).
`qat_*` and `grace_*` are the constrained LoRA runs initialized independently
from `released_bf16`. The `released_grace_*` rows are the authors' full-method
research checkpoints. Local SFT is the previous vanilla-initialized run.

{{scores}}

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

{{paper}}

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

{{pairs}}

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

{{ci}}

The MMStar deficits are substantial relative to this sample uncertainty at
both bit widths. W4's SEED improvement is also clearer than the ScienceQA
gains. Both ScienceQA intervals and W8's SEED interval include zero. AI2D
leans against GRACE, with W4's interval especially close to zero. These are
descriptive per-benchmark intervals without multiple-comparison correction;
they do not measure training-seed variability or uncertainty in the paper's
results. No confidence interval for the four-suite mean is asserted.

### MMStar diagnosis

All six broad MMStar categories decline under GRACE relative to matched QAT:

{{categories}}

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

{{old}}

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

'''
for key,value in tables.items():
    report=report.replace('{{'+key+'}}',value)
assert '{{' not in report
old=p.read_text()
old=old.replace('This workflow evaluates all **10 checkpoints × 7 benchmarks** from `experiment.json`',
                'The full workflow supports **10 checkpoints × 7 benchmarks** from `experiment.json`')
old=old.replace("MMMU's standard scorer is local. Upstream nevertheless initializes an unused\njudge while importing shared utilities. The wrapper disables that unused object\nand verifies the selected function is `mmmu_process_results`; accidental access\nto the disabled judge raises an error. It does not supply fake API credentials.",
"MMMU's standard scorer is local, but the current runtime still triggers unused\njudge initialization on a second task import. The initial wrapper guard proved\ninsufficient; see the failure analysis above. MMMU is excluded from this round.\nHallusionBench and MMBench remain judge-based tasks.")
old=old.replace('## Validation completed on this machine','## Initial CPU validation (before the completed GPU round)')
old=old.replace('Actual model loading, GPU memory requirements, dataset downloads and judge\navailability still need the staged execution below.',
'GPU inference and dataset loading have since completed for the four suites\nreported above. MMMU failed during loading; judge availability for the two\njudge-based suites remains untested.')
old=old.replace('--suites scienceqa,ai2d,mmmu,mmstar,seedbench', '--suites scienceqa,ai2d,mmstar,seedbench')
old=old.replace('# Then all ten models on the same five suites;', '# Then all ten models on the same four suites;')
old=old.replace('Use the five non-judge suites for the initial smoke test:', 'Use the four completed non-judge suites for this round:')
old=old.replace('That\ncrossover is deliberately deferred until this harness comparison is complete.',
'The four-suite comparison above is complete; any crossover is a separate\nfollow-up experiment and has not been launched.')
first=old.index('\n')
p.write_text(old[:first+1]+'\n'+report+old[first+1:].lstrip())
