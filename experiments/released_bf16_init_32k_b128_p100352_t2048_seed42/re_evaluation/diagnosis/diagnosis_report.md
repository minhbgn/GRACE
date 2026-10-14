# Diagnosis of the completed reference re-evaluation — 2026-09-26

> Follow-up: [four-suite diagnosis](four_suite_followup.md) adds label-bias measurements and reference rescoring. Per the user's instruction, the shared W4/W8 resolution test proposed below is withdrawn; preserve native processor settings. Historical findings below remain unchanged.

Recommendation: do one bounded round of targeted diagnosis (option 1), then use a
validated, frozen local protocol as the primary controlled comparison (a qualified
option 3). Do not drop benchmarks merely because their paper gap exceeds 10 pp
(option 2). Keep the paper table as an external reference and report unresolved
discrepancies rather than representing local results as exact reproduction.

This analysis verified all 35 completion artifacts, result coverage, score
arithmetic and checkpoint paths. It used saved answers only: no GPU inference,
new downloads, live judges, benchmark-score edits or stage-two gate changes.
Reproducible evidence: [analysis script](analyze_saved_answers.py) and
[machine-readable results](saved_answer_analysis.json). Run the script using
GRACE's `.venv-lmms-eval/bin/python`.

## What the pattern tells us

The released BF16 checkpoint matches six of seven published columns to within
0.70 pp. Its exception is MMBench. The vanilla gaps are much larger and have
opposite signs across tasks. Released W4/W8 MMMU responses are already single
letters, so the vanilla response-format explanation cannot explain their MMMU
gaps. There is no evidence for one global failure that explains all columns.

## 1. Nondeterminism: possible, but a low-priority primary explanation

The archived evaluators explicitly set `do_sample=False` and call `model.eval()`.
The checkpoint's saved `do_sample=true` is overridden by the generation call.
Consequently these are greedy runs, not stochastic temperature-0.7 samples.
GPU floating-point operations and software changes can still alter predictions;
identical seeds do not guarantee reproduction across platforms or versions
([PyTorch reproducibility guidance](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/randomness.rst)).

We have not rerun identical inputs, so run-to-run variance is unmeasured. There
is no basis to blame the 8–17 pp gaps on randomness before testing the much more
concrete problems below. A fixed representative subset repeated under identical
settings is sufficient for an initial determinism check; no need to repeat the
entire matrix first.

## 2. Answer parsing and output truncation: confirmed problems, unequal impact

MMMU MC generation is capped at eight new tokens. The custom parser picks the
first character anywhere in the output that belongs to the valid answer letters.
For vanilla 2B, the saved response to `validation_Accounting_2` is
`To solve for the missing account balance,`; it is parsed as A. Another response,
`To determine the highest possible beta estimate for`, is parsed as B and happens
to match its gold B. These are not trustworthy extracted answers. Parsing can
create both false positives and false negatives.

| MMMU MC diagnostic | Vanilla 2B | Vanilla 8B | Released BF16 | Released W4 | Released W8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MC questions | 847 | 847 | 847 | 847 | 847 |
| Single-letter raw answers | 49 | 118 | 847 | 847 | 847 |
| Responses re-encoding to at least the 8-token cap | 429 | 341 | 0 | 0 | 0 |
| No standalone uppercase A–J anywhere | 174 | 28 | 0 | 0 | 0 |

Re-encoded lengths are a **truncation proxy**: original output token IDs and
finish reasons were not saved. Some capped outputs already contain a valid
leading answer, so these counts are not counts of lost correct answers. However,
the unfinished reasoning prefixes are direct evidence that short generation can
fail to reach an answer, particularly for vanilla 2B. A better regex cannot
recover an answer that was never generated.

MMStar also has a definite parser defect. Vanilla 2B sometimes outputs
`C: A forest`, but the last-standalone-letter parser selects A instead of the
explicit leading C. We found 31 leading-answer/parser disagreements for 2B,
versus one for 8B and none for the releases. Correcting only these explicit
leading conflicts changes 2B from 46.87% to **47.40%**, and 8B from 62.73% to
**62.80%**. This particular parser defect does not explain the whole gap.
Vanilla 2B has 299/1,500 MMStar responses re-encoding to at least its 64-token cap;
295 begin with reasoning prefixes. Vanilla 8B has 38 such responses, 36 with
reasoning prefixes. Further generation/prompt diagnosis is warranted.

MMMU's open subset has only 53 questions. Applying the pinned official-style
open parsing/matching helpers to the existing responses raises each vanilla
model by only 2/900 = **0.22 pp**, with no change for the three released models.
This is an isolated helper diagnostic using saved gold answers, not a certified
new full benchmark score. The [official MMMU scorer](https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py)
supports rule-based MC/open scoring; a paid judge is not required for that path.

## 3. Judges and metric definitions: distinguish these carefully

All seven **executed local scripts** are judge-free. For the selected standard
benchmark paths, HallusionBench uses a GPT correctness judge; MMBench uses
answer matching with an optional judge fallback plus circular aggregation.
ScienceQA, AI2D, MMMU standard MC/open, SEED-image and MMStar do not require a
judge. Judge availability cannot explain those five suites by itself.

Having a predetermined reference answer is normal for both judged and unjudged
benchmarks. The judge interprets whether a free-form prediction matches that
reference; it does not replace the benchmark's reference labels or reconstruct
a missing final answer.

### MMBench: aggregation already explains much of the discrepancy

The local metric averages 4,329 option-rotation rows. Circular evaluation groups
them into 1,164 underlying questions and requires all available rotations in a
group to be correct. This matches the grouping logic in the pinned harness;
the [official MMBench repository](https://github.com/open-compass/MMBench)
documents CircularEval.

Holding the saved predictions and their existing extracted choices fixed:

| Model | Paper % | Local row accuracy % | Circular saved-choice % | Circular minus paper pp |
| --- | ---: | ---: | ---: | ---: |
| Vanilla 2B | 78.40 | 83.81 | 77.75 | -0.65 |
| Vanilla 8B | 84.50 | 89.79 | 85.82 | +1.32 |
| Released BF16 | 86.40 | 94.66 | 91.15 | +4.75 |
| Released W4 | 84.60 | 92.63 | 88.66 | +4.06 |
| Released W8 | 85.50 | 92.49 | 88.40 | +2.90 |

This is a useful diagnostic, not the complete official submission pipeline:
vanilla extraction remains the original parser, and no judge was run. All three
released models produced only bare option letters on MMBench, so their saved
answers do not need semantic disambiguation. Their remaining 3–5 pp discrepancy
cannot plausibly be repaired just by interpreting those letters with a judge.

### HallusionBench: judge difference is real, but insufficient as a blanket explanation

The [official evaluation code](https://github.com/tianyi-lab/HallusionBench/blob/main/evaluation.py)
uses GPT-based correctness assessment. Our local comparison includes 1,129 rows,
whereas the chosen image-only harness task covers 951. Nevertheless, every raw
vanilla response here is exactly `Yes` or `No`, so there is no ambiguous prose
for a judge to interpret. Switching just to image-only rule scoring increases
vanilla scores from 62.44/72.90 to **64.25/73.82**, farther above the paper's
51.4/61.1. The scope change alone therefore does not resolve the mismatch.

A small official-judge cross-check can test scoring/label consistency, but we
should not assume that paying for a judge will remove the whole gap. Keep the
same task split, prompt, judge version and metric for every model.

## 4. Wrong model downloads: evidence weighs against a local mix-up

For all five references, this run's recorded repository, revision and full file
hash mapping agree with the earlier locally rehashed, Hub-verified audit. Model
paths also agree with the actual saved results. See the
[previous integrity audit](../../reports/evaluation_integrity_audit.md).
This check compared manifests; it did not re-download or independently rehash
all model weights again today.

The evidence supports evaluation of the intended published artifacts, including
Instruct variants of vanilla 2B/8B. It cannot establish which private paths or
checkpoint revisions produced the authors' table. An author-side mix-up remains
untestable without their records; the present evidence does not justify claiming
that one occurred.

## 5. Undocumented settings: plausible and still unresolved

Processor settings are a known confound: vanilla and released BF16 allow up to
16,777,216 pixels, W4 allows 200,704, and W8 allows 50,176. The narrow W8 image
budget is a reasonable hypothesis for some losses, but only a controlled
resolution comparison can test it. W4/W8 emit single letters on all 847 MMMU MC
questions, so their remaining MMMU deficit needs a different explanation from
vanilla clipping. Full-test versus image-only ScienceQA also changes scores;
for the teacher, image-only is 94.60%, so that switch does not close its already
positive gap to the paper's 85.0%.

The [GRACE paper](https://arxiv.org/html/2601.22709v5#S4) and
[public repository](https://github.com/ForeverBlue816/GRACE) still do not establish
one unambiguous Table 6 evaluation recipe. Exact reconstruction from the current
public materials is underdetermined; it is not proven permanently impossible.
The authors could resolve this with exact commands, evaluator revisions,
processor overrides, dataset versions/splits, checkpoint revisions and raw
predictions. No message has been sent to them during this analysis.

## Next decision

1. **Do a bounded diagnostic round.** First reuse saved predictions for circular
   MMBench scoring and parser audits. Then test longer generation on a fixed
   representative MMMU/MMStar subset, logging token IDs and finish reasons.
   Use the same settings for every reference, including releases. Separately
   test a shared image budget for W4/W8, with no simultaneous prompt changes.
   Repeat a subset identically to measure nondeterminism. Add official judging
   where the selected scoring protocol calls for it. Validate changes against
   answer content, not against closeness to the paper's target number.
2. **Do not drop tasks based on observed paper gaps.** This would select evidence
   according to the desired result and conceal known failure modes. If a task
   cannot be run validly, report it as unavailable with the methodological
   reason; label any remaining-task aggregate explicitly. A >10 pp discrepancy
   alone is not such a reason.
3. **Make controlled local evaluation the primary comparison once validated.**
   Evaluate the released references and all four LoRA models using the same
   explicit resolution policy, prompts, output caps, scorers and judge settings.
   Preserve all seven suites. Compare GRACE-LoRA against matched LoRA-QAT and
   against the released models re-evaluated locally. Keep paper numbers in a
   separate external-reference table, with unresolved gaps documented.

Stop trying to recover unpublished settings once known implementation faults
are fixed, representative output review passes and repeatability is established.
At that point, failure to enter ±3 pp on every published cell is a reproduction
limitation, not a reason to block internally valid experiments indefinitely.
The current native-processor replay is valuable reproduction evidence, but its
unequal per-model pixel budgets make it unsuitable as the sole matched-method
comparison. Keep it archived and create a separate validated local protocol.
