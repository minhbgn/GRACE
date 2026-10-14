# Experiment report: released_bf16_init_32k_b128_p100352_t2048_seed42

**Protocol recovery update (2026-09-25):** The [source audit and new evaluation workflows](../../evaluation/README_paper_protocol.md) recover all seven upstream script recipes and add a separate corrected standard harness protocol. Confirmed differences include checkpoint-native image budgets and MMStar's 64-token launcher limit; MMMU task loading and open-answer scoring bugs are fixed for the new harness protocol. CPU checks passed; no new GPU results or paper-tolerance claim are included here. Historical scores below are unchanged.

**GRACE-LoRA is feasible within the two-GPU, five-hour budget and is competitive with the released full-method checkpoints on the seven-suite mean, but the added distillation does not beat matched LoRA-QAT.** Starting from the released BF16 student, GRACE-LoRA scores **73.32% at W4 and 73.92% at W8**. These are **0.54 percentage points below the released W4 model and 1.65 points above the released W8 model**, evaluated locally under the same protocol. However, matched QAT scores **74.44% and 76.15%**, respectively, with lower runtime and memory. GRACE's large MMStar regressions prevent interpreting its near-parity with the releases as uniformly retained quality.

Each GRACE run completes in approximately **2.39 process hours / 4.77 GPU-hours**, with **26.07 GiB peak PyTorch reserved memory per GPU**. This demonstrates an affordable additional training stage using an already trained initialization. It does not measure the cost of reproducing the author's complete pipeline from vanilla weights. The most useful next step is to diagnose loss balance, teacher suitability and native-to-export changes before increasing training scale.

Report prepared **2026-09-23** from completed local artifacts. All four full runs and all 70 merged evaluations were validated; no new training or inference was performed for this report. The structure follows the [first report](../constrained_32k_seed42/experiment_report.md) and [second report](../constrained_32k_b128_p100352_t2048_seed42/experiment_report.md). Scores and deltas use unrounded results, with two decimals displayed. The mean assigns equal weight to seven suites and is not an official leaderboard aggregate.

## 1. Experimental setup and what changed

The main change from the previous experiment is the **student initialization**: every new run independently starts from `ForeverBlue/Qwen3-VL-2B-GRACE-BF16`, rather than vanilla Qwen3-VL-2B-Instruct. This released initialization is the **BF16 SFT baseline**, explicitly identified in the [authors' training README](../../README.md#1-bf16-sft-baseline); its `GRACE-BF16` name does not imply prior distillation. QAT uses answer cross-entropy (CE) with simulated quantization. GRACE adds confidence-weighted prediction distillation (GDKD), visual relationship alignment (RCKA), and the adaptive IB controller, using frozen vanilla Qwen3-VL-8B-Instruct as teacher. In this report, **GRACE-LoRA** and **LoRA-QAT** denote the local constrained methods; **released full GRACE** denotes the author's evaluation-only W4/W8 checkpoints.

| Setting | Completed constrained runs |
| --- | --- |
| Hardware / pause budgets | 2 × A100 40GB; 34 GiB peak reserved and five training-loop hours per invocation |
| Shared data | 32,768 validated single-image examples from ShareGPT4V mix665k; one pass |
| Seeds | Training 42; subset selection 2026 |
| Batch / accumulation / updates | 2 GPUs × micro-batch 1 × accumulation 64 = 128; 256 updates |
| Training inputs | 4,096–100,352 image pixels; strict 2,048-token total input limit |
| Quantization | Separate W4G128 and W8G128 runs; 196 eligible language linear modules |
| LoRA | Attention q/k/v/o; rank 32, alpha 64, dropout 0.05; 112 adapted modules |
| Trainable parameters | 12,845,056 LoRA + 25,174,016 merger + 2,752,512 scales = **40,771,584** |
| Frozen parameters | Base language weights, embeddings/head, vision tower; scales outside adapted layers |
| Learning rates / schedule | LoRA/scales 2e-4; merger 1e-5; cosine, 3% LR warmup, zero weight decay |
| GDKD | Temperature 2; component weights 1/4; no GDKD warmup |
| RCKA | Weight 3; penultimate layer; 25-update warmup |
| Controller | Target 3; step size .003; initial beta .5; bounds [.1, 1]; EMA .99 |
| Memory controls | BF16, gradient checkpointing, ZeRO-2, no training KV cache |

The inventories establish which parameters are eligible to learn, not that every tensor changed. The visual merger contributes **61.7%** of the trainable count; this is attention LoRA **plus merger and selected scales**, rather than adapter-only training. Both local and released low-bit configs record 196 quantized modules and skip the vision tower, merger, embeddings and output head. Quantization group size is 128; W4/W8 are weight precisions, not activation precisions.

The [subset comparison](subset_comparison.json) shows **identical example identities, order, retained conversations and token hashes** to the previous larger-input experiment. Both manifests have records SHA256 `3cab4012409207fb29f1461eb1d1c9f8e0c6135f2283dfd34cfee1585dbd2342`. The saved W4 training settings differ only in initialization and its added weight hash; the previously recorded training/data/loss/quantization source hashes and Torch/Transformers versions match, while the launcher changed. This makes the historical W4 comparison informative about initialization, though it remains a single-seed experiment with updated evaluation bookkeeping.

### Comparison with the full-training reference

| Resource or setting | Constrained GRACE-LoRA | Repository full-GRACE launcher reference |
| --- | --- | --- |
| GPU allocation | 2 × A100 40GB | 16 × A100 80GB |
| Nominal aggregate device capacity | 80 GB | 1,280 GB |
| Data exposure | 32,768 examples, one pass | Full mix665k, approximately 665,000 examples; one epoch default |
| Trainable scope | Attention adapters, merger, selected scales | Full language-model and merger training |
| Effective batch | 128 | 512 |
| Maximum image area | 100,352 pixels, at most 98 merged image tokens | 200,704 pixels, at most 196 merged image tokens |
| Declared training token limit | 2,048, enforced by subset preparation | 8,192; flattened collator semantics differ |
| Model / scale learning rate | 2e-4 adapters / 2e-4 scales | 1e-5 model / 2e-4 scales |
| RCKA warmup | 25 updates | 400 updates |

These are [repository launcher defaults](../../qwen-vl-finetune/scripts/finetune_qwen3vl_2b_grace.slurm), **not an authenticated recipe for the downloaded releases**. Relative to that allocation, we use eight times fewer devices, one-sixteenth of the nominal aggregate memory capacity and approximately 4.9% of the examples. These ratios do not measure memory utilization, GPU-hours or speedup. Changes in data, schedule and trainable scope prevent attributing the release gap solely to LoRA. The local reference launcher also defaults to vanilla initialization, whereas the [saved release card](provenance/released_bf16_model_card.md) identifies the released BF16 checkpoint as the W4/W8 initialization.

## 2. Evaluation protocol and complete results

All ten rows use the same question identities, gold labels, image-area budget of **4,096–100,352 pixels**, greedy decoding, one beam and repetition penalty 1. Each model uses its explicitly selected processor after compatibility checks. Generation is capped at eight tokens except MMMU open answers, which permit 32; MMMU multiple-choice remains at eight. The training limit of 2,048 tokens is **not** applied as evaluation truncation.

HallusionBench uses aAcc; MMBench uses English-dev per-row accuracy; ScienceQA includes its full test set; MMMU uses the existing local mixed-answer scorer. All failures remain in the denominator. Detailed results and categories are in the [validated comparison](evaluation/comparison.md), with machine-readable [scores](evaluation/comparison.csv) and [deltas](evaluation/deltas.csv).

| Model / method | HallB (1,129) | MMBench (4,329) | ScienceQA (4,241) | AI2D (3,088) | MMMU (900) | SEED-IMG (14,233) | MMStar (1,500) | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla 2B, BF16 | 59.88 | 83.37 | 82.95 | 74.81 | 40.00 | 72.46 | 44.67 | 65.45 |
| Previous local SFT, BF16 | 64.13 | 83.85 | 82.72 | 74.94 | 43.00 | 73.06 | 52.47 | 67.74 |
| Vanilla 8B teacher, BF16 | 70.33 | 89.10 | 92.36 | 82.09 | 51.33 | 76.13 | 59.13 | 74.35 |
| Released BF16 initialization | 66.34 | 92.95 | 85.03 | 80.31 | 70.00 | 72.29 | 65.73 | 76.09 |
| Released full GRACE W4 | 63.86 | 92.47 | 83.68 | 78.82 | 61.22 | 72.25 | 64.67 | 73.85 |
| LoRA-QAT W4 | 63.86 | 92.31 | 82.88 | 79.37 | 66.33 | 72.44 | 63.87 | 74.44 |
| **GRACE-LoRA W4** | **64.92** | **92.28** | **83.73** | **79.05** | **64.78** | **73.04** | **55.40** | **73.32** |
| Released full GRACE W8 | 60.85 | 91.55 | 82.95 | 78.11 | 56.89 | 71.99 | 63.60 | 72.28 |
| LoRA-QAT W8 | 64.92 | 93.32 | 85.00 | 80.76 | 70.67 | 72.63 | 65.73 | **76.15** |
| **GRACE-LoRA W8** | **66.43** | **92.47** | **85.71** | **79.44** | **67.44** | **72.77** | **53.20** | **73.92** |

**LoRA-QAT W8 has the highest mean**, just +0.06 pp above its BF16 initialization; that small difference does not establish improvement or equivalence statistically. It does show strong observed preservation of the starting quality. QAT W4 loses 1.66 pp to BF16 but exceeds released full W4 by 0.58 pp. QAT W8 exceeds released full W8 by 3.87 pp. The released BF16 model itself exceeds vanilla 2B by **10.65 pp** and the vanilla teacher by **1.74 pp**.

Both GRACE-LoRA variants beat vanilla 2B on all seven suites, by **+7.87 / +8.48 pp** on the mean. They exceed previous local SFT by **+5.58 / +6.19 pp**, but that SFT checkpoint started from vanilla and is not a matched continued-training control. These gains cannot be credited to the new distillation stage: its initialization was already stronger, and both final GRACE models score below it.

### Comparison with the paper's claimed results

**Added 2026-09-25.** The reference is the Qwen3-VL comparison in **Table 6 of
arXiv v5**, also reproduced in the authors' repository and model cards. The
LLaVA and Qwen2-VL result tables concern different backbones. Each cell below
shows **paper → our original custom-protocol evaluation** for the same named
reference model; these are not the later lmms-eval scores.
[Paper, Table 6](https://arxiv.org/html/2601.22709v5#A1.T6),
[repository results](https://github.com/ForeverBlue816/GRACE#-results).

| Reference model | HallB | MMBench | ScienceQA | AI2D | MMMU | SEED | MMStar | Mean | Δ mean (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla 2B, BF16 | 51.4 → 59.88 | 78.4 → 83.37 | 81.4 → 82.95 | 76.9 → 74.81 | 53.4 → 40.00 | 71.2 → 72.46 | 58.3 → 44.67 | 67.3 → 65.45 | -1.85 |
| Vanilla 8B teacher, BF16 | 61.1 → 70.33 | 84.5 → 89.10 | 85.0 → 92.36 | 85.7 → 82.09 | 69.6 → 51.33 | 77.5 → 76.13 | 70.9 → 59.13 | 76.3 → 74.35 | -1.95 |
| Released BF16 initialization | 66.9 → 66.34 | 86.4 → 92.95 | 86.2 → 85.03 | 81.3 → 80.31 | 72.1 → 70.00 | 76.7 → 72.29 | 67.3 → 65.73 | 76.7 → 76.09 | -0.61 |
| Released full GRACE W4 | 65.4 → 63.86 | 84.6 → 92.47 | 84.3 → 83.68 | 79.5 → 78.82 | 70.5 → 61.22 | 75.1 → 72.25 | 65.8 → 64.67 | 75.0 → 73.85 | -1.15 |
| Released full GRACE W8 | 66.1 → 60.85 | 85.5 → 91.55 | 85.3 → 82.95 | 80.4 → 78.11 | 71.3 → 56.89 | 75.9 → 71.99 | 66.5 → 63.60 | 75.9 → 72.28 | -3.62 |

The paper's one-decimal values and reported `Avg.` are retained as published.
Local scores and differences are calculated from the unrounded
[report data](reports/report_analysis.json); the
[machine-readable comparison](reports/custom_protocol_paper_comparison.json)
records both sets of values. Because the paper rounds its cells and average
separately, the average of its displayed cells can differ slightly from its
reported `Avg.`. The mean-gap column subtracts the reported `Avg.`, not a
reconstructed mean of rounded cells.

For the low-bit releases, the per-benchmark differences are **local minus paper**:

| Released checkpoint | HallB | MMBench | ScienceQA | AI2D | MMMU | SEED | MMStar | Δ mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Released full GRACE W4 | -1.54 | +7.87 | -0.62 | -0.68 | -9.28 | -2.85 | -1.13 | -1.15 |
| Released full GRACE W8 | -5.25 | +6.05 | -2.35 | -2.29 | -14.41 | -3.91 | -2.90 | -3.62 |

**The seven-suite headline gap is −1.15 pp for W4 and −3.62 pp for W8, with
substantial cancellation between benchmarks.** MMMU is the largest negative
difference for both releases: −9.28 and −14.41 pp. MMBench moves in the opposite
direction, by +7.87 and +6.05 pp. SEED and HallusionBench also contribute to W8's
shortfall. Thus, the W4 mean being relatively close does not establish faithful
reproduction at the task level.

The discrepancy is not unique to the low-bit releases. The vanilla 8B teacher
is −18.27 pp on MMMU and −11.77 pp on MMStar, but +9.23 pp on HallusionBench and
+7.36 pp on ScienceQA. Such large, opposing changes in an independently released
baseline are consistent with evaluation sensitivity; they cannot be explained
solely by our constrained training or the authors' low-bit optimization.

The released W8 checkpoint also remains below released W4 on every local suite:
**72.28 versus 73.85**, whereas the paper reports W8 ahead by 0.9 pp. That
reversed ordering is an observation about these artifacts under our protocol,
not evidence that eight-bit training intrinsically performs worse.

#### Why this is not a matched reproduction of the paper

Our internal comparison holds the local evaluator fixed across all ten models.
Matching benchmark names does not establish that the authors used identical
splits, prompts, image budgets or scoring. In particular:

- This report uses **4,096–100,352 pixels**, eight output tokens for most tasks,
  and custom Transformers evaluators. The exact author settings for the
  published checkpoint scores have not been recovered.
- **ScienceQA** here combines 2,017 image and 2,224 text-only questions. The later
  harness round uses image questions only; the paper's exact evaluation scope
  must be confirmed before calling either configuration an exact match.
- **HallusionBench** here uses rule-based scoring over the local image/non-image
  collection. **MMBench** uses per-row English-dev accuracy. These differ from
  the image-only judged HallusionBench and rolling/judge-assisted MMBench tasks
  selected for the pinned harness workflow.
- **MMMU** uses the local mixed-answer scorer, including simplified open-answer
  matching. **MMStar** uses question-level accuracy, whereas the later pinned
  harness reports an equal mean over L2-category accuracies. Answer extraction
  can also respond differently to models' output formats.

Consequently, a local method numerically exceeding a published headline is not
a matched-method win. For example, LoRA-QAT W8's 76.15 is above the paper's W8
75.9, but the valid controlled contrast in this report is against the **same
release evaluated locally**, at 72.28. Likewise, GRACE-LoRA W8 exceeds that
local release by 1.65 pp while remaining below the paper's headline. The paper's
Qwen3 table has no matched QAT-only row; our QAT ablation supplies that control
for this constrained experiment.

#### What the subsequent checks established

The later [integrity audit](reports/evaluation_integrity_audit.md) verified all
ten checkpoint files against their evaluation manifests and the downloaded
references against official Hugging Face commits. It found no swapped paths or
corrupted reference downloads. For the **four completed lmms-eval suites**, it
also verified source datasets, shared prompts/settings and independent
explicit-choice rescoring. That latter scoring audit concerns the harness
round; it is not an independent validation of every custom scorer in this
original seven-suite report.

The [lmms-eval comparison](lmms_evaluation.md) now gives released W4/W8 four-suite
means of **77.68 / 76.72**. Relative to the paper's same four named columns,
these are +1.50 / −0.31 pp, but MMMU and the two judge-based suites remain
unavailable in that round. These are **four-suite means** and cannot replace,
be compared directly with, or be combined with the seven-suite means above.
The authentic W4/W8 ordering remains reversed in that evaluation as well.

The evidence therefore supports competitive local accuracy under explicitly
recorded protocols. The unresolved paper discrepancy calls for a successful
standard MMMU evaluation, the authors' exact commands/raw predictions, and
controlled changes to resolution, prompts and scoring. It does not justify
assuming that the downloaded models are wrong or that the paper's claimed
numbers have been reproduced.

### Distillation benefit and gap to the full method

Every entry below is **GRACE-LoRA minus the named reference**, in percentage points; positive favors the local GRACE model. This sign convention makes quality gains easy to compare. The generated `deltas.csv` instead defines “gap to full method” as release minus local, so those particular signs are reversed there.

| Benchmark | W4 − QAT W4 | W4 − full W4 | W4 − BF16 init | W8 − QAT W8 | W8 − full W8 | W8 − BF16 init |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HallusionBench | +1.06 | +1.06 | −1.42 | +1.51 | +5.58 | +0.09 |
| MMBench | −0.02 | −0.18 | −0.67 | −0.85 | +0.92 | −0.49 |
| ScienceQA | +0.85 | +0.05 | −1.30 | +0.71 | +2.76 | +0.68 |
| AI2D | −0.32 | +0.23 | −1.26 | −1.33 | +1.33 | −0.87 |
| MMMU | −1.56 | +3.56 | −5.22 | −3.22 | +10.56 | −2.56 |
| SEED-Bench-IMG | +0.60 | +0.79 | +0.75 | +0.15 | +0.78 | +0.48 |
| MMStar | −8.47 | −9.27 | −10.33 | −12.53 | −10.40 | −12.53 |
| **Mean** | **−1.12** | **−0.54** | **−2.78** | **−2.23** | **+1.65** | **−2.17** |

**W4 is close to the released mean, but has a different capability profile.** GRACE-LoRA exceeds full W4 on five suites, particularly MMMU (+3.56 pp), while losing 9.27 pp on MMStar. Its mean is 99.27% of the release's mean, but this score ratio is not “99% of capabilities retained.” Relative to matched QAT, it wins HallusionBench, ScienceQA and SEED and loses the other four suites.

**W8 exceeds the released reference on six suites, but still fails the matched distillation test.** Its +1.65 pp mean advantage includes +10.56 pp on MMMU and +5.58 pp on HallusionBench, alongside −10.40 pp on MMStar. It loses to QAT on the same four suites as W4. Increasing precision raises QAT by **1.71 pp**, but GRACE by only **0.61 pp**; GRACE's MMStar score actually falls 2.20 pp from W4 to W8. Thus, low precision alone is an insufficient explanation for the added-objective deficit.

The released W8 checkpoint scores below released W4 on **all seven local suites**, the reverse of the mean ordering in the archived release card. Its local mean is 72.28, versus the card's published 75.9; full W4 is 73.85 locally versus 75.0 in the card. Those published values use a different evaluation setup and are not substituted into our comparison. This unexpected W8 ordering motivated the subsequent [integrity audit](reports/evaluation_integrity_audit.md), which verified checkpoint identity and found no download/path error; the intended author evaluation protocol remains unresolved, so “better than full GRACE” is not established as a general claim. The local result is valid for these pinned artifacts and this protocol; it does not prove that full W8 training is intrinsically worse. Correcting all nine reported released-W8 HallusionBench parse failures could raise its mean by at most **0.11 pp**, insufficient to explain the 1.65 pp local-GRACE advantage.

## 3. Task-level behavior and paired predictions

### MMStar is the main regression, across all six categories

| MMStar category, 250 examples each | QAT W4 | GRACE W4 | Δ (pp) | QAT W8 | GRACE W8 | Δ (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Coarse perception | 80.00 | 74.40 | −5.60 | 79.60 | 72.80 | −6.80 |
| Fine-grained perception | 58.80 | 52.00 | −6.80 | 59.60 | 49.60 | −10.00 |
| Instance reasoning | 78.40 | 70.00 | −8.40 | 78.80 | 64.40 | −14.40 |
| Logical reasoning | 60.40 | 54.00 | −6.40 | 63.60 | 50.80 | −12.80 |
| Math | 56.40 | 36.80 | −19.60 | 58.40 | 35.20 | −23.20 |
| Science & technology | 49.20 | 45.20 | −4.00 | 54.40 | 46.40 | −8.00 |

MMStar contributes **−1.21 / −1.79 pp** to the W4/W8 seven-suite mean differences from QAT. As a diagnostic only, excluding it changes the six-suite differences to **+0.10 / −0.51 pp**. This identifies the concentration of the loss; dropping an unfavorable suite after observing it would not be a valid replacement success criterion. All local QAT/GRACE MMStar files record zero inference errors and zero parse failures. The broad category losses, especially math and reasoning, deserve direct inspection of disagreements.

### HallusionBench and ScienceQA show real measured trade-offs

| Additional metric | QAT W4 | GRACE W4 | QAT W8 | GRACE W8 | BF16 init | Teacher |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HallusionBench qAcc, 455 groups | 29.89 | 31.87 | 31.21 | 33.63 | 33.19 | 38.90 |
| HallusionBench fAcc, 346 groups | 31.21 | 36.13 | 32.08 | 37.57 | 36.99 | 47.40 |
| ScienceQA image, 2,017 questions | 84.28 | 85.47 | 86.22 | 86.61 | 86.07 | 93.36 |
| ScienceQA text, 2,224 questions | 81.61 | 82.15 | 83.90 | 84.89 | 84.08 | 91.46 |

GRACE improves HallusionBench qAcc/fAcc over QAT by **+1.98/+4.91 pp at W4** and **+2.42/+5.49 pp at W8**. This extends the QAT-relative consistency benefit observed in both earlier experiments. W8 slightly exceeds BF16 on aAcc/qAcc/fAcc, but by only **1 row / 2 question groups / 2 figure groups**. These small differences should not be called robust improvements. The teacher remains substantially better on all three HallusionBench metrics.

ScienceQA improves over QAT in both image and text subsets at both precisions. This is consistent with useful guidance on a task where the teacher is stronger, although an ablation is required to attribute it to GDKD or RCKA. Conversely, **13 of W4's 14 net lost MMMU answers and 26 of W8's 29 are multiple-choice**. The local open-answer scorer therefore cannot explain most of the MMMU deficit to QAT.

### Which answers change?

Rows are aligned by suite-specific identities, not file order. A win means GRACE correct / QAT wrong; a loss means GRACE wrong / QAT correct.

| Suite | W4 wins / losses | W8 wins / losses |
| --- | ---: | ---: |
| HallusionBench | 62 / 50 | 58 / 41 |
| MMBench | 83 / 84 | 92 / 129 |
| ScienceQA | 158 / 122 | 166 / 136 |
| AI2D | 117 / 127 | 121 / 162 |
| MMMU | 48 / 62 | 44 / 73 |
| SEED-Bench-IMG | 627 / 541 | 689 / 668 |
| MMStar | 84 / 211 | 61 / 249 |

For example, the W4 MMBench difference is just one net answer, whereas W8 MMStar loses 188 net answers. These counts identify useful diagnostic samples and avoid treating all displayed gaps alike. They are not significance tests. Grouped HallusionBench questions, repeated MMBench variants and shared images require appropriate grouped uncertainty analysis; additional training seeds address a separate source of variation.

## 4. What this adds to the previous two experiments

| Experiment | Initialization | Image / training-token cap | QAT W4 mean | GRACE W4 mean | GRACE−QAT |
| --- | --- | --- | ---: | ---: | ---: |
| First constrained pilot | Vanilla 2B | 50,176 / 1,024 | 63.99 | 63.39 | −0.59 pp |
| Larger-input pilot | Vanilla 2B | 100,352 / 2,048 | 65.89 | 64.96 | −0.93 pp |
| **Current released-init experiment** | Released BF16 | 100,352 / 2,048 | **74.44** | **73.32** | **−1.12 pp** |

Relative to the second pilot, current W4 QAT gains **8.54 pp** and GRACE gains **8.35 pp**. The initialization change yields far larger absolute gains than the earlier input/batch change, but GRACE's deficit to QAT grows by another **0.19 pp**. The W8 extension also has a negative GRACE−QAT result. Thus, **starting from vanilla explained much of the low absolute performance, but correcting that mismatch did not establish a benefit from the current constrained distillation objective**.

The current vanilla and previous-SFT evaluations reproduce the previous experiment's saved result rows exactly on every suite. Together with the identical training manifest and core implementation hashes, this strengthens the historical comparison. It still does not convert the three configurations into repeated training seeds or establish that all cross-experiment differences have one cause.

This experiment also resolves an important earlier uncertainty: the releases are now evaluated under the same local protocol, rather than compared through published headline means. What remains unresolved is why the local added objectives lose to QAT, how much conversion to the final export costs each method, and how the released W8 result changes under its intended evaluation environment.

## 5. Training diagnostics and plausible explanations

### The controller saturates even earlier with the stronger initialization

| Logged GRACE diagnostic | Previous vanilla-init W4 | Current W4 | Current W8 |
| --- | ---: | ---: | ---: |
| First update at beta = 1 | 8 | 4 | 5 |
| Updates at beta = 1 | 249/256 (97.27%) | 253/256 (98.83%) | 252/256 (98.44%) |
| Last 25 updates: mean CE | 1.221 | 1.091 | 1.081 |
| Last 25 updates: mean GDKD | 8.119 | 9.143 | 8.681 |
| Last 25 updates: mean RCKA | 0.438 | 0.466 | 0.463 |
| Last 25 updates: weighted RCKA, coefficient 3 | 1.313 | 1.399 | 1.390 |

These are **last-micro-batch snapshots at optimizer boundaries**, not losses averaged over each 128-example accumulation cycle. They should not be interpreted as precise dataset-level loss means. The current first-25-update GDKD averages are 26.66/24.95, and GDKD has full weight from the start. By the tail, its numerical contribution is still about eight times CE, with beta fixed at its cap.

The implemented update is `beta ← clip(beta + .003 × (EMA(GDKD) − 3), .1, 1)`. High disagreement increases distillation pressure. Lowering the target would therefore not weaken distillation. The fast saturation and weak quality outcome make a smaller fixed weight or delayed GDKD a concrete diagnostic experiment. Loss magnitude alone does not establish gradient dominance, gradient conflict or causation; those need measurements on fixed batches.

### Teacher suitability changes after adopting released BF16

The teacher is better than the BF16 initialization on HallusionBench, ScienceQA, AI2D and SEED, but worse on **MMBench (89.10 vs 92.95), MMMU (51.33 vs 70.00), and MMStar (59.13 vs 65.73)**. The same teacher is therefore not uniformly a quality ceiling for this student. Continued matching may erode capabilities already present in the initialization, especially when confidence is not calibrated to correctness. This is a hypothesis consistent with the affected tasks, not proof: the training distribution differs from the benchmarks, GDKD uses teacher-forced token distributions, and RCKA also acts on representations.

In particular, both final GRACE MMStar scores fall below **both** initialization and teacher. Simply moving accuracy toward the teacher's score cannot explain the whole loss. Objective interaction, optimization capacity and export effects remain plausible contributors. AI2D also loses to QAT despite the teacher's higher score, so choosing tasks solely by teacher benchmark accuracy would be insufficient.

### Export behavior and restricted adaptation remain unresolved

The [quantized LoRA implementation](../../qwen-vl-finetune/qwenvl/train/qat_modules.py) trains the function `Q(W)x + LoRA(x)` but merges and requantizes for export as `Q(W + ΔW)x`. Rounding makes these functions unequal. Native versus exported accuracy has not been measured here, so final-artifact losses cannot yet be assigned entirely to training. The W8 regression makes a W4-only rounding explanation inadequate, but it does not eliminate conversion effects.

Attention-only LoRA and selected scale updates may limit joint compensation for quantization and teacher alignment, while the 32k one-pass budget limits optimization. However, QAT W8 preserves the BF16 mean using the same capacity and data. Resource limits alone therefore do not explain why adding GRACE hurts. A no-training quantization control and BF16 continued LoRA-SFT control are absent, leaving quantization, ordinary continued adaptation and export effects partially entangled.

The current experiment folder contains **no saved stress-profile run summaries or learning-diagnostic summaries**. Earlier experiments' short-profile findings about sampled scale/merger updates cannot be promoted to evidence that these new runs updated those tensors correctly. The full-run inventories, losses and completion records establish feasibility and execution; a focused optimizer-state/full-tensor audit is still needed to establish effective updates of each parameter category.

## 6. Runtime, memory and the practical constraint

| Method | Updates | Training-loop hours | Process hours | GPU-hours, 2 GPUs | Peak allocated GiB/GPU | Peak reserved GiB/GPU | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QAT W4 | 256 | 1.876 | 1.924 | 3.848 | 7.833 | 8.475 | 74.44 |
| GRACE W4 | 256 | 2.340 | 2.390 | 4.780 | 24.696 | 26.068 | 73.32 |
| QAT W8 | 256 | 1.889 | 1.941 | 3.883 | 7.833 | 8.475 | 76.15 |
| GRACE W8 | 256 | 2.332 | 2.382 | 4.765 | 24.696 | 26.068 | 73.92 |

Each run has one recorded successful invocation, no resume and no budget stop. Total cost for the four additional runs is **8.64 process hours / 17.28 GPU-hours**. These totals exclude data preparation, evaluation and the author's prior training. Process timing wraps `torchrun`, including loading, checkpointing and export; time outside the recorded Trainer intervals is approximately **2.9–3.1 minutes per run**. Loading and export are not separately instrumented.

Adding GRACE costs **28.0 minutes (+24.2%) at W4** and **26.5 minutes (+22.7%) at W8** relative to matched QAT. Peak reserved memory increases by **17.59 GiB/GPU**, to **3.08×** QAT. The replicated BF16 teacher is a major memory burden even though its weights are frozen and the student uses LoRA. Measured GRACE peaks are 7.93 GiB below the 34 GiB pause threshold; this is training-loop allocator headroom, not guaranteed free device memory during loading/export. Memory numbers are rank maxima at logged training boundaries, not sums across the two GPUs or complete device occupancy.

**For the chosen mean-quality, time and training-memory objectives, QAT is the better measured choice at each precision.** GRACE remains a task-specific option if HallusionBench consistency or ScienceQA gains justify losses elsewhere. W8 QAT offers much better observed quality than W4 QAT for almost identical local training cost. This does not settle deployment choice: all these research exports store floating-point tensors, and neither packed W4/W8 inference memory nor speed was benchmarked. The similar training costs are not evidence that real INT8 and INT4 deployment cost the same.

### What can be said about cost versus full GRACE?

The released checkpoints contain Trainer time records, which should be disclosed separately rather than ignored:

| Released artifact | Saved global step / configured max | Saved Trainer runtime |
| --- | ---: | ---: |
| BF16 initialization | 3,289 / 3,290 | 7,654.34 s = 2.13 h |
| Full W4 reference | 2,890 / 2,892 | 12,678.35 s = 3.52 h |
| Full W8 reference | 2,890 / 2,892 | 2,281.61 s = 0.63 h |

Sources are the released [BF16](../../models/Qwen3-VL-2B-GRACE-BF16/trainer_state.json), [W4](../../models/Qwen3-VL-2B-GRACE-W4G128/trainer_state.json), and [W8](../../models/Qwen3-VL-2B-GRACE-W8G128/trainer_state.json) Trainer states. Their provenance files hash these assets. **These time records do not establish device count/type, resumed segments, or complete runtime history.** The separate README explicitly identifies the BF16 checkpoint's source as SFT; the remaining timing uncertainty does not make that objective ambiguous. Their update counts also caution against treating the current full-GRACE launcher defaults as their exact executed recipe. In particular, the W8 saved time is shorter than our local stage, so these files do not support a blanket claim that LoRA is faster in wall-clock time.

The defensible resource conclusion is that our additional stage **actually completed on two 40GB GPUs in under 2.4 process hours**, compared with a repository reference allocation of sixteen 80GB GPUs. A measured full-method GPU-hour reduction or end-to-end speedup remains unavailable. The cost of producing the released BF16 initialization is inherited and cannot be erased from a from-scratch reproduction claim.

## 7. Evidence, validation and limitations

The report audit revalidated **70 merged result files**, including expected counts, unique question identities, matching gold labels, correct totals/accuracies, model/processor paths, and the saved evaluation policy. Regenerating the complete comparison in memory matched `evaluation/comparison.md` exactly. There are **zero recorded inference errors across all ten models**. Reported parse failures total 0 for the four local low-bit models, BF16, released W4 and SFT; 9 for released W8, 4 for vanilla 2B and 2 for the teacher. MMMU does not supply a separate parse-failure count. This validates consistency of saved results, not every scoring decision or historical inference operation.

All four full runs have contiguous updates 1–256, matching Trainer/run summaries, matching pinned launch recipes, `EXPORT_COMPLETE`, merged-adapter metadata, and 40,771,584 trainable parameters. Current training-source hashes match both saved bit-width recipes. Full-data completion is supported directly; no separate current stress-profile artifacts were found. The report did not rerun GPU preflight, inference, a full checkpoint-tensor audit, statistical significance tests or a data-contamination audit. Single-seed and category-level observations remain exploratory.

Pinned release revisions are BF16 `82f9a267340f65a3619be0e19daf6582d3998a5f`, W4 `57dcaeed3fd7739a0566bfc4b8651f451068825c`, and W8 `32c351aed1388ae602d59f7c26101dd16aca8c80`. BF16 has the explicitly pinned weight hash in the README. W4/W8 provenance records local download metadata and explicit repository mapping; that is not an independent audit of the author's training history. External published figures above come from the experiment's archived cards rather than a newly fetched leaderboard.

Primary evidence:

- [Experiment setup and commands](README.md), [model matrix](experiment.json), [W4 recipe](w4g128/comparison_recipe.json), [W8 recipe](w8g128/comparison_recipe.json), and [subset identity comparison](subset_comparison.json).
- [Evaluation comparison](evaluation/comparison.md), [protocol](evaluation/protocol.json), and [derived report statistics with result-file hashes](reports/report_analysis.json).
- [Runtime summary](reports/runtime_memory.md) and [unrounded runtime data](reports/runtime_memory.json).
- Full-run summaries: [QAT W4](w4g128/runs/qat/run_summary.json), [GRACE W4](w4g128/runs/grace/run_summary.json), [QAT W8](w8g128/runs/qat/run_summary.json), [GRACE W8](w8g128/runs/grace/run_summary.json).
- GRACE step logs: [W4](w4g128/runs/grace/step_metrics.jsonl), [W8](w8g128/runs/grace/step_metrics.jsonl).
- Release provenance: [BF16](provenance/released_bf16.json), [W4](provenance/released_grace_w4.json), [W8](provenance/released_grace_w8.json); archived cards: [BF16](provenance/released_bf16_model_card.md), [W4](provenance/released_grace_w4_model_card.md), [W8](provenance/released_grace_w8_model_card.md).
- [Evaluation validator](../../evaluation/summarize_constrained_evals.py), [GRACE objective integration](../../qwen-vl-finetune/qwenvl/train/train_qwen_grace.py), [IB controller](../../qwen-vl-finetune/qwenvl/train/grace_modules.py), and [quantization/LoRA/export implementation](../../qwen-vl-finetune/qwenvl/train/qat_modules.py).

To revalidate the complete evaluation comparison on CPU without rewriting results, run from the workspace root:

```bash
python3 reproduction/GRACE/evaluation/summarize_constrained_evals.py \
  --matrix \
  --results-dir reproduction/GRACE/experiments/released_bf16_init_32k_b128_p100352_t2048_seed42/evaluation
```
