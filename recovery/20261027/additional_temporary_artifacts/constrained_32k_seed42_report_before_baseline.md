# Experiment report: constrained_32k_seed42

This experiment tested whether GRACE's teacher-guided training improves a small, 4-bit vision-language model when training is limited to two A100 40GB GPUs. All three methods completed the planned training and were evaluated on the same questions. Under this setup, **GRACE did not improve average accuracy over QAT or ordinary supervised fine-tuning**, although it improved several HallusionBench metrics.

## 1. What we trained

All students started from **Qwen3-VL-2B-Instruct**. Only GRACE used the frozen **Qwen3-VL-8B-Instruct** teacher.

| Method | Plain-language description | Training objective | Weight precision |
| --- | --- | --- | --- |
| SFT | Learn directly from the dataset's correct answers | Answer prediction loss (CE) | BF16 |
| QAT | Learn from the answers while simulating 4-bit weight rounding | CE | W4G128 |
| GRACE | QAT plus guidance from the larger teacher's predictions and visual representations | CE + confidence-gated distillation + visual relationship alignment | W4G128 |

W4G128 means 4-bit weights with one quantization scale per group of 128 weights. The QAT and GRACE exports store those rounded weights in BF16 tensors; these runs measure accuracy, not packed INT4 inference speed or memory savings.

There is **no BF16 GRACE run** in this experiment. Here, “GRACE−QAT” and “GRACE−SFT” mean differences between the three rows above, rather than separate methods named GRACE-QAT and GRACE-SFT.

## 2. Configuration and differences from the original setup

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

The subset is approximately **4.9% of mix665k**. Sampling preserves image-source proportions, rather than verified task proportions; approximately 56% of retained examples use COCO train images. Text-only and video records are excluded. Long conversations lose complete trailing question/answer pairs until they fit. The full-training launchers use a flattened collator, so their declared sequence lengths should not be treated as equivalent strict input limits.

Our image cap is **75% lower than the GRACE launcher's cap**. With Qwen3-VL's 16-pixel patches and 2 × 2 merging, it permits at most **49 visual tokens**, compared with 196 under the larger cap. The saved evaluation processor uses the same reduced budget. This reduction affects both what the student can see and what the teacher can teach.

For context, the paper reports 1.3M image-text pairs, three epochs, eight GH200 GPUs, an effective batch of 128 and a 2,048-token limit. Its Table 6 specifies RCKA weight 1 and controller settings different from ours: target .35, step size .0015, initial weight 1, bounds [.1, 5]. Thus, this is a constrained adaptation rather than an exact paper reproduction. [Original paper, §4 and Appendix D.1](https://arxiv.org/html/2601.22709v1#S4)

## 3. Results and relative performance

Scores below are percentages. A difference of one percentage point (pp) means, for example, moving from 70% to 71%. “Relative change” divides the score difference by the comparison method's score.

| Benchmark | SFT | QAT | GRACE | GRACE−QAT (pp) | GRACE−SFT (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| HallusionBench aAcc | 59.43 | 55.62 | 57.93 | +2.30 | −1.51 |
| MMBench English dev, per-row | 82.95 | 81.54 | 80.78 | −0.76 | −2.17 |
| ScienceQA, full test | 81.35 | 80.71 | 79.86 | −0.85 | −1.49 |
| AI2D | 72.57 | 69.95 | 69.88 | −0.06 | −2.69 |
| MMMU validation | 44.33 | 41.89 | 39.11 | −2.78 | −5.22 |
| SEED-Bench-IMG | 70.95 | 70.74 | 70.86 | +0.12 | −0.08 |
| MMStar | 49.13 | 47.47 | 45.33 | −2.13 | −3.80 |
| **Unweighted mean** | **65.82** | **63.99** | **63.39** | **−0.59** | **−2.42** |

Deltas use unrounded scores. Each benchmark has equal weight in this descriptive mean.

Across seven benchmarks, GRACE was **0.93% relatively lower than QAT** and **3.68% relatively lower than SFT**. It exceeded QAT on two benchmarks and fell below SFT on all seven primary metrics. Its largest deficits were on MMMU and MMStar. However, HallusionBench's stricter grouped metrics improved: qAcc was 24.62% versus QAT's 21.98% and SFT's 23.96%; fAcc was 34.10% versus 30.35% and 30.92%. These require all answers within a question or figure group to be correct.

### How this differs from the original reported gains

| Experiment/reference | GRACE score | QAT comparison | BF16 comparison | GRACE−QAT | GRACE−BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Our seven-suite mean** | 63.39 | 63.99 | 65.82, matched SFT | **−0.59 pp / −0.93% relative** | **−2.42 pp / −3.68% relative** |
| Paper Table 2: LLaVA-1.5-7B, INT4 | 67.2 | 65.9 | 66.5, baseline | +1.3 pp / +1.97% relative | +0.7 pp / +1.05% relative |
| Paper Table 4: Qwen2-VL-2B, INT4, three tasks | 77.2 | 71.3 | 72.7, baseline | +5.9 pp / +8.27% relative | +4.5 pp / +6.19% relative |

Paper values come from [Tables 2 and 4](https://arxiv.org/html/2601.22709v1#S4). The original BF16 rows are labeled **baseline**, not a separately documented SFT run matched to ours. Different backbones, training and evaluation prevent a controlled comparison of effect sizes. The meaningful observation is that the paper reports positive gains while our matched pilot produces negative average gaps.

The repository's Qwen3-VL release reports **75.0 for W4G128 GRACE** and **76.7 for BF16 GRACE**. Our W4 GRACE mean of 63.39 is numerically **11.61 pp, or 15.47%, below** its W4 result. Our SFT mean is 10.88 pp below its BF16 GRACE result, but these use different objectives: ordinary SFT does not include teacher guidance. The release table supplies neither a standalone QAT score nor a matched SFT score, so it cannot establish original Qwen3-VL GRACE−QAT or GRACE−SFT gains. [Repository results](https://github.com/ForeverBlue816/GRACE#-results)

The model descriptions also say the released W4/W8 students initialize from a **BF16 GRACE checkpoint**. Our runs initialize directly from the original 2B model, omitting that prior teacher-guided training stage. This distinction applies to the released Qwen3-VL pipeline, not automatically to the paper's experiments. [Repository model descriptions](https://github.com/ForeverBlue816/GRACE#-model-zoo)

These cross-report numbers are indicative, not equivalent leaderboard measurements. Our local evaluation uses rule-based scoring, per-row MMBench accuracy, full-test ScienceQA and a simplified MMMU open-answer scorer. Prompting, answer parsing and benchmark scope can change scores. There were no recorded inference errors; a small number of parse failures remained in the denominator. One training seed also leaves repeatability unresolved.

## 4. Possible reasons for the outcome

The following are hypotheses, with the supporting observations stated explicitly.

1. **Teacher guidance may have outweighed learning the correct answers.** The logged distillation weight reached its maximum of 1 at update 7 and stayed there for 506/512 updates. Over the last 50 logged updates, mean GDKD was approximately 7.59 versus CE of 1.30; weighted visual alignment contributed approximately 0.93. GDKD remained above the controller target of 3. This suggests the controller was mostly saturated and the objective was dominated numerically by teacher matching. Loss size alone does not establish gradient dominance or causality, but this is the strongest specific diagnostic from the logs.

2. **Much less data and training may have limited knowledge transfer.** We used 32,768 examples and one pass, and skipped the released Qwen3-VL pipeline's BF16 GRACE initialization stage. A source-proportional subset does not guarantee enough science, diagrams or difficult reasoning examples. A larger teacher's capabilities may require more varied supervision and a longer schedule to transfer.

3. **Low resolution removes useful visual detail.** At most 49 image tokens must represent the whole image. Small text, diagram labels and fine spatial relationships can be lost. The teacher also receives the reduced input, so its supervision may be less useful. This could help explain lower absolute scores, but it cannot by itself explain GRACE losing to QAT: both receive the same image budget.

4. **LoRA gives the student fewer ways to adapt.** Only attention adapters and the visual projector learn, alongside selected quantization scales. QAT/GRACE have about 40.77M trainable parameters, including 12.85M adapter parameters and 2.75M scales. The frozen base and feed-forward weights may restrict adaptation to both the teacher and 4-bit rounding. This differs substantially from updating the full language model.

5. **The exported model may behave differently from the training model.** Training applies quantization to the base weights and then adds the LoRA update: `Q(W) + ΔW`. Export merges first and then quantizes: `Q(W + ΔW)`. Rounding makes these expressions unequal. Both QAT and GRACE use this procedure, but their learned updates can incur different conversion losses. Native-versus-export accuracy has not been measured here.

6. **Scoring differences and one-seed variation may contribute.** Shared questions and evaluation code make the local comparison useful, but response-format differences can still interact with the local parsers. Cross-report score gaps additionally include protocol and backbone differences.

## 5. Suggested next experiments

First, inspect raw benchmark answers and compare native versus exported models on a small fixed validation set. This checks whether parsing or export changes explain part of the gap. Next, test weaker distillation or a longer distillation warmup, with CE, GDKD, RCKA and controller trajectories recorded separately. Then increase the visual budget and training data one factor at a time, retaining matched QAT and SFT controls. Repeat the selected configuration with additional seeds. Evaluate an untuned base model and the released GRACE checkpoint under the same local protocol to separate training effects from evaluation differences.

| Method | Recorded training-loop time | Recorded peak PyTorch reserved memory |
| --- | ---: | ---: |
| SFT | 1.45 h | 6.59 GiB |
| QAT | 1.90 h | 6.64 GiB |
| GRACE | 2.37 h | 24.13 GiB |

These are recorded run-summary measurements, not total end-to-end time or total GPU occupancy. All methods completed 512 updates and produced final exports.

## Supervisor-ready description

> We ran a resource-constrained GRACE experiment using a Qwen3-VL-2B student and an 8B teacher on two A100 40GB GPUs. To fit the available resources, we trained attention LoRA adapters and the visual projector, used 32,768 examples, limited inputs to 1,024 tokens and reduced images to at most 50,176 pixels. SFT, QAT and GRACE shared the same data and effective batch of 64, completing 512 updates each. GRACE averaged 63.39%, compared with 63.99% for QAT and 65.82% for SFT: gaps of −0.59 and −2.42 percentage points. This differs from the positive gains reported in the original paper, although the setups are not directly comparable. GRACE improved hallucination consistency, but its distillation controller stayed at maximum weight for almost the entire run. We suspect the loss balance, limited training capacity, smaller dataset and low resolution contributed. We propose checking export/scoring effects and testing weaker teacher guidance before scaling up.

## Local evidence

- [Pinned experiment configuration](comparison_recipe.json)
- [Detailed evaluation comparison](evaluation/comparison.md), based on the 21 merged result files
- Training summaries and per-update logs under `runs/{sft,qat,grace}/`
- [Constrained-training documentation](../../qwen-vl-finetune/scripts/README_constrained.md)
- [Full-training GRACE launcher](../../qwen-vl-finetune/scripts/finetune_qwen3vl_2b_grace.slurm)
- [Evaluation protocol notes](../../evaluation/README_lmms_constrained.md)
