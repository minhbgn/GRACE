# Experiment report: constrained_32k_seed42

**Provenance correction (2026-10-05):** The authors' [BF16 SFT baseline](../../README.md#1-bf16-sft-baseline) is the source of the released `GRACE-BF16` checkpoint. References below to its initialization now identify SFT rather than prior distillation; historical results are unchanged.

This experiment tested whether GRACE's teacher-guided training improves a small, 4-bit vision-language model when training is limited to two A100 40GB GPUs. All three trained methods and the vanilla Qwen3-VL-2B-Instruct baseline were evaluated on the same questions. Under this setup, **GRACE did not improve average accuracy over vanilla, QAT or ordinary supervised fine-tuning**: their seven-suite means were 63.39%, 63.72%, 63.99% and 65.82%, respectively. GRACE nevertheless improved three primary benchmark scores over vanilla and several HallusionBench metrics.

## 1. What we trained

All trained students started from **Qwen3-VL-2B-Instruct**. Only GRACE used the frozen **Qwen3-VL-8B-Instruct** teacher. **Vanilla** is the original instruction-tuned 2B checkpoint, with no additional training in this experiment; it is distinct from our SFT run.

| Method | Plain-language description | Training objective | Weight precision |
| --- | --- | --- | --- |
| Vanilla | Evaluate the original Qwen3-VL-2B-Instruct checkpoint | No additional training | BF16 |
| SFT | Learn directly from the dataset's correct answers | Answer prediction loss (CE) | BF16 |
| QAT | Learn from the answers while simulating 4-bit weight rounding | CE | W4G128 |
| GRACE | QAT plus guidance from the larger teacher's predictions and visual representations | CE + confidence-gated distillation + visual relationship alignment | W4G128 |

W4G128 means 4-bit weights with one quantization scale per group of 128 weights. The QAT and GRACE exports store those rounded weights in BF16 tensors; these runs measure accuracy, not packed INT4 inference speed or memory savings.

There is **no BF16 GRACE run** in this experiment. Here, “GRACE−QAT” and “GRACE−SFT” mean differences between the trained methods above, rather than separate methods named GRACE-QAT and GRACE-SFT.

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

Our image cap is **75% lower than the GRACE launcher's cap**. With Qwen3-VL's 16-pixel patches and 2 × 2 merging, it permits at most **49 visual tokens**, compared with 196 under the larger cap. The saved evaluation processor uses the same reduced budget. The completed vanilla evaluation loads the original model weights but reuses the SFT processor, including its tokenizer and chat template; all seven baseline result files record a minimum of 4,096 and maximum of 50,176 pixels. This makes vanilla a control under the same local visual budget, rather than a measurement at the upstream processor's original resolution. The reduction affects both what the student can see and what the teacher can teach.

For context, the paper reports 1.3M image-text pairs, three epochs, eight GH200 GPUs, an effective batch of 128 and a 2,048-token limit. Its Table 6 specifies RCKA weight 1 and controller settings different from ours: target .35, step size .0015, initial weight 1, bounds [.1, 5]. Thus, this is a constrained adaptation rather than an exact paper reproduction. [Original paper, §4 and Appendix D.1](https://arxiv.org/html/2601.22709v1#S4)

## 3. Results and relative performance

Scores below are percentages. A difference of one percentage point (pp) means, for example, moving from 70% to 71%. “Relative change” divides the score difference by the comparison method's score.

| Benchmark | Vanilla | SFT | QAT | GRACE | GRACE−Vanilla (pp) | GRACE−QAT (pp) | GRACE−SFT (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HallusionBench aAcc | 57.31 | 59.43 | 55.62 | 57.93 | +0.62 | +2.30 | −1.51 |
| MMBench English dev, per-row | 82.26 | 82.95 | 81.54 | 80.78 | −1.48 | −0.76 | −2.17 |
| ScienceQA, full test | 80.90 | 81.35 | 80.71 | 79.86 | −1.04 | −0.85 | −1.49 |
| AI2D | 72.18 | 72.57 | 69.95 | 69.88 | −2.30 | −0.06 | −2.69 |
| MMMU validation | 40.67 | 44.33 | 41.89 | 39.11 | −1.56 | −2.78 | −5.22 |
| SEED-Bench-IMG | 70.22 | 70.95 | 70.74 | 70.86 | +0.65 | +0.12 | −0.08 |
| MMStar | 42.53 | 49.13 | 47.47 | 45.33 | +2.80 | −2.13 | −3.80 |
| **Unweighted mean** | **63.72** | **65.82** | **63.99** | **63.39** | **−0.33** | **−0.59** | **−2.42** |

Deltas use unrounded scores. Each benchmark has equal weight in this descriptive mean.

Against vanilla, SFT improved the mean by **2.09 pp (+3.28% relative)** and improved all seven primary scores. QAT improved the mean by **0.27 pp (+0.42% relative)**, with gains on MMMU, SEED-Bench-IMG and MMStar. GRACE was **0.33 pp lower (−0.52% relative)**, with gains on HallusionBench (+0.62 pp), SEED-Bench-IMG (+0.65 pp) and MMStar (+2.80 pp), offset by losses on MMBench, ScienceQA, AI2D and MMMU. Its largest loss against vanilla was AI2D (−2.30 pp). The ordering is **SFT > QAT > Vanilla > GRACE** on this descriptive mean; the small QAT/vanilla and GRACE/vanilla gaps have not been tested for statistical significance.

Across seven benchmarks, GRACE was **0.93% relatively lower than QAT** and **3.68% relatively lower than SFT**. It exceeded QAT on two benchmarks and fell below SFT on all seven primary metrics. Its largest deficits against the trained controls were on MMMU and MMStar. In particular, GRACE's MMStar gain over vanilla is smaller than QAT's +4.93 pp and SFT's +6.60 pp, so it does not demonstrate a benefit from teacher guidance over the matched training controls.

HallusionBench's stricter grouped metrics improved over all three controls: GRACE qAcc was 24.62%, versus vanilla's 22.64%, QAT's 21.98% and SFT's 23.96%; fAcc was 34.10%, versus vanilla/SFT's 30.92% and QAT's 30.35%. These require all answers within a question or figure group to be correct. The ScienceQA loss against vanilla occurs in both its image subset (−1.64 pp) and text subset (−0.49 pp), so that loss is not confined to image-containing questions.

| Additional metric | Vanilla | SFT | QAT | GRACE | GRACE−Vanilla (pp) | GRACE−QAT (pp) | GRACE−SFT (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HallusionBench qAcc | 22.64 | 23.96 | 21.98 | 24.62 | +1.98 | +2.64 | +0.66 |
| HallusionBench fAcc | 30.92 | 30.92 | 30.35 | 34.10 | +3.18 | +3.76 | +3.18 |
| ScienceQA test (image) | 81.31 | 81.56 | 80.22 | 79.67 | −1.64 | −0.55 | −1.88 |
| ScienceQA test (text) | 80.53 | 81.16 | 81.16 | 80.04 | −0.49 | −1.12 | −1.12 |

HallusionBench qAcc/fAcc use 455/346 groups; ScienceQA image/text use 2,017/2,224 questions. Denominators match across all four methods.

### How this differs from the original reported gains

| Experiment/reference | GRACE score | QAT comparison | BF16 comparison | GRACE−QAT | GRACE−BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Our seven-suite mean vs. vanilla** | 63.39 | 63.99 | 63.72, vanilla baseline | **−0.59 pp / −0.93% relative** | **−0.33 pp / −0.52% relative** |
| Our seven-suite mean vs. SFT | 63.39 | 63.99 | 65.82, matched SFT | −0.59 pp / −0.93% relative | −2.42 pp / −3.68% relative |
| Paper Table 2: LLaVA-1.5-7B, INT4 | 67.2 | 65.9 | 66.5, baseline | +1.3 pp / +1.97% relative | +0.7 pp / +1.05% relative |
| Paper Table 4: Qwen2-VL-2B, INT4, three tasks | 77.2 | 71.3 | 72.7, baseline | +5.9 pp / +8.27% relative | +4.5 pp / +6.19% relative |
| Repository release: Qwen3-VL-2B, W4G128 | 75.0 | Not reported | 67.3, vanilla baseline | Not reported | +7.7 pp / +11.44% relative |

Paper values come from [Tables 2 and 4](https://arxiv.org/html/2601.22709v1#S4). The original BF16 rows are labeled **baseline**, not a separately documented SFT run matched to ours. Different backbones, training and evaluation prevent a controlled comparison of effect sizes. The completed vanilla control now lets us distinguish improvement over the starting checkpoint from differences among trained methods: our GRACE mean is below both vanilla and matched SFT, whereas the paper reports gains over its BF16 baselines.

The repository's Qwen3-VL release explicitly uses **vanilla Qwen3-VL-2B-Instruct** as its baseline and reports a mean of **67.3**, compared with **75.0 for W4G128 GRACE** and **76.7 for BF16 GRACE**. Its reported W4 gain over vanilla is +7.7 pp; ours is −0.33 pp. Release deltas use the published rounded means.

Our W4 GRACE mean of 63.39 is numerically **11.61 pp, or 15.47%, below** its W4 result. Our SFT mean is 10.88 pp below the released BF16 SFT result. Both are SFT baselines, but their data, trainable scope, training budgets and evaluation differ. The release table supplies a BF16 SFT reference but no standalone QAT row or SFT control matched to our constrained recipe, so it cannot establish the corresponding controlled Qwen3-VL GRACE−QAT or GRACE−SFT gains. [Repository results](https://github.com/ForeverBlue816/GRACE#-results)

The released W4/W8 students initialize from the **released BF16 SFT checkpoint**, named `GRACE-BF16`. The authors' [training README](../../README.md#1-bf16-sft-baseline) explicitly identifies the BF16 SFT launcher as its source. Our pilot runs initialize directly from the original 2B model, omitting that preceding SFT stage. This distinction applies to the released Qwen3-VL pipeline, not automatically to every experiment in the paper.

Our local vanilla mean of 63.72 is itself **3.58 pp below** the release baseline of 67.3. Because vanilla has no additional experiment training, that difference shows that the absolute release/local gap is not solely a property of our GRACE training; evaluation protocol and preprocessing differences must be considered. A vanilla evaluation at the upstream visual budget would help assess resolution effects, but the current data do not isolate their contribution.

These cross-report numbers are indicative, not equivalent leaderboard measurements. Our local evaluation uses rule-based scoring, per-row MMBench accuracy, full-test ScienceQA and a simplified MMMU open-answer scorer. Prompting, answer parsing and benchmark scope can change scores. There were no recorded inference errors in any of the four methods; a small number of parse failures remained in the denominator. Vanilla had four reported parse failures (one ScienceQA, two AI2D, one MMStar); MMMU does not report a separate parse-failure count. One training seed also leaves repeatability unresolved.

## 4. Possible reasons for the outcome

The following are hypotheses, with the supporting observations stated explicitly.

1. **Teacher guidance may have outweighed learning the correct answers.** The logged distillation weight reached its maximum of 1 at update 7 and stayed there for 506/512 updates. Over the last 50 logged updates, mean GDKD was approximately 7.59 versus CE of 1.30; weighted visual alignment contributed approximately 0.93. GDKD remained above the controller target of 3. This suggests the controller was mostly saturated and the objective was dominated numerically by teacher matching. Loss size alone does not establish gradient dominance or causality, but this is the strongest specific diagnostic from the logs. Vanilla now shows that GRACE did not improve the starting checkpoint on average, while ordinary SFT did; this is consistent with investigating the added objectives, but does not establish that distillation caused the loss.

2. **Much less data and training may have limited knowledge transfer.** We used 32,768 examples and one pass, and skipped the released Qwen3-VL pipeline's BF16 SFT initialization stage. A source-proportional subset does not guarantee enough science, diagrams or difficult reasoning examples. A larger teacher's capabilities may require more varied supervision and a longer schedule to transfer. However, SFT improved every primary score over vanilla on this same subset, so limited data did not prevent all useful adaptation.

3. **Low resolution removes useful visual detail.** At most 49 image tokens must represent the whole image. Small text, diagram labels and fine spatial relationships can be lost. The teacher also receives the reduced input, so its supervision may be less useful. This could help explain lower absolute scores, but it cannot by itself explain GRACE losing to vanilla, QAT or SFT: all four evaluations use the same image budget. The ScienceQA loss also appears in the text-only subset.

4. **LoRA gives the student fewer ways to adapt.** Only attention adapters and the visual projector learn, alongside selected quantization scales. QAT/GRACE have about 40.77M trainable parameters, including 12.85M adapter parameters and 2.75M scales. The frozen base and feed-forward weights may restrict adaptation to both the teacher and 4-bit rounding. This differs substantially from updating the full language model. SFT's +2.09 pp mean gain over vanilla shows that these adapters can improve accuracy under the constrained setup; LoRA capacity alone is not enough to explain the lack of a GRACE gain.

5. **The exported model may behave differently from the training model.** Training applies quantization to the base weights and then adds the LoRA update: `Q(W) + ΔW`. Export merges first and then quantizes: `Q(W + ΔW)`. Rounding makes these expressions unequal. Both QAT and GRACE use this procedure, but their learned updates can incur different conversion losses. Native-versus-export accuracy has not been measured here. A no-training W4 quantization control is also absent, so QAT/GRACE versus vanilla combine training and quantization effects rather than isolating either one.

6. **Scoring differences and one-seed variation may contribute.** Shared questions and evaluation code make the local comparison useful, but response-format differences can still interact with the local parsers. The few reported parse failures do not explain the mean ordering: even making all reported GRACE parse failures correct would add only about 0.03 pp to its seven-suite mean. MMMU's simplified open-answer scorer still warrants inspection. Cross-report score gaps additionally include protocol and backbone differences.

## 5. Suggested next experiments

The matched vanilla evaluation is complete. First, inspect raw benchmark answers and compare native versus exported models on a small fixed validation set to check for export or scoring effects. Add a no-training W4 control to separate quantization loss from the benefit of QAT adaptation. Evaluate the released GRACE checkpoint under the same local protocol, and run vanilla at the upstream visual budget in a separate comparison to assess preprocessing effects.

Next, test weaker distillation or a longer distillation warmup, recording CE, GDKD, RCKA and controller trajectories. Increase the visual budget and training data one factor at a time, retaining vanilla, QAT and SFT controls. Repeat the selected configuration with additional seeds.

| Method | Recorded training-loop time | Recorded peak PyTorch reserved memory |
| --- | ---: | ---: |
| Vanilla | No experiment training | Not measured for training |
| SFT | 1.45 h | 6.59 GiB |
| QAT | 1.90 h | 6.64 GiB |
| GRACE | 2.37 h | 24.13 GiB |

These are recorded run-summary measurements, not total end-to-end time or total GPU occupancy. All three trained methods completed 512 updates and produced final exports; vanilla underwent no experiment training.

## Supervisor-ready description

> We ran a resource-constrained GRACE experiment using a Qwen3-VL-2B student and an 8B teacher on two A100 40GB GPUs. To fit the available resources, we trained attention LoRA adapters and the visual projector, used 32,768 examples, limited inputs to 1,024 tokens and reduced images to at most 50,176 pixels. SFT, QAT and GRACE shared the same data and effective batch of 64, completing 512 updates each. We also evaluated the original Qwen3-VL-2B-Instruct weights with the same saved processor and questions. Vanilla averaged 63.72%, SFT 65.82%, QAT 63.99% and GRACE 63.39%. GRACE was −0.33 pp below vanilla, −0.59 pp below QAT and −2.42 pp below SFT, while SFT improved all seven primary scores over vanilla. GRACE improved three primary scores over vanilla and HallusionBench grouped consistency over all controls, but showed no average improvement over the starting checkpoint. This differs from the positive gains reported in the original paper, although the setups are not directly comparable. Its distillation controller stayed at maximum weight for almost the entire run. We suspect the loss balance, limited training capacity, smaller dataset and low resolution contributed. We propose checking export/scoring effects and testing weaker teacher guidance before scaling up.

## Local evidence

- [Pinned experiment configuration](comparison_recipe.json)
- [Detailed evaluation comparison](evaluation/comparison.md), based on the 28 merged result files across vanilla, SFT, QAT and GRACE
- [Vanilla evaluation configuration and protocol](../../evaluation/README_vanilla_baseline.md); original weights and processor budget recorded in `evaluation/baseline/*.json`
- Training summaries and per-update logs under `runs/{sft,qat,grace}/`
- [Constrained-training documentation](../../qwen-vl-finetune/scripts/README_constrained.md)
- [Full-training GRACE launcher](../../qwen-vl-finetune/scripts/finetune_qwen3vl_2b_grace.slurm)
- [Evaluation protocol notes](../../evaluation/README_lmms_constrained.md)

## 6. Validate configuration and training behavior in code

Exported-model benchmark scores measure the quality of the final artifact. To understand how that artifact was produced, validate a chain of evidence: **saved configuration → executed training branch → processed inputs → loss and gradients → optimizer updates → native checkpoint → exported weights**. A declared setting, a nonzero loss, and an updated parameter are different observations; each needs its own check.

### 6.1 Identify the executed entry point and the historical configuration

This experiment launches `qwenvl.train.train_qwen_constrained`, as shown in [run_constrained.py:95](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained.py:95). It imports `GraceTrainer` and student/teacher helpers from `train_qwen_grace.py`, plus QAT utilities from `qat_modules.py`. The standalone `train()` functions in `train_qwen.py`, `train_qwen_qat.py` and `train_qwen_grace.py`, and the full-training Slurm scripts, are separate entry points. Their defaults do not establish this experiment's configuration.

Use [comparison_recipe.json](comparison_recipe.json) and each method's `runs/<method>/launch_recipe.json` as the historical configuration records. `config.json` describes the model/export; it is not a complete training recipe. For runtime evidence, also read `trainable_parameters.json`, `trainer_state.json`, `step_metrics.jsonl`, `run_summary.json` and the native checkpoint's scheduler, optimizer, RNG and GRACE IB state.

The source-hash check performed while adding this appendix found that **every recorded training/data Python file and the ZeRO JSON still matches the saved implementation hash, while `scripts/run_constrained.py` differs**. Its current resolved values still match the pinned configuration in the inspected environment. Therefore, the line references below describe the current files, and today's launcher dry-run must be compared with the saved recipe before using it as evidence of a historical run. The pinned hashes do not cover the entire installed Transformers/DeepSpeed implementation or prove the identity of model-weight files; preserve the environment and checkpoint provenance when reproducing the run.

### 6.2 Where settings are defined and where they take effect

Line links point to the current local workspace. Use the named function and the saved hash to relocate a reference if a file later changes.

| Configuration | This experiment | Definition / implementation | How to validate its effect |
| --- | --- | --- | --- |
| Student, teacher, subset and seeds | 2B-Instruct / 8B-Instruct; 32,768 records; training seed 42, subset seed 2026 | [run_constrained.py:25](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained.py:25); [run_constrained.py:50](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained.py:50) | Read saved `comparison_recipe.json`; verify the GRACE-only teacher branch in [train_qwen_constrained.py:239](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_constrained.py:239). |
| Batch size and duration | 2 ranks × micro-batch 1 × accumulation 32 = 64; 32,768 / 64 = 512 updates | [run_constrained.py:33](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained.py:33); [run_constrained.py:68](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained.py:68); [run_constrained.py:95](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained.py:95) | The launcher fixes two ranks. ZeRO batch/accumulation fields are `auto`, resolved from Trainer; check runtime values rather than reading `auto` as a numeric setting. |
| Image and sequence limits | 4,096–50,176 pixels; ≤1,024 input tokens | [run_constrained.py:34](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained.py:34); [data_processor.py:44](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/data/data_processor.py:44) | Processed tensors are checked in [constrained_data.py:53](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/data/constrained_data.py:53); complete trailing Q/A pairs are removed in [constrained_data.py:91](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/data/constrained_data.py:91). |
| LoRA and trainable modules | Rank 32, alpha 64, dropout .05; q/k/v/o adapters and visual merger | [run_constrained.py:36](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained.py:36); [run_constrained.py:41](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained.py:41); [train_qwen_constrained.py:55](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_constrained.py:55) | Follow [train_qwen_qat.py:116](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_qat.py:116) then [train_qwen_constrained.py:222](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_constrained.py:222): the later language-model/head freeze overrides `tune_mm_llm=True`. Inspect `trainable_parameters.json`. |
| QAT scope and arithmetic | W4G128 on eligible language-model attention/MLP linears; vision, merger, embeddings and head skipped | [qat_modules.py:312](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/qat_modules.py:312); [qat_modules.py:356](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/qat_modules.py:356); [qat_modules.py:169](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/qat_modules.py:169) | Only adapted layers retain trainable scales; other eligible linears use frozen quantizers. The quantizer rounds/clamps weight codes and dequantizes for floating-point `F.linear`. |
| Optimizer and schedule | LoRA LR 2e−4, scale LR 2e−4, merger LR 1e−5; cosine, 3% warmup, zero weight decay | [run_constrained.py:38](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained.py:38); [run_constrained.py:72](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained.py:72); [train_qwen_grace.py:380](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_grace.py:380) | Scale groups are built in [qat_modules.py:432](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/qat_modules.py:432); the merger is split into its own LR group. Inspect actual optimizer groups and scheduler/checkpoint state. |
| Precision, memory and training mode | BF16, TF32, gradient checkpointing, flattened attention, no data packing, ZeRO-2 | [run_constrained.py:65](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained.py:65); [run_constrained.py:71](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained.py:71); [zero2_lora_memory_safe.json:2](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/scripts/zero2_lora_memory_safe.json:2) | BF16 storage/compute and fake quantization do not imply a packed INT4 kernel. Flattening is a collator/attention representation; it does not relax the per-record token cap. |
| Method selection and objective | SFT/QAT: CE; GRACE: CE + β·GDKD + 3·warmup·RCKA | [train_qwen_constrained.py:202](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_constrained.py:202); [train_qwen_grace.py:427](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_grace.py:427); [train_qwen_grace.py:519](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_grace.py:519) | SFT uses [train_qwen_constrained.py:49](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_constrained.py:49) without fake quantization. QAT shares the CE branch but its linear modules still quantize. |
| GDKD settings and supervision | Temperature 2, target/non-target weights 1/4; 128-token chunks; checkpointed chunks; no DKD warmup | [run_constrained.py:43](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained.py:43); [run_constrained.py:86](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained.py:86); [grace_modules.py:105](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/grace_modules.py:105) | Check next-token shifting, `labels != -100`, entropy gates and normalization by gate sum. Answer masks are created in [data_processor.py:234](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/data/data_processor.py:234). |
| RCKA feature selection and ramp | Layer index −2; image-token relationships; coefficient rises from 0 to 3 over 50 optimizer updates | [run_constrained.py:45](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained.py:45); [train_qwen_grace.py:131](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_grace.py:131); [train_qwen_grace.py:148](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_grace.py:148); [grace_modules.py:237](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/grace_modules.py:237) | Inspect selected student hidden-state/teacher hook outputs and image grid boundaries. Require ≥4 tokens and nondegenerate Gram matrices. A nonzero raw RCKA at global step 0 has zero weighted contribution. |
| IB controller | τ=3, η=.003, β starts .5 and stays within [.1,1]; EMA decay .99 | [run_constrained.py:46](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained.py:46); [train_qwen_grace.py:184](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_grace.py:184); [grace_modules.py:335](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/grace_modules.py:335) | Launcher overrides the controller class defaults. Accumulation/rank averaging and update timing are in [train_qwen_grace.py:491](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_grace.py:491). |
| Completion and export | Native checkpoint first; merge adapters; QAT bake; save ordinary HF weights and quantization sidecar | [train_qwen_constrained.py:184](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_constrained.py:184); [train_qwen_constrained.py:273](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_constrained.py:273); [train_qwen_constrained.py:290](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_constrained.py:290) | Check `complete`, `profile`, global step and checkpoint state together. `EXPORT_COMPLETE` also exists for profile exports and is insufficient on its own. |

From `reproduction/GRACE`, these commands show the **current** resolved configuration and command without writing files or allocating GPUs:

```bash
conda activate grace
python qwen-vl-finetune/scripts/run_constrained.py --method sft --dry-run
python qwen-vl-finetune/scripts/run_constrained.py --method qat --dry-run
python qwen-vl-finetune/scripts/run_constrained.py --method grace --dry-run
```

Environment variables can override launcher defaults. Compare the printed values with the pinned recipe, including `GLOBAL_BATCH`, `MAX_PIXELS`, `MODEL_MAX_LENGTH`, learning rates and seeds; do not assume that the terminal environment is clean. The main runtime parser is [train_qwen_constrained.py:194](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_constrained.py:194), and method selection explicitly sets `grace_enable`/`qat_enable` after parsing.

### 6.3 Read-only checks on the completed run

Run this standard-library audit from `reproduction/GRACE`. It checks historical recipes, subset identity, source hashes, completion, trainable-parameter counts and log boundaries. A `CHANGED` source is reported for investigation rather than silently accepted as the original implementation.

```python
import hashlib
import json
from pathlib import Path

root = Path.cwd()
project = root / "qwen-vl-finetune"
experiment = root / "experiments/constrained_32k_seed42"
recipe = json.loads((experiment / "comparison_recipe.json").read_text())
c = recipe["config"]
assert (c["subset_size"], c["global_batch"], c["grad_accum"], c["max_steps"]) == (32768, 64, 32, 512)
assert (c["min_pixels"], c["max_pixels"], c["model_max_length"]) == (4096, 50176, 1024)
assert 2 * 1 * c["grad_accum"] == c["global_batch"]
assert c["subset_size"] == c["global_batch"] * c["max_steps"]
manifest = json.loads((experiment / "subset.json").read_text())
assert len(manifest["records"]) == c["subset_size"]
assert manifest["records_sha256"] == recipe["subset_sha256"]
record_bytes = json.dumps(manifest["records"], sort_keys=True, ensure_ascii=False).encode()
assert hashlib.sha256(record_bytes).hexdigest() == manifest["records_sha256"]
for name, expected in recipe["implementation"].items():
    actual = hashlib.sha256((project / name).read_bytes()).hexdigest()
    print("MATCH" if actual == expected else "CHANGED", name)
for method, expected_count in (("sft", 38019072), ("qat", 40771584), ("grace", 40771584)):
    run = experiment / "runs" / method
    launch = json.loads((run / "launch_recipe.json").read_text())
    assert launch["comparison"] == recipe
    assert launch["method"] == method and launch["profile"] is False
    summary = json.loads((run / "run_summary.json").read_text())
    state = json.loads((run / "trainer_state.json").read_text())
    assert summary["complete"] is True and summary["profile"] is False
    assert summary["stop_reason"] is None
    assert summary["global_step"] == state["global_step"] == c["max_steps"]
    assert (run / "EXPORT_COMPLETE").is_file()
    inventory = json.loads((run / "trainable_parameters.json").read_text())
    assert sum(p["parameters"] for p in inventory.values()) == expected_count
    assert all("merger" in n or n.endswith(("lora_A", "lora_B", "log_scale")) for n in inventory)
    if method == "sft":
        assert not any(n.endswith("log_scale") for n in inventory)
    rows = [json.loads(line) for line in (run / "step_metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == list(range(1, c["max_steps"] + 1))
    if method == "grace":
        assert all(0.1 <= r["beta"] <= 1.0 for r in rows)
        assert rows[0]["rcka_scale"] == 0.0 and rows[-1]["rcka_scale"] == 1.0
    print(method, "validated", expected_count, "trainable parameters", len(rows), "updates")
```

The recorded inventories contain **38,019,072 trainable parameters for SFT** and **40,771,584 for QAT/GRACE**: 12,845,056 LoRA parameters and 25,174,016 merger parameters in all three, plus 2,752,512 trainable scales only in QAT/GRACE. These inventories establish what was eligible for training, not that every parameter changed. The contiguous-log check above applies to this completed run; resumed runs may need segment-aware inspection rather than assuming one uninterrupted log.

For a deeper input check, reconstruct records through `ConstrainedDataset.__getitem__` at [constrained_data.py:200](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/data/constrained_data.py:200) with the saved processor. It verifies image size/mtime, preprocessing token hashes, actual sequence length and visual grids. Inspect decoded non-ignored labels to confirm that supervision covers assistant answers and intended end markers, with prompts masked. The assistant-mask implementation uses specific token IDs, so merely finding some non-ignored labels is not enough to establish correct masking. Image size/mtime checks are not full image-content hashes.

### 6.4 Validate the forward pass, loss, backward pass and optimizer separately

**Follow the arithmetic of each method.** SFT computes `W·x + (alpha/r)·B·A·dropout(x)` with no fake quantization. QAT/GRACE compute `Q(W)·x + (alpha/r)·B·A·dropout(x)` at [qat_modules.py:289](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/qat_modules.py:289), where `Q(W)` is groupwise rounded/clamped and dequantized. There is no activation quantization or packed INT4 GEMM in this path. Non-adapted eligible MLP linears also quantize, but their weights/scales are frozen; their first-forward cache can change the stored weight by materializing `Q(W)` without any optimizer update. Compare frozen values after runtime setup and first-use materialization, rather than misclassifying that initialization as learning.

**Check teacher and loss inputs.** [train_qwen_grace.py:293](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_grace.py:293) freezes the BF16 teacher, switches it to evaluation mode and attaches its hidden-state hook. The teacher forward at [train_qwen_grace.py:450](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_grace.py:450) runs under `no_grad` on the student's input tensors with labels removed; it uses teacher forcing, not generated teacher answers. GDKD aligns next-token logits at supervised answer positions and applies normalized entropy-based weights, rather than dropping examples with a hard confidence threshold. RCKA compares centered visual-token Gram matrices per image; it is not a pixel reconstruction loss or text-token alignment. Inspect layer indices and captured tensor shapes against the model implementation, since the student's hidden-state tuple and teacher's decoder-module hook are different indexing mechanisms.

**Check the weighted objective and controller timing.** The loss is `CE + beta × dkd_scale × GDKD + 3 × rcka_scale × RCKA`. Raw logged loss magnitudes do not establish gradient dominance. The IB update is `beta = clip(beta + .003 × (EMA(GDKD) − 3), .1, 1)` and therefore increases the distillation weight when GDKD is above target. It fires at the accumulation boundary after rank averaging, inside `compute_loss`, before constructing the final micro-batch loss; the new beta can apply to that last micro-batch while earlier micro-batches used the preceding beta. `step_metrics.jsonl` records the last micro-batch's `_grace_logs` at each optimizer boundary, not the mean CE/GDKD/RCKA over all 64 examples. Reconstructing the controller requires accumulation/rank means, which cannot be recovered exactly from these rounded last-micro-batch rows alone.

**Check gradient flow and actual updates.** The existing diagnostic hooks are [grace_profiler.py:361](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/grace_profiler.py:361); optimizer-boundary callbacks are at [grace_profiler.py:635](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/grace_profiler.py:635). Inspect `profiling/learning_diagnostics.jsonl` and `learning_diagnostics_summary.json` for teacher trainability, backward signals, loss coefficients and sampled parameter deltas. With ZeRO, an empty `.grad` after a step is not sufficient evidence of disconnection: gradients can be partitioned or cleared. Hooks before the optimizer, plus optimizer-state/parameter comparisons, are stronger evidence. Expect some initially zero LoRA-A gradients because B starts at zero; assess multiple steps. The diagnostic parameter deltas are sampled changes from the start, and gradient norms combine micro-batch backward signals; they are not exhaustive tensor comparisons or the final clipped optimizer-gradient norm.

For any trainable category with gradients but no measured update, verify membership and LR in the actual optimizer groups, scheduler LR at that step, whether the optimizer stepped, and parameter/master-weight precision. BF16 rounding can hide small changes. Use full-tensor comparisons and FP32 optimizer state over more steps before concluding that scales or the merger are either frozen or successfully updated. If the concern is which loss changes a particular parameter, perform per-term gradient checks on a small diagnostic batch and compare CE-only, CE+GDKD and CE+RCKA branches with controlled weights, inputs and dropout/RNG. A loss-output backward hook establishes participation in the combined objective, not its individual gradient contribution to every trainable module.

Non-finite distillation terms are replaced with a detached zero at [train_qwen_grace.py:418](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_grace.py:418); missing/degenerate RCKA inputs can also produce zero. Inspect warnings, actual tensors and term-gradient connectivity rather than interpreting every logged zero as healthy convergence. Loss normalization also matters: [train_qwen_constrained.py:255](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_constrained.py:255) disables accumulated-item loss kwargs so Trainer applies gradient accumulation to this per-example loss.

### 6.5 What the saved profile already proves, and what remains unresolved

The [three-step GRACE diagnostic summary](profiles/grace/profiling/learning_diagnostics_summary.json) records a frozen teacher, nonzero LoRA-B and QAT-scale backward signals, both GDKD and post-warmup RCKA connected to backward, a changed sampled LoRA-B tensor, and an unchanged sampled frozen MLP scale. However, **`qat_scale_parameter_changed` is false**, and the sampled trainable scale and merger norm show zero delta over the three-step profile. That does not prove all scales/merger tensors were unchanged, but it also does not establish successful updates for those categories. The full 512-step runs have inventories and step logs, but no saved learning-diagnostics summary; the short profile cannot prove their entire optimization history.

The diagnostics refresh their baseline after framework dtype/device setup at [grace_profiler.py:479](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/grace_profiler.py:479), preventing setup casting from being mistaken for learning. Profile datasets intentionally repeat the eight longest retained records at [constrained_data.py:192](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/data/constrained_data.py:192). They exercise the memory and graph envelope, not the representative training distribution or convergence. Timing summaries identify where compute occurs, while gradient/update diagnostics address whether learning occurs; neither substitutes for the other.

For additional dynamic checks, use the launcher's `--profile` mode in a **fresh `EXPERIMENT_DIR`**, compare its dry-run configuration with this pinned recipe, and survey GPUs before launching. The launcher enables learning diagnostics only for GRACE profile runs at [run_constrained.py:90](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained.py:90); enabling equivalent checks for SFT/QAT requires an explicit diagnostic configuration change. Do not reuse the completed experiment directory or interpret profile exports as final trained results. No new training or GPU run was launched to add this appendix.

### 6.6 Validate native checkpoints against exports before attributing a score gap to training

The native [GRACE checkpoint-512](runs/grace/checkpoint-512/) retains adapters, QAT scales and DeepSpeed optimizer state. Restore it through the matching constrained model/wrapper and native resume path; loading it as an ordinary unwrapped HF model may discard adapter/scale keys and test a different function. Inspect missing/unexpected keys, restored global step, optimizer/scheduler, RNG and `grace_ib_state.json`.

Compare a fixed validation batch in three states: the reconstructed native model before merge, after LoRA merge, and after QAT bake/HF reload. Switch to evaluation mode, freeze dropout/RNG choices, use the same processor, weights and inputs, and measure next-token logits, CE and task predictions in addition to accuracy. Export mutates the model in place, so use separate copies/restorations for each state. The merge is [qat_modules.py:456](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/qat_modules.py:456) and the bake is [qat_modules.py:518](/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune/qwenvl/train/qat_modules.py:518): native QAT uses `Q(W) + delta`, while the merged export uses `Q(W + delta)`. Equality is not guaranteed because rounding is nonlinear; an observed difference must be measured rather than inferred to be the cause of a benchmark loss. SFT's merge should preserve its evaluation-mode linear function up to numerical precision.

Finally, check that the HF export has merged adapters, ordinary floating-point weights, the expected quantization sidecar, matching processor settings and no leftover adapter/scale keys in its HF weight file. Packed INT4 deployment requires a separate conversion and kernel validation. The [evaluation summarizer](../../evaluation/summarize_constrained_evals.py) validates final predictions and matching questions, but it does not test any of the training-gradient, optimizer-update or native-versus-export properties above. Only the combined evidence supports a claim about what the implementation actually learned.

