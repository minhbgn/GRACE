# Evaluation protocol, upstream GRACE comparison, and next-step recommendation

This follows the [experiment report](experiment_report.md). The main conclusions are:

1. Both experiments evaluate the same benchmark questions and labels. Their image preprocessing differs; the saved training token limits are not enforced as evaluation input-truncation limits.
2. The constrained implementation retains the core GRACE loss mathematics, but changes the trainable model, data, optimization and export. It is a substantial adaptation, not an equivalent small-hardware execution of the original training recipe.
3. Evaluating released GRACE-BF16 weights is the recommended immediate next step. Whether that can replace training reproduction depends on whether the research contribution is inference-time compression or compression incorporated into distillation.

## 1. Evaluation scripts and token budgets

### Actual call paths

The old experiment is the default target of [run_all_constrained_evals.sh](../../evaluation/run_all_constrained_evals.sh). The new [run_local_evaluation.sh](run_local_evaluation.sh) sets its own `EXPERIMENT_DIR`, pins `baseline sft qat grace`, and calls that same shared evaluator. Its resume/keep-going behavior and final summary generation concern orchestration, not scoring.

I generated both sets of dry-run commands with the same Python interpreter and environment. All **28 command pairs differ only in experiment-specific values of `--model_path`, `--processor_path`, and `--output`**. Dataset arguments, evaluator scripts, attention-backend arguments and output-token limits match.

| Suite | Shared evaluator, relative to `evaluation/` | Split / scope | Maximum generated tokens |
| --- | --- | --- | ---: |
| HallusionBench | `HallusionBench/eval_hallusion.py` | Same local benchmark rows | 8 |
| MMBench | `MMBench/eval_mmbench.py` | English dev, per-row | 8 |
| ScienceQA | `ScienceQA/eval_scienceqa.py` | Full test | 8 |
| AI2D | `AI2D/eval_ai2d.py` | Test | 8 |
| MMMU | `mmmu/eval_mmmu.py` | Validation | 32 |
| SEED-Bench-IMG | `SEEDBench/eval_seedbench.py` | Image subset | 8 |
| MMStar | `MMStar/eval_mmstar.py` | Same local benchmark rows | 8 |

All seven use greedy generation (`do_sample=False`). The prior result audit verified identical unique question identities and gold labels across all four methods and both experiments. Changes in rank count or merged-file row order do not change that conclusion; comparisons must align by identity.

Against the upstream checkout, each of these seven evaluator files has only three kinds of changes: an optional `--processor_path`, loading the processor from that path when provided, and recording processor path/image size in the result metadata. Their prompt construction, answer parsers and correctness logic have not changed in that diff. Historical execution environments and complete raw dataset bytes were not archived by this audit, so this is stronger evidence about the observed question sets and current code than about every historical runtime detail.

### Three different limits must be distinguished

| Limit | Previous | Current | Effect in these evaluations |
| --- | ---: | ---: | --- |
| Saved tokenizer `model_max_length` | 1,024 | 2,048 | No corresponding input truncation requested by these scripts |
| Image-processor maximum area | 50,176 pixels | 100,352 pixels | Changes resized images and visual-token counts |
| Generation `max_new_tokens` | 8, or 32 for MMMU | Same | No extra answer-generation budget |

The evaluator first renders the chat template with `tokenize=False`, then calls `processor(..., padding=True, return_tensors="pt")`. It does not set `truncation=True` or pass an input `max_length`. A saved tokenizer limit alone does not request truncation or change the architecture's context capacity.

I confirmed this using the actual saved SFT processors, entirely on CPU and without loading model weights:

- The same long text prompt produces **2,509 input tokens under both processors**, exceeding both saved limits. The tokenizer emits a length warning but retains the tokens. This is a preprocessing check, not an inference test of that prompt.
- The same synthetic 1,024 × 1,024 image produces a `[1, 14, 14]` patch grid under the old processor and `[1, 18, 18]` under the new processor: **49 versus 81 merged visual tokens**.

The theoretical area-derived ceilings are 49/98 tokens per image, but resizing respects patch/merge multiples and aspect ratio, so a square need not attain 98. MMMU can contain multiple images, so per-image limits should not be confused with a limit on total input visual tokens.

Vanilla loads the original model weights and the corresponding experiment's **SFT processor only**. Thus it also receives the larger image budget in the new experiment. All current result files record that budget; old vanilla files record the old budget, and old trained-model processors retain the old configuration.

The supported interpretation is: **the new evaluation supplies more visual detail to all four models; it does not provide longer generated answers or demonstrably recover text previously cut off at 1,024 tokens.** All four seven-suite means increase, although vanilla and SFT decline on MMMU. More visual tokens are therefore a plausible contributor to the broad gains, not a guarantee of improvement on every task.

Vanilla's +1.72 pp mean improvement, and its ScienceQA image gain of +4.31 pp with unchanged text predictions, support this interpretation. For the trained models, weight changes and preprocessing changes remain entangled. Evaluate old/current checkpoints under a shared processor to separate them.

## 2. What “original GRACE” means in this comparison

For a stable code reference, I compared the working tree to the GRACE repository's checked-in upstream commit **`b73fa9b4d427aa64ed3ad274def641efb3145877`**, using `git show HEAD:<file>` and `git diff`. This avoids mistakenly treating the locally modified shared trainer as the original implementation.

The reference launcher is [finetune_qwen3vl_2b_grace.slurm](../../qwen-vl-finetune/scripts/finetune_qwen3vl_2b_grace.slurm), which is unchanged from that commit and declares 4 nodes × 4 A100-80GB GPUs. Its defaults are a reference recipe, not a verified training log for every released checkpoint. The paper itself reports LLaVA/Qwen2-VL experiments with a different hardware/data schedule, so this comparison concerns the released **Qwen3-VL code path**, not an exact reconstruction of all paper experiments. [GRACE paper, training setup](https://arxiv.org/html/2601.22709v1#S4)

| Dimension | Upstream 16-GPU Qwen3-VL GRACE launcher | Current constrained experiment | Why it matters |
| --- | --- | --- | --- |
| Entry point | `train_qwen_grace` | `train_qwen_constrained` → shared `GraceTrainer` | Local wrapper changes data, trainability, normalization and export |
| Student / teacher | Vanilla 2B / frozen 8B | Same | Backbone pair is preserved |
| Trainable student | Full language model, including embeddings/head, plus visual merger | Attention LoRA rank 32 plus visual merger; base language model/head frozen | Very different optimization space |
| Quantization | W4G128 on eligible language linears; learned scales across replaced layers | Same quantization scope; only adapted attention scales train | Frozen MLP weights/scales cannot adapt to quantization or teacher supervision |
| Visual merger | BF16, trainable | BF16, trainable | Merger remaining BF16 is inherited from this Qwen3-VL code, not a new LoRA deviation |
| Training data | ShareGPT4V mix665k, about 665k records | 32,768 selected single-image records | About 4.9% of nominal data; additional modality/content restrictions |
| Duration | One epoch by default | One pass, 256 updates | Much less total data exposure |
| Global batch | 16 × 2 × 16 = 512 | 2 × 1 × 64 = 128 | Different update count, micro-batch weighting and controller schedule |
| Image maximum / minimum area | 200,704 / 784 | 100,352 / 4,096 | Different evidence available to teacher and student |
| Declared token limit | 8,192 | Strict 2,048 | Upstream flattened preprocessing does not enforce an equivalent strict cap |
| Base/adaptation LR | 1e-5 for dense training | 2e-4 for LoRA, 1e-5 for merger | A LoRA LR is not equivalent to a dense-weight LR |
| Scale LR / weight decay | 2e-4 / 1e-3 | 2e-4 / zero | Optimization differs even beyond trainable capacity |
| RCKA weight / warmup | 3 / 400 updates | 3 / 25 updates | Much earlier full alignment in updates and examples |
| GDKD/controller settings | T=2, component weights 1/4; target 3, eta .003, beta .5→[.1,1]; no DKD warmup | Same | Controller saturation was not introduced by changing these numerical defaults |
| Data ordering / loss averaging | Modality-length grouping; flattened micro-batch 2 | No modality-length grouping; micro-batch 1, per-example means | Gradient accumulation need not recreate original token/example weighting |
| DeepSpeed | ZeRO-2, communication overlap | ZeRO-2, smaller buckets, no overlap | Primarily memory/throughput changes, not new teacher-loss formulas |
| Export | Bake the trained dense weights' quantization | Merge LoRA into frozen base, then quantize/bake | Creates a native/export function mismatch specific to the adapter path |

The previous experiment used an even smaller image/token budget and 512 updates; the corresponding comparison is in its [report](../constrained_32k_seed42/experiment_report.md). Neither constrained run contains a BF16 KD-only experiment: local SFT has no teacher loss. The released `GRACE-BF16` checkpoint is also an SFT baseline according to the authors' [training README](../../README.md#1-bf16-sft-baseline). The local and released SFT checkpoints differ in training recipe and scale; their distinction is not SFT versus distillation.

### Core mathematics that was preserved

An AST comparison of upstream and current `grace_modules.py` found these definitions unchanged:

- `decoupled_kd_per_token` and `confidence_weights`;
- `_gram`, `_center`, `_cka`, `_cka_safe`, and `rcka_loss`;
- `AdaptiveIBController`.

The teacher remains frozen and receives the same inputs as the student with labels removed. The total objective still combines CE, beta-scaled GDKD and warmed-up RCKA. Visual alignment remains per image/video item; larger gradient accumulation does not create a larger set of tokens for a single RCKA computation.

The main loss implementation addition is checkpointed GDKD chunks: selection/softmax intermediates are recomputed during backward rather than all retained. The current recipe uses chunk size 128 instead of the upstream default 2,048. All four existing CPU tests in [test_grace_dkd_memory.py](../../tests/test_grace_dkd_memory.py) passed during this audit. They cover loss/gradient agreement between the retained original path and checkpointed path in FP32/BF16, empty supervision, memory retention and invalid chunk sizes. The unchanged base loss definitions plus these checks are evidence that this memory optimization preserves the intended loss, not proof of end-to-end training equivalence.

Teacher hooks, non-finite-term guards and the per-update adaptive controller already existed upstream. Their presence should not be counted as newly invented constrained-GRACE behavior. Added diagnostics, safe pausing, native checkpoints, processor saving and export error checks largely concern execution and auditability. ZeRO-3 teacher-loading protections exist in the modified shared code but are not activated by these ZeRO-2 recipes.

### Changes that can materially affect the research conclusion

**LoRA changes the method being tested.** Original full-weight QAT optimizes a function of the form `Q(W)` with trainable W and scales. The local attention layers train `Q(W_frozen) + ΔW_LoRA`, and non-adapted quantized MLP weights/scales remain frozen. This is not merely the same optimizer running more slowly on fewer GPUs. Both adaptation capacity and how the model compensates for quantization change.

**LoRA export changes the forward function.** The local export uses `Q(W_frozen + ΔW_LoRA)`, which need not equal the native training function. A difference is mathematically expected in general, but its actual benchmark cost remains unmeasured. Since QAT and GRACE learn different deltas/scales, sharing an export routine does not ensure equal conversion loss. Native/export comparisons remain necessary before assigning all deficits to learning.

**Data and weighting change.** The constrained dataset drops entire trailing conversation pairs to fit and rejects invalid samples, rather than silently clipping multimodal streams. This protects input validity but changes the supervision. Its micro-batch-one losses average within individual examples; accumulating them is not generally equivalent to averaging all supervised tokens/gates across the original flattened micro-batches. The local trainer explicitly disables accumulated-item loss kwargs so Trainer scales these per-example losses by accumulation. That normalization should be checked against the intended objective when porting to another Transformers version; this audit does not establish a bug in the authors' original environment.

**Successful gradients are not proof of all intended updates.** The profile shows LoRA updates and backward connectivity, but sampled scale and merger-norm tensors did not change. This warrants full-tensor and optimizer-state inspection, not the assertion that the whole merger or every scale failed to learn.

The justified conclusion is therefore: **this constrained LoRA/W4 adaptation completes within budget, but does not recover a general GRACE advantage.** It does not prove that GRACE is computationally infeasible on the machine, that LoRA is the cause, or that the core algorithm fails. Matching the original global batch through more accumulation would not restore full-weight optimization, missing training examples or the released initialization history. Conversely, a 16-GPU launcher is not proof that exactly 16 GPUs are an algorithmic requirement; memory strategy and acceptable runtime are separate constraints.

### Joint training versus the released checkpoints' reported initialization

**The upstream launcher implements one joint KD+QAT training stage from vanilla Qwen3-VL-2B-Instruct by default. It does not require preceding BF16 distillation.** For the released Qwen3-VL checkpoints, the documented preceding initialization stage is BF16 SFT. The vanilla-initialized pilots omit that SFT initialization, not a prior KD stage.

The upstream source at commit `b73fa9b` establishes the execution order:

1. The Slurm launcher defaults `MODEL_NAME_OR_PATH` to `Qwen3-VL-2B-Instruct`, and enables both `qat_enable` and `grace_enable`.
2. `train_qwen_grace.py` loads that student and replaces eligible linears with QAT modules **before** constructing the trainer or calling `trainer.train()`.
3. Each QAT forward rounds/clamps/dequantizes the weights with learned scales. CE, GDKD and RCKA are computed using this quantization-aware student and the frozen teacher in the same training loop.
4. After training, `bake_qat_weights_inplace` materializes the learned quantized weights for saving. This export does not perform another teacher-free training stage; packed INT4 conversion is another representation/deployment operation.

The paper's §3.4 describes joint optimization of weights and quantization scales during distillation. The original paper reports LLaVA/Qwen2-VL experiments; its Qwen3-VL-specific initialization should not be inferred from that paper. [GRACE paper, §3.4](https://arxiv.org/html/2601.22709v1#S3.SS4)

The released W4/W8 models use the released BF16 student as initialization. The authors explicitly identify [BF16 SFT](../../README.md#1-bf16-sft-baseline) as the source of `GRACE-BF16`; the shared project name does not imply GDKD/RCKA training. Our matched branches are **SFT → QAT (CE only)** and **SFT → GRACE (CE + GDKD + RCKA with QAT)**. Using the GRACE trainer with this SFT initialization performs joint KD+QAT in the subsequent stage.

The vanilla initialization in the GRACE launcher defaults differs from the SFT initialization documented for the releases. The checked-in `finetune_qwen3vl_2b_bf16.slurm` calls the ordinary SFT trainer, and the README explicitly connects that launcher to the released BF16 checkpoint. Exact release job settings may still require run records, but the stated SFT objective should not be treated as unresolved.

For the earlier pilots, starting from vanilla matches the upstream GRACE launcher's default. It is not evidence of a mandatory omitted KD stage. Those pilots differ from the release pipeline by skipping the preceding SFT initialization. Exact commands, data and schedules remain relevant to full recipe reproduction; the authors' explicit identification of the BF16 source as SFT already resolves that objective distinction.

### BF16 provenance correction and saved quantization scales

**Correction recorded 2026-10-05:** use the explicit [Training → BF16 SFT baseline](../../README.md#1-bf16-sft-baseline) statement for release provenance. Broad model-card descriptions such as “trained under the GRACE framework” and the `GRACE-BF16` name do not override the documented SFT source. Earlier interpretations of those labels as evidence of BF16 distillation are corrected above. Archived upstream cards and recorded results remain unchanged.

The deployment instructions do explain **W4 QAT research checkpoint → packed INT4**. Their phrase “BF16 QAT checkpoint” means quantized/dequantized W4 weights stored in BF16 tensors, not the separate unquantized `GRACE-BF16` release. The packing command consumes `GRACE-W4G128/qat_quantized_weights.bin`; it is not a documented direct conversion of the unquantized BF16 release into the same learned W4 model. [Deployment instructions](https://github.com/ForeverBlue816/GRACE#-deployment-int4--awq-inference)

The Hugging Face API file listings were checked directly, without downloading released model weights:

| Release | Revision checked | `qat_quantized_weights.bin` present? |
| --- | --- | --- |
| GRACE-BF16 | `82f9a267340f65a3619be0e19daf6582d3998a5f` | No |
| GRACE-W4G128 | `57dcaeed3fd7739a0566bfc4b8651f451068825c` | Yes |
| GRACE-W8G128 | `32c351aed1388ae602d59f7c26101dd16aca8c80` | Yes |
| GRACE-W4G128-AWQ | `2a4cde13a84962f0fe17861057a86746652524e0` | No; deployment uses packed model tensors instead |

The training source saves scales as `<module>.log_scale` alongside quantized/dequantized `<module>.weight` and optional bias in the sidecar. The ordinary final HF weight file has scale parameters stripped; native training checkpoints retain their training state. `qat_save_quantized` enables the sidecar and is true in the upstream launcher. The packing code reads those log-scales, computes `s = exp(log_scale)`, and stores reshaped FP16 `scales` with packed integer weights/zero points. It reuses the learned quantizer rather than fitting a new calibration-based AWQ quantizer. Integer-code preservation is checked; FP16 scale storage can introduce rounding differences.

For a direct local artifact check, our current GRACE sidecar contains **196 scale tensors / 11,010,048 scale entries**, while its final `model.safetensors` contains no `log_scale` keys. These include frozen initialized MLP scales as well as the 112 attention-layer scale tensors eligible for optimization; presence does not imply every scale was learned or changed. This verifies our local export's contents, not the contents of the remote sidecar binaries.

Full-precision KD without QAT has no learned weight-quantization scales to recover. Saved W4 scales belong to the corresponding QAT-trained weights and should not be transplanted onto the earlier unquantized BF16 model as if that reproduced the W4 checkpoint.

## 3. Should we move to released GRACE-BF16?

**Yes for the next evaluation baseline. Whether to stop training reproduction depends on the research question.** The checkpoint is a standard full-precision GRACE student, without QAT weight compression. It avoids the local LoRA merge/quantize path and lets us first test whether the evaluation and token-reduction implementation behave sensibly. Its published scores must still be checked locally under an explicit protocol. [Released checkpoint and intended use](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-BF16)

There are three distinct milestones:

| Milestone | What it establishes | Is original retraining required? |
| --- | --- | --- |
| Re-evaluate released weights | A trusted local starting accuracy and inference pipeline | No |
| Apply pruning/merging to released weights | Compatibility and accuracy/efficiency of post-training token reduction | No |
| Integrate token reduction into teacher-guided learning | Whether training compensates for token loss or learns a better reduction policy | A verified training path and matched training controls are needed; full original retraining is not necessarily needed |

Inference-only integration is a valid empirical study, but does not establish the benefit of compression **during** distillation. A released checkpoint can also initialize a smaller continuation study: identical starting weights, one continuation without reduction, one with reduction, and a CE-only reduction control. That studies adaptation of an already-distilled model; it should not be described as reproducing distillation from vanilla.

### A concrete, bounded initial study

First evaluate vanilla 2B and released GRACE-BF16 without added reduction under a common, explicitly recorded image budget, prompts, decoding and scorers. Inspect the released processor and author evaluation configuration rather than silently reusing our constrained SFT processor. If trying to match published scores, keep that reproduction protocol separately named from a common-budget research comparison. Pin model/code revisions and save actual visual-token counts. Resolve a major unexplained baseline discrepancy before adding a pruning algorithm.

Then construct the four-cell comparison:

| Weights | Full visual tokens | Added pruning/merging |
| --- | --- | --- |
| Vanilla 2B BF16 | A | B |
| Released GRACE 2B BF16 | C | D |

Keep image resolution fixed across cells and sweep retained fractions, for example 100%, 75%, 50% and 25%. Include one well-defined pruning method and one merging method initially, with simple random/uniform retention controls as appropriate. Match insertion layer and actual retained counts where possible. Report both absolute scores and degradation from each model's own full-token baseline. The interaction `(D − C) − (B − A)` measures whether the released GRACE model loses less accuracy under reduction, rather than merely starting from higher accuracy. It remains a comparison of model histories, not proof that GRACE's loss alone caused robustness without matched training controls.

Because our current image caps already compress visual information heavily, do not treat 49/98 tokens as an uncompressed architecture baseline. Use a justified higher-resolution reference with actual token counts, and include **lower-resolution, no-pruning** as a separate control at comparable language-model token counts. Downsampling and feature pruning discard information at different stages; this control directly tests whether the added algorithm improves the trade-off beyond simply shrinking images.

Measure accuracy, prefill/time-to-first-token, end-to-end latency and peak allocated memory with identical hardware and generation settings. Include reduction/attention-extraction overhead and separate preprocessing, vision-encoder and language-model costs where practical. Removing tokens after the vision encoder cannot remove vision-encoder work already performed. Token-count or FLOP reductions alone are insufficient evidence of wall-clock speedup.

Use the existing seven suites for continuity, and add a detail-sensitive benchmark such as TextVQA, OCRBench or DocVQA. A later multi-turn check would test whether prompt-dependent selection discards information needed by subsequent questions. LLMC+ explicitly studies attention versus similarity, prune versus merge, and vulnerabilities on detail-sensitive and multi-turn tasks. Its reported model families include Qwen2.5-VL, so the paper alone does not establish that a method is directly compatible with Qwen3-VL. [LLMC+ paper](https://arxiv.org/html/2508.09981v2)

After BF16 behavior is understood, add released W4 and an appropriate quantization control if quantization/token-reduction interactions are part of the intended claim. BF16 experiments alone cannot establish that interaction. Keep rounded-weight research exports distinct from packed-INT4 implementations when reporting memory or speed.

### Integration points that need deliberate implementation

Qwen3-VL uses a spatial merger, multimodal position encoding and DeepStack feature injection. The local Transformers implementation indexes DeepStack additions by a visual-position mask, so arbitrary removal of embeddings without updating the associated tensors can invalidate the forward pass. Pruning after some language layers, after the merger, or inside the vision encoder are different interventions. Attention-based methods may also require information unavailable from the chosen fast-attention execution path. These issues need explicit handling and a 100%-retention equivalence check. [Qwen3-VL architecture](https://github.com/QwenLM/Qwen3-VL#introduction)

For training-time integration, existing RCKA assumes corresponding teacher/student visual tokens. Pruning the student to M tokens while leaving the teacher at N creates incompatible M×M and N×N Gram matrices. Selecting the corresponding teacher tokens or aggregating teacher features with the same grouping operator can provide an M-token comparison; choosing that mapping is part of the proposed method. It is not enough to pass the old image grid through unchanged. GDKD likewise needs alignment of corresponding supervised answer positions after sequence lengths change. This is a concrete reason to understand and verify the training implementation before attempting joint integration.

## 4. Why the supervisor may have requested reproduction

The local [initialization plan](../../../../plans/1_initialization_plan.md) records the mentor's guidance to reproduce one baseline before extending it, and explicitly describes incorporating compression **during distillation**. The [reproduction plan](../../../../plans/2_reproduction_plan.md) also discusses inference-time integration as a possible direction. This supports a likely rationale, but does not establish the supervisor's current acceptance criterion.

Reproduction can serve several purposes:

- Establish that the baseline and evaluator are trustworthy before attributing changes to a new reduction method.
- Make the training path understandable and modifiable, including teacher/student alignment and quantization behavior.
- Provide a matched no-reduction training control, so an improvement cannot be explained merely by additional fine-tuning.
- Demonstrate that the proposed contribution is a training method rather than a combination of existing inference tools.

Those purposes do not logically require spending unlimited time reproducing the full released training run. Some can be satisfied by released-checkpoint evaluation plus a smaller, verified continuation experiment. The two completed constrained runs are useful feasibility and implementation evidence, but do not yet satisfy full training-result reproduction.

I recommend proposing a scoped change to the supervisor: pause further broad tuning of the current constrained recipe; validate released BF16 weights now; agree whether the next milestone is post-training compression robustness or a trainable compression method; retain targeted training correctness checks if the latter is required.

Suggested wording:

> Our two constrained LoRA/W4 runs complete, but GRACE remains below matched QAT on the seven-suite mean. The core loss formulas are preserved, while trainable capacity, data exposure, optimization and export differ substantially from the full-training recipe. The larger-input run also changes evaluation preprocessing, which benefits vanilla without training. I propose validating the released GRACE-BF16 checkpoint next. For the reproduction milestone, do you require retraining from the original student, or would reproducing released-checkpoint evaluation plus validating a smaller continuation-training control be sufficient? Is the intended contribution inference-time token reduction, or token reduction integrated into distillation?

## 5. Checks performed and boundaries

- Compared 28 old/new dry-run command pairs after normalizing interpreter/environment.
- Inspected all seven evaluator diffs against the pinned upstream checkout; cross-experiment question/gold equality is documented in the previous result audit.
- Ran CPU-only text/image preprocessing checks with both saved processors.
- Compared upstream/current loss definitions structurally and inspected training, quantization, data and export changes.
- Ran all four existing CPU DKD memory/equivalence tests successfully.
- Checked primary paper/model sources for checkpoint scope and token-reduction methodology.

No training, GPU inference, model-weight downloads, code rollback or launch-configuration edits were performed. This is a source/artifact audit with small CPU checks, not proof of full numerical equivalence, successful updates of every trainable tensor, or a measured runtime for any proposed next experiment.
