from pathlib import Path
p = Path('plans/3_llmc_grace_proposal.md')
s = p.read_text()
s = s.replace('released_bf16_init_32k_b128_p589824_t2048_seed42', 'released_bf16_init_32k_b128_p589824_t4096_seed42')
s = s.replace('Prepared 2026-10-05. **Planning update approved on 2026-10-05. Subsequent request: prepare and memory-check larger-resolution QAT/GRACE controls before LLMC+ work.**', 'Prepared 2026-10-05. **Updated by user request on 2026-10-12: proceed with the completed 576-cap / 4,096-total-token configuration, learn LLMC+, integrate token reduction with QAT W4, then choose further work from the measured results.**')
s = s.replace('Use these completed unpruned controls for later within-checkpoint retention sweeps; the rest of the approved LLMC+ sequence remains unchanged. The older table below is preserved as historical evidence.', '''The older table below is preserved as historical evidence.

**Selected configuration:** keep minimum/maximum image area **4,096 / 589,824 pixels**, maximum merged visual count **576**, total training cap **4,096**, the saved 32,768-example manifest/order, batch **128**, **256** updates and seed **42**. Use the completed **QAT W4** checkpoint as the primary unpruned control and **QAT W8** as the quality reference. Existing GRACE checkpoints remain secondary comparisons; new KD training is conditional, not part of the first required matrix. Keep this configuration fixed through the first LLMC+ integration study; further image-budget expansion is deferred.

QAT took about **1.96 elapsed hours on two A100s**, with **9.55 GiB peak reserved per GPU**. Versus the old budget, QAT time rose **1.2–2.0%**, while reserved memory rose **12.7%** (about 1.1 GiB). GRACE time rose about **5%** and reserved memory about **4.1%**. These are adaptation measurements, not inference speedups. The next steps are toolkit learning/reproduction → inference-only QAT W4 composition → a focused joint-QAT comparison → analysis and a conditional extension or alternative method.''')
s = s.replace('[Latest local native/greedy256]', '[Historical 98-cap local native/greedy256]').replace('Current development comparison; MMBench', 'Historical comparison; MMBench').replace('Same latest protocol, W8', 'Same 98-cap protocol, W8').replace('QAT is the strongest evaluated local low-bit model on this mean.', 'QAT leads GRACE on this mean.')
s = s.replace('**Task trade-offs remain important.** In the latest evaluation,', '**Task trade-offs remain important.** In the historical 98-cap evaluation,')
s = s.replace('HallusionBench grouped consistency is a recurring measured benefit;', 'The new 576-cap results preserve the same task ordering, with MMStar deficits **7.93/13.20 pp** and math deficits **19.20/25.20 pp** at W4/W8. HallusionBench grouped consistency is a recurring measured benefit;')
s = s.replace('**Cost correction.** The released-init GRACE runs', '**Historical 98-cap cost correction.** The released-init GRACE runs')
def section(start, end, replacement):
    global s
    a, b = s.index(start), s.index(end)
    s = s[:a] + replacement.rstrip() + '\n\n' + s[b:]
section('## A bounded baseline check before joint training', '## Proposed sequence', '''## A bounded baseline check before joint training

Proceed with the completed 576-cap configuration. The former historical MMBench-bookkeeping and documentation-cleanup items are removed from the active plan by user decision; their archived limitations remain recorded in the reports. The remaining checks concern integration correctness, not reopening baseline selection. LLMC+ learning and token-only reproduction can proceed independently.

1. On a small fixed development set, compare the native QAT W4 adapter/checkpoint with its merged/baked export. Native `Q(W) + delta` and exported `Q(W + delta)` need not agree. Establish the same computation for the sequential and joint-QAT comparisons; extend this check to GRACE only if a KD comparison is pursued.
2. Verify full-tensor scale/merger updates and optimizer state before interpreting learned compensation during joint QAT. Short profiles found gradients but no sampled scale change; this neither proves all scales were frozen nor confirms successful scale learning. If a correction changes the forward/training function, create matched corrected controls in a new experiment and preserve the completed baseline.
3. **Conditional on returning to KD:** diagnose controller saturation and teacher suitability without an expansive tuning sweep. Beta is capped for about 98% of the latest GRACE updates. Measure per-term gradient norms and cosines on fixed batches; large scalar losses alone do not establish gradient conflict. Do not assume lowering the controller target weakens KD: the implemented update has the opposite direction.

The no-training pruning study can use the completed exported artifacts while native training checks continue. Claims about learned compensation require the verified training/export path; KD diagnostics are not a prerequisite for CE-only QAT integration.''')
s = s.replace('### A. Reproduce a defined LLMC+ slice, with no training', '### A. Learn the LLMC+ toolkit and reproduce a defined token-only slice')
s = s.replace('- After plan approval, clone into', '- Clone into')
s = s.replace('- First target: LLaVA-1.5-7B,', '- Learn the configuration and execution path: model loading → processor → token selector and insertion layer → generation/cache → evaluator. Write a short local usage guide with commands for the full-token baseline and one reducer, config fields, output locations and how to verify actual retained counts. This toolkit-learning step must distinguish token reduction from LLMC+\'s quantization/calibration options; preserve our existing QAT weights when composing methods.\n- First target: LLaVA-1.5-7B,')
s = s.replace('The [current model registry]', 'The [model registry at the initial inspection]').replace('includes Qwen2-VL and Qwen2.5-VL, but no explicit Qwen3-VL class in the inspected version.', 'included Qwen2-VL and Qwen2.5-VL, but no explicit Qwen3-VL class. Recheck the pinned version during setup.')
section('### B. Establish inference-only composition', '### C.', '''### B. Learn and validate LLMC+ composition with the frozen QAT W4 checkpoint

Use the completed [576-cap QAT W4 experiment](../reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/README.md) as the primary target. Map the chosen LLMC+ reducer onto Qwen3-VL's forward path; verify compatibility rather than assuming toolkit support. Preserve the trained W4 computation and scales; do not silently replace QAT with the toolkit's PTQ/calibration pipeline.

First verify that enabling the integration with retention **1.0** reproduces the unmodified checkpoint's outputs within a declared numerical tolerance. Check spatial merging, original multimodal positions, attention masks, cache handling, text/answer preservation and DeepStack feature injection. Document selector placement, retained-count rounding/minimum, causal behavior and backend requirements before interpreting accuracy changes.

Then screen one selected reducer at retained fractions **1.0, 0.75, 0.50 and 0.25**, with random or uniform spatial selection at matched counts. Use development data to choose the reducer, layer and budget. Keep the native **4,096 / 589,824-pixel processor bounds** and **4,096 training-token cap** fixed; no image-budget expansion is part of this first study. Actual starting visual counts vary (training mean 254.64, maximum 576); measure the evaluation distribution independently. LLaVA's fixed 576-token input does not establish equal redundancy in Qwen3-VL.

Log actual tokens before/after reduction per image/layer, text/output lengths, selection overhead, vision time, prefill, decode, end-to-end latency and peak memory. Time on a fixed device with warmup, synchronization and repeated samples; distinguish fixed-generation microbenchmarks from natural task latency. Preserve the dense FlashAttention 2 baseline and measure any required backend change: exposing full attention matrices can erase pruning gains.

Confirm selected QAT W4 comparisons on the existing seven-suite protocol plus a detail-sensitive task such as TextVQA or DocVQA, keeping new task scores separate from the established mean. QAT W8 is the quality reference; BF16 and existing GRACE sweeps are secondary analyses if needed, not a mandatory five-model grid. Preserve each reference's native processor and describe unequal-budget comparisons as checkpoint-plus-processor behavior, not a precision-only effect. Record the result even if inference-only composition does not help; use the failure diagnosis to decide whether joint QAT could reasonably compensate.''')
section('### C.', '## Mathematical hypotheses', '''### C. Integrate token reduction into CE-only QAT W4

The core study now compares **sequential composition with joint QAT**, rather than requiring four QAT/GRACE training arms. Choose one reducer, insertion layer and retention budget on development data. Reuse the completed unpruned QAT W4 checkpoint when the verified implementation/export path is unchanged. Train the joint arm from the **same released BF16 SFT initialization**, matching the saved manifest/order, LoRA scope, optimizer, seed, batch 128 and 256 updates. The image ceiling remains 576 and the total training cap 4,096. Do not add extra adaptation to the existing QAT checkpoint and describe it as an equal-exposure comparison without a matched continuation control.

| Comparison | Pruning during training | Objective | Evaluation | Role |
| --- | --- | --- | --- | --- |
| A: completed QAT W4 | None | CE-only QAT | Full tokens and selected pruning budget | Reused unpruned control and sequential-composition baseline |
| B: joint QAT W4 + reducer | Selected budget | CE-only QAT | Selected budget; full-token diagnostic | Tests adaptation to reduced inputs |

**B versus A at the same inference retention budget** tests the benefit of joint training. Both pruned variants must also be compared with unpruned A for quality and actual resource savings. A useful sequential result remains valuable even if B offers no additional gain. First compare equal data exposure/updates and report measured GPU-hours; an equal-compute study is a separate extension, not implied by equal update counts.

Before a full run, validate finite loss, effective scale/merger updates, memory, checkpoint/export behavior and the retention-1.0 path. Keep every text/answer token, preserve original spatial identity, and make training selection causal without using future ground-truth answers. Multi-turn selection/cache behavior must match deployment. Hard top-k has no ordinary gradient through the chosen indices; QAT can adapt retained features and weights without learning the discrete selection rule. State this limitation explicitly.

**Optional later GRACE comparison:** only if results motivate a specific teacher-compensation hypothesis, add C (unpruned GRACE, reusing the completed control if valid) and D (GRACE with the same reducer), then compare D versus B. The full-resolution teacher must see the same source image/resolution; align answer logits by original supervised-token identity. For retained visual indices I, compare H_S with teacher rows H_T[I]; if merging is used, define a merge matrix M and compare with M H_T. A rectangular student/teacher Gram comparison is not valid CKA. These correspondence checks and controller diagnostics become required only for that extension. Existing GRACE deficits do not justify making new KD runs part of the first required matrix.

### D. Analyze the results against declared quality and resource targets

Answer two questions separately: **does LLMC+ token reduction give QAT W4 a useful quality/resource trade-off, and does joint QAT improve that trade-off over inference-only composition?** Before final evaluation, specify acceptable mean/task-specific quality loss, the target resource (latency, peak memory or both), a minimum useful improvement, development/confirmation splits and a total compute budget. Numerical thresholds are not yet selected; do not infer them from final test outcomes.

Report the full-token control, inference-only pruning, matched-count random/uniform control and joint QAT together. Include per-task scores, the unchanged seven-suite mean, actual retained-count distributions and measured costs. Keep MMStar math/fine-grained categories, HallusionBench grouped metrics and separate OCR/document diagnostics visible. For multi-turn diagnostics, report absolute turn accuracy alongside conditional consistency. Stratify by original visual count, text length and task so an aggregate gain cannot hide failures on small images or detail-heavy inputs.

Use paired, appropriately grouped bootstrap intervals for evaluation uncertainty, grouping repeated questions/rotations and shared images. Training-seed uncertainty requires additional training seeds. Preserve a genuinely held-out confirmation set because existing test results already informed baseline selection. Analyze selection overlap/ranking margins, activation distortion and component timing to test the hypotheses below rather than selecting an explanation only after a favorable score.

Fewer tokens or theoretical FLOPs alone do not establish faster inference. Attribute latency gains from token reduction under the current floating execution separately from packed INT4/INT8 acceleration. The existing simulated low-bit exports do not establish smaller deployed weight storage. Validate packing, supported kernels and export equivalence before claiming combined deployment compression.

### E. Let the evidence determine the next experiment

| Observed result | Next step |
| --- | --- |
| Useful quality/resource improvement; joint QAT improves on sequential composition | Expand analysis of the selected method: paired seeds, held-out/detail/multi-turn confirmation, then a bounded retention/layer sweep and QAT W8 comparison. Study which tasks, token counts and quantization errors explain the gain. |
| Inference-only pruning helps, but joint QAT adds no useful gain | Keep the simpler sequential composition as the successful result. Investigate the missing adaptation benefit with a focused diagnostic before spending more training hours. |
| Quality regresses beyond the declared limits | Inspect lost detail, token selection, causal/cache correctness and quantization sensitivity. Try one targeted change—retention, layer, an alternative pruning/merging rule, or quantization-aware selection—at a time. |
| Tokens/FLOPs fall but measured latency or memory does not improve | Profile selector, vision encoder, attention, MLP and backend overhead. Change implementation or choose a method targeting the measured bottleneck before broadening evaluation. |
| Selected token-reduction methods remain ineffective after bounded checks | Investigate an alternative pruning method or sparse attention on the same 576-cap QAT W4 control; record the negative result and its limits. |

For a sparse-attention branch, distinguish **removing tokens** from **restricting token-to-token connections**. State whether the intervention targets the vision encoder or language model. Preserve causality and compare an actual sparse kernel with the existing dense FlashAttention 2 execution; a dense mask alone is not evidence of acceleration. Evaluate whether keeping more tokens with fewer connections beats pruning at matched measured resources. Our training sequences average only about 479 total tokens, so short-sequence overhead and evaluation-length distributions must be measured before committing to this branch. Sparse attention does not by itself reduce weight storage or necessarily shrink the KV cache.

The aim is stronger combined compression at acceptable quality, not the smallest token count in isolation. Keep the 576 cap fixed for these comparisons. A future larger-image-budget study must be separately motivated by the results and recorded as a new experiment; it is not the automatic response to disappointing pruning results. Stop a branch when bounded diagnostics show no useful resource/quality benefit.''')
s = s.replace('**H3: teacher guidance can compensate for missing context, but can also conflict with CE.**', '**H3 (conditional KD extension): teacher guidance can compensate for missing context, but can also conflict with CE.**')
s = s.replace('Recommended sequence and exercises are in [the reading guide](llmc_qat_reading_guide.md).', 'Recommended sequence and exercises are in [the reading guide](llmc_qat_reading_guide.md). For this revision, prioritize LLMC+ configuration/selector mechanics, FastV/DART and Qwen3-VL mapping, followed by LSQ/export arithmetic and hypotheses H1/H2/H4. Study DKD/CKA correspondence if the optional KD branch is activated; study sparse attention and kernel overhead if the alternative-method branch is selected.')
s = s.replace('**under a small adaptation budget, when does joint weight-only QAT and visual-token reduction outperform sequential composition, and when does distillation help or harm?**', '**under a small adaptation budget, when does joint weight-only QAT and visual-token reduction outperform sequential composition on the 576-cap Qwen3-VL workload?** Distillation and sparse attention are conditional extensions of this question.')
start = s.index('## Approved scope of the to-do update')
s = s[:start] + '''## Approved scope of the to-do update

The user requested this revision on **2026-10-12**. [The active to-do plan](todo_plan.md) remains the authoritative status checklist; historical results and limitations stay archived. The adopted sequence is:

- [x] Select the completed 576-cap / 4,096-total-token configuration, QAT W4 as the primary control and QAT W8 as the quality reference; defer further image expansion.
- [ ] Learn LightCompress in an isolated pinned environment and write a local usage/configuration guide.
- [ ] Reproduce the selected LLaVA-1.5-7B / FastV / DART token-only slice with measured quality and inference cost.
- [ ] Validate Qwen3-VL integration at retention 1.0, then test inference-only reduction on the completed QAT W4 checkpoint.
- [ ] Complete the relevant native/export and scale-update checks; specify one reducer/budget, quality/resource targets and compute allocation before joint training.
- [ ] Run the focused sequential-versus-joint CE-QAT W4 comparison, reusing the completed control when valid.
- [ ] Analyze quality, actual resource use and failure modes; expand promising findings or investigate a targeted alternative pruning/sparse-attention method according to Section E.
- [ ] Study the relevant theory during setup and job waits, maintaining a claim/evidence/falsification ledger; add KD work only if the results motivate it.

This request updates the plan; it does not launch LLMC+ installation, implementation or GPU workloads. The larger-budget training/evaluation is already complete. Survey GPU availability before later jobs. Future compute allocation and numerical quality/resource thresholds remain to be specified before execution.
'''
p.write_text(s)
print('Updated proposal.')
