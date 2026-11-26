from pathlib import Path
p = Path('plans/todo_plan.md')
s = p.read_text().replace('released_bf16_init_32k_b128_p589824_t2048_seed42', 'released_bf16_init_32k_b128_p589824_t4096_seed42')
s = s.replace('**Current experimental control:** CE-only LoRA-QAT initialized from the released BF16 SFT checkpoint (`GRACE-BF16`); LoRA-GRACE retained as a matched experimental arm.', '**Current experimental control:** completed **576-cap / 4,096-total-token CE-only QAT W4**, initialized from the released BF16 SFT checkpoint (`GRACE-BF16`); QAT W8 is the quality reference. Existing GRACE checkpoints are secondary comparisons; new KD training is conditional.')
s = s.replace('## Current focus — approved 2026-10-05', '## Current focus — revised by user request 2026-10-12')
a = s.index('The user approved this sequence:')
b = s.index('Earlier week-based checklists', a)
s = s[:a] + '''The adopted sequence is **keep the completed 576-cap configuration → learn LLMC+ and reproduce a token-only slice → incorporate a selected reducer with QAT W4 at inference → test joint CE-only QAT → analyze and branch on the results**. Positive results lead to deeper analysis and confirmation; mixed or negative results lead to a bounded diagnosis and, if needed, alternative pruning or sparse attention. The original four-arm joint-training requirement is superseded by the focused QAT comparison; new GRACE training is optional. Further image-budget expansion is deferred.

This update changes the plan, not the completed experiments. It does not launch installations or GPU jobs. The historical MMBench-bookkeeping and configuration-cleanup tasks are removed from active prerequisites as requested; their report limitations remain archived. Future compute allocation and numerical quality/resource thresholds remain to be specified before execution.

''' + s[b:]
s = s.replace('### Immediate priority — larger visual budget, requested 2026-10-05', '### Completed larger-budget controls and selected configuration')
s = s.replace('- [ ] Discuss the completed report with the supervisor before LLMC+ integration; retain QAT as the main control and investigate MMStar regressions.', '- [x] Adopt the completed 576-cap / 4,096-total-token recipe as the next-study baseline: QAT W4 primary, QAT W8 quality reference, existing GRACE secondary. Preserve the saved manifest/order, batch 128, 256 updates and seed 42. Keep MMStar regressions visible in analysis; new KD diagnostics are conditional.')
s = s.replace('This raises the unpruned input ceiling; it does not implement token reduction or guarantee redundancy. Keep native processors fixed within each new retention sweep. Larger images compete with conversation tokens; the user has now raised the total limit to 4,096 as well. Report both changes.', 'The selected native processor bounds are **4,096 / 589,824 pixels**. The ceiling is not a fixed 576-token input or proof of redundancy. Keep the processor fixed within each retention sweep and measure evaluation token distributions separately. QAT took about **1.96 hours on two A100s**, using **9.55 GiB peak reserved per GPU**: time increased **1.2–2.0%**, memory **12.7%** versus the old controls. These are training measurements, not measured inference savings.')
a = s.index('### 1. Close bounded baseline checks')
b = s.index('\n---\n\n## A. Foundation', a)
s = s[:a] + '''### 1. Complete bounded integration checks alongside toolkit learning

- [ ] Compare the native QAT W4 checkpoint and merged/baked export on fixed development inputs; measure logits, CE and predictions so sequential and joint comparisons use the same verified computation.
- [ ] Verify full-tensor scale/merger updates and optimizer state before interpreting joint-QAT compensation. If a correction changes computation, create matched corrected controls separately and preserve the completed baseline.
- [ ] **Only if returning to KD:** extend export checks to GRACE and inspect CE/GDKD/RCKA gradients, beta saturation and teacher suitability before selecting one loss diagnostic.

LLMC+ learning/reproduction and inference-only composition can proceed independently of native training checks. Historical bookkeeping recovery and documentation cleanup are no longer active prerequisites; KD diagnostics do not block CE-only QAT.

### 2. Learn LLMC+ and reproduce a selected token-only slice

- [ ] Clone [LightCompress](https://github.com/ModelTC/LightCompress) into `reproduction/LLMC/`; pin a paper-compatible commit and isolated environment.
- [ ] Learn the model/configuration/selector/evaluation path and write a local usage guide with full-token and reducer commands, configuration fields, output locations, insertion-layer numbering, and retained-versus-removed fractions.
- [ ] Distinguish token reduction from the toolkit's quantization/calibration path; preserve our existing QAT weights and scales during later composition.
- [ ] Freeze the reproduction manifest: LLMC+ v2, LLaVA-1.5-7B checkpoint/processor, datasets, metrics, prompts, decoding, attention backend and exact pruning semantics.
- [ ] Survey GPU availability, profile batch-one representative/longest inputs and estimate evaluation GPU-hours before expanding. Repeat the availability survey before later GPU workloads.
- [ ] Validate the fixed 576-token LLaVA reference, then FastV and DART retaining 192, 128 and 64 tokens: seven configurations targeting the selected Table 5 slice.
- [ ] Evaluate GQA, MMBench EN, MME, POPE, TextVQA, VizWiz and ScienceQA with the paper's definitions. Label reduced coverage as partial reproduction; predeclare per-suite tolerances before observing results.
- [ ] Record scores, actual tokens, latency and memory, plus bounded detail/multi-turn diagnostics. Explain the selected method before porting it to Qwen3-VL.

### 3. Incorporate LLMC+ token reduction with the completed QAT W4 checkpoint

- [ ] Recheck toolkit support at the pinned commit; map the reducer onto Qwen3-VL spatial merging, multimodal positions, masks, cache handling and DeepStack injection.
- [ ] Verify retention **1.0** matches the original QAT W4 outputs within a declared numerical tolerance; preserve the existing quantized computation rather than substituting PTQ.
- [ ] Screen one reducer at retained fractions **1.0, 0.75, 0.50 and 0.25**, with random or uniform spatial selection at matched counts. Choose the reducer/layer/budget on development data.
- [ ] Keep the **576 visual ceiling / 4,096 training-token cap** and native processor fixed. Log actual before/after counts, selection overhead, vision/prefill/decode/end-to-end time and memory against the dense FlashAttention 2 baseline.
- [ ] Confirm selected comparisons on all seven existing suites plus separate OCR/document diagnostics; preserve the existing seven-suite mean and evaluation policy.
- [ ] Analyze inference-only viability before a full joint run. Use QAT W8 as the quality reference; BF16 and existing GRACE sweeps are optional targeted comparisons, not a required five-model matrix. Unequal native processor budgets do not isolate precision effects.

### 4. Test sequential composition against joint CE-only QAT W4

- [ ] Finalize falsifiable hypotheses, development/held-out splits, one reducer/layer/budget, quality-loss/task guardrails, a minimum useful measured resource gain and total GPU-hour allocation.
- [ ] Validate causal selection without future-answer leakage, text/answer preservation, original positions/cache, finite losses, memory, scale/merger updates and export behavior.
- [ ] Train B from the same released BF16 SFT initialization with matched saved data/order, seed, trainable scope, optimizer, batch 128 and 256 updates; reuse completed A if computation/protocol is unchanged. Any extra continuation training requires a matched continuation control.

| Arm | Training tokens | Objective | Evaluation |
| --- | --- | --- | --- |
| A: completed QAT W4 | Full | CE-only QAT | Full and selected pruning budget |
| B: joint QAT W4 + reducer | Pruned | CE-only QAT | Selected budget; full-token diagnostic |

- [ ] Compare B versus inference-pruned A at the same retention budget for adaptation benefit; compare both with full-token A for the quality/resource trade-off. Equal updates establish equal exposure, not equal compute; report measured GPU-hours.
- [ ] Add new GRACE arms only if a specific hypothesis motivates them after this study. Then validate teacher/student answer identity and retained-token RCKA correspondence and compare pruned GRACE directly with B.

### 5. Analyze results and choose the next branch

- [ ] Compile unpruned A, inference-pruned A, matched-count random/uniform selection and B in one report: task scores, seven-suite mean, separate detail/multi-turn diagnostics, actual token distributions, latency components and peak memory.
- [ ] Preserve MMStar categories and HallusionBench grouped metrics; stratify by original visual count and text length. Use grouped paired evaluation uncertainty and distinguish it from training-seed variation.
- [ ] **If the quality/resource trade-off improves:** expand analysis and held-out confirmation; prioritize paired seeds, detail/multi-turn robustness, a bounded layer/retention sweep and then QAT W8. Connect observed gains to activation distortion, selection stability and measured bottlenecks.
- [ ] **If only inference-only composition helps:** retain that simpler result; investigate the absence of joint-training gain with one focused diagnostic.
- [ ] **If quality deteriorates:** diagnose lost detail, selection, placement and quantization sensitivity; change one factor or test one alternative pruning/merging rule at a time.
- [ ] **If theoretical savings do not yield actual savings:** profile selector/backend, vision, attention and MLP costs before expanding methods or training.
- [ ] **If bounded token-reduction attempts remain ineffective:** assess an alternative pruning method or sparse attention on the same QAT W4/576-cap control. Specify vision-encoder versus language-model scope, preserve causality and compare actual sparse execution with dense FlashAttention 2 at matched measured resources.
- [ ] Retain negative results, stop branches without useful benefit, and keep image-budget expansion separate and conditional on evidence.
- [ ] Validate export equivalence, packing and supported low-bit kernels before claiming deployed weight-storage savings or combined low-bit acceleration; token and attention sparsity alone do not establish those claims.

### 6. Study theory before implementation and during job waits

- [ ] Follow the [reading guide](llmc_qat_reading_guide.md), prioritizing LLMC+/FastV/DART mechanics, Qwen3-VL integration, LSQ/export arithmetic and pruning/quantization interaction (H1/H2/H4 in the proposal).
- [ ] Compare the already identified related work with our weight-only LoRA-QAT setting before asserting novelty; distinguish authors' findings from local evidence.
- [ ] Produce the reproduction manifest, toolkit usage guide, QAT/reducer integration map and quantization/export arithmetic note before joint training.
- [ ] Study DKD/CKA correspondence and gradient interaction only as required for the optional KD extension; study sparse connectivity, causal patterns and kernel overhead for the alternative-attention branch.
- [ ] Maintain a claim/assumption/prediction/falsification ledger and connect each analysis to a hypothesis.
''' + s[b:]
s = s.replace('see the 2026-10-05 current-focus section.', 'see the current-focus section revised 2026-10-12.')
p.write_text(s)

p = Path('CODEX.md')
s = p.read_text()
s = s.replace('- **Approved direction (2026-10-05):** bounded baseline checks, selected LLMC+ token-only reproduction, inference-only Qwen3-VL pruning comparisons, then the controlled QAT/GRACE joint-training study. See [the approved proposal](plans/3_llmc_grace_proposal.md).', '- **Current direction (revised by user 2026-10-12):** keep the completed 576-cap / 4,096-total-token recipe; learn LLMC+, reproduce a token-only slice, compose with QAT W4, then compare sequential pruning with joint CE-only QAT. Expand useful results or investigate alternative pruning/sparse attention after bounded failure analysis. QAT W8 is the quality reference; new GRACE/KD training and further image expansion are conditional. See [the approved proposal](plans/3_llmc_grace_proposal.md).')
a = s.index('Constrained GRACE/QAT experiments and several evaluation rounds')
b = s.index('\n## Repository map', a)
s = s[:a] + '''Constrained GRACE/QAT experiments are complete on **two A100 40GB GPUs**. Use the dated current-focus section of [plans/todo_plan.md](plans/todo_plan.md), rather than historical checklists. The selected baseline is `released_bf16_init_32k_b128_p589824_t4096_seed42`: maximum 589,824 pixels (576 merged visual tokens), minimum 4,096 pixels, 4,096 total training tokens, 32,768 saved examples, batch 128, 256 updates, seed 42. The experiment directory was previously named `..._t2048_seed42`; use the current `t4096` path for navigation without rewriting archived provenance.

All four training runs and 28/28 latest local evaluation pairs are complete and CPU-audited. QAT W4/W8 means are **75.77/77.37**, GRACE **74.56/74.86**, with GRACE−QAT **−1.22/−2.51 pp**. Average visual count is **254.64**, maximum total length **2,209**; one example was replaced and 99 positions changed relative to the old subset. QAT costs about 1.96 elapsed hours and 9.55 GiB peak reserved per GPU; its time rose 1.2–2.0% and reserved memory 12.7% versus old controls. GRACE costs about 2.50 hours and 27.13 GiB. These are training measurements, not inference speedups.

The user chose QAT W4 for the first LLMC+ integration study, with QAT W8 as the quality reference. Reuse completed controls when the verified computation/protocol is unchanged. The required joint study is unpruned/sequential QAT versus joint CE-only QAT; new GRACE training is optional. Historical MMBench bookkeeping recovery and old-report configuration cleanup are removed from active prerequisites by explicit user request; historical scores pass audit, while their missing markers remain recorded. Keep native/export and effective-update checks relevant to joint-training interpretation; defer teacher/controller checks unless returning to KD. Do not rerun completed work, expand the image cap by default, or claim the shared SFT initialization was distilled. No LLMC+ implementation or workloads have been launched by the planning revision.
''' + s[b:]
a = s.index('Follow the approved current-focus checklist', s.index('## Useful next actions'))
s = s[:a] + '''Follow [the current-focus checklist](plans/todo_plan.md): learn LightCompress and reproduce the selected token-only slice, validate inference-only composition on the frozen QAT W4/576-cap control, then run the focused sequential-versus-joint QAT comparison. Analyze measured quality and resource use before broadening successful findings or considering alternative pruning/sparse attention. Preserve the existing native evaluation policy, historical results and shared SFT initialization. A planning update alone does not launch workloads.
'''
p.write_text(s)
print('Updated to-do plan and project guide.')
