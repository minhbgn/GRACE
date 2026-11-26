from pathlib import Path
root=Path('/raid/lvchengtao/Minh/Distillation/reproduction/GRACE')
report=root/'experiments/constrained_32k_seed42/experiment_report.md'
project=root/'qwen-vl-finetune'
def code(file,line,label=None):
    return f'[{label or Path(file).name+":"+str(line)}]({project/file}:{line})'
rows=[
('Student, teacher, subset and seeds', '2B-Instruct / 8B-Instruct; 32,768 records; training seed 42, subset seed 2026', code('scripts/run_constrained.py',25)+'; '+code('scripts/run_constrained.py',50), 'Read saved `comparison_recipe.json`; verify the GRACE-only teacher branch in '+code('qwenvl/train/train_qwen_constrained.py',239)+'.'),
('Batch size and duration', '2 ranks × micro-batch 1 × accumulation 32 = 64; 32,768 / 64 = 512 updates', code('scripts/run_constrained.py',33)+'; '+code('scripts/run_constrained.py',68)+'; '+code('scripts/run_constrained.py',95), 'The launcher fixes two ranks. ZeRO batch/accumulation fields are `auto`, resolved from Trainer; check runtime values rather than reading `auto` as a numeric setting.'),
('Image and sequence limits', '4,096–50,176 pixels; ≤1,024 input tokens', code('scripts/run_constrained.py',34)+'; '+code('qwenvl/data/data_processor.py',44), 'Processed tensors are checked in '+code('qwenvl/data/constrained_data.py',53)+'; complete trailing Q/A pairs are removed in '+code('qwenvl/data/constrained_data.py',91)+'.'),
('LoRA and trainable modules', 'Rank 32, alpha 64, dropout .05; q/k/v/o adapters and visual merger', code('scripts/run_constrained.py',36)+'; '+code('scripts/run_constrained.py',41)+'; '+code('qwenvl/train/train_qwen_constrained.py',55), 'Follow '+code('qwenvl/train/train_qwen_qat.py',116)+' then '+code('qwenvl/train/train_qwen_constrained.py',222)+': the later language-model/head freeze overrides `tune_mm_llm=True`. Inspect `trainable_parameters.json`.'),
('QAT scope and arithmetic', 'W4G128 on eligible language-model attention/MLP linears; vision, merger, embeddings and head skipped', code('qwenvl/train/qat_modules.py',312)+'; '+code('qwenvl/train/qat_modules.py',356)+'; '+code('qwenvl/train/qat_modules.py',169), 'Only adapted layers retain trainable scales; other eligible linears use frozen quantizers. The quantizer rounds/clamps weight codes and dequantizes for floating-point `F.linear`.'),
('Optimizer and schedule', 'LoRA LR 2e−4, scale LR 2e−4, merger LR 1e−5; cosine, 3% warmup, zero weight decay', code('scripts/run_constrained.py',38)+'; '+code('scripts/run_constrained.py',72)+'; '+code('qwenvl/train/train_qwen_grace.py',380), 'Scale groups are built in '+code('qwenvl/train/qat_modules.py',432)+'; the merger is split into its own LR group. Inspect actual optimizer groups and scheduler/checkpoint state.'),
('Precision, memory and training mode', 'BF16, TF32, gradient checkpointing, flattened attention, no data packing, ZeRO-2', code('scripts/run_constrained.py',65)+'; '+code('scripts/run_constrained.py',71)+'; '+code('scripts/zero2_lora_memory_safe.json',2), 'BF16 storage/compute and fake quantization do not imply a packed INT4 kernel. Flattening is a collator/attention representation; it does not relax the per-record token cap.'),
('Method selection and objective', 'SFT/QAT: CE; GRACE: CE + β·GDKD + 3·warmup·RCKA', code('qwenvl/train/train_qwen_constrained.py',202)+'; '+code('qwenvl/train/train_qwen_grace.py',427)+'; '+code('qwenvl/train/train_qwen_grace.py',519), 'SFT uses '+code('qwenvl/train/train_qwen_constrained.py',49)+' without fake quantization. QAT shares the CE branch but its linear modules still quantize.'),
('GDKD settings and supervision', 'Temperature 2, target/non-target weights 1/4; 128-token chunks; checkpointed chunks; no DKD warmup', code('scripts/run_constrained.py',43)+'; '+code('scripts/run_constrained.py',86)+'; '+code('qwenvl/train/grace_modules.py',105), 'Check next-token shifting, `labels != -100`, entropy gates and normalization by gate sum. Answer masks are created in '+code('qwenvl/data/data_processor.py',234)+'.'),
('RCKA feature selection and ramp', 'Layer index −2; image-token relationships; coefficient rises from 0 to 3 over 50 optimizer updates', code('scripts/run_constrained.py',45)+'; '+code('qwenvl/train/train_qwen_grace.py',131)+'; '+code('qwenvl/train/train_qwen_grace.py',148)+'; '+code('qwenvl/train/grace_modules.py',237), 'Inspect selected student hidden-state/teacher hook outputs and image grid boundaries. Require ≥4 tokens and nondegenerate Gram matrices. A nonzero raw RCKA at global step 0 has zero weighted contribution.'),
('IB controller', 'τ=3, η=.003, β starts .5 and stays within [.1,1]; EMA decay .99', code('scripts/run_constrained.py',46)+'; '+code('qwenvl/train/train_qwen_grace.py',184)+'; '+code('qwenvl/train/grace_modules.py',335), 'Launcher overrides the controller class defaults. Accumulation/rank averaging and update timing are in '+code('qwenvl/train/train_qwen_grace.py',491)+'.'),
('Completion and export', 'Native checkpoint first; merge adapters; QAT bake; save ordinary HF weights and quantization sidecar', code('qwenvl/train/train_qwen_constrained.py',184)+'; '+code('qwenvl/train/train_qwen_constrained.py',273)+'; '+code('qwenvl/train/train_qwen_constrained.py',290), 'Check `complete`, `profile`, global step and checkpoint state together. `EXPORT_COMPLETE` also exists for profile exports and is insufficient on its own.'),
]
section='''

## 6. Validate configuration and training behavior in code

Exported-model benchmark scores measure the quality of the final artifact. To understand how that artifact was produced, validate a chain of evidence: **saved configuration → executed training branch → processed inputs → loss and gradients → optimizer updates → native checkpoint → exported weights**. A declared setting, a nonzero loss, and an updated parameter are different observations; each needs its own check.

### 6.1 Identify the executed entry point and the historical configuration

This experiment launches `qwenvl.train.train_qwen_constrained`, as shown in ENTRY_LINK. It imports `GraceTrainer` and student/teacher helpers from `train_qwen_grace.py`, plus QAT utilities from `qat_modules.py`. The standalone `train()` functions in `train_qwen.py`, `train_qwen_qat.py` and `train_qwen_grace.py`, and the full-training Slurm scripts, are separate entry points. Their defaults do not establish this experiment's configuration.

Use [comparison_recipe.json](comparison_recipe.json) and each method's `runs/<method>/launch_recipe.json` as the historical configuration records. `config.json` describes the model/export; it is not a complete training recipe. For runtime evidence, also read `trainable_parameters.json`, `trainer_state.json`, `step_metrics.jsonl`, `run_summary.json` and the native checkpoint's scheduler, optimizer, RNG and GRACE IB state.

The source-hash check performed while adding this appendix found that **every recorded training/data Python file and the ZeRO JSON still matches the saved implementation hash, while `scripts/run_constrained.py` differs**. Therefore, the line references below describe the current files, and today's launcher dry-run must be compared with the saved recipe before using it as evidence of a historical run. The pinned hashes do not cover the entire installed Transformers/DeepSpeed implementation or prove the identity of model-weight files; preserve the environment and checkpoint provenance when reproducing the run.

### 6.2 Where settings are defined and where they take effect

Line links point to the current local workspace. Use the named function and the saved hash to relocate a reference if a file later changes.

| Configuration | This experiment | Definition / implementation | How to validate its effect |
| --- | --- | --- | --- |
'''.replace('ENTRY_LINK',code('scripts/run_constrained.py',95))
section+='\n'.join('| '+' | '.join(r)+' |' for r in rows)
section+='''

From `reproduction/GRACE`, these commands show the **current** resolved configuration and command without writing files or allocating GPUs:

```bash
conda activate grace
python qwen-vl-finetune/scripts/run_constrained.py --method sft --dry-run
python qwen-vl-finetune/scripts/run_constrained.py --method qat --dry-run
python qwen-vl-finetune/scripts/run_constrained.py --method grace --dry-run
```

Environment variables can override launcher defaults. Compare the printed values with the pinned recipe, including `GLOBAL_BATCH`, `MAX_PIXELS`, `MODEL_MAX_LENGTH`, learning rates and seeds; do not assume that the terminal environment is clean. The main runtime parser is PARSER_LINK, and method selection explicitly sets `grace_enable`/`qat_enable` after parsing.

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
record_bytes = json.dumps(manifest["records"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
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

For a deeper input check, reconstruct records through `ConstrainedDataset.__getitem__` at DATASET_LINK with the saved processor. It verifies image size/mtime, preprocessing token hashes, actual sequence length and visual grids. Inspect decoded non-ignored labels to confirm that supervision covers assistant answers and intended end markers, with prompts masked. The assistant-mask implementation uses specific token IDs, so merely finding some non-ignored labels is not enough to establish correct masking. Image size/mtime checks are not full image-content hashes.

### 6.4 Validate the forward pass, loss, backward pass and optimizer separately

**Follow the arithmetic of each method.** SFT computes `W·x + (alpha/r)·B·A·dropout(x)` with no fake quantization. QAT/GRACE compute `Q(W)·x + (alpha/r)·B·A·dropout(x)` at QAT_FORWARD_LINK, where `Q(W)` is groupwise rounded/clamped and dequantized. There is no activation quantization or packed INT4 GEMM in this path. Non-adapted eligible MLP linears also quantize, but their weights/scales are frozen; their first-forward cache can change the stored weight by materializing `Q(W)` without any optimizer update. Compare frozen values after runtime setup and first-use materialization, rather than misclassifying that initialization as learning.

**Check teacher and loss inputs.** TEACHER_LINK freezes the BF16 teacher, switches it to evaluation mode and attaches its hidden-state hook. The teacher forward at TEACHER_FORWARD_LINK runs under `no_grad` on the student's input tensors with labels removed; it uses teacher forcing, not generated teacher answers. GDKD aligns next-token logits at supervised answer positions and applies normalized entropy-based weights, rather than dropping examples with a hard confidence threshold. RCKA compares centered visual-token Gram matrices per image; it is not a pixel reconstruction loss or text-token alignment. Inspect layer indices and captured tensor shapes against the model implementation, since the student's hidden-state tuple and teacher's decoder-module hook are different indexing mechanisms.

**Check the weighted objective and controller timing.** The loss is `CE + beta × dkd_scale × GDKD + 3 × rcka_scale × RCKA`. Raw logged loss magnitudes do not establish gradient dominance. The IB update is `beta = clip(beta + .003 × (EMA(GDKD) − 3), .1, 1)` and therefore increases the distillation weight when GDKD is above target. It fires at the accumulation boundary after rank averaging, inside `compute_loss`, before constructing the final micro-batch loss; the new beta can apply to that last micro-batch while earlier micro-batches used the preceding beta. `step_metrics.jsonl` records the last micro-batch's `_grace_logs` at each optimizer boundary, not the mean CE/GDKD/RCKA over all 64 examples. Reconstructing the controller requires accumulation/rank means, which cannot be recovered exactly from these rounded last-micro-batch rows alone.

**Check gradient flow and actual updates.** The existing diagnostic hooks are DIAGNOSTICS_LINK. Inspect `profiling/learning_diagnostics.jsonl` and `learning_diagnostics_summary.json` for teacher trainability, backward signals, loss coefficients and sampled parameter deltas. With ZeRO, an empty `.grad` after a step is not sufficient evidence of disconnection: gradients can be partitioned or cleared. Hooks before the optimizer, plus optimizer-state/parameter comparisons, are stronger evidence. Expect some initially zero LoRA-A gradients because B starts at zero; assess multiple steps. The diagnostic parameter deltas are sampled changes from the start, and gradient norms combine micro-batch backward signals; they are not exhaustive tensor comparisons or the final clipped optimizer-gradient norm.

For any trainable category with gradients but no measured update, verify membership and LR in the actual optimizer groups, scheduler LR at that step, whether the optimizer stepped, and parameter/master-weight precision. BF16 rounding can hide small changes. Use full-tensor comparisons and FP32 optimizer state over more steps before concluding that scales or the merger are either frozen or successfully updated. If the concern is which loss changes a particular parameter, perform per-term gradient checks on a small diagnostic batch and compare CE-only, CE+GDKD and CE+RCKA branches with controlled weights, inputs and dropout/RNG. A loss-output backward hook establishes participation in the combined objective, not its individual gradient contribution to every trainable module.

Non-finite distillation terms are replaced with a detached zero at SANITIZE_LINK; missing/degenerate RCKA inputs can also produce zero. Inspect warnings, actual tensors and term-gradient connectivity rather than interpreting every logged zero as healthy convergence. Loss normalization also matters: LOSS_NORMALIZATION_LINK disables accumulated-item loss kwargs so Trainer applies gradient accumulation to this per-example loss.

### 6.5 What the saved profile already proves, and what remains unresolved

The [three-step GRACE diagnostic summary](profiles/grace/profiling/learning_diagnostics_summary.json) records a frozen teacher, nonzero LoRA-B and QAT-scale backward signals, both GDKD and post-warmup RCKA connected to backward, a changed sampled LoRA-B tensor, and an unchanged sampled frozen MLP scale. However, **`qat_scale_parameter_changed` is false**, and the sampled trainable scale and merger norm show zero delta over the three-step profile. That does not prove all scales/merger tensors were unchanged, but it also does not establish successful updates for those categories. The full 512-step runs have inventories and step logs, but no saved learning-diagnostics summary; the short profile cannot prove their entire optimization history.

The diagnostics refresh their baseline after framework dtype/device setup at BASELINE_REFRESH_LINK, preventing setup casting from being mistaken for learning. Profile datasets intentionally repeat the eight longest retained records at PROFILE_DATA_LINK. They exercise the memory and graph envelope, not the representative training distribution or convergence. Timing summaries identify where compute occurs, while gradient/update diagnostics address whether learning occurs; neither substitutes for the other.

For additional dynamic checks, use the launcher's `--profile` mode in a **fresh `EXPERIMENT_DIR`**, compare its dry-run configuration with this pinned recipe, and survey GPUs before launching. The launcher enables learning diagnostics only for GRACE profile runs at PROFILE_FLAG_LINK; enabling equivalent checks for SFT/QAT requires an explicit diagnostic configuration change. Do not reuse the completed experiment directory or interpret profile exports as final trained results. No new training or GPU run was launched to add this appendix.

### 6.6 Validate native checkpoints against exports before attributing a score gap to training

The native [GRACE checkpoint-512](runs/grace/checkpoint-512/) retains adapters, QAT scales and DeepSpeed optimizer state. Restore it through the matching constrained model/wrapper and native resume path; loading it as an ordinary unwrapped HF model may discard adapter/scale keys and test a different function. Inspect missing/unexpected keys, restored global step, optimizer/scheduler, RNG and `grace_ib_state.json`.

Compare a fixed validation batch in three states: the reconstructed native model before merge, after LoRA merge, and after QAT bake/HF reload. Switch to evaluation mode, freeze dropout/RNG choices, use the same processor, weights and inputs, and measure next-token logits, CE and task predictions in addition to accuracy. Export mutates the model in place, so use separate copies/restorations for each state. The merge is MERGE_LINK and the bake is BAKE_LINK: native QAT uses `Q(W) + delta`, while the merged export uses `Q(W + delta)`. Equality is not guaranteed because rounding is nonlinear; an observed difference must be measured rather than inferred to be the cause of a benchmark loss. SFT's merge should preserve its evaluation-mode linear function up to numerical precision.

Finally, check that the HF export has merged adapters, ordinary floating-point weights, the expected quantization sidecar, matching processor settings and no leftover adapter/scale keys in its HF weight file. Packed INT4 deployment requires a separate conversion and kernel validation. The [evaluation summarizer](../../evaluation/summarize_constrained_evals.py) validates final predictions and matching questions, but it does not test any of the training-gradient, optimizer-update or native-versus-export properties above. Only the combined evidence supports a claim about what the implementation actually learned.
'''
replacements={
'PARSER_LINK':code('qwenvl/train/train_qwen_constrained.py',194),
'DATASET_LINK':code('qwenvl/data/constrained_data.py',200),
'QAT_FORWARD_LINK':code('qwenvl/train/qat_modules.py',289),
'TEACHER_LINK':code('qwenvl/train/train_qwen_grace.py',293),
'TEACHER_FORWARD_LINK':code('qwenvl/train/train_qwen_grace.py',450),
'DIAGNOSTICS_LINK':code('qwenvl/train/grace_profiler.py',361)+'; optimizer-boundary callbacks are at '+code('qwenvl/train/grace_profiler.py',635),
'SANITIZE_LINK':code('qwenvl/train/train_qwen_grace.py',418),
'LOSS_NORMALIZATION_LINK':code('qwenvl/train/train_qwen_constrained.py',255),
'BASELINE_REFRESH_LINK':code('qwenvl/train/grace_profiler.py',479),
'PROFILE_DATA_LINK':code('qwenvl/data/constrained_data.py',192),
'PROFILE_FLAG_LINK':code('scripts/run_constrained.py',90),
'MERGE_LINK':code('qwenvl/train/qat_modules.py',456),
'BAKE_LINK':code('qwenvl/train/qat_modules.py',518),
}
for key,value in replacements.items():section=section.replace(key,value)
assert '## 6. Validate configuration' not in report.read_text()
report.write_text(report.read_text().rstrip()+section+'\n')
print('Appended configuration and training-validation guide.')
