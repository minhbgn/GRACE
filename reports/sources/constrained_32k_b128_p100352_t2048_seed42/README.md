# Larger-batch/input GRACE experiment

This preset keeps the 32,768-example, seed-42, two-GPU LoRA comparison and changes:

| Setting | Previous experiment | New experiment |
| --- | ---: | ---: |
| Effective batch | 64 | **128** |
| Gradient accumulation (2 GPUs, micro-batch 1) | 32 | **64** |
| Maximum image pixels | 50,176 | **100,352** |
| Maximum total input tokens | 1,024 | **2,048** |
| One-pass optimizer updates | 512 | **256** |
| RCKA warmup updates (automatic proportional scaling) | 50 | **25** |

The pixel limit is image area, not one edge length. It allows up to 98 merged visual tokens with Qwen3-VL's patch/merge settings; actual counts depend on image dimensions. Training and the saved evaluation processor use the new limits. This tests three changes together; it cannot attribute an accuracy difference to one setting alone.

The manifest is prepared again under the new input limits. The same source quotas and subset seed are used, but changed eligibility can change selected examples, and retained conversation turns can differ. Do not copy the old `subset.json` or checkpoints. Existing shared training code and the old pinned recipe remain untouched.

## Commands

Run from the workspace root. Activate `grace`, or use its Python explicitly:

```bash
export PYTHON=/home/air/miniconda3/envs/grace/bin/python

# Inspect resolved settings without GPU access, preparation or output writes.
bash reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained_b128_p100352_t2048.sh all --dry-run

# Validate/select the new shared subset on CPU; no model allocation.
bash reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained_b128_p100352_t2048.sh grace --prepare-only

# Choose TWO free A100 GPU IDs in PCI order; inspect ownership first.
gpuwho
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1  # replace with two free A100 IDs

# Start with GRACE to test the most memory-intensive method (10 updates).
bash reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained_b128_p100352_t2048.sh grace --profile

# Then collect the two matched controls, if the GRACE profile completed.
bash reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained_b128_p100352_t2048.sh sft --profile
bash reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained_b128_p100352_t2048.sh qat --profile

# Produce a combined table, with the previous experiment's stress profiles.
"$PYTHON" reproduction/GRACE/qwen-vl-finetune/scripts/summarize_constrained_runtime.py \
  --experiment reproduction/GRACE/experiments/constrained_32k_b128_p100352_t2048_seed42 \
  --baseline reproduction/GRACE/experiments/constrained_32k_seed42 --require-complete
```

`all --profile` also runs SFT, QAT and GRACE sequentially and produces the combined table. It needs both selected GPUs free. The wrapper surveys ownership if `gpuwho` is available; the existing launcher requires at least 36 GiB free per selected A100. It never kills other jobs.

The preset is [the environment configuration](../../qwen-vl-finetune/scripts/recipes/constrained_32k_b128_p100352_t2048_seed42.env). `EXPERIMENT_DIR` can select a fresh directory. Set `PROFILE_STEPS` before the first profile to change its length; do not change it while resuming an existing profile. Completed profiles are protected against overwriting. For a repeat, use a new experiment directory.

## What gets measured

- `profiles/{method}/step_metrics.jsonl`: rank-max update/cycle timing and peak allocated/reserved memory.
- `profiles/{method}/run_summary.json`: completion, stop reason and recorded training runtime.
- `profiles/{method}/EXPORT_COMPLETE`: confirms the merge/export path completed.
- `profiles/runtime_memory_{methods}.md` and `.json`: summary and optional baseline comparison. The first update of each invocation is excluded from timing medians; memory peaks include all logged updates.

The existing profile repeats the eight longest retained examples and saves every update. It exercises long-input memory and the export path; it is **not representative throughput**. Cycle times include checkpoint work. The report also gives update-only timing, examples/second and a rough 256-update time projection. Detailed GRACE diagnostics add overhead. Loading/export peaks and total device occupancy are not captured by these training-loop PyTorch metrics.

Batch accumulation does not hold 128 examples' activations simultaneously. Peak memory is primarily affected by the per-pass sequence/image sizes, teacher, gradients and optimizer state. The previous GRACE run's 24.13 GiB reserved memory is a useful measured starting point, but does not guarantee this profile fits. The 34 GiB memory and five-hour budgets pause at completed updates; an actual OOM can occur before those checks.

## Full runs after profiling

```bash
# Inspect the full-training command first.
bash reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained_b128_p100352_t2048.sh grace --train --dry-run

# Train the three matched methods sequentially; 256 updates each.
bash reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained_b128_p100352_t2048.sh all --train

# If a run pauses on a budget, resume that method with the same configuration.
bash reproduction/GRACE/qwen-vl-finetune/scripts/run_constrained_b128_p100352_t2048.sh grace --train --resume
```

Full runs write under `runs/`, separately from profiles. Summarizing with `--kind runs` compares actual-data timing to the old actual-data runs. Automatic summaries stop the wrapper if a method paused or failed to export. Keep the comparison's other settings fixed. At batch 128 there are half as many optimizer/controller updates per pass; the launcher also scales RCKA warmup to 25 updates. Neither fact is a claim that the larger batch improves accuracy.

For local evaluation of the new final exports:

```bash
bash reproduction/GRACE/experiments/constrained_32k_b128_p100352_t2048_seed42/run_local_evaluation.sh --check
bash reproduction/GRACE/experiments/constrained_32k_b128_p100352_t2048_seed42/run_local_evaluation.sh
```

The wrapper uses the shared scripts in `evaluation/` and runs the same seven-suite
comparison as `constrained_32k_seed42`: the original vanilla model plus the final
`runs/{sft,qat,grace}` exports. On success it writes the merged results and validated
`comparison.md`/`comparison.csv` under this experiment's `evaluation/` directory.
Normal runs are resumable: completed merged results are skipped, incomplete suites are
retried, and a failed suite does not prevent the remaining suites and methods from being
attempted. The command still exits nonzero and prints a failure list if anything failed;
run it again to retry those suites. The comparison files are written only after all 28
results are complete. Use `--dry-run` to print the 28 evaluation commands. `OUT_DIR`,
`NPROC`, `CUDA_VISIBLE_DEVICES`, `ATTN`, and `PYTHON` remain overridable; the four-method
set is intentionally pinned. Use the final `runs/` exports for accuracy; the `profiles/`
models are stress-test artifacts.
