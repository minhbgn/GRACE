# Matched SFT / QAT / GRACE on two A100 40GB GPUs

These three launchers use one shared trainer, dataset manifest, seed, adapter
initialization, projector, optimizer grouping and training schedule:

| Launcher | Training objective | Student quantization |
|---|---|---|
| `finetune_qwen3vl_2b_sft_lora_constrained.sh` | CE | BF16 |
| `finetune_qwen3vl_2b_qat_lora_constrained.sh` | CE | W4G128 fake quantization |
| `finetune_qwen3vl_2b_grace_lora_constrained.sh` | CE + GDKD + RCKA, adaptive IB | Same W4G128 fake quantization |

Only GRACE constructs the frozen Qwen3-VL-8B teacher. All students start from
the original Qwen3-VL-2B-Instruct checkpoint, not a previously distilled model.
The existing full-dataset launchers remain separate.

## Default recipe

- Two GPUs, per-device batch 1, accumulation 32, global batch 64.
- 32,768 examples, one pass, exactly 512 optimizer updates.
- One image per example, no video or text-only records. Source proportions
  follow the eligible ShareGPT4V mix665k records; source is inferred from image
  path prefixes. This is source stratification, not verified task stratification.
- 1,024 total tokens including expanded visual tokens; 50,176 maximum image
  pixels and 4,096 minimum pixels (enough for at least four RCKA visual tokens).
- LoRA rank 32, alpha 64, dropout 0.05 on language-model q/k/v/o projections.
- Frozen language-model base weights, embeddings, output head and vision tower;
  the BF16 visual merger/projector trains in all three methods.
- Adapter LR 2e-4, projector LR 1e-5, quantization-scale LR 2e-4, cosine schedule,
  3% LR warmup, zero weight decay, max gradient norm 1.
- QAT/GRACE adapt the scales in LoRA-targeted layers. Other eligible language
  linears use frozen initialized scales, as in the existing native LoRA recipe.
- GRACE: T=2, DKD alpha=1/beta=4, RCKA weight=3, RCKA warmup=50 updates;
  IB tau=3, eta=.003, beta0=.5, beta bounds [.1,1], EMA decay=.99.
- BF16, FlashAttention 2, gradient checkpointing, no KV cache, ZeRO-2 with the
  existing 5M-element buckets and no communication overlap.
- Save every 50 updates, keep two native checkpoints and a final merged export.
- Pause and checkpoint after a completed update if PyTorch peak reserved memory
  exceeds 34 GiB or training-loop elapsed time reaches five hours per invocation.

The wall-time and memory figures are budgets, not measured guarantees. These
checks run at optimizer boundaries: they cannot rescue an allocation that has
already OOMed. Model loading, scale initialization, data preparation and final
export take additional time. Another user's allocation can also cause an OOM.
Run the profile first and reserve the two selected GPUs while training.

## Run from the repository root

Activate the existing GRACE environment, or set `PYTHON` to its executable:

```bash
conda activate grace
export PYTHON=python
export EXPERIMENT_DIR="$PWD/reproduction/GRACE/experiments/constrained_32k_seed42"
export CUDA_VISIBLE_DEVICES=0,1  # replace with two available A100 physical IDs
```

The defaults resolve model/data paths relative to `reproduction/GRACE`.
`BASE_MODEL_DIR`, `TEACHER_MODEL_PATH` and `SHAREGPT4V_ROOT` override those paths.
No models or evaluation datasets are downloaded by these scripts.

Inspect the command without loading models, touching GPUs or writing files:

```bash
bash reproduction/GRACE/qwen-vl-finetune/scripts/finetune_qwen3vl_2b_grace_lora_constrained.sh --dry-run
```

Prepare the shared subset on CPU once (no GPUs are needed for this command):

```bash
bash reproduction/GRACE/qwen-vl-finetune/scripts/finetune_qwen3vl_2b_sft_lora_constrained.sh --prepare-only
```

Preparation checks decoding, token length, image-grid alignment, RoPE indices,
and supervised labels. Long records lose complete trailing question/answer
pairs; records whose first pair cannot fit are rejected. Each source fills its
fixed quota from a deterministic shuffle. If a source cannot fill its quota,
preparation fails explicitly. Nothing silently substitutes a different record
at training time. The first ordinary/profile launch also prepares the subset
if it does not exist. All later methods reuse it.

Profile all three sequentially on the two selected GPUs, run from /raid/lvchengtao/Minh/Distillation/reproduction/GRACE/qwen-vl-finetune:

```bash
for method in sft qat grace; do
  bash "./scripts/finetune_qwen3vl_2b_${method}_lora_constrained.sh" --profile || break
done
```

Each profile executes three full optimizer updates on repeated longest retained
examples, saves native checkpoints, and exercises the final merge/export. It
uses the production micro-batch, accumulation and input bounds. This is a
stress profile, not a representative-data speed benchmark or an accuracy run.
Outputs go to `profiles/{sft,qat,grace}`, separate from real training.
`PROFILE_STEPS` can increase the number of profiling updates.

Inspect `run_summary.json`, `step_metrics.jsonl` and `EXPORT_COMPLETE` in each
profile directory. `complete=true` and `EXPORT_COMPLETE` are both needed.
GRACE additionally writes detailed diagnostics under `profiling/`; check that
CE, GDKD and RCKA are finite, RCKA is nonzero, and adapter/scale gradients exist.
`cycle_seconds` includes intervening checkpoint/data work; the detailed profiler
also reports data wait and rank-max timing. Exclude startup when extrapolating.
For 512 updates, a sum of about 105 seconds/update across the three methods
corresponds to 15 hours of training, leaving nine hours for preparation,
profiling, export, evaluation and contingency. Account for your actual I/O.

Then run the three training jobs, sequentially:

```bash
bash reproduction/GRACE/qwen-vl-finetune/scripts/finetune_qwen3vl_2b_sft_lora_constrained.sh
bash reproduction/GRACE/qwen-vl-finetune/scripts/finetune_qwen3vl_2b_qat_lora_constrained.sh
bash reproduction/GRACE/qwen-vl-finetune/scripts/finetune_qwen3vl_2b_grace_lora_constrained.sh
```

Check each summary for completion before proceeding. A budget pause returns
success after saving a checkpoint but has `complete=false` and no final export.
An existing completed output is never overwritten. Do not use the profile
models or incomplete runs in the final comparison.

## Pause, resume and change budgets

To pause at the next scheduled checkpoint:

```bash
touch "$EXPERIMENT_DIR/runs/grace/STOP_AFTER_CHECKPOINT"
```

Resume with the same environment/configuration and two available GPUs:

```bash
bash reproduction/GRACE/qwen-vl-finetune/scripts/finetune_qwen3vl_2b_grace_lora_constrained.sh --resume
```

Native checkpoints retain adapters, scales, DeepSpeed optimizer state, schedule,
RNG and (for GRACE) IB state. RNG and HF scheduler state are stored as JSON so
the pinned PyTorch 2.5 environment does not need to load these through
Transformers' pickle loader. DeepSpeed still uses its own native checkpoint
format. No safety-check bypass is installed. Distributed GPU resume still needs
the hardware preflight; CPU tests do not establish distributed restart parity.

`comparison_recipe.json` pins the shared configuration, subset checksum,
processor files, implementation hashes and torch/Transformers versions. Each
method also records its run recipe and trainable parameter inventory. A mismatch
fails before training; keep one experiment directory per comparison. Runs on
modified code or different seeds must use a new directory.

If the 32k profile is too slow, decide on the smaller comparison before viewing
benchmark outcomes:

```bash
export SUBSET_SIZE=16384
export EXPERIMENT_DIR="$PWD/reproduction/GRACE/experiments/constrained_16k_seed42"
```

All three methods then use 256 updates and RCKA warmup 25. The manifest is
reselected deterministically with smaller source quotas; it is not guaranteed
to be an exact nested subset because quotas use largest-remainder rounding.
Other shared overrides include `GLOBAL_BATCH`, `MODEL_MAX_LENGTH`, `MAX_PIXELS`,
`MIN_PIXELS`, `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`, `LEARNING_RATE`,
`MM_PROJECTOR_LR`, `QAT_SCALE_LR`, `SEED`, `SUBSET_SEED`, `SAVE_STEPS`,
`SAVE_TOTAL_LIMIT`, `MAX_TRAIN_HOURS` and `PEAK_MEMORY_GIB`. Export overrides
before launching any method. The GPU count and micro-batch remain two and one.

## Evaluate the actual exported models

Final HF model directories are `$EXPERIMENT_DIR/runs/{sft,qat,grace}`. The
`checkpoint-*` subdirectories contain native training modules and are **not**
plain HF inference models. The final exports include the processor with the
same pixel budget for all three methods.

For example, with an already available MMStar TSV:

```bash
export MMSTAR_TSV=/path/to/MMStar.tsv
for method in sft qat grace; do
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
    reproduction/GRACE/evaluation/MMStar/eval_mmstar.py \
    --model_path "$EXPERIMENT_DIR/runs/$method" \
    --tsv "$MMSTAR_TSV" \
    --output "$EXPERIMENT_DIR/mmstar_$method.jsonl" \
    --max_new_tokens 8 || break
done
```

Preselect the benchmark suite (e.g. MMStar, AI2D, image-containing ScienceQA),
check training/evaluation overlap, and use identical splits, prompts, resolution
and decoding. Existing AI2D and ScienceQA evaluators are under `evaluation/`.
Record inference errors alongside accuracy; do not silently omit failed cases.
For an untuned base-model comparison, explicitly apply the same evaluation pixel
budget: its original saved processor has a much larger default.

SFT is a BF16 quality reference; QAT versus GRACE is the matched low-bit
comparison. For a common W4 deployment comparison, separately quantize the SFT
export using a documented identical calibration/evaluation protocol. Do not call
the BF16 SFT row an INT4 baseline.

The native LoRA formulation is `Q(W)x + LoRA(x)`. Final export merges then
quantizes, giving `Q(W + delta_W)x`; these need not be equivalent. This is a
documented adaptation shared by QAT and GRACE, not exact full-weight GRACE.
Judge the primary low-bit accuracy from the final exports. Keep native
checkpoints if investigating conversion loss. Final QAT weights use BF16
storage on the quantization grid; they are not packed INT4 or evidence of INT4
memory/throughput gains. `qat_quantized_weights.bin` retains scales for the
repository's later packing workflow. LLMC+ integration is intentionally deferred
until this comparison establishes a usable development baseline.

Report all three outcomes, per-task scores, training time and peak memory.
One seed is a pilot, not evidence of seed robustness; a later matched seed repeat
is needed for that claim. No launcher selects configurations based on GRACE winning.

## CPU validation

```bash
CUDA_VISIBLE_DEVICES='' python reproduction/GRACE/tests/test_constrained_training.py
CUDA_VISIBLE_DEVICES='' python reproduction/GRACE/tests/test_grace_dkd_memory.py
CUDA_VISIBLE_DEVICES='' python reproduction/GRACE/tests/test_grace_resume.py
```
