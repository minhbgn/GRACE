# Pausing GRACE LoRA and switching GPU count

Use `TRAINING_BACKEND=ddp` from the beginning of a new run. The launcher still
uses ZeRO-2 by default for existing fixed-GPU workflows. The installed DeepSpeed
rejects loading ordinary ZeRO optimizer shards with a different data-parallel
world size. DDP saves an unsharded optimizer, allowing this switch without
resetting Adam moments, scheduler, global step, or the adaptive IB state.
DDP replicates optimizer state and gradients, so it needs more memory per GPU
than ZeRO-2. GPU memory and throughput must be checked on the actual workload.

Environment prerequisite: the checked local `grace` environment and repository
requirements use PyTorch 2.5.1 with Transformers 5.9.0. This Transformers version
refuses to load optimizer/RNG checkpoints with PyTorch <2.6. Before starting,
prepare a training environment with PyTorch >=2.6, matching torchvision, and
FlashAttention rebuilt for that PyTorch/CUDA combination. The launcher checks
this before starting even a fresh DDP run. Do not bypass the safety check. The
shared environment and repository dependency pins have not been changed.

From `qwen-vl-finetune`, with a compatible GRACE environment activated:

```bash
export TRAINING_BACKEND=ddp
export OUTPUT_DIR=/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/models/grace-lora-portable
# Choose GPUs after checking their availability yourself.
CUDA_VISIBLE_DEVICES=2,4 bash scripts/finetune_qwen3vl_2b_grace_lora_ngpu.sh 2
```

In another terminal, request a pause using the same output directory:

```bash
touch /raid/lvchengtao/Minh/Distillation/reproduction/GRACE/models/grace-lora-portable/STOP_AFTER_CHECKPOINT
```

The run finishes the next scheduled checkpoint (every 50 optimizer updates),
waits for all ranks, writes `PAUSED_AT_CHECKPOINT`, consumes the stop request,
and exits successfully without the final merge/export. Wait for the launcher
to print `PAUSED` and exit before restarting. Do not use Ctrl-C to request this
pause and do not run two launchers against the same output directory.

```bash
# In the original shell, OUTPUT_DIR and TRAINING_BACKEND are still exported.
CUDA_VISIBLE_DEVICES=0,1,2,4 bash scripts/finetune_qwen3vl_2b_grace_lora_ngpu.sh 4
```

The latest checkpoint is resumed automatically. Defaults are `SAVE_STEPS=50`
and `SAVE_TOTAL_LIMIT=5`; full resumable checkpoints are retained. With
`PER_DEVICE_BATCH=4` and `TARGET_GLOBAL_BATCH=512`, automatic accumulation is:

| GPUs | Micro-batch per GPU | Accumulation | Effective batch |
| --- | --- | --- | --- |
| 2 | 4 | 64 | 512 |
| 4 | 4 | 32 | 512 |

Do not carry a manually exported `GRAD_ACCUM=64` into the 4-GPU restart. Let the
launcher compute it. Keep the same model, dataset, LoRA/QAT configuration,
micro-batch, learning rates, and training duration. In particular, `MAX_STEPS`
is the total target step, not additional steps after resuming. Do not lower it
to pause: that also changes the cosine schedule. Checkpoint recipe validation
rejects incompatible changes and rejects legacy checkpoints without a recipe.
It compares settings and paths, not file contents: do not modify the dataset or
model files in place between launches. Keep the same software environment.

GPU-count changes do not reproduce a bitwise-identical uninterrupted run.
Rank-local RNG state cannot be mapped exactly onto new ranks, and sample
assignment and floating-point reduction order can change. Trainer uses the
restored global step and the new accumulation to skip consumed batches; epoch
end rounding can vary with world size. Keep `ignore_data_skip=False` (default).
GRACE's IB controller already updates once per optimizer step and its saved
beta/EMA are restored. A normal completed run still produces the merged export.

Legacy ZeRO checkpoints cannot be directly used for this DDP workflow. Retain
them for same-topology training using the previous launcher, or plan an explicit
checkpoint migration; loading only model weights would reset training state.

Validation: shell syntax and three CPU regression tests passed. A two-process
CPU Trainer run paused successfully at its scheduled checkpoint. The four-process
resume reached optimizer loading, where the installed PyTorch version was
rejected; a complete GPU/GRACE resume has not yet been validated.
