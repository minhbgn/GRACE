# Memory-safe QAT-LoRA launcher

From the repository root:

```bash
PER_DEVICE_BATCH=4 bash reproduction/GRACE/qwen-vl-finetune/scripts/finetune_qwen3vl_2b_grace_lora_4gpu_memory_safe.sh
```

The configurable `finetune_qwen3vl_2b_grace_lora_ngpu.sh` uses the same
protections by default. Four GPUs use accumulation 32 for global batch 512
unless `GRAD_ACCUM` is explicitly set. The default physical GPU pool remains
`0,1,2,4`; override `CUDA_VISIBLE_DEVICES` as needed.

The recipe retains the full-training memory-safe launcher's 2048-token
per-example limit, 50176-pixel per-image budget, flattened FlashAttention,
gradient checkpointing, and checkpoint interval of 50 optimizer steps.
For LoRA it uses ZeRO-2 with 5-million-element communication buckets and
communication overlap disabled. ZeRO stage and native checkpoint format
remain unchanged from the previous LoRA launcher, as does the default output
directory and automatic resume behavior.

`DKD_CHECKPOINT_CHUNKS=True` moves supervised-position selection into
checkpointed sequence chunks. Backward recomputes the KL intermediates rather
than retaining every chunk's softmax tensors. This trades computation for
lower activation memory without changing the distillation objective.
`DKD_CHUNK_SIZE` defaults to 256; 128 reduces chunk workspace further.
Original full student/teacher logits and student gradients still occupy
memory proportional to the total sequence length.

This is not a fixed VRAM reservation or a worst-case startup preflight.
It addresses retained DKD intermediates and communication buffers, but GPU
capacity still needs validation for the chosen data budgets and concurrent
workloads. Changes take effect in newly started training processes.

CPU regression checks (in the GRACE environment):

```bash
python reproduction/GRACE/tests/test_grace_dkd_memory.py
```

The checks cover loss/gradient equivalence in FP32 and BF16, ignored labels,
noncontiguous inputs, inference, invalid chunks, and absence of retained
vocabulary-sized DKD intermediates. They do not measure end-to-end GPU peaks
or distributed throughput.
