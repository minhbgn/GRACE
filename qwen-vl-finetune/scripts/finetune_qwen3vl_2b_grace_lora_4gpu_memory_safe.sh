#!/usr/bin/env bash
# Four-GPU entry point for memory-safe QAT-LoRA GRACE (micro-batch 4).
# The shared launcher provides 2K tokens/example, reduced image resolution,
# FlashAttention, gradient checkpointing, recomputed DKD chunks, and small
# non-overlapping ZeRO-2 buckets. All environment overrides remain available.
# These reduce peaks; they cannot guarantee capacity against other GPU jobs
# or arbitrarily increased token/image budgets. Checkpoints remain native
# ZeRO-2 QAT-LoRA; final export merges LoRA as in the shared launcher.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if (( $# == 0 )); then
    set -- "${NPROC_PER_NODE:-4}"
fi
exec bash "${SCRIPT_DIR}/finetune_qwen3vl_2b_grace_lora_ngpu.sh" "$@"
