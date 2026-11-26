#!/usr/bin/env bash
# End-to-end two-A100 QAT-LoRA GRACE run.
#
# This wrapper keeps the full reference effective batch of 512:
#   2 GPUs x micro-batch 2 x gradient accumulation 128 = 512
# It delegates the model/training settings to the maintained LoRA launcher and
# explicitly enables the final LoRA merge, QAT bake, and standalone HF export.
#
# Physical GPU indices follow nvidia-smi/PCI order. On conti-sym7870, index 3
# is the DGX Display adapter, so select two A100 indices (for example 0,1).
#
# Examples:
#   ./scripts/finetune_qwen3vl_2b_grace_lora_2gpu.sh
#   CUDA_VISIBLE_DEVICES=2,4 ./scripts/finetune_qwen3vl_2b_grace_lora_2gpu.sh
#   MAX_STEPS=20 OUTPUT_DIR=/path/to/smoke-output \
#       ./scripts/finetune_qwen3vl_2b_grace_lora_2gpu.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GRACE_ROOT=${GRACE_ROOT:-"$(dirname -- "$(dirname -- "${SCRIPT_DIR}")")"}
BASE_LAUNCHER="${SCRIPT_DIR}/finetune_qwen3vl_2b_grace_lora_4gpu_memory_safe.sh"

QAT_BITS=${QAT_BITS:-4}
QAT_GROUP_SIZE=${QAT_GROUP_SIZE:-128}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-2}
TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH:-512}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [ "${#GPU_IDS[@]}" -ne 2 ]; then
    echo "ERROR: this launcher requires exactly two CUDA devices; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}." >&2
    exit 1
fi
for gpu_id in "${GPU_IDS[@]}"; do
    case "${gpu_id}" in
        ''|*[!0-9]*)
            echo "ERROR: invalid physical GPU index: ${gpu_id}" >&2
            exit 1
            ;;
        3)
            echo "ERROR: physical GPU 3 is the DGX Display adapter on this host; choose A100 indices from 0,1,2,4." >&2
            exit 1
            ;;
    esac
done

denominator=$((2 * PER_DEVICE_BATCH))
if [ $((TARGET_GLOBAL_BATCH % denominator)) -ne 0 ]; then
    echo "ERROR: TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH} is not divisible by 2 x PER_DEVICE_BATCH=${PER_DEVICE_BATCH}." >&2
    exit 1
fi
GRAD_ACCUM=${GRAD_ACCUM:-$((TARGET_GLOBAL_BATCH / denominator))}

FINAL_MODEL_NAME="Qwen3-VL-2B-GRACE-W${QAT_BITS}G${QAT_GROUP_SIZE}-LORA"
RUN_NAME=${RUN_NAME:-"${FINAL_MODEL_NAME}"}
OUTPUT_DIR=${OUTPUT_DIR:-"${GRACE_ROOT}/models/${FINAL_MODEL_NAME}"}

echo "Two-GPU end-to-end QAT-LoRA GRACE"
echo "  physical A100s=${CUDA_VISIBLE_DEVICES}"
echo "  micro_batch=${PER_DEVICE_BATCH} accumulation=${GRAD_ACCUM} effective_batch=$((2 * PER_DEVICE_BATCH * GRAD_ACCUM))"
echo "  final merged model=${OUTPUT_DIR}"

NPROC_PER_NODE=2 \
QAT_BITS="${QAT_BITS}" \
QAT_GROUP_SIZE="${QAT_GROUP_SIZE}" \
PER_DEVICE_BATCH="${PER_DEVICE_BATCH}" \
GRAD_ACCUM="${GRAD_ACCUM}" \
PROFILE_STEPS=0 \
PROFILE_DEEPSPEED_COMMS=False \
LEARNING_DIAGNOSTICS_STEPS=0 \
SKIP_FINAL_EXPORT=False \
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
"${BASE_LAUNCHER}"

CONFIG_PATH="${OUTPUT_DIR}/config.json"
WEIGHT_COUNT=$(find "${OUTPUT_DIR}" -maxdepth 1 -type f \
    \( -name 'model*.safetensors' -o -name 'pytorch_model*.bin' \) | wc -l)
if [ ! -f "${CONFIG_PATH}" ] || [ "${WEIGHT_COUNT}" -eq 0 ]; then
    echo "ERROR: training returned successfully but no standalone model was found in ${OUTPUT_DIR}." >&2
    exit 1
fi

python - "${OUTPUT_DIR}" <<'PY'
import json
import pathlib
import sys

from safetensors import safe_open

output = pathlib.Path(sys.argv[1])
config = json.loads((output / "config.json").read_text())
lora = config.get("qat_lora", {})
if not lora.get("merged", False):
    raise SystemExit("ERROR: final config does not mark qat_lora.merged=true")

for path in sorted(output.glob("model*.safetensors")):
    with safe_open(path, framework="pt", device="cpu") as handle:
        auxiliary = [
            key for key in handle.keys()
            if any(fragment in key for fragment in ("lora_A", "lora_B", "log_scale"))
        ]
    if auxiliary:
        raise SystemExit(
            f"ERROR: merged model still contains {len(auxiliary)} LoRA/QAT auxiliary tensors in {path.name}"
        )
print("Final export check PASS: merged=true and no LoRA/QAT auxiliary tensors remain.")
PY

echo "DONE: final merged standalone model is ready at ${OUTPUT_DIR}"
echo "NOTE: checkpoint-* subdirectories remain resumable native QAT-LoRA checkpoints."
