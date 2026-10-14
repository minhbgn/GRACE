#!/usr/bin/env bash
# Opt-in BF16 SFT launcher for 4 x A100 40 GB.
#
# This file intentionally does not wrap or modify
# finetune_qwen3vl_2b_bf16_4gpu.sh. The original launcher remains the fallback.
# Memory-safe differences:
#   - micro-batch 2 with accumulation 32 (effective global batch 256)
#   - flattened/variable-length FlashAttention batches (no batch padding)
#   - non-offloaded ZeRO-3 instead of ZeRO-2
#   - Liger fused linear cross-entropy by default
#   - a separate output directory, so it cannot resume/overwrite the old run
#
# Optional experiments:
#   USE_LIGER=0 ...       isolate flattening + ZeRO-3 without Liger
#   MAX_STEPS=5 ...       short integration/throughput test
#   MODEL_MAX_LENGTH=2048 ... cap the very small long-context tail
#   MAX_PIXELS=50176 ...  use the lower-resolution Qwen/QAT image budget

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-"$(dirname -- "${SCRIPT_DIR}")"}
GRACE_ROOT=${GRACE_ROOT:-"$(dirname -- "${PROJECT_DIR}")"}

BASE_MODEL_DIR=${BASE_MODEL_DIR:-"${GRACE_ROOT}/models/Qwen3-VL-2B-Instruct"}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"${BASE_MODEL_DIR}"}
SHAREGPT4V_ROOT=${SHAREGPT4V_ROOT:-"${GRACE_ROOT}/ShareGPT4V"}
DATASETS=${DATASETS:-sharegpt4v_mix665k}

LEARNING_RATE=${LEARNING_RATE:-2e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-3}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-}

NPROC_PER_NODE=${NPROC_PER_NODE:-4}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-2}
GRAD_ACCUM=${GRAD_ACCUM:-32}

MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-8192}
MAX_PIXELS=${MAX_PIXELS:-200704}
MIN_PIXELS=${MIN_PIXELS:-784}
SAVE_STEPS=${SAVE_STEPS:-50}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}

# Liger >= 0.6.4 added Qwen3-VL support. Transformers applies the supported
# Qwen3-VL kernels, including fused linear cross-entropy, when this is enabled.
USE_LIGER=${USE_LIGER:-1}

RUN_NAME=${RUN_NAME:-Qwen3-VL-2B-SFT-BF16-MEMSAFE}
OUTPUT_DIR=${OUTPUT_DIR:-"${GRACE_ROOT}/models/${RUN_NAME}"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"${PROJECT_DIR}/scripts/zero3.json"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_HOME=${HF_HOME:-"${GRACE_ROOT}/.cache/huggingface"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"${HF_HOME}/datasets"}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-"${OUTPUT_DIR}/wandb"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=900
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

for executable in python torchrun; do
    command -v "${executable}" >/dev/null 2>&1 || {
        echo "ERROR: ${executable} is unavailable; activate the GRACE environment." >&2
        exit 1
    }
done

for path in \
    "${BASE_MODEL_DIR}" \
    "${SHAREGPT4V_ROOT}/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json" \
    "${SHAREGPT4V_ROOT}/data/sam/images" \
    "${DEEPSPEED_CONFIG}"; do
    [ -e "${path}" ] || { echo "ERROR: required path does not exist: ${path}" >&2; exit 1; }
done

LIGER_ARGS=()
case "${USE_LIGER,,}" in
    1|true|yes)
        if ! python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version

try:
    installed = version("liger-kernel")
except PackageNotFoundError:
    raise SystemExit(1)

if Version(installed) < Version("0.6.4"):
    raise SystemExit(1)
print(f"Liger Kernel: {installed}")
PY
        then
            echo "ERROR: USE_LIGER=1 requires liger-kernel>=0.6.4 in the active environment." >&2
            echo "Install it with: pip install 'liger-kernel>=0.6.4'" >&2
            echo "Or test flattening + ZeRO-3 alone with USE_LIGER=0." >&2
            exit 1
        fi
        LIGER_ARGS=(--use_liger_kernel True)
        ;;
    0|false|no)
        echo "WARNING: Liger is disabled; the large-vocabulary loss will use more memory." >&2
        ;;
    *)
        echo "ERROR: USE_LIGER must be one of 1/0, true/false, or yes/no." >&2
        exit 1
        ;;
esac

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${HF_DATASETS_CACHE}"

# Only resume checkpoints created by this launcher/output directory. In
# particular, do not load ZeRO-2 optimizer shards into this ZeRO-3 run.
LATEST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
if [ -n "${LATEST_CKPT}" ] && [ -d "${LATEST_CKPT}" ]; then
    echo "RESUME: loading memory-safe model and Trainer state from ${LATEST_CKPT}"
    MODEL_NAME_OR_PATH=${LATEST_CKPT}
    for filename in preprocessor_config.json video_preprocessor_config.json; do
        if [ ! -f "${LATEST_CKPT}/${filename}" ] && [ -f "${BASE_MODEL_DIR}/${filename}" ]; then
            cp -v "${BASE_MODEL_DIR}/${filename}" "${LATEST_CKPT}/${filename}"
        fi
    done
else
    echo "FRESH: starting memory-safe BF16 SFT from ${MODEL_NAME_OR_PATH}"
fi

LIMIT_ARGS=(--num_train_epochs "${NUM_TRAIN_EPOCHS}")
if [ -n "${MAX_STEPS}" ]; then
    case "${MAX_STEPS}" in
        *[!0-9]*|'') echo "ERROR: MAX_STEPS must be a positive integer." >&2; exit 1 ;;
    esac
    if [ "${MAX_STEPS}" -le 0 ]; then
        echo "ERROR: MAX_STEPS must be greater than zero." >&2
        exit 1
    fi
    LIMIT_ARGS=(--max_steps "${MAX_STEPS}")
fi

EFFECTIVE_BATCH=$((NPROC_PER_NODE * PER_DEVICE_BATCH * GRAD_ACCUM))
echo "Memory-safe BF16 SFT"
echo "  GPUs=${NPROC_PER_NODE} micro_batch=${PER_DEVICE_BATCH} accumulation=${GRAD_ACCUM} effective_batch=${EFFECTIVE_BATCH}"
echo "  flatten=True ZeRO=$(basename "${DEEPSPEED_CONFIG}") Liger=${USE_LIGER}"
echo "  max_length=${MODEL_MAX_LENGTH} max_pixels=${MAX_PIXELS}"
echo "  output=${OUTPUT_DIR}"
if [ -n "${MAX_STEPS}" ]; then
    echo "  test_limit=${MAX_STEPS} optimizer steps"
fi

cd "${PROJECT_DIR}"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m qwenvl.train.train_qwen \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --dataset_use "${DATASETS}" \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 True \
    --tf32 True \
    --output_dir "${OUTPUT_DIR}" \
    "${LIMIT_ARGS[@]}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --per_device_eval_batch_size "${PER_DEVICE_BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --max_pixels "${MAX_PIXELS}" \
    --min_pixels "${MIN_PIXELS}" \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --ddp_find_unused_parameters False \
    --group_by_modality_length True \
    --dataloader_drop_last True \
    --remove_unused_columns False \
    --run_name "${RUN_NAME}" \
    --report_to wandb \
    "${LIGER_ARGS[@]}"
