#!/usr/bin/env bash
# Direct single-node QAT-only launcher for 4 x A100 40 GB.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-"$(dirname -- "${SCRIPT_DIR}")"}
GRACE_ROOT=${GRACE_ROOT:-"$(dirname -- "${PROJECT_DIR}")"}

BASE_MODEL_DIR=${BASE_MODEL_DIR:-"${GRACE_ROOT}/models/Qwen3-VL-2B-Instruct"}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"${BASE_MODEL_DIR}"}
SHAREGPT4V_ROOT=${SHAREGPT4V_ROOT:-"${GRACE_ROOT}/ShareGPT4V"}
DATASETS=${DATASETS:-sharegpt4v_mix665k}

QAT_BITS=${QAT_BITS:-4}
QAT_GROUP_SIZE=${QAT_GROUP_SIZE:-128}
QAT_SCALE_LR=${QAT_SCALE_LR:-2e-4}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}

# Reference: 16 GPUs x micro-batch 4 x accumulation 4 = 256.
# Local:      4 GPUs x micro-batch 2 x accumulation 32 = 256.
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-2}
GRAD_ACCUM=${GRAD_ACCUM:-32}

MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-8192}
MAX_PIXELS=${MAX_PIXELS:-50176}
MIN_PIXELS=${MIN_PIXELS:-784}
SAVE_STEPS=${SAVE_STEPS:-500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}

RUN_NAME=${RUN_NAME:-"Qwen3-VL-2B-QAT-W${QAT_BITS}G${QAT_GROUP_SIZE}"}
OUTPUT_DIR=${OUTPUT_DIR:-"${GRACE_ROOT}/models/${RUN_NAME}"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"${PROJECT_DIR}/scripts/zero2.json"}

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

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${HF_DATASETS_CACHE}"
LATEST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
if [ -n "${LATEST_CKPT}" ] && [ -d "${LATEST_CKPT}" ]; then
    echo "RESUME: loading model and Trainer state from ${LATEST_CKPT}"
    MODEL_NAME_OR_PATH=${LATEST_CKPT}
    for filename in preprocessor_config.json video_preprocessor_config.json; do
        if [ ! -f "${LATEST_CKPT}/${filename}" ] && [ -f "${BASE_MODEL_DIR}/${filename}" ]; then
            cp -v "${BASE_MODEL_DIR}/${filename}" "${LATEST_CKPT}/${filename}"
        fi
    done
else
    echo "FRESH: starting QAT from ${MODEL_NAME_OR_PATH}"
fi

EFFECTIVE_BATCH=$((NPROC_PER_NODE * PER_DEVICE_BATCH * GRAD_ACCUM))
echo "QAT W${QAT_BITS}G${QAT_GROUP_SIZE}: ${NPROC_PER_NODE} GPUs, micro-batch ${PER_DEVICE_BATCH}, accumulation ${GRAD_ACCUM}, effective batch ${EFFECTIVE_BATCH}"
echo "Output: ${OUTPUT_DIR}"

cd "${PROJECT_DIR}"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m qwenvl.train.train_qwen_qat \
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
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --per_device_eval_batch_size "$((PER_DEVICE_BATCH * 2))" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --max_pixels "${MAX_PIXELS}" \
    --min_pixels "${MIN_PIXELS}" \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay 1e-3 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
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
    --qat_enable True \
    --qat_bits "${QAT_BITS}" \
    --qat_group_size "${QAT_GROUP_SIZE}" \
    --qat_scale_lr "${QAT_SCALE_LR}" \
    --qat_save_quantized True
