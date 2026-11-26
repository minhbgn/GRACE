#!/usr/bin/env bash
# Direct, single-node launcher for GRACE on 4 x A100 40 GB GPUs.
#
# Defaults assume this repository's models/ and ShareGPT4V/ directories.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-"$(dirname -- "${SCRIPT_DIR}")"}
GRACE_ROOT=${GRACE_ROOT:-"$(dirname -- "${PROJECT_DIR}")"}

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"${GRACE_ROOT}/models/Qwen3-VL-2B-Instruct"}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"${GRACE_ROOT}/models/Qwen3-VL-8B-Instruct"}
SHAREGPT4V_ROOT=${SHAREGPT4V_ROOT:-"${GRACE_ROOT}/ShareGPT4V"}

BASE_MODEL_DIR=${MODEL_NAME_OR_PATH}
DATASETS=${DATASETS:-sharegpt4v_mix665k}

QAT_BITS=${QAT_BITS:-4}
QAT_GROUP_SIZE=${QAT_GROUP_SIZE:-128}
QAT_SCALE_LR=${QAT_SCALE_LR:-2e-4}

DKD_TEMPERATURE=${DKD_TEMPERATURE:-2.0}
DKD_ALPHA=${DKD_ALPHA:-1.0}
DKD_BETA=${DKD_BETA:-4.0}
RCKA_WEIGHT=${RCKA_WEIGHT:-3.0}
RCKA_LAYER=${RCKA_LAYER:--2}
IB_TAU=${IB_TAU:-3.0}
IB_ETA=${IB_ETA:-0.003}
IB_BETA_INIT=${IB_BETA_INIT:-0.5}
IB_BETA_MIN=${IB_BETA_MIN:-0.1}
IB_BETA_MAX=${IB_BETA_MAX:-1.0}
DKD_WARMUP_STEPS=${DKD_WARMUP_STEPS:-0}
RCKA_WARMUP_STEPS=${RCKA_WARMUP_STEPS:-400}

LEARNING_RATE=${LEARNING_RATE:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-3}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
# The upstream launcher defaults to one epoch, but its own comment says the
# reported paper run uses three. Use one here only for a pipeline smoke test.
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}

# Reference GRACE: 16 GPUs x micro-batch 2 x accumulation 16 = 512.
# This launcher:     4 GPUs x micro-batch 1 x accumulation 128 = 512.
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-1}
GRAD_ACCUM=${GRAD_ACCUM:-128}

MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-8192}
MAX_PIXELS=${MAX_PIXELS:-200704}
MIN_PIXELS=${MIN_PIXELS:-784}
SAVE_STEPS=${SAVE_STEPS:-500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}

RUN_NAME=${RUN_NAME:-"Qwen3-VL-2B-GRACE-W${QAT_BITS}G${QAT_GROUP_SIZE}"}
OUTPUT_DIR=${OUTPUT_DIR:-"${GRACE_ROOT}/models/${RUN_NAME}"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"${PROJECT_DIR}/scripts/zero2.json"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_HOME=${HF_HOME:-"${PROJECT_DIR}/.cache/huggingface"}
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
    if ! command -v "${executable}" >/dev/null 2>&1; then
        echo "ERROR: ${executable} is not available; activate the GRACE training environment." >&2
        exit 1
    fi
done

for path in \
    "${BASE_MODEL_DIR}" \
    "${TEACHER_MODEL_PATH}" \
    "${SHAREGPT4V_ROOT}/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json" \
    "${SHAREGPT4V_ROOT}/data/sam/images" \
    "${DEEPSPEED_CONFIG}"; do
    if [ ! -e "${path}" ]; then
        echo "ERROR: required path does not exist: ${path}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${HF_DATASETS_CACHE}"

# Match the upstream launcher's behavior when OUTPUT_DIR already contains a
# checkpoint. Trainer then restores optimizer, scheduler, and RNG state.
LATEST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
if [ -n "${LATEST_CKPT}" ] && [ -d "${LATEST_CKPT}" ]; then
    echo "RESUME: loading student and Trainer state from ${LATEST_CKPT}"
    MODEL_NAME_OR_PATH=${LATEST_CKPT}
    for filename in preprocessor_config.json video_preprocessor_config.json; do
        if [ ! -f "${LATEST_CKPT}/${filename}" ] && [ -f "${BASE_MODEL_DIR}/${filename}" ]; then
            cp -v "${BASE_MODEL_DIR}/${filename}" "${LATEST_CKPT}/${filename}"
        fi
    done
else
    echo "FRESH: starting GRACE from ${MODEL_NAME_OR_PATH}"
fi

EFFECTIVE_BATCH=$((NPROC_PER_NODE * PER_DEVICE_BATCH * GRAD_ACCUM))
echo "GRACE W${QAT_BITS}G${QAT_GROUP_SIZE}: ${NPROC_PER_NODE} GPUs, micro-batch ${PER_DEVICE_BATCH}, accumulation ${GRAD_ACCUM}, effective batch ${EFFECTIVE_BATCH}"
echo "Student: ${MODEL_NAME_OR_PATH}"
echo "Teacher: ${TEACHER_MODEL_PATH}"
echo "Output:  ${OUTPUT_DIR}"
if [ "${EFFECTIVE_BATCH}" -ne 512 ]; then
    echo "WARNING: effective batch differs from the GRACE reference value of 512." >&2
fi

cd "${PROJECT_DIR}"

torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m qwenvl.train.train_qwen_grace \
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
    --qat_enable True \
    --qat_bits "${QAT_BITS}" \
    --qat_group_size "${QAT_GROUP_SIZE}" \
    --qat_scale_lr "${QAT_SCALE_LR}" \
    --qat_save_quantized True \
    --grace_enable True \
    --teacher_model_path "${TEACHER_MODEL_PATH}" \
    --dkd_temperature "${DKD_TEMPERATURE}" \
    --dkd_alpha "${DKD_ALPHA}" \
    --dkd_beta "${DKD_BETA}" \
    --rcka_weight "${RCKA_WEIGHT}" \
    --rcka_layer "${RCKA_LAYER}" \
    --ib_tau "${IB_TAU}" \
    --ib_eta "${IB_ETA}" \
    --ib_beta_init "${IB_BETA_INIT}" \
    --ib_beta_min "${IB_BETA_MIN}" \
    --ib_beta_max "${IB_BETA_MAX}" \
    --dkd_warmup_steps "${DKD_WARMUP_STEPS}" \
    --rcka_warmup_steps "${RCKA_WARMUP_STEPS}"

