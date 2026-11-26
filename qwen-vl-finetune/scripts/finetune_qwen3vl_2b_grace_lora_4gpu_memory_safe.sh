#!/usr/bin/env bash
# QAT-aware LoRA GRACE launcher for one node with 4 x A100 40 GB GPUs.
#
# Trainable student state:
#   - LoRA A/B on the selected language-model linears
#   - LSQ log_scale only on those LoRA-targeted linears
#   - the visual merger (--tune_mm_mlp True), matching the full-GRACE recipe
# Frozen student state:
#   - all language-model base weights
#   - initialized LSQ log_scale on non-LoRA language-model linears
#   - the vision tower and lm_head
#
# At the end, LoRA is merged into the base weights and the merged weights are
# baked through QAT. The output directory is therefore a standalone, vanilla
# Qwen model, not an adapter-only checkpoint.
#
# Useful overrides:
#   MAX_STEPS=5 ...                       integration test
#   PER_DEVICE_BATCH=1 GRAD_ACCUM=128 ... lower activation memory
#   LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj ...
#                                            add MLP adapters

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-"$(dirname -- "${SCRIPT_DIR}")"}
GRACE_ROOT=${GRACE_ROOT:-"$(dirname -- "${PROJECT_DIR}")"}

BASE_MODEL_DIR=${BASE_MODEL_DIR:-"${GRACE_ROOT}/models/Qwen3-VL-2B-Instruct"}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"${BASE_MODEL_DIR}"}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"${GRACE_ROOT}/models/Qwen3-VL-8B-Instruct"}
SHAREGPT4V_ROOT=${SHAREGPT4V_ROOT:-"${GRACE_ROOT}/ShareGPT4V"}
DATASETS=${DATASETS:-sharegpt4v_mix665k}

QAT_BITS=${QAT_BITS:-4}
QAT_GROUP_SIZE=${QAT_GROUP_SIZE:-128}
QAT_SCALE_LR=${QAT_SCALE_LR:-2e-4}

LORA_R=${LORA_R:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}

DKD_TEMPERATURE=${DKD_TEMPERATURE:-2.0}
DKD_ALPHA=${DKD_ALPHA:-1.0}
DKD_BETA=${DKD_BETA:-4.0}
DKD_CHUNK_SIZE=${DKD_CHUNK_SIZE:-256}
RCKA_WEIGHT=${RCKA_WEIGHT:-3.0}
RCKA_LAYER=${RCKA_LAYER:--2}
IB_TAU=${IB_TAU:-3.0}
IB_ETA=${IB_ETA:-0.003}
IB_BETA_INIT=${IB_BETA_INIT:-0.5}
IB_BETA_MIN=${IB_BETA_MIN:-0.1}
IB_BETA_MAX=${IB_BETA_MAX:-1.0}
DKD_WARMUP_STEPS=${DKD_WARMUP_STEPS:-0}
RCKA_WARMUP_STEPS=${RCKA_WARMUP_STEPS:-400}

# LoRA generally needs a higher LR than full-model GRACE. The QAT scale LR is
# still independently configurable above.
LEARNING_RATE=${LEARNING_RATE:-2e-4}
MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
MAX_STEPS=${MAX_STEPS:-}
PROFILE_STEPS=${PROFILE_STEPS:-0}
PROFILE_DEEPSPEED_COMMS=${PROFILE_DEEPSPEED_COMMS:-False}
SKIP_FINAL_EXPORT=${SKIP_FINAL_EXPORT:-False}
LEARNING_DIAGNOSTICS_STEPS=${LEARNING_DIAGNOSTICS_STEPS:-0}

# 4 GPUs x micro-batch 2 x accumulation 64 = reference global batch 512.
# LoRA eliminates dense-weight gradients and optimizer state. ZeRO-2 avoids
# ZeRO-3 parameter-gather traffic for the now-frozen 2B student backbone.
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-2}
GRAD_ACCUM=${GRAD_ACCUM:-64}

MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
MAX_PIXELS=${MAX_PIXELS:-50176}
MIN_PIXELS=${MIN_PIXELS:-784}
SAVE_STEPS=${SAVE_STEPS:-50}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}

FINAL_MODEL_NAME="Qwen3-VL-2B-GRACE-W${QAT_BITS}G${QAT_GROUP_SIZE}-LORA"
RUN_NAME=${RUN_NAME:-"${FINAL_MODEL_NAME}"}
OUTPUT_DIR=${OUTPUT_DIR:-"${GRACE_ROOT}/models/${FINAL_MODEL_NAME}"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"${PROJECT_DIR}/scripts/zero2.json"}

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
# On conti-sym7870, physical index 3 is the DGX Display adapter; the fourth
# A100 is physical index 4.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,4}
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
    "${TEACHER_MODEL_PATH}" \
    "${SHAREGPT4V_ROOT}/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json" \
    "${SHAREGPT4V_ROOT}/data/sam/images" \
    "${DEEPSPEED_CONFIG}"; do
    [ -e "${path}" ] || {
        echo "ERROR: required path does not exist: ${path}" >&2
        exit 1
    }
done

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${HF_DATASETS_CACHE}"

# Intermediate checkpoint directories retain native LoRA and QAT scale state
# so Trainer can resume optimizer/scheduler/model state exactly. Only the
# final model at OUTPUT_DIR is merged and stripped of auxiliary tensors.
LATEST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
if [ -n "${LATEST_CKPT}" ] && [ -d "${LATEST_CKPT}" ]; then
    echo "RESUME: loading QAT-LoRA GRACE state from ${LATEST_CKPT}"
    MODEL_NAME_OR_PATH=${LATEST_CKPT}
    for filename in preprocessor_config.json video_preprocessor_config.json; do
        if [ ! -f "${LATEST_CKPT}/${filename}" ] && [ -f "${BASE_MODEL_DIR}/${filename}" ]; then
            cp -v "${BASE_MODEL_DIR}/${filename}" "${LATEST_CKPT}/${filename}"
        fi
    done
else
    echo "FRESH: starting QAT-LoRA GRACE from ${MODEL_NAME_OR_PATH}"
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
echo "QAT-LoRA GRACE W${QAT_BITS}G${QAT_GROUP_SIZE}"
echo "  GPUs=${NPROC_PER_NODE} micro_batch=${PER_DEVICE_BATCH} accumulation=${GRAD_ACCUM} effective_batch=${EFFECTIVE_BATCH}"
echo "  LoRA r=${LORA_R} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT} targets=${LORA_TARGET_MODULES}"
echo "  max_length=${MODEL_MAX_LENGTH} max_pixels=${MAX_PIXELS} DKD_chunk=${DKD_CHUNK_SIZE}"
echo "  student=${MODEL_NAME_OR_PATH}"
echo "  teacher=${TEACHER_MODEL_PATH}"
echo "  final_model=${OUTPUT_DIR}"
if [ "${EFFECTIVE_BATCH}" -ne 512 ]; then
    echo "WARNING: effective batch differs from the GRACE reference value of 512." >&2
fi
if [ -n "${MAX_STEPS}" ]; then
    echo "  test_limit=${MAX_STEPS} optimizer steps"
fi

cd "${PROJECT_DIR}"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
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
    --lora_enable True \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --lora_target_modules "${LORA_TARGET_MODULES}" \
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
    --mm_projector_lr "${MM_PROJECTOR_LR}" \
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
    --dkd_chunk_size "${DKD_CHUNK_SIZE}" \
    --rcka_weight "${RCKA_WEIGHT}" \
    --rcka_layer "${RCKA_LAYER}" \
    --ib_tau "${IB_TAU}" \
    --ib_eta "${IB_ETA}" \
    --ib_beta_init "${IB_BETA_INIT}" \
    --ib_beta_min "${IB_BETA_MIN}" \
    --ib_beta_max "${IB_BETA_MAX}" \
    --dkd_warmup_steps "${DKD_WARMUP_STEPS}" \
    --rcka_warmup_steps "${RCKA_WARMUP_STEPS}" \
    --profile_steps "${PROFILE_STEPS}" \
    --profile_deepspeed_comms "${PROFILE_DEEPSPEED_COMMS}" \
    --skip_final_export "${SKIP_FINAL_EXPORT}" \
    --learning_diagnostics_steps "${LEARNING_DIAGNOSTICS_STEPS}"

case "${SKIP_FINAL_EXPORT,,}" in
    true|1|yes)
        echo "DONE: profiling/training checkpoint saved under ${OUTPUT_DIR}; final LoRA merge/QAT bake was skipped."
        ;;
    *)
        echo "DONE: merged standalone model at ${OUTPUT_DIR}"
        ;;
esac
