#!/usr/bin/env bash
# Configurable end-to-end QAT-LoRA GRACE launcher. Defaults to 4 GPUs.
# Usage: ./scripts/finetune_qwen3vl_2b_grace_lora_ngpu.sh [NUM_GPUS]
# Example: CUDA_VISIBLE_DEVICES=2,4 ./scripts/finetune_qwen3vl_2b_grace_lora_ngpu.sh 2
# Default GPU pool: 0,1,2,4; skips this host's display adapter.
# Uses micro-batch 4 and rounds accumulation up toward TARGET_GLOBAL_BATCH=512.
# Checkpointed DKD and bounded ZeRO-2 buckets reduce variable-batch peaks.
# GRAD_ACCUM overrides automatic accumulation. TRAINING_BACKEND=ddp allows GPU-count changes.
# touch OUTPUT_DIR/STOP_AFTER_CHECKPOINT to pause; completion exports the merged model.
# See README_grace_gpu_switch.md for the 2-to-4 GPU workflow.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-"$(dirname -- "${SCRIPT_DIR}")"}
GRACE_ROOT=${GRACE_ROOT:-"$(dirname -- "${PROJECT_DIR}")"}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: $0 [NUM_GPUS] (default: 4)"
    echo "Use TRAINING_BACKEND=ddp from a fresh run to allow GPU-count changes."
    echo "Pause: touch OUTPUT_DIR/STOP_AFTER_CHECKPOINT (defaults: save every 50, retain 5)."
    exit 0
fi
if (( $# > 1 )); then
    echo "ERROR: expected at most one argument: NUM_GPUS (default: 4)." >&2
    exit 1
fi
NUM_GPUS=${1-4}
QAT_BITS=${QAT_BITS:-4}
QAT_GROUP_SIZE=${QAT_GROUP_SIZE:-128}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-2}
TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH:-512}
for name in NUM_GPUS PER_DEVICE_BATCH TARGET_GLOBAL_BATCH; do
    if [[ ! "${!name}" =~ ^[1-9][0-9]{0,5}$ ]]; then
        echo "ERROR: ${name} must be a positive integer of at most 6 digits." >&2
        exit 1
    fi
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ ! -v CUDA_VISIBLE_DEVICES ]]; then
    DEFAULT_GPU_IDS=(0 1 2 4)
    if (( NUM_GPUS > ${#DEFAULT_GPU_IDS[@]} )); then
        echo "ERROR: default pool has 4 GPUs; set CUDA_VISIBLE_DEVICES explicitly for ${NUM_GPUS} GPUs." >&2
        exit 1
    fi
    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${DEFAULT_GPU_IDS[*]:0:NUM_GPUS}")
fi
export CUDA_VISIBLE_DEVICES
if [[ ! "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES must be comma-separated physical GPU indices." >&2
    exit 1
fi
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#GPU_IDS[@]} != NUM_GPUS )); then
    echo "ERROR: requested ${NUM_GPUS} GPUs but CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}." >&2
    exit 1
fi
declare -A SEEN_GPU_IDS=()
for gpu_id in "${GPU_IDS[@]}"; do
    if [[ ! "${gpu_id}" =~ ^(0|[1-9][0-9]{0,5})$ ]]; then
        echo "ERROR: invalid physical GPU index: ${gpu_id}" >&2
        exit 1
    fi
    if [[ -n "${SEEN_GPU_IDS[$gpu_id]:-}" ]]; then
        echo "ERROR: duplicate physical GPU index: ${gpu_id}" >&2
        exit 1
    fi
    SEEN_GPU_IDS[$gpu_id]=1
done

denominator=$((NUM_GPUS * PER_DEVICE_BATCH))
GRAD_ACCUM=${GRAD_ACCUM-$(((TARGET_GLOBAL_BATCH + denominator - 1) / denominator))}
if [[ ! "${GRAD_ACCUM}" =~ ^[1-9][0-9]{0,5}$ ]]; then
    echo "ERROR: GRAD_ACCUM must be a positive integer of at most 6 digits." >&2
    exit 1
fi
if (( denominator * GRAD_ACCUM != TARGET_GLOBAL_BATCH )); then
    echo "NOTE: effective batch is $((denominator * GRAD_ACCUM)); target is ${TARGET_GLOBAL_BATCH}."
fi

BASE_MODEL_DIR=${BASE_MODEL_DIR:-"${GRACE_ROOT}/models/Qwen3-VL-2B-Instruct"}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"${BASE_MODEL_DIR}"}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"${GRACE_ROOT}/models/Qwen3-VL-8B-Instruct"}
SHAREGPT4V_ROOT=${SHAREGPT4V_ROOT:-"${GRACE_ROOT}/ShareGPT4V"}
DATASETS=${DATASETS:-sharegpt4v_mix665k}

QAT_SCALE_LR=${QAT_SCALE_LR:-2e-4}

LORA_R=${LORA_R:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}

DKD_TEMPERATURE=${DKD_TEMPERATURE:-2.0}
DKD_ALPHA=${DKD_ALPHA:-1.0}
DKD_BETA=${DKD_BETA:-4.0}
DKD_CHUNK_SIZE=${DKD_CHUNK_SIZE:-256}
DKD_CHECKPOINT_CHUNKS=${DKD_CHECKPOINT_CHUNKS:-True}
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
# This end-to-end launcher always performs the final merged export.
PROFILE_STEPS=0
PROFILE_DEEPSPEED_COMMS=False
SKIP_FINAL_EXPORT=False
LEARNING_DIAGNOSTICS_STEPS=0

# 4 GPUs x micro-batch 4 x accumulation 32 = reference global batch 512.
# LoRA eliminates dense-weight gradients and optimizer state. ZeRO-2 avoids
# ZeRO-3 parameter-gather traffic for the now-frozen 2B student backbone.
NPROC_PER_NODE=${NUM_GPUS}

MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
MAX_PIXELS=${MAX_PIXELS:-50176}
MIN_PIXELS=${MIN_PIXELS:-784}
SAVE_STEPS=${SAVE_STEPS:-50}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}

FINAL_MODEL_NAME="Qwen3-VL-2B-GRACE-W${QAT_BITS}G${QAT_GROUP_SIZE}-LORA"
RUN_NAME=${RUN_NAME:-"${FINAL_MODEL_NAME}"}
OUTPUT_DIR=${OUTPUT_DIR:-"${GRACE_ROOT}/models/${FINAL_MODEL_NAME}"}
TRAINING_BACKEND=${TRAINING_BACKEND:-zero2}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"${PROJECT_DIR}/scripts/zero2_lora_memory_safe.json"}

BACKEND_ARGS=()
case "${TRAINING_BACKEND}" in
    ddp) ;;
    zero2) BACKEND_ARGS=(--deepspeed "${DEEPSPEED_CONFIG}") ;;
    *) echo "ERROR: TRAINING_BACKEND must be ddp or zero2." >&2; exit 1 ;;
esac

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

# Store the resolved training recipe in every checkpoint. GPU count and
# accumulation may change together; model/data/optimizer/schedule must match.
RECIPE_KEYS=(TRAINING_BACKEND BASE_MODEL_DIR MODEL_NAME_OR_PATH TEACHER_MODEL_PATH
    DATASETS SHAREGPT4V_ROOT QAT_BITS QAT_GROUP_SIZE QAT_SCALE_LR LORA_R LORA_ALPHA
    LORA_DROPOUT LORA_TARGET_MODULES DKD_TEMPERATURE DKD_ALPHA DKD_BETA
    DKD_CHUNK_SIZE DKD_CHECKPOINT_CHUNKS RCKA_WEIGHT RCKA_LAYER IB_TAU IB_ETA
    IB_BETA_INIT IB_BETA_MIN IB_BETA_MAX DKD_WARMUP_STEPS RCKA_WARMUP_STEPS
    LEARNING_RATE MM_PROJECTOR_LR WEIGHT_DECAY WARMUP_RATIO MAX_GRAD_NORM
    NUM_TRAIN_EPOCHS MAX_STEPS MODEL_MAX_LENGTH MAX_PIXELS MIN_PIXELS PER_DEVICE_BATCH)
RECIPE_ARGS=()
for name in "${RECIPE_KEYS[@]}"; do RECIPE_ARGS+=("${name}=${!name}"); done
export GRACE_LAUNCH_RECIPE="${OUTPUT_DIR}/launch_recipe.json"
python "${SCRIPT_DIR}/grace_resume_recipe.py" "${OUTPUT_DIR}" "${NUM_GPUS}" \
    "$((denominator * GRAD_ACCUM))" "${DEEPSPEED_CONFIG}" "${RECIPE_ARGS[@]}"

# Intermediate checkpoint directories retain native LoRA and QAT scale state
# so Trainer can restore optimizer/scheduler/model state. Only the
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
echo "  physical GPUs=${CUDA_VISIBLE_DEVICES}"
echo "  GPUs=${NPROC_PER_NODE} micro_batch=${PER_DEVICE_BATCH} accumulation=${GRAD_ACCUM} effective_batch=${EFFECTIVE_BATCH}"
echo "  LoRA r=${LORA_R} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT} targets=${LORA_TARGET_MODULES}"
echo "  max_length=${MODEL_MAX_LENGTH} max_pixels=${MAX_PIXELS} DKD_chunk=${DKD_CHUNK_SIZE}"
echo "  DKD_checkpoint_chunks=${DKD_CHECKPOINT_CHUNKS} deepspeed=${DEEPSPEED_CONFIG}"
echo "  student=${MODEL_NAME_OR_PATH}"
echo "  teacher=${TEACHER_MODEL_PATH}"
echo "  final_model=${OUTPUT_DIR}"
echo "  backend=${TRAINING_BACKEND}; to pause: touch ${OUTPUT_DIR}/STOP_AFTER_CHECKPOINT"
if [ "${EFFECTIVE_BATCH}" -ne 512 ]; then
    echo "WARNING: effective batch differs from the GRACE reference value of 512." >&2
fi
if [ -n "${MAX_STEPS}" ]; then
    echo "  test_limit=${MAX_STEPS} optimizer steps"
fi

cd "${PROJECT_DIR}"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m qwenvl.train.train_qwen_grace \
    "${BACKEND_ARGS[@]}" \
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
    --dkd_checkpoint_chunks "${DKD_CHECKPOINT_CHUNKS}" \
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

if [ -f "${OUTPUT_DIR}/PAUSED_AT_CHECKPOINT" ]; then
    echo "PAUSED: $(cat "${OUTPUT_DIR}/PAUSED_AT_CHECKPOINT")"
    echo "Restart this launcher with the same OUTPUT_DIR and TRAINING_BACKEND."
    exit 0
fi

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
