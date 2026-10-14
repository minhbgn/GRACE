#!/usr/bin/env bash
# Isolated GRACE timing run: 10 ordinary optimizer steps, followed optionally
# by one communication-instrumented step. The latter is separate because the
# DeepSpeed communication logger synchronizes every collective.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-"$(dirname -- "${SCRIPT_DIR}")"}
GRACE_ROOT=${GRACE_ROOT:-"$(dirname -- "${PROJECT_DIR}")"}

PROFILE_RUN_NAME=${PROFILE_RUN_NAME:-Qwen3-VL-2B-GRACE-W4G128-PROFILE-10STEP}
PROFILE_OUTPUT_DIR=${PROFILE_OUTPUT_DIR:-"${GRACE_ROOT}/models/${PROFILE_RUN_NAME}"}
RUN_COMM_PROFILE=${RUN_COMM_PROFILE:-1}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

mkdir -p "${PROFILE_OUTPUT_DIR}/profiling"

MONITOR_PID=""
SYSTEM_METRICS="${PROFILE_OUTPUT_DIR}/profiling/nvidia_smi.csv"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
        --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,clocks.mem \
        --format=csv \
        --loop=1 >"${SYSTEM_METRICS}" 2>&1 &
    MONITOR_PID=$!
fi

cleanup() {
    if [ -n "${MONITOR_PID}" ]; then
        kill "${MONITOR_PID}" >/dev/null 2>&1 || true
        wait "${MONITOR_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

echo "PROFILE: 10-step lightweight timing run"
MAX_STEPS=10 \
PROFILE_STEPS=10 \
PROFILE_DEEPSPEED_COMMS=False \
RUN_NAME="${PROFILE_RUN_NAME}" \
OUTPUT_DIR="${PROFILE_OUTPUT_DIR}" \
SAVE_STEPS=1000 \
"${SCRIPT_DIR}/finetune_qwen3vl_2b_grace_4gpu_memory_safe.sh"

if [ "${RUN_COMM_PROFILE}" = "1" ]; then
    COMM_RUN_NAME="${PROFILE_RUN_NAME}-COMM"
    COMM_OUTPUT_DIR="${GRACE_ROOT}/models/${COMM_RUN_NAME}"
    echo "PROFILE: separate 1-step DeepSpeed communication run"
    MAX_STEPS=1 \
    PROFILE_STEPS=1 \
    PROFILE_DEEPSPEED_COMMS=True \
    RUN_NAME="${COMM_RUN_NAME}" \
    OUTPUT_DIR="${COMM_OUTPUT_DIR}" \
    SAVE_STEPS=1000 \
    DEEPSPEED_CONFIG="${SCRIPT_DIR}/zero3_profile_comms.json" \
    "${SCRIPT_DIR}/finetune_qwen3vl_2b_grace_4gpu_memory_safe.sh"
fi

echo "PROFILE: timing output at ${PROFILE_OUTPUT_DIR}/profiling"
