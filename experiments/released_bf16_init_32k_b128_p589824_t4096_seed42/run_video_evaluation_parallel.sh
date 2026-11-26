#!/usr/bin/env bash
# Disjoint lmms-eval video matrices; 720p, VideoMME short/medium only.
set -euo pipefail

PARALLEL_EXP="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PARALLEL_RUNNER="$PARALLEL_EXP/run_video_evaluation.sh"
PARALLEL_ACTION="${1:-plan}"
PARALLEL_REFERENCE_GPU="${2:-2}"
PARALLEL_LOCAL_GPU="${3:-4}"
PARALLEL_REFERENCE_METHODS=vanilla_8b_teacher,released_bf16,released_grace_w4,released_grace_w8
PARALLEL_LOCAL_METHODS=grace_w4,grace_w8
PARALLEL_ROOT="${VIDEO_PARALLEL_OUTPUT:-$PARALLEL_EXP/video_evaluation/lmms_parallel_no_local_sft_v1}"

if [[ $# -gt 3 || ! "$PARALLEL_ACTION" =~ ^(plan|run)$ ||
      ! "$PARALLEL_REFERENCE_GPU" =~ ^[0-9]+$ || ! "$PARALLEL_LOCAL_GPU" =~ ^[0-9]+$ ||
      "$PARALLEL_REFERENCE_GPU" == "$PARALLEL_LOCAL_GPU" ]]; then
    printf 'Usage: bash %s [plan|run] [reference GPU=2] [local GPU=4]\n' "$0" >&2
    exit 2
fi

printf 'Priority phase (sequential): qat_w4, qat_w8, vanilla_2b\nThen GPU %s: %s\nGPU %s: %s\nResults: %s\n' \
    "$PARALLEL_REFERENCE_GPU" "$PARALLEL_REFERENCE_METHODS" \
    "$PARALLEL_LOCAL_GPU" "$PARALLEL_LOCAL_METHODS" "$PARALLEL_ROOT"

run_parallel_group() {
    local group_gpu="$1" group_methods="$2" group_output="$3"
    bash "$PARALLEL_RUNNER" "$PARALLEL_ACTION" --gpu "$group_gpu" \
        --methods "$group_methods" --output "$group_output"
}

# Keep each model in its usual output group; no other models launch until
# all three priority models have been attempted in the requested order.
PARALLEL_PRIORITY_STATUS=0
run_parallel_group "$PARALLEL_LOCAL_GPU" qat_w4 "$PARALLEL_ROOT/local" || PARALLEL_PRIORITY_STATUS=1
run_parallel_group "$PARALLEL_LOCAL_GPU" qat_w8 "$PARALLEL_ROOT/local" || PARALLEL_PRIORITY_STATUS=1
run_parallel_group "$PARALLEL_REFERENCE_GPU" vanilla_2b "$PARALLEL_ROOT/references" || PARALLEL_PRIORITY_STATUS=1

if [[ "$PARALLEL_ACTION" == plan ]]; then
    run_parallel_group "$PARALLEL_REFERENCE_GPU" "$PARALLEL_REFERENCE_METHODS" "$PARALLEL_ROOT/references"
    run_parallel_group "$PARALLEL_LOCAL_GPU" "$PARALLEL_LOCAL_METHODS" "$PARALLEL_ROOT/local"
    exit 0
fi

mkdir -p "$PARALLEL_ROOT"
bash "$PARALLEL_RUNNER" run --gpu "$PARALLEL_REFERENCE_GPU" \
    --methods "$PARALLEL_REFERENCE_METHODS" --output "$PARALLEL_ROOT/references" \
    >>"$PARALLEL_ROOT/references.log" 2>&1 &
PARALLEL_REFERENCE_PID=$!
bash "$PARALLEL_RUNNER" run --gpu "$PARALLEL_LOCAL_GPU" \
    --methods "$PARALLEL_LOCAL_METHODS" --output "$PARALLEL_ROOT/local" \
    >>"$PARALLEL_ROOT/local.log" 2>&1 &
PARALLEL_LOCAL_PID=$!
stop_parallel_groups() {
    kill -TERM "$PARALLEL_REFERENCE_PID" "$PARALLEL_LOCAL_PID" 2>/dev/null || true
    wait "$PARALLEL_REFERENCE_PID" "$PARALLEL_LOCAL_PID" 2>/dev/null || true
    exit 130
}
trap stop_parallel_groups INT TERM
PARALLEL_REFERENCE_STATUS=0
PARALLEL_LOCAL_STATUS=0
wait "$PARALLEL_REFERENCE_PID" || PARALLEL_REFERENCE_STATUS=$?
wait "$PARALLEL_LOCAL_PID" || PARALLEL_LOCAL_STATUS=$?
printf 'Reference exit=%s; local exit=%s\nComparisons: %s/{references,local}/full/comparison.md\n' \
    "$PARALLEL_REFERENCE_STATUS" "$PARALLEL_LOCAL_STATUS" "$PARALLEL_ROOT"
if (( PARALLEL_PRIORITY_STATUS || PARALLEL_REFERENCE_STATUS || PARALLEL_LOCAL_STATUS )); then
    exit 1
fi
