#!/usr/bin/env bash
# Short correctness test for QAT-LoRA GRACE. Physical GPUs 0,1,2 are used by
# default; override VALIDATION_CUDA_VISIBLE_DEVICES on a shared machine.
# This diagnoses learning; it is not a clean throughput benchmark because
# gradient hooks and parameter snapshots intentionally add overhead.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-"$(dirname -- "${SCRIPT_DIR}")"}
GRACE_ROOT=${GRACE_ROOT:-"$(dirname -- "${PROJECT_DIR}")"}

VALIDATION_CUDA_VISIBLE_DEVICES=${VALIDATION_CUDA_VISIBLE_DEVICES:-0,1,2}
IFS=',' read -r -a VALIDATION_GPU_IDS <<< "${VALIDATION_CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE=${#VALIDATION_GPU_IDS[@]}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${VALIDATION_CUDA_VISIBLE_DEVICES}"
export WANDB_MODE=disabled

VALIDATION_STEPS=${VALIDATION_STEPS:-5}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-2}
SAMPLES_PER_GPU_STEP=${SAMPLES_PER_GPU_STEP:-12}
if [ $((SAMPLES_PER_GPU_STEP % PER_DEVICE_BATCH)) -ne 0 ]; then
    echo "ERROR: SAMPLES_PER_GPU_STEP=${SAMPLES_PER_GPU_STEP} is not divisible by PER_DEVICE_BATCH=${PER_DEVICE_BATCH}." >&2
    exit 1
fi
GRAD_ACCUM=${GRAD_ACCUM:-$((SAMPLES_PER_GPU_STEP / PER_DEVICE_BATCH))}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
MAX_PIXELS=${MAX_PIXELS:-50176}
VALIDATION_TAG=${VALIDATION_TAG:-"$(date +%Y%m%d-%H%M%S)"}
OUTPUT_DIR=${OUTPUT_DIR:-"${GRACE_ROOT}/models/profiling/grace-lora-learning-${VALIDATION_TAG}"}
mkdir -p "${OUTPUT_DIR}"

echo "QAT-LoRA GRACE learning validation"
echo "  physical GPUs=${VALIDATION_CUDA_VISIBLE_DEVICES} (${NPROC_PER_NODE} ranks, PCI bus order)"
echo "  optimizer steps=${VALIDATION_STEPS}"
echo "  micro_batch=${PER_DEVICE_BATCH} accumulation=${GRAD_ACCUM}"
echo "  global samples/step=$((NPROC_PER_NODE * PER_DEVICE_BATCH * GRAD_ACCUM))"
echo "  output=${OUTPUT_DIR}"

NPROC_PER_NODE="${NPROC_PER_NODE}" \
MAX_STEPS="${VALIDATION_STEPS}" \
PROFILE_STEPS=0 \
PROFILE_DEEPSPEED_COMMS=False \
LEARNING_DIAGNOSTICS_STEPS="${VALIDATION_STEPS}" \
SKIP_FINAL_EXPORT=True \
PER_DEVICE_BATCH="${PER_DEVICE_BATCH}" \
GRAD_ACCUM="${GRAD_ACCUM}" \
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH}" \
MAX_PIXELS="${MAX_PIXELS}" \
SAVE_STEPS=1000 \
RUN_NAME="Qwen3-VL-2B-GRACE-LORA-LEARNING-CHECK" \
OUTPUT_DIR="${OUTPUT_DIR}" \
"${SCRIPT_DIR}/finetune_qwen3vl_2b_grace_lora_4gpu_memory_safe.sh" \
    2>&1 | tee "${OUTPUT_DIR}/learning_validation.log"

SUMMARY="${OUTPUT_DIR}/profiling/learning_diagnostics_summary.json"
[ -f "${SUMMARY}" ] || {
    echo "ERROR: learning diagnostic summary was not produced: ${SUMMARY}" >&2
    exit 1
}

python - "${SUMMARY}" <<'PY'
import json
import sys

path = sys.argv[1]
summary = json.load(open(path))
checks = summary.get("checks", {})

print("\nLearning validation checks")
for name, passed in checks.items():
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")

print("\nFinal sampled parameter deltas")
for category, result in summary.get("final_parameter_deltas", {}).items():
    delta = result.get("sample_max_abs_delta_from_start", 0.0)
    print(f"  {category:<18} {delta:.8e}  {result.get('sample_parameter', '')}")

failed = [name for name, passed in checks.items() if not passed]
if failed:
    print("\nFAILED checks: " + ", ".join(failed), file=sys.stderr)
    raise SystemExit(2)
print(f"\nAll learning checks passed. Summary: {path}")
PY
