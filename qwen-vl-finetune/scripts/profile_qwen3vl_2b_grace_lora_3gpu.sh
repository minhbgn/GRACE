#!/usr/bin/env bash
# Compare full GRACE with QAT-LoRA GRACE and find the largest viable
# micro-batch. Physical GPUs 0,1,2 are used by default; the selection can be
# overridden with PROFILE_CUDA_VISIBLE_DEVICES.
#
# Defaults run:
#   full GRACE: batch 1, accumulation 12
#   QAT-LoRA:   batch 1/2, accumulation 12/6
#
# Every case therefore processes 12 samples per selected GPU per optimizer
# step, making tokens/sec comparable despite different micro-batches when the
# same GPU selection is used for all cases.
# The first profiled optimizer step is excluded from the steady-state summary.
# Final model export is disabled: this is a speed/memory test, not training.
#
# Examples:
#   BATCH_SIZES="1 2 3" ./scripts/profile_qwen3vl_2b_grace_lora_3gpu.sh
#   PROFILE_CUDA_VISIBLE_DEVICES=0,1 BATCH_SIZES="1 2" RUN_BASELINE=0 \
#       ./scripts/profile_qwen3vl_2b_grace_lora_3gpu.sh
#   RUN_BASELINE=0 PROFILE_OPTIMIZER_STEPS=5 \
#       ./scripts/profile_qwen3vl_2b_grace_lora_3gpu.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-"$(dirname -- "${SCRIPT_DIR}")"}
GRACE_ROOT=${GRACE_ROOT:-"$(dirname -- "${PROJECT_DIR}")"}

# This default intentionally selects only physical GPUs 0,1,2.
PROFILE_CUDA_VISIBLE_DEVICES=${PROFILE_CUDA_VISIBLE_DEVICES:-0,1,2}
IFS=',' read -r -a PROFILE_GPU_IDS <<< "${PROFILE_CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE=${#PROFILE_GPU_IDS[@]}
RUN_BASELINE=${RUN_BASELINE:-1}
BATCH_SIZES=${BATCH_SIZES:-"1 2"}
PROFILE_OPTIMIZER_STEPS=${PROFILE_OPTIMIZER_STEPS:-3}
SAMPLES_PER_GPU_STEP=${SAMPLES_PER_GPU_STEP:-12}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
MAX_PIXELS=${MAX_PIXELS:-50176}
MIN_PIXELS=${MIN_PIXELS:-784}

PROFILE_TAG=${PROFILE_TAG:-"$(date +%Y%m%d-%H%M%S)"}
PROFILE_ROOT=${PROFILE_ROOT:-"${GRACE_ROOT}/models/profiling/grace-lora-3gpu-${PROFILE_TAG}"}
BASELINE_LAUNCHER="${SCRIPT_DIR}/finetune_qwen3vl_2b_grace_4gpu_memory_safe.sh"
LORA_LAUNCHER="${SCRIPT_DIR}/finetune_qwen3vl_2b_grace_lora_4gpu_memory_safe.sh"

case "${PROFILE_OPTIMIZER_STEPS}" in
    *[!0-9]*|'') echo "ERROR: PROFILE_OPTIMIZER_STEPS must be an integer." >&2; exit 1 ;;
esac
if [ "${PROFILE_OPTIMIZER_STEPS}" -lt 2 ]; then
    echo "ERROR: use at least 2 optimizer steps so one warmup step can be excluded." >&2
    exit 1
fi
case "${SAMPLES_PER_GPU_STEP}" in
    *[!0-9]*|'') echo "ERROR: SAMPLES_PER_GPU_STEP must be an integer." >&2; exit 1 ;;
esac
if [ "${SAMPLES_PER_GPU_STEP}" -le 0 ]; then
    echo "ERROR: SAMPLES_PER_GPU_STEP must be positive." >&2
    exit 1
fi

for launcher in "${BASELINE_LAUNCHER}" "${LORA_LAUNCHER}"; do
    [ -x "${launcher}" ] || {
        echo "ERROR: launcher is missing or not executable: ${launcher}" >&2
        exit 1
    }
done

mkdir -p "${PROFILE_ROOT}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${PROFILE_CUDA_VISIBLE_DEVICES}"
export WANDB_MODE=disabled

ACTIVE_MONITOR_PID=""
stop_monitor() {
    if [ -n "${ACTIVE_MONITOR_PID}" ]; then
        kill "${ACTIVE_MONITOR_PID}" >/dev/null 2>&1 || true
        wait "${ACTIVE_MONITOR_PID}" >/dev/null 2>&1 || true
        ACTIVE_MONITOR_PID=""
    fi
}
trap stop_monitor EXIT INT TERM

start_monitor() {
    local output_file=$1
    stop_monitor
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        nvidia-smi \
            --id="${PROFILE_CUDA_VISIBLE_DEVICES}" \
            --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,clocks.mem \
            --format=csv \
            --loop=1 >"${output_file}" 2>&1 &
        ACTIVE_MONITOR_PID=$!
    fi
}

run_case() {
    local case_name=$1
    local launcher=$2
    local micro_batch=$3
    local grad_accum=$4
    local output_dir="${PROFILE_ROOT}/${case_name}"
    local log_file="${output_dir}/launcher.log"
    local status_file="${output_dir}/status.txt"

    mkdir -p "${output_dir}/profiling"
    {
        echo "case=${case_name}"
        echo "physical_gpus=${PROFILE_CUDA_VISIBLE_DEVICES}"
        echo "nproc_per_node=${NPROC_PER_NODE}"
        echo "micro_batch=${micro_batch}"
        echo "gradient_accumulation=${grad_accum}"
        echo "global_samples_per_optimizer_step=$((NPROC_PER_NODE * micro_batch * grad_accum))"
        echo "profile_optimizer_steps=${PROFILE_OPTIMIZER_STEPS}"
        echo "model_max_length=${MODEL_MAX_LENGTH}"
        echo "max_pixels=${MAX_PIXELS}"
    } | tee "${output_dir}/configuration.txt"

    start_monitor "${output_dir}/profiling/nvidia_smi.csv"
    set +e
    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    CUDA_VISIBLE_DEVICES="${PROFILE_CUDA_VISIBLE_DEVICES}" \
    MAX_STEPS="${PROFILE_OPTIMIZER_STEPS}" \
    PROFILE_STEPS="${PROFILE_OPTIMIZER_STEPS}" \
    PROFILE_DEEPSPEED_COMMS=False \
    SKIP_FINAL_EXPORT=True \
    PER_DEVICE_BATCH="${micro_batch}" \
    GRAD_ACCUM="${grad_accum}" \
    MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH}" \
    MAX_PIXELS="${MAX_PIXELS}" \
    MIN_PIXELS="${MIN_PIXELS}" \
    SAVE_STEPS=1000 \
    RUN_NAME="${case_name}" \
    OUTPUT_DIR="${output_dir}" \
    "${launcher}" 2>&1 | tee "${log_file}"
    local rc=${PIPESTATUS[0]}
    set -e
    stop_monitor

    if [ "${rc}" -eq 0 ]; then
        echo "PASS" | tee "${status_file}"
    else
        echo "FAIL exit_code=${rc}" | tee "${status_file}"
    fi
    return "${rc}"
}

echo "Profiling directory: ${PROFILE_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (${NPROC_PER_NODE} physical GPU(s), PCI bus order)"

if [ "${RUN_BASELINE}" = "1" ]; then
    if ! run_case "full_grace_mb1" "${BASELINE_LAUNCHER}" 1 "${SAMPLES_PER_GPU_STEP}"; then
        echo "WARNING: full-GRACE baseline failed; continuing with LoRA sweep." >&2
    fi
fi

LORA_SUCCEEDED=0
for micro_batch in ${BATCH_SIZES}; do
    case "${micro_batch}" in
        *[!0-9]*|'') echo "ERROR: invalid batch size: ${micro_batch}" >&2; exit 1 ;;
    esac
    if [ "${micro_batch}" -le 0 ]; then
        echo "ERROR: batch sizes must be positive: ${micro_batch}" >&2
        exit 1
    fi
    if [ $((SAMPLES_PER_GPU_STEP % micro_batch)) -ne 0 ]; then
        echo "ERROR: SAMPLES_PER_GPU_STEP=${SAMPLES_PER_GPU_STEP} is not divisible by batch ${micro_batch}." >&2
        echo "Choose a common multiple of every requested batch size." >&2
        exit 1
    fi
    grad_accum=$((SAMPLES_PER_GPU_STEP / micro_batch))
    if run_case "lora_mb${micro_batch}" "${LORA_LAUNCHER}" "${micro_batch}" "${grad_accum}"; then
        LORA_SUCCEEDED=$((LORA_SUCCEEDED + 1))
    else
        echo "LoRA batch ${micro_batch} failed; not trying larger batches." >&2
        break
    fi
done

python - "${PROFILE_ROOT}" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for case_dir in sorted(root.iterdir()):
    if not case_dir.is_dir():
        continue
    status_path = case_dir / "status.txt"
    summary_path = case_dir / "profiling" / "grace_timing_summary.json"
    status = status_path.read_text().strip() if status_path.exists() else "UNKNOWN"
    row = {"case": case_dir.name, "status": status}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        metrics = (
            summary.get("steady_state_excluding_first_mean")
            or summary.get("all_steps_mean")
            or {}
        )
        row.update({
            "tokens_per_second": metrics.get("global_tokens_per_second"),
            "peak_allocated_gib": metrics.get("peak_allocated_gib_rank_max"),
            "peak_reserved_gib": metrics.get("peak_reserved_gib_rank_max"),
            "step_wall_seconds": (
                metrics.get("total_step_wall_ms_rank_max", 0.0) / 1000.0
            ),
            "student_seconds": (
                metrics.get("student_backbone_ms_rank_max", 0.0) / 1000.0
            ),
            "teacher_seconds": (
                metrics.get("teacher_backbone_ms_rank_max", 0.0) / 1000.0
            ),
            "gdkd_seconds": metrics.get("gdkd_ms_rank_max", 0.0) / 1000.0,
            "backward_seconds": (
                metrics.get("backward_ms_rank_max", 0.0) / 1000.0
            ),
        })
    rows.append(row)

baseline = next(
    (r.get("tokens_per_second") for r in rows if r["case"] == "full_grace_mb1"),
    None,
)
for row in rows:
    tps = row.get("tokens_per_second")
    row["throughput_vs_full"] = tps / baseline if tps and baseline else None

columns = [
    "case", "status", "tokens_per_second", "throughput_vs_full",
    "peak_allocated_gib", "peak_reserved_gib", "step_wall_seconds",
    "student_seconds", "teacher_seconds", "gdkd_seconds", "backward_seconds",
]
csv_path = root / "profile_comparison.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print("\nSteady-state comparison (first optimizer step excluded)")
print("case                 status          tok/s    vs_full  alloc_GiB  reserved_GiB")
for row in rows:
    def fmt(key, pattern, missing="-"):
        value = row.get(key)
        return pattern.format(value) if value is not None else missing
    print(
        f"{row['case']:<20} {row['status']:<15} "
        f"{fmt('tokens_per_second', '{:>8.1f}')} "
        f"{fmt('throughput_vs_full', '{:>8.2f}x')} "
        f"{fmt('peak_allocated_gib', '{:>10.2f}')} "
        f"{fmt('peak_reserved_gib', '{:>13.2f}')}"
    )
print(f"\nCSV summary: {csv_path}")
PY

if [ "${LORA_SUCCEEDED}" -eq 0 ]; then
    echo "ERROR: no QAT-LoRA profile case completed successfully." >&2
    exit 1
fi

echo "DONE: inspect ${PROFILE_ROOT}/profile_comparison.csv and per-case launcher logs."
