#!/usr/bin/env bash
# Two-stage, resumable evaluation with the recovered upstream local protocol.
set -euo pipefail
EXPERIMENT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GRACE_PATH="$(cd -- "$EXPERIMENT_PATH/../.." && pwd)"
PYTHON_PATH="${LMMS_PYTHON:-$GRACE_PATH/.venv-lmms-eval/bin/python}"
OUTPUT_PATH="$EXPERIMENT_PATH/re_evaluation"
RUNNER_PATH="$GRACE_PATH/evaluation/run_upstream_matrix.py"
GATE_PATH="$GRACE_PATH/evaluation/check_re_evaluation_gate.py"
REFERENCE_MODELS=vanilla_2b,vanilla_8b_teacher,released_bf16,released_grace_w4,released_grace_w8
LORA_MODELS=qat_w4,grace_w4,qat_w8,grace_w8

usage() {
  cat <<'HELP'
Usage: bash run_re_evaluation.sh stage1|stage2|status [options]

  stage1  Evaluate five reference checkpoints on all seven local suites (35 jobs).
          SFT reference = released_bf16, not the previous local sft_2b export.
  stage2  Verify stage-one acceptance, then evaluate four LoRA checkpoints (28 jobs).
  status  Recheck stage-one results and write stage1_gate.{md,json}; no GPU work.

Options:
  --gpu N          Physical GPU in PCI order; required for actual stage1/stage2 runs.
  --plan           Print selected evaluation commands; no GPU work or output writes.
  --check          CPU model/dataset preflight; no GPU work or output writes.
  --accept-barely   Allow ±5 pp per reference/suite instead of the default ±3 pp.
  -h, --help       Show this help.

Results: re_evaluation/full/<model>/<suite>/attempt_*/
Logs:    re_evaluation/logs/
Gate:    re_evaluation/stage1_gate.md and stage1_gate.json
Rerun the same command to resume. Stage two never starts automatically.
All seven local scorers are unjudged and need no API key.
Use gpuwho before selecting a free GPU. LMMS_PYTHON overrides the Python executable.
HELP
}

if [[ $# -eq 0 || "$1" == -h || "$1" == --help ]]; then usage; exit 0; fi
STAGE="$1"
shift
case "$STAGE" in stage1|stage2|status) ;; *) usage >&2; exit 1 ;; esac
ACTION=run
GPU_INDEX=
GATE_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || { echo '--gpu requires one physical GPU index' >&2; exit 1; }
      GPU_INDEX="$2"; shift 2 ;;
    --plan|--check)
      [[ "$ACTION" == run ]] || { echo 'Choose only one of --plan and --check' >&2; exit 1; }
      ACTION="${1#--}"; shift ;;
    --accept-barely) GATE_ARGS=(--accept-barely); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done
[[ -x "$PYTHON_PATH" ]] || { echo "Python not executable: $PYTHON_PATH" >&2; exit 1; }
# Resolve this experiment consistently even if another shell set EXPERIMENT_DIR.
export EXPERIMENT_DIR="$EXPERIMENT_PATH"
export PYTHONPATH="$GRACE_PATH/qwen-vl-finetune/scripts:$GRACE_PATH/third_party/lmms-eval-v0.7.1${PYTHONPATH:+:$PYTHONPATH}"
COMMON_ARGS=(--experiment "$EXPERIMENT_PATH/experiment.json" --output "$OUTPUT_PATH")
MODELS="$REFERENCE_MODELS"
[[ "$STAGE" != stage2 ]] || MODELS="$LORA_MODELS"

if [[ "$STAGE" != status && "$ACTION" != run ]]; then
  exec "$PYTHON_PATH" "$RUNNER_PATH" "$ACTION" "${COMMON_ARGS[@]}" --methods "$MODELS"
fi
if [[ "$STAGE" == status && "$ACTION" != run ]]; then
  echo 'status cannot be combined with --plan or --check' >&2; exit 1
fi
if [[ "$STAGE" != status && -z "$GPU_INDEX" ]]; then
  echo 'Specify --gpu N after surveying free GPUs with gpuwho.' >&2; exit 1
fi
mkdir -p "$OUTPUT_PATH/logs"
LOG_PATH="$OUTPUT_PATH/logs/${STAGE}_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
exec > >(tee -a "$LOG_PATH") 2>&1

check_gate() {
  if [[ ! -f "$OUTPUT_PATH/full/protocol.json" ]]; then
    echo 'Stage one has no full evaluation protocol yet. Run stage1 first.' >&2
    return 2
  fi
  "$PYTHON_PATH" "$GATE_PATH" "${COMMON_ARGS[@]}" "${GATE_ARGS[@]}"
}

if [[ "$STAGE" == status ]]; then check_gate; exit $?; fi
if [[ "$STAGE" == stage2 ]]; then
  # Missing, failed, altered or out-of-tolerance references prevent GPU launch.
  check_gate
fi
"$PYTHON_PATH" "$RUNNER_PATH" run "${COMMON_ARGS[@]}" --methods "$MODELS" --gpu "$GPU_INDEX"
if [[ "$STAGE" == stage1 ]]; then
  if check_gate; then
    echo "Stage one is complete and within tolerance. Stage two is available: bash $0 stage2 --gpu $GPU_INDEX ${GATE_ARGS[*]}"
  else
    GATE_STATUS=$?
    [[ "$GATE_STATUS" -eq 2 ]] || exit "$GATE_STATUS"
    echo 'Stage-one evaluations are complete, but the performance gate did not pass. Review stage1_gate.md.'
    echo 'Stage two remains blocked; --accept-barely permits at most ±5 pp per pair.'
  fi
else
  "$PYTHON_PATH" "$RUNNER_PATH" summarize "${COMMON_ARGS[@]}" --methods "$REFERENCE_MODELS,$LORA_MODELS"
  echo "Both stages complete. Combined results: $OUTPUT_PATH/full/paper_comparison.md"
fi
