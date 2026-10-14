#!/usr/bin/env bash
# Larger input/batch recipe using the existing matched trainer, without changing it.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GRACE_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

usage() {
  cat <<'EOF'
Usage: bash run_constrained_b128_p100352_t2048.sh {sft|qat|grace|all} [mode] [--resume]
Modes: --profile (default), --train, --prepare-only, --dry-run
--dry-run prints resolved settings and the full-training command without execution.
Set PYTHON to the grace environment's Python and CUDA_VISIBLE_DEVICES to two free A100 IDs.
PROFILE_STEPS defaults to 10. EXPERIMENT_DIR may specify a fresh output directory.
EOF
}

METHOD=${1:-}
case "$METHOD" in sft|qat|grace|all) shift ;; -h|--help) usage; exit 0 ;; *) usage >&2; exit 2 ;; esac
MODE=--profile
MODE_SET=false
DRY_RUN=false
RESUME=false
for arg in "$@"; do
  case "$arg" in
    --profile|--train|--prepare-only)
      if "$MODE_SET"; then echo 'Choose only one execution mode.' >&2; exit 2; fi
      MODE=$arg; MODE_SET=true ;;
    --dry-run) DRY_RUN=true ;;
    --resume) RESUME=true ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
if "$RESUME" && { "$DRY_RUN" || [[ "$MODE" == --prepare-only ]]; }; then
  echo '--resume requires a profile or training execution.' >&2; exit 2
fi

source "${SCRIPT_DIR}/recipes/constrained_32k_b128_p100352_t2048_seed42.env"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${GRACE_ROOT}/experiments/constrained_32k_b128_p100352_t2048_seed42}"
PYTHON_BIN=${PYTHON:-python}
command -v "$PYTHON_BIN" >/dev/null || { echo 'Activate the grace environment or set PYTHON.' >&2; exit 1; }

ARGS=()
if "$DRY_RUN"; then
  ARGS+=(--dry-run)
elif [[ "$MODE" != --train ]]; then
  ARGS+=("$MODE")
fi
if "$RESUME"; then ARGS+=(--resume); fi

METHODS=("$METHOD")
if [[ "$METHOD" == all ]]; then METHODS=(sft qat grace); fi
# The subset is shared; preparing it once is sufficient.
if [[ "$MODE" == --prepare-only ]]; then METHODS=(grace); fi
if ! "$DRY_RUN" && [[ "$MODE" != --prepare-only ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo 'Set CUDA_VISIBLE_DEVICES to two free A100 GPU IDs before execution.' >&2; exit 2
  fi
  # Read-only ownership survey when the helper is available; never reclaim GPUs.
  if command -v gpuwho >/dev/null; then gpuwho; fi
fi

for method in "${METHODS[@]}"; do
  "$PYTHON_BIN" "${SCRIPT_DIR}/run_constrained.py" --method "$method" "${ARGS[@]}"
  if ! "$DRY_RUN" && [[ "$MODE" != --prepare-only ]]; then
    OUTPUT_KIND=runs
    [[ "$MODE" != --profile ]] || OUTPUT_KIND=profiles
    "$PYTHON_BIN" "${SCRIPT_DIR}/summarize_constrained_runtime.py" \
      --experiment "$EXPERIMENT_DIR" --kind "$OUTPUT_KIND" --methods "$method" \
      --require-complete
  fi
done
if ! "$DRY_RUN" && [[ "$MODE" != --prepare-only ]]; then
  "$PYTHON_BIN" "${SCRIPT_DIR}/summarize_constrained_runtime.py" \
    --experiment "$EXPERIMENT_DIR" --kind "$OUTPUT_KIND" --methods "${METHODS[@]}" \
    --baseline "${GRACE_ROOT}/experiments/constrained_32k_seed42" --require-complete
fi
