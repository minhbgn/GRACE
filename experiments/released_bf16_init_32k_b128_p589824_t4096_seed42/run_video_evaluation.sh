#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR_VIDEO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GRACE_ROOT_VIDEO="$(cd -- "$EXPERIMENT_DIR_VIDEO/../.." && pwd)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "${LMMS_PYTHON:-$GRACE_ROOT_VIDEO/.venv-lmms-eval/bin/python}" \
  "$GRACE_ROOT_VIDEO/evaluation/runners/run_lmms_matrix.py" \
  --video "${1:-plan}" --experiment "$EXPERIMENT_DIR_VIDEO/evaluation.json" \
  --output "$EXPERIMENT_DIR_VIDEO/video_evaluation/no_local_sft_v1" "${@:2}"
