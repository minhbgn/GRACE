#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR_VIDEO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GRACE_ROOT_VIDEO="$(cd -- "$EXPERIMENT_DIR_VIDEO/../.." && pwd)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "${LMMS_PYTHON:-$GRACE_ROOT_VIDEO/.venv-lmms-eval/bin/python}" \
  "$GRACE_ROOT_VIDEO/evaluation/run_video_matrix.py" \
  "${1:-plan}" --experiment "$EXPERIMENT_DIR_VIDEO/experiment.json" "${@:2}"
