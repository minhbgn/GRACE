#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
unset EXPERIMENT_DIR BASE_MODEL_DIR TEACHER_MODEL_PATH SFT_MODEL_DIR
PYTHON_PATH="${LMMS_PYTHON:-$ROOT/.venv-lmms-eval/bin/python}"
# Evaluation-only model selection excludes local SFT; use a fresh pinned output root.
exec "$PYTHON_PATH" "$ROOT/evaluation/runners/run_local_development.py" \
  --experiment "$HERE/evaluation.json" --output "$HERE/local_development_layout2_no_local_sft_v1" \
  --protocol local_native_greedy256_v1 --methods lora "$@"
