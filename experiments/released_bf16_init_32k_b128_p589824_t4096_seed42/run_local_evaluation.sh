#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
unset EXPERIMENT_DIR BASE_MODEL_DIR TEACHER_MODEL_PATH SFT_MODEL_DIR
PYTHON_PATH="${LMMS_PYTHON:-$ROOT/.venv-lmms-eval/bin/python}"
# This output root is separate from all historical evaluations.
exec "$PYTHON_PATH" "$ROOT/evaluation/run_local_development.py" \
  --experiment "$HERE/experiment.json" --output "$HERE/local_development" \
  --protocol local_native_greedy256_v1 --methods lora "$@"
