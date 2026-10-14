#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
source "$HERE/recipe.env"
# Never redirect this experiment into a previous run through a stale shell export.
unset EXPERIMENT_DIR BASE_MODEL_DIR TEACHER_MODEL_PATH SFT_MODEL_DIR
PYTHON_PATH="${PYTHON:-/home/air/miniconda3/envs/grace/bin/python}"
exec "$PYTHON_PATH" "$ROOT/qwen-vl-finetune/scripts/run_constrained_matrix.py" --config "$HERE/experiment.json" "$@"
