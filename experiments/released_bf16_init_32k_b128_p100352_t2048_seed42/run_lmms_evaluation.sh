#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GRACE_PATH="$(cd -- "$EXPERIMENT_PATH/../.." && pwd)"
PYTHON_PATH="${LMMS_PYTHON:-$GRACE_PATH/.venv-lmms-eval/bin/python}"
export PYTHONPATH="$GRACE_PATH/third_party/lmms-eval-v0.7.1${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_PATH" "$GRACE_PATH/evaluation/run_lmms_matrix.py" --experiment "$EXPERIMENT_PATH/experiment.json" "$@"
