#!/usr/bin/env bash
# Separate corrected local protocol. Does not invoke the paper-reproduction gate.
set -euo pipefail
EVAL_HUB="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GRACE_PATH="$(cd -- "$EVAL_HUB/../.." && pwd)"
PYTHON_PATH="${LMMS_PYTHON:-$GRACE_PATH/.venv-lmms-eval/bin/python}"
# Resolve each --experiment against its JSON directory, not an old shell export.
unset EXPERIMENT_DIR
exec "$PYTHON_PATH" "$GRACE_PATH/evaluation/run_local_development.py" "$@"
