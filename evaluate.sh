#!/usr/bin/env bash
set -euo pipefail
EVAL_GRACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PYTHON="${LMMS_PYTHON:-$EVAL_GRACE_ROOT/.venv-lmms-eval/bin/python}"
if [[ ! -x "$EVAL_PYTHON" && -z "${LMMS_PYTHON:-}" ]]; then
    EVAL_PYTHON=/home/air/miniconda3/envs/grace/bin/python
fi
if [[ ! -x "$EVAL_PYTHON" ]]; then
    echo "Missing evaluation Python: $EVAL_PYTHON. Set LMMS_PYTHON or run evaluation/data_tools/prepare_lmms_eval.sh." >&2
    exit 1
fi
exec "$EVAL_PYTHON" "$EVAL_GRACE_ROOT/evaluation/runners/evaluate.py" "$@"
