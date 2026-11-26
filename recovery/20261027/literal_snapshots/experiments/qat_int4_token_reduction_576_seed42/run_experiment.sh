#!/usr/bin/env bash
set -euo pipefail
TOKEN_EXP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOKEN_GRACE_ROOT="$(cd -- "$TOKEN_EXP_DIR/../.." && pwd)"
TOKEN_ACTION="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "${LMMS_PYTHON:-$TOKEN_GRACE_ROOT/.venv-lmms-eval/bin/python}" \
  "$TOKEN_GRACE_ROOT/evaluation/runners/run_token_reduction_matrix.py" \
  "$TOKEN_ACTION" --experiment "$TOKEN_EXP_DIR/experiment.json" "$@"
