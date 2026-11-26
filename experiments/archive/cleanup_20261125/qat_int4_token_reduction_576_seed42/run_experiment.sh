#!/usr/bin/env bash
set -euo pipefail
TOKEN_EXP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOKEN_GRACE_ROOT="$(cd -- "$TOKEN_EXP_DIR/../.." && pwd)"
TOKEN_ACTION="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
TOKEN_HAS_GPU=false
for TOKEN_ARG in "$@"; do
  case "$TOKEN_ARG" in --gpu|--gpu=*|--gpus|--gpus=*) TOKEN_HAS_GPU=true ;; esac
done
case "$TOKEN_ACTION" in
  plan|run|kernel-check)
    if [[ "$TOKEN_HAS_GPU" == false ]]; then set -- --gpus 0,2 "$@"; fi ;;
esac
unset EXPERIMENT_DIR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
TOKEN_PYTHON="${LMMS_PYTHON:-$TOKEN_GRACE_ROOT/.venv-lmms-eval/bin/python}"
if [[ ! -x "$TOKEN_PYTHON" ]]; then
  echo "Missing evaluation Python: $TOKEN_PYTHON. Set LMMS_PYTHON to a compatible environment or run evaluation/prepare_lmms_eval.sh." >&2
  exit 1
fi
exec "$TOKEN_PYTHON" \
  "$TOKEN_GRACE_ROOT/evaluation/runners/run_token_reduction_matrix.py" \
  "$TOKEN_ACTION" --experiment "$TOKEN_EXP_DIR/experiment.json" "$@"
