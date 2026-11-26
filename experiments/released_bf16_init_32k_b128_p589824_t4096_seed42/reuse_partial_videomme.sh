#!/usr/bin/env bash
# CPU-only validation/import of saved VideoMME short/medium predictions.
set -euo pipefail
PARTIAL_VIDEO_EXP="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PARTIAL_VIDEO_ROOT="$(cd -- "$PARTIAL_VIDEO_EXP/../.." && pwd)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "${LMMS_PYTHON:-$PARTIAL_VIDEO_ROOT/.venv-lmms-eval/bin/python}" \
    "$PARTIAL_VIDEO_ROOT/evaluation/runners/reuse_partial_videomme.py" \
    --experiment "$PARTIAL_VIDEO_EXP/experiment.json" "$@"
