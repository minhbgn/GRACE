#!/usr/bin/env bash
set -euo pipefail
RETRY_EXP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RETRY_GRACE_ROOT="$(cd -- "$RETRY_EXP_DIR/../.." && pwd)"
RETRY_PYTHON="${LMMS_PYTHON:-/home/air/miniconda3/envs/grace/bin/python}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset EXPERIMENT_DIR
exec "$RETRY_PYTHON" "$RETRY_GRACE_ROOT/evaluation/runners/run_parallel_video_evaluation.py" "$@"
