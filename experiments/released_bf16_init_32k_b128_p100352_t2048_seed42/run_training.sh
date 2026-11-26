#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
source "$ROOT/qwen-vl-finetune/scripts/recipes/constrained_32k_b128_p100352_t2048_seed42.env"
exec "${PYTHON:-python}" "$ROOT/qwen-vl-finetune/scripts/run_constrained_matrix.py" --config "$HERE/experiment.json" "$@"
