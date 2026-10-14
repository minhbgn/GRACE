#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
exec "${PYTHON:-python}" "$ROOT/evaluation/run_constrained_matrix_evals.py" --config "$HERE/experiment.json" "$@"
