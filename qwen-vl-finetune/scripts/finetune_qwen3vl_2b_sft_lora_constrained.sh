#!/usr/bin/env bash
# Matched BF16 SFT: 2 A100-40GB, fixed subset, one epoch. See README_constrained.md.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python}" "${SCRIPT_DIR}/run_constrained.py" --method sft "$@"
