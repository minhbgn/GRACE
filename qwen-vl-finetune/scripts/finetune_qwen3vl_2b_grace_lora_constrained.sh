#!/usr/bin/env bash
# Matched W4G128 GRACE: QAT + GDKD + RCKA + adaptive IB control.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python}" "${SCRIPT_DIR}/run_constrained.py" --method grace "$@"
