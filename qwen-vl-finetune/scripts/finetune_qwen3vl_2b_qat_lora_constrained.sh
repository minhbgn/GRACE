#!/usr/bin/env bash
# Matched W4G128 QAT: same student/data/adapters as GRACE, CE only.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python}" "${SCRIPT_DIR}/run_constrained.py" --method qat "$@"
