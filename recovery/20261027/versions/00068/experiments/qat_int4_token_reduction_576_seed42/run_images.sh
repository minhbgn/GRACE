#!/usr/bin/env bash
set -euo pipefail
TOKEN_IMAGE_EXP="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOKEN_IMAGE_ACTION="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
exec bash "$TOKEN_IMAGE_EXP/run_experiment.sh" "$TOKEN_IMAGE_ACTION" \
  --suites hallusionbench,mmbench,scienceqa,ai2d,mmmu,seedbench_img,mmstar "$@"
