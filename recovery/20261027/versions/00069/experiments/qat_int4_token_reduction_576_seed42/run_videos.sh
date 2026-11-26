#!/usr/bin/env bash
set -euo pipefail
TOKEN_VIDEO_EXP="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOKEN_VIDEO_ACTION="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
exec bash "$TOKEN_VIDEO_EXP/run_experiment.sh" "$TOKEN_VIDEO_ACTION" \
  --suites tempcompass,videomme,videommmu_perception,videommmu_comprehension,videommmu_adaptation "$@"
