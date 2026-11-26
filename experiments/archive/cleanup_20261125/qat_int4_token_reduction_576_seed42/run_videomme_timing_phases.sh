#!/usr/bin/env bash
# Manually run each phase in tmux. No phase automatically launches another.
set -euo pipefail
TIMING_PHASE_EXP="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TIMING_PHASE_ROOT="$(cd -- "$TIMING_PHASE_EXP/../.." && pwd)"
TIMING_PHASE_PYTHON="${LMMS_PYTHON:-/home/air/miniconda3/envs/grace/bin/python}"
TIMING_PHASE_RUNNER="$TIMING_PHASE_ROOT/evaluation/runners/run_videomme_timing_subset.py"
TIMING_PHASE_SEED="${TIMING_SUBSET_SEED:-42}"
TIMING_PHASE_OUTPUT="${TIMING_OUTPUT:-$TIMING_PHASE_EXP/results_videomme_timing50_seed${TIMING_PHASE_SEED}_phased_20261123}"
TIMING_PHASE_GPUS="${TIMING_GPUS:-4}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
case "${1:-help}" in
    prepare)
        exec "$TIMING_PHASE_PYTHON" "$TIMING_PHASE_RUNNER" prepare \
            --output "$TIMING_PHASE_OUTPUT" --seed "$TIMING_PHASE_SEED"
        ;;
    resolution)
        exec "$TIMING_PHASE_PYTHON" "$TIMING_PHASE_RUNNER" run \
            --output "$TIMING_PHASE_OUTPUT" --gpus "$TIMING_PHASE_GPUS" --wait \
            --configurations resolution_r050,resolution_r025,resolution_r010
        ;;
    check)
        exec "$TIMING_PHASE_PYTHON" "$TIMING_PHASE_RUNNER" check-projection --output "$TIMING_PHASE_OUTPUT"
        ;;
    remaining)
        # Rebuild the check from hashed completions rather than trust a stale report.
        "$TIMING_PHASE_PYTHON" "$TIMING_PHASE_RUNNER" check-projection --output "$TIMING_PHASE_OUTPUT"
        "$TIMING_PHASE_PYTHON" - "$TIMING_PHASE_OUTPUT/projection_check.json" <<'PY'
import json
from pathlib import Path
import sys
report = json.loads(Path(sys.argv[1]).read_text())
if report['recommendation'] != 'ready_for_manual_review_before_uniform_merge':
    raise SystemExit('Projection checks need review; remaining methods were not launched. Read projection_check.md.')
PY
        exec "$TIMING_PHASE_PYTHON" "$TIMING_PHASE_RUNNER" run \
            --output "$TIMING_PHASE_OUTPUT" --gpus "$TIMING_PHASE_GPUS" --wait \
            --configurations baseline_r100,uniform_r050,uniform_r025,uniform_r010,merge_r050,merge_r025,merge_r010
        ;;
    summarize)
        exec "$TIMING_PHASE_PYTHON" "$TIMING_PHASE_RUNNER" summarize --output "$TIMING_PHASE_OUTPUT"
        ;;
    *)
        echo 'Usage: bash run_videomme_timing_phases.sh {prepare|resolution|check|remaining|summarize}'
        echo "Output: $TIMING_PHASE_OUTPUT"
        echo 'Overrides: TIMING_GPUS (default 4), TIMING_SUBSET_SEED (default 42), TIMING_OUTPUT'
        ;;
esac
