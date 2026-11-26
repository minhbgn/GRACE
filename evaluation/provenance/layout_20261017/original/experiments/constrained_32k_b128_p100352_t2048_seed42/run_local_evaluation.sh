#!/usr/bin/env bash
# Evaluate the final exports from this experiment with the shared constrained protocol.
set -euo pipefail

EXPERIMENT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GRACE_ROOT=$(cd -- "$EXPERIMENT_DIR/../.." && pwd)
EVALUATOR="$GRACE_ROOT/evaluation/run_all_constrained_evals.sh"
SUMMARIZER="$GRACE_ROOT/evaluation/summarize_constrained_evals.py"

# Match constrained_32k_seed42: vanilla plus every final training export. Pin the
# set here so a stale METHODS value in the calling shell cannot omit a model.
METHODS="baseline sft qat grace"
OUT_DIR=${OUT_DIR:-"$EXPERIMENT_DIR/evaluation"}
if [[ -z "${PYTHON:-}" && -x /home/air/miniconda3/envs/grace/bin/python ]]; then
    PYTHON=/home/air/miniconda3/envs/grace/bin/python
fi
PYTHON=${PYTHON:-python}

export EXPERIMENT_DIR METHODS OUT_DIR PYTHON

if (( $# == 0 )); then
    # Preserve completed suites on restart and do not let one failed suite prevent
    # the remaining methods/suites from being attempted.
    bash "$EVALUATOR" --resume --keep-going
else
    bash "$EVALUATOR" "$@"
fi

# A normal full run should also produce the same validated comparison artifacts
# as the earlier constrained experiment. Checks and dry-runs do not write them.
if (( $# == 0 )); then
    "$PYTHON" "$SUMMARIZER" --results-dir "$OUT_DIR" --write
fi
