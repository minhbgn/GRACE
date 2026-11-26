#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/recipe.env"
unset EXPERIMENT_DIR BASE_MODEL_DIR TEACHER_MODEL_PATH SFT_MODEL_DIR
exec "${PYTHON:-/home/air/miniconda3/envs/grace/bin/python}" "$HERE/profile_memory.py" "$@"
