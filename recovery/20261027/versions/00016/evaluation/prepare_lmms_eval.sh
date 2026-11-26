#!/usr/bin/env bash
# Prepare source templates and isolated evaluation dependencies; no inference.
set -euo pipefail
GRACE_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_PATH="$GRACE_PATH/third_party/lmms-eval-v0.7.1"
VENV_PATH="$GRACE_PATH/.venv-lmms-eval"
BASE_PYTHON="${GRACE_PYTHON:-/home/air/miniconda3/envs/grace/bin/python}"
COMMIT=88b23e2bfa16a1edbc16e9e238ed82130b3a4f56
if [[ ! -d "$SOURCE_PATH" ]]; then
  mkdir -p "$GRACE_PATH/third_party"
  git clone --depth 1 --branch v0.7.1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git "$SOURCE_PATH"
fi
[[ "$(git -C "$SOURCE_PATH" rev-parse HEAD)" == "$COMMIT" ]] || { echo 'Unexpected harness commit' >&2; exit 1; }
if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV_PATH"
fi
"$VENV_PATH/bin/python" -m pip install --no-deps decord==0.6.0 setuptools==80.9.0 wheel==0.45.1
"$VENV_PATH/bin/python" -m pip install --no-deps --no-build-isolation -e "$SOURCE_PATH"
export PYTHONPATH="$SOURCE_PATH${PYTHONPATH:+:$PYTHONPATH}"
"$VENV_PATH/bin/python" "$GRACE_PATH/evaluation/run_lmms_matrix.py" check
