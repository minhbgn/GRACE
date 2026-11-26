#!/usr/bin/env bash
# Download the three gated video suites; see --help and README_datasets.md.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python3}" -u "$SCRIPT_DIR/download_video_benchmarks.py" "$@"
