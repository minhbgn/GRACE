#!/usr/bin/env bash
# Continue the report's model order: GRACE W4, QAT W8, GRACE W8, on PCI GPU 0.
# Keep the existing short/medium output namespace, completion skips and resume.
set -euo pipefail

FROM_GRACE_EXP="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FROM_GRACE_ACTION="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
export VIDEO_GPU=0
exec bash "$FROM_GRACE_EXP/run_video_evaluation_short_medium.sh" "$FROM_GRACE_ACTION" \
    --methods grace_w4,qat_w8,grace_w8 "$@"
