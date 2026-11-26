#!/usr/bin/env bash
# lmms-eval 720p, all models: QAT W4, QAT W8, vanilla 2B first, on GPU 0.
# VideoMME long is excluded centrally. Completed pairs skip; failed pairs retry.
# Override with --gpu / --methods; --methods "" selects all experiment models.
# plan/check/summarize do not load models. Rerun the same run command to retry incomplete pairs.
set -euo pipefail

VIDEO_GPU="${VIDEO_GPU:-0}"
VIDEO_METHODS="${VIDEO_METHODS:-}"
SHORT_MEDIUM_EXP="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SHORT_MEDIUM_ACTION="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
case "$SHORT_MEDIUM_ACTION" in
    plan|check|prepare|run|summarize) ;;
    *) printf 'Usage: bash %s [plan|check|prepare|run|summarize] [runner options...]\n' "$0" >&2; exit 2 ;;
esac

exec bash "$SHORT_MEDIUM_EXP/run_video_evaluation.sh" "$SHORT_MEDIUM_ACTION" \
    --gpu "$VIDEO_GPU" --methods "$VIDEO_METHODS" --skip-videomme-long "$@"
