#!/usr/bin/env bash
# Compatibility entry point. Full command reference: experiments/README.md.
set -euo pipefail
TOKEN_LAUNCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$TOKEN_LAUNCH_ROOT/evaluate.sh" "$@"
