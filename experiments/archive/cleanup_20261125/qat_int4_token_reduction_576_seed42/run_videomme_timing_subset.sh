#!/usr/bin/env bash
set -euo pipefail
TIMING_EXP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TIMING_GRACE_ROOT="$(cd -- "$TIMING_EXP_DIR/../.." && pwd)"
TIMING_PYTHON="${LMMS_PYTHON:-/home/air/miniconda3/envs/grace/bin/python}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [[ "${1:-}" == queue ]]; then
    shift
    exec "$TIMING_PYTHON" - "$TIMING_GRACE_ROOT" "$@" <<'PY'
from pathlib import Path
import shlex
import subprocess
import sys
root = Path(sys.argv[1])
output = root / 'experiments/qat_int4_token_reduction_576_seed42/results_videomme_timing50_seed42'
if not (output / 'protocol.json').exists():
    raise SystemExit('Run prepare first')
session = 'videomme-timing'
exists = subprocess.run(['tmux', 'has-session', '-t', '=' + session], capture_output=True)
if exists.returncode == 0:
    print(f'Tmux session already exists: tmux attach -t {session}')
    raise SystemExit(0)
command = 'exec ' + shlex.join([sys.executable, '-u',
    str(root / 'evaluation/runners/run_videomme_timing_subset.py'), 'run', '--wait', *sys.argv[2:]])
subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(root), command], check=True)
subprocess.run(['tmux', 'set-window-option', '-t', session + ':0', 'remain-on-exit', 'on'], check=True)
subprocess.run(['tmux', 'pipe-pane', '-o', '-t', session + ':0.0',
                'cat >> ' + shlex.quote(str(output / 'queue.log'))], check=True)
pid = subprocess.check_output(['tmux', 'display-message', '-p', '-t', session + ':0.0', '#{pane_pid}'], text=True).strip()
(output / 'queue.pid').write_text(pid + '\n')
(output / 'tmux_session.txt').write_text(session + '\n')
print(f'Queued timing controller PID {pid} in tmux: tmux attach -t {session}')
PY
fi
exec "$TIMING_PYTHON" "$TIMING_GRACE_ROOT/evaluation/runners/run_videomme_timing_subset.py" "${@:-prepare}"
