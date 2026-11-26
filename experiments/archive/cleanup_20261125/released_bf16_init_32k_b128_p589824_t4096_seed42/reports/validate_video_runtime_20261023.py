#!/usr/bin/env python3
"""Validate real GPU workers and kill only a child launched by this check."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
OUT = EXP / 'reports/resilient_runtime_validation_20261023'
OUT.mkdir(exist_ok=False)
RUN = EXP / 'run_video_evaluation.sh'
command = ['bash', str(RUN), 'run', '--gpu', '0', '--methods', 'qat_w4,grace_w4,qat_w8,grace_w8',
           '--suites', 'tempcompass', '--smoke', '3', '--output', str(OUT / 'smoke')]
killed = None
with (OUT / 'matrix.log').open('w') as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        while process.poll() is None:
            if killed is None:
                for path in sorted((OUT / 'smoke/smoke_3/grace_w4/tempcompass').glob('attempt_*/worker.json')):
                    attempt = path.parent
                    pred = attempt / 'predictions.jsonl'
                    status = attempt / 'status.json'
                    if pred.exists() and pred.stat().st_size > 0 and status.exists():
                        state = json.loads(status.read_text())
                        if state['stage'] == 'complete': continue
                        pid = json.loads(path.read_text())['pid']
                        cmdline = Path(f'/proc/{pid}/cmdline')
                        if cmdline.exists() and str(attempt / 'job.json').encode() in cmdline.read_bytes().split(b'\0'):
                            os.kill(pid, signal.SIGKILL)
                            killed = dict(pid=pid, attempt=str(attempt), status=state)
                            (OUT / 'injected_sigkill.json').write_text(json.dumps(killed, indent=2)+'\n')
                            print('Injected SIGKILL into own GRACE W4 validation worker', pid, flush=True)
                            break
            time.sleep(.025)
    except BaseException:
        process.terminate()
        process.wait(timeout=30)
        raise
if process.returncode or killed is None:
    raise RuntimeError(f'Validation failed: code={process.returncode}, killed={killed}; see {OUT}')
results = []
for method in ['qat_w4','grace_w4','qat_w8','grace_w8']:
    pair = OUT / 'smoke/smoke_3' / method / 'tempcompass'
    done = json.loads((pair / 'COMPLETE.json').read_text())
    assert done['summary']['count'] == 3 and done['summary']['generation_cutoffs'] == 0
    for name, sha in done['sha256'].items():
        assert hashlib.sha256((pair / done['attempt'] / name).read_bytes()).hexdigest() == sha
    attempts = sorted(pair.glob('attempt_*'))
    if method == 'grace_w4':
        assert len(attempts) == 2
        assert json.loads((attempts[0] / 'exit.json').read_text())['signal'] == 'SIGKILL'
        assert json.loads((attempts[1] / 'resume.json').read_text())['predictions'] >= 1
    results.append(dict(method=method, count=3, attempts=len(attempts), complete=str(pair/'COMPLETE.json')))
(OUT / 'validation.json').write_text(json.dumps(dict(passed=True, injected_sigkill=killed, models=results),indent=2)+'\n')
print(json.dumps(results), flush=True)
