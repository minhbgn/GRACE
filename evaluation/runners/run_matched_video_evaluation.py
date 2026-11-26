#!/usr/bin/env python3
"""Fresh adaptive-policy QAT video evaluations, without historical answer reuse.

Accepts the same arguments as run_parallel_video_evaluation.py. Completions and
answer checkpoints within this launcher's output are still resumable.
"""
import argparse
from pathlib import Path
import sys
from unittest.mock import patch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.runners import run_parallel_video_evaluation as parallel

retry = parallel.retry
matrix = retry.matrix
DEFAULT_OUTPUT = retry.EXP / 'outputs/reruns/video_matched'
MODE = dict(schema=1, historical_completion_reuse=False,
            resume='verified completions and answer checkpoints in this output only')


def resolved(output, suite, config):
    directory = output / 'qat_w4' / suite / config
    done = matrix.verified_completion(directory)
    return done, 'retry' if done else None, directory


def inventory(output, suites):
    jobs, completed = [], []
    for suite in suites:
        for method, retention in retry.CONFIGS:
            config = matrix.configuration_id(method, retention)
            if resolved(output, suite, config)[0] is not None:
                completed.append(dict(suite=suite, config=config))
            else:
                jobs.append(dict(model='qat_w4', suite=suite, method=method, retention=retention))
    return jobs, [], completed


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('action', nargs='?', default='plan')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--smoke', type=int)
    args, _ = parser.parse_known_args()
    if '-h' in sys.argv or '--help' in sys.argv:
        parallel.main()
        return
    base = args.output.resolve()
    output = base / (f'smoke_{args.smoke}' if args.smoke else 'full')
    for protected in (retry.SOURCE, retry.DEFAULT_OUTPUT.resolve()):
        if output == protected or protected in output.parents or output in protected.parents:
            parser.error('Use a separate --output base for matched reruns')
    marker = output / 'matched_rerun.json'
    if (output / 'protocol.json').exists() and not marker.exists():
        parser.error('Existing output belongs to another launcher; use a fresh --output base')
    if marker.exists() and matrix.read(marker) != MODE:
        parser.error('Matched rerun mode changed; use a fresh --output base')
    if args.action in ('prepare', 'check', 'run'):
        matrix.pin(marker, MODE)
    own_file = str(Path(__file__).resolve().relative_to(matrix.ROOT))
    with patch.object(retry, 'resolved', resolved), patch.object(retry, 'inventory', inventory), \
         patch.object(parallel, 'EXTENSION_FILES', [*parallel.EXTENSION_FILES, own_file]), \
         patch.object(sys, 'argv', [*sys.argv, '--output', str(base)]):
        parallel.main()


if __name__ == '__main__':
    main()
