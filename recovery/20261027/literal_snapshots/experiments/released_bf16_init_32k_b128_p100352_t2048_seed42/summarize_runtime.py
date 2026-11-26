#!/usr/bin/env python3
"""Report cost of the four new constrained runs (CPU only)."""
import argparse
import json
import os
from pathlib import Path
import sys
from common import HERE, PROJECT, model_paths, read, validate_export
sys.path.insert(0, str(PROJECT / 'scripts'))
from summarize_constrained_runtime import read_measurement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('runs', 'profiles'), default='runs')
    args = parser.parse_args()
    experiment = Path(os.environ.get('EXPERIMENT_DIR', HERE)).resolve()
    rows = []
    for bits in (4, 8):
        for method in ('qat', 'grace'):
            group = experiment / f'w{bits}g128'
            output = group / args.kind / method
            data = read_measurement(group, args.kind, method, 1)
            if data is None or not data['complete']:
                raise ValueError(f'Missing or incomplete output: {output}')
            if args.kind == 'runs':
                validate_export(output, bits, model_paths(experiment)['released_bf16'])
            invocations = [json.loads(line) for line in (output / 'invocations.jsonl').read_text().splitlines()]
            wall = sum(row['wall_seconds'] for row in invocations)
            training = sum(row.get('train_runtime_seconds', 0) for row in invocations)
            row = dict(data, result_id=f'{method}_w{bits}',
                       trainable_parameters=sum(p['parameters'] for p in read(output / 'trainable_parameters.json').values()),
                       process_wall_hours=wall / 3600, process_gpu_hours=2 * wall / 3600,
                       recorded_training_hours=training / 3600,
                       outside_recorded_training_hours=(wall - training) / 3600)
            rows.append(row)
    lines = ['# Constrained-stage runtime and memory', '',
             f'Kind: {args.kind}. Costs cover only these four new runs, including logged resumed/failed invocations.', '',
             '| Result | Updates | Trainable parameters | Process hours | GPU-hours (2 GPUs) | Recorded training hours | Other hours | Peak allocated GiB | Peak reserved GiB |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        values = [row['result_id'], row['steps'], row['trainable_parameters']]
        values += [f'{row[key]:.3f}' for key in ('process_wall_hours', 'process_gpu_hours', 'recorded_training_hours',
                                                'outside_recorded_training_hours', 'peak_allocated_gib', 'peak_reserved_gib')]
        lines.append('| ' + ' | '.join(map(str, values)) + ' |')
    lines += ['', 'Process time is measured around torchrun and includes loading, training, checkpoints and export. '
              'Other hours are process time outside recorded Trainer intervals, including loading/export and failures without a Trainer summary; those components are not separately instrumented. '
              'CPU preparation/provenance and the author’s prior training are excluded. A hard kill of the launcher can prevent recording its invocation.', '',
              'Memory is the maximum PyTorch allocated/reserved value across ranks at logged training boundaries; loading/export peaks and total device occupancy are not measured. '
              'Profiles repeat the longest examples and do not measure representative training throughput.']
    report_dir = experiment / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    name = 'runtime_memory' if args.kind == 'runs' else 'profile_runtime_memory'
    (report_dir / f'{name}.md').write_text('\n'.join(lines) + '\n')
    (report_dir / f'{name}.json').write_text(json.dumps(rows, indent=2) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
