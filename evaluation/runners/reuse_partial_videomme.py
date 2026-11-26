#!/usr/bin/env python3
"""Certify saved short/medium predictions from unfinished VideoMME attempts, on CPU."""
from __future__ import annotations

import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import datetime as dt
import fcntl
import hashlib
import json

from evaluation.paths import EVALUATION_ROOT
from evaluation.protocols.video_population import VIDEOMME_SHORT_MEDIUM, select_population
from evaluation.protocols.video_protocol import DEFAULT_DATA, SUITES, digest, load_suite
from evaluation.runners.run_video_matrix import (DEFAULT, Experiment, completed, pin, read,
    select, sha, summary_from_rows, validate_attempt, write)
from evaluation.runners.run_video_matrix_resilient import summarize
from evaluation.workers.video_reuse import compatible_policy
from evaluation.workers.video_runtime import validate_prefix


def read_prefix(path, count):
    """Read only the selected population; a broken excluded tail is irrelevant."""
    records, checksum = [], hashlib.sha256()
    with Path(path).open('rb') as stream:
        for index in range(count):
            line = stream.readline()
            if not line or not line.endswith(b'\n'):
                raise ValueError(f'Only {index} complete prediction records; need {count}')
            records.append(json.loads(line))
            checksum.update(line)
    return records, checksum.hexdigest()


def validate_candidate(source_root, attempt, job, identity, model_provenance):
    """Validate provenance and every selected prediction without a full-run marker."""
    source_identity = read(source_root / 'protocol.json')
    if source_identity['limit'] is not None or source_identity['experiment'] != identity['experiment']:
        raise ValueError('Source is smoke or belongs to another experiment')
    if not compatible_policy(source_identity['policy'], job['protocol']):
        raise ValueError('Source inference policy differs')
    required = {'run_video_matrix.py', 'video_protocol.py', 'video_worker.py', 'local_scoring.py', 'paths.py'}
    required |= {'video_worker_resilient.py', 'video_runtime.py'} & set(source_identity['code'])
    if any(source_identity['code'].get(k) != identity['code'].get(k) for k in required):
        raise ValueError('Source computation code differs')
    if any(identity['environment'].get(k) != v for k, v in source_identity['environment'].items()):
        raise ValueError('Source package versions differ')
    if read(source_root / job['method'] / 'model_provenance.json') != model_provenance:
        raise ValueError('Source model provenance differs')
    original_bytes = (attempt / 'job.json').read_bytes()
    original = json.loads(original_bytes)
    if any(original[k] != job[k] for k in ('method', 'suite', 'model_path')):
        raise ValueError('Source job identity differs')
    if original['protocol'] != source_identity['policy'] or original['rows'][:len(job['rows'])] != job['rows']:
        raise ValueError('Source policy or ordered question identities differ')
    processor_bytes = (attempt / 'processor.json').read_bytes()
    processor = json.loads(processor_bytes)
    if processor['effective_policy'] != original['protocol']:
        raise ValueError('Source processor policy differs')
    records, checksum = read_prefix(attempt / 'predictions.jsonl', len(job['rows']))
    validate_prefix(records, job)
    if any(r['context_limit'] != processor['context_limit'] for r in records):
        raise ValueError('Source context telemetry differs')
    # No long-question results or whole-run summary are needed to certify this subset.
    return records, processor_bytes, dict(
        source_attempt=str(attempt.resolve()), source_run_identity=digest(source_identity),
        source_protocol=original['protocol'], source_questions=len(original['rows']),
        selected_questions=len(records), source_prefix_sha256=checksum,
        source_job_sha256=hashlib.sha256(original_bytes).hexdigest(),
        source_processor_sha256=hashlib.sha256(processor_bytes).hexdigest(),
        source_was_partial=not (attempt.parent / 'COMPLETE.json').exists(),
        importer_sha256=sha(Path(__file__)),
        note='Certified selected predictions from a snapshot of an unfinished run; no model inference. '
             'Processor metadata retains the source policy. Excluded long-question tail was not used.')


def import_candidate(job_dir, job, records, processor_bytes, audit):
    attempt = job_dir / ('attempt_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    attempt.mkdir(parents=True)
    write(attempt / 'job.json', job)
    with (attempt / 'predictions.jsonl').open('w') as stream:
        for row in records:
            stream.write(json.dumps(row) + '\n')
    (attempt / 'processor.json').write_bytes(processor_bytes)
    write(attempt / 'summary.json', summary_from_rows(records))
    write(attempt / 'reuse.json', audit)
    summary = validate_attempt(attempt, job['rows'], job['protocol'])
    pin(job_dir / 'COMPLETE.json', dict(attempt=attempt.name, summary=summary,
        sha256={name: sha(attempt / name) for name in
                ('job.json', 'predictions.jsonl', 'summary.json', 'processor.json', 'reuse.json')}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'import'], nargs='?', default='plan')
    parser.add_argument('--experiment', type=Path, default=DEFAULT / 'experiment.json')
    parser.add_argument('--target', type=Path, help='Initialized short/medium full/ directory; auto-detected by default')
    parser.add_argument('--source', type=Path, action='append', help='Source full/ directory; default scan this experiment')
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--methods', help='Default all experiment models')
    args = parser.parse_args()
    exp = Experiment(args.experiment)
    root = exp.directory / 'video_evaluation'
    if args.target is None:
        candidates = [p.parent for p in root.rglob('protocol.json')
                      if read(p).get('policy', {}).get('videomme_population') == VIDEOMME_SHORT_MEDIUM
                      and read(p).get('limit') is None]
        if len(candidates) != 1:
            parser.error('Specify --target: expected one initialized short/medium full/ directory')
        args.target = candidates[0]
    target = args.target.resolve()
    identity = read(target / 'protocol.json')
    if (identity['limit'] is not None or identity['experiment'] != exp.spec
            or identity['policy'].get('videomme_population') != VIDEOMME_SHORT_MEDIUM):
        raise ValueError('Target must be the initialized 1800-question run for this experiment')
    # Keep the existing target identity byte-for-byte; normal resume must still pass.
    for name, checksum in identity['code'].items():
        paths = [EVALUATION_ROOT / group / name for group in ('runners', 'workers', 'protocols', '')]
        file = next((p for p in paths if p.exists()), None)
        if file is None or sha(file) != checksum:
            raise ValueError(f'Target pinned computation changed: {name}')
    rows, info = load_suite('videomme', args.data_root)
    rows, info = select_population('videomme', rows, info, True)
    if len(rows) != 1800 or info['status'] != 'ready' or read(target / 'datasets/videomme.json') != info:
        raise ValueError('Target selected dataset changed or is incomplete')
    sources = args.source or [p.parent for p in root.rglob('protocol.json') if read(p).get('limit') is None]
    sources = sorted({p.resolve() for p in sources} - {target})
    reports = []
    with (target / '.matrix.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for method in select(args.methods, exp.models):
            destination = target / method / 'videomme'
            if completed(destination, rows, identity['policy']):
                reports.append(dict(method=method, status='already_complete'))
                continue
            model_provenance = read(target / method / 'model_provenance.json')
            job = dict(method=method, suite='videomme', model_path=str(exp.paths[method]), rows=rows,
                       protocol=identity['policy'], scratch=None, run_identity=digest(identity))
            attempts = sorted({(source, attempt.resolve()) for source in sources
                               for attempt in (source / method / 'videomme').glob('attempt_*')},
                              key=lambda pair: (pair[1].name, str(pair[0])), reverse=True)
            rejected, chosen = [], None
            for source, attempt in attempts:
                try:
                    records, processor_bytes, audit = validate_candidate(source, attempt, job, identity, model_provenance)
                except (ValueError, KeyError, OSError) as error:
                    rejected.append(dict(attempt=str(attempt), reason=str(error)))
                    continue
                chosen = dict(method=method, status='eligible', source_attempt=str(attempt), count=len(records))
                if args.action == 'import':
                    summary = import_candidate(destination, job, records, processor_bytes, audit)
                    chosen.update(status='imported', accuracy_percent=100 * summary['accuracy'])
                break
            report = chosen or dict(method=method, status='pending')
            report['rejected'] = rejected
            reports.append(report)
            print(json.dumps({k: v for k, v in report.items() if k != 'rejected'}), flush=True)
        if args.action == 'import':
            write(target / 'partial_videomme_reuse.json', dict(target=str(target), models=reports))
            datasets = {'videomme': rows}
            for suite in SUITES:
                if suite == 'videomme' or not (target / 'datasets' / (suite + '.json')).exists():
                    continue
                other_rows, other_info = load_suite(suite, args.data_root)
                if other_info['status'] == 'ready':
                    if read(target / 'datasets' / (suite + '.json')) != other_info:
                        raise ValueError(f'Target dataset changed: {suite}')
                    datasets[suite] = other_rows
            summarize(target, list(exp.models), datasets, identity['policy'])
    print(json.dumps(dict(target=str(target), models=reports), indent=2))


if __name__ == '__main__':
    main()
