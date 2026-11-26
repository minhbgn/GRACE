#!/usr/bin/env python3
"""Fork saved results for the user-confirmed, scheduling-only memory change.

Dry-run by default. No GPUs are used. The original results are never rewritten.
Only the exact old/new source hashes inspected for this transition are accepted.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.protocols.evaluation_checkpoints import digest, records

CHANGES = {
    'evaluation/protocols/parallel_evaluation.py': (
        'c5f6b0348518cc7299b38d71bc134f27db1787e4e4f2101f08727bd03db7de00',
        '67842bbffd29601723938545d32dfbfdb6417ed37ac74d3d950ac50c04925cdf'),
    'evaluation/runners/run_parallel_video_evaluation.py': (
        'ada72accac5ad81ea13d5bd37769321b78e820b8fc3dc30b799bb156b41bffee',
        '9d9b461bf316ab204cd27c0c421c951acbf59ea0ad451e8fa3dd4d323d38e0b7'),
}
IDENTITY_KEYS = (
    'model', 'model_path', 'suite', 'method', 'retention', 'policy', 'execution',
    'sidecar_sha256', 'limit', 'resume_provenance', 'extension_code')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def identity(job):
    return digest({key: job[key] for key in IDENTITY_KEYS})


def inspect(source):
    old = read(source / 'parallel_implementation.json')
    if old['schema'] != 1 or old['limit'] is not None:
        raise ValueError('Expected full-population schema 1 results')
    new_code = {name: sha(ROOT / name) for name in old['code']}
    changed = {name for name in new_code if new_code[name] != old['code'][name]}
    if changed != set(CHANGES) or any(
        (old['code'][name], new_code[name]) != pair for name, pair in CHANGES.items()
    ):
        raise ValueError('Source hashes differ from the reviewed scheduling-only transition')
    pending = []
    completed = 0
    for directory in sorted((source / 'qat_w4').glob('*/*')):
        if not directory.is_dir():
            continue
        marker = directory / 'COMPLETE.json'
        if marker.exists():
            done = read(marker)
            for name, checksum in done['sha256'].items():
                if sha(directory / done['attempt'] / name) != checksum:
                    raise ValueError(f'Completed artifact changed: {directory / name}')
            completed += 1
            continue
        for attempt in sorted(directory.glob('attempt_*')):
            rows = records(attempt / 'answers.checkpoint.jsonl')
            if not rows:
                continue
            job = read(attempt / 'job.json')
            if job['extension_code'] != old['code']:
                raise ValueError(f'Unexpected attempt code: {attempt}')
            before = identity(job)
            if any(row['identity'] != before for row in rows):
                raise ValueError(f'Unexpected checkpoint identity: {attempt}')
            job['extension_code'] = new_code
            pending.append((attempt.relative_to(source), rows, job, before, identity(job)))
    return old, dict(old, code=new_code), pending, completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=EXP / 'results_video_retry_gpu1',
                        help='Existing output base; /full is appended')
    parser.add_argument('--output', type=Path, default=EXP / 'results_video_retry_gpu0_scheduler_resume',
                        help='NEW output base; must not already exist')
    parser.add_argument('--apply', action='store_true', help='Create and migrate the copy')
    args = parser.parse_args()
    source = args.source.resolve() / 'full'
    destination = args.output.resolve()
    if destination.exists() or destination == source or source in destination.parents:
        parser.error('--output must be a new directory outside the source results')
    # The controller owns this lock until all its children and holders exit.
    with (source / '.matrix.lock').open('r') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Stop the old controller and wait for it to exit before migrating')
        old, new, pending, completed = inspect(source)
        report = dict(
            source=str(source), output=str(destination / 'full'),
            reason='User-confirmed scheduling-only memory budget change; evaluation computation unchanged',
            old_implementation=old, new_implementation=new,
            completed_retry_configurations=completed,
            checkpoints=[dict(attempt=str(path), answers=len(rows),
                              old_identity=before, new_identity=after)
                         for path, rows, job, before, after in pending])
        print(json.dumps(report, indent=2), flush=True)
        if not args.apply:
            print('DRY RUN: no results changed. Stop the old run, then rerun with --apply.')
            return
        # A sibling staging directory prevents a failed copy becoming runnable.
        staging = destination.with_name(destination.name + '.migration_in_progress')
        staging.mkdir()
        target = staging / 'full'
        shutil.copytree(source, target)
        for path, rows, job, before, after in pending:
            attempt = target / path
            write(attempt / 'job.json', job)
            with (attempt / 'answers.checkpoint.jsonl').open('w') as stream:
                for row in rows:
                    row = dict(row, identity=after)
                    stream.write(json.dumps(dict(row, checksum=digest(row))) + '\n')
            assert len(records(attempt / 'answers.checkpoint.jsonl')) == len(rows)
        write(target / 'parallel_implementation.json', new)
        report['created_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        report['migration_script_sha256'] = sha(Path(__file__))
        write(target / 'scheduler_resume_migration.json', report)
        staging.rename(destination)
        print(f'Ready. Resume with --output {destination}')


if __name__ == '__main__':
    main()
