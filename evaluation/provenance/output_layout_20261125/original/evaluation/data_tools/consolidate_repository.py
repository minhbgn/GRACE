#!/usr/bin/env python3
"""Plan, apply or verify the reversible 2026-11-25 artifact consolidation.

Moves preserve bytes and timestamps. Relative compatibility links preserve pinned
absolute paths in old jobs; no completion, prediction or protocol is rewritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / 'results/migration_20261125.json'
TOKEN = 'qat_int4_token_reduction_576_seed42'
ARCHIVE = ROOT / 'experiments/archive/cleanup_20261125'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def planned_moves():
    moves = []
    for exp in sorted((ROOT / 'experiments').iterdir()):
        if not exp.is_dir() or exp.name == 'archive':
            continue
        for source in sorted(exp.iterdir()):
            if source.is_symlink():
                continue
            if source.is_dir() and (source.name.startswith('results') or source.name in
                    {'outputs', 'predictions', 'local_development', 'video_evaluation',
                     'lmms_evaluation', 'memory_checks', 'reports'}):
                category = 'runs' if source.name.startswith('results') else 'evidence'
                dest = ROOT / 'results' / exp.name / category / source.name
                moves.append((source, dest, True))
        # Dated diagnostics/report builders are superseded by compile_results.py.
        helpers = list((exp / 'reports').rglob('*.py'))
        if exp.name == TOKEN:
            helpers += [exp / name for name in ('compile_report.py', 'compile_video_progress.py',
                'resume_after_scheduler_change.py', 'run_remaining_qat_videos_gpu1.sh',
                'run_images.sh', 'run_videos.sh', 'run_videomme_timing_subset.sh',
                'run_videomme_timing_phases.sh')]
        # Move helpers before their enclosing reports directory.
        for source in sorted(helpers):
            if source.is_file():
                moves.insert(0, (source, ARCHIVE / exp.name / source.relative_to(exp), False))
    # Restored duplicate root implementations bypass the organized package.
    for source in sorted((ROOT / 'evaluation').iterdir()):
        if source.is_file() and (source.suffix in {'.py', '.sh'} and source.name not in
                {'__init__.py', 'paths.py'} or source.name.startswith('README') or
                source.name == 'differences_from_org.md'):
            moves.append((source, ROOT / 'evaluation/provenance/cleanup_20261125/root' / source.name, False))
    return moves


def verify(record):
    count = 0
    for move in record['moves']:
        destination = ROOT / move['destination']
        if move['compatibility_link']:
            source = ROOT / move['source']
            if not source.is_symlink() or source.resolve() != destination.resolve():
                raise ValueError(f'Broken compatibility link: {source}')
        for name, checksum in move['files'].items():
            path = destination / name if destination.is_dir() else destination
            if sha(path) != checksum:
                raise ValueError(f'Artifact changed: {path}')
            count += 1
    print(f'Verified {count} relocated files and all compatibility links.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'apply', 'verify'], nargs='?', default='plan')
    args = parser.parse_args()
    if RECORD.exists():
        record = json.loads(RECORD.read_text())
        verify(record)
        return
    if args.action == 'verify':
        parser.error('No migration record exists')
    moves = planned_moves()
    if args.action == 'plan':
        for source, dest, link in moves:
            print(f'{source.relative_to(ROOT)} -> {dest.relative_to(ROOT)}' + (' [compatibility link]' if link else ''))
        return
    record = dict(schema=1, date='2026-11-25', moves=[])
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    for source, dest, link in moves:
        if dest.exists() or dest.is_symlink():
            raise FileExistsError(dest)
        files = {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*'))
                 if p.is_file() and '__pycache__' not in p.parts} if source.is_dir() else {'': sha(source)}
        entry = dict(source=str(source.relative_to(ROOT)), destination=str(dest.relative_to(ROOT)),
                     compatibility_link=link, files=files)
        dest.parent.mkdir(parents=True, exist_ok=True)
        source.rename(dest)
        if link:
            source.symlink_to(os.path.relpath(dest, source.parent), target_is_directory=True)
        record['moves'].append(entry)
        RECORD.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    verify(record)


if __name__ == '__main__':
    main()
