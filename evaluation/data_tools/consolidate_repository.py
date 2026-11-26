#!/usr/bin/env python3
"""Verify preserved artifacts after the completed output/report relocation.

The original migration manifest is immutable. The explicit second relocation
map resolves its destinations without retaining surface-level aliases.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.protocols.artifact_layout import LAYOUT, resolve_artifact, sha

RECORD = ROOT / 'reports/provenance/migration_20261125.json'


def verify():
    record = json.loads(RECORD.read_text())
    layout = json.loads((LAYOUT / 'paths.json').read_text())
    removed = set(layout['removed_aliases'])
    count = 0
    for move in record['moves']:
        destination = resolve_artifact(move['destination'])
        if move['compatibility_link'] and move['source'] not in removed:
            source = ROOT / move['source']
            if not source.is_symlink() or source.resolve() != destination.resolve():
                raise ValueError(f'Broken evidence link: {source}')
        for name, checksum in move['files'].items():
            # Individual report artifacts may have moved to figures/diagnostics.
            old_path = Path(move['destination']) / name if name else Path(move['destination'])
            path = resolve_artifact(old_path)
            if sha(path) != checksum:
                raise ValueError(f'Artifact changed: {path}')
            count += 1
    for name in removed:
        if (ROOT / name).is_symlink():
            raise ValueError(f'Obsolete surface alias remains: {name}')
    print(f'Verified {count} relocated files; obsolete surface aliases are removed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'verify'], nargs='?', default='verify')
    args = parser.parse_args()
    if args.action == 'plan':
        print((LAYOUT / 'paths.json').read_text())
        return
    verify()


if __name__ == '__main__':
    main()
