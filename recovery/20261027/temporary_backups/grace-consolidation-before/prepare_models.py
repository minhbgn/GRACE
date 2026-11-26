#!/usr/bin/env python3
"""Record local checkpoint provenance, optionally acquire immutable author releases."""
import argparse
import os
from pathlib import Path
from common import (HERE, REPOS, BF16_REVISION, model_paths, pin, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Download the three author releases; network required')
    parser.add_argument('--models', nargs='+', choices=tuple(REPOS), default=tuple(REPOS))
    args = parser.parse_args()
    experiment = Path(os.environ.get('EXPERIMENT_DIR', HERE)).resolve()
    paths = model_paths(experiment)
    for name in args.models:
        if args.download and name.startswith('released_'):
            from huggingface_hub import HfApi, snapshot_download
            repo = REPOS[name]
            revision = BF16_REVISION if name == 'released_bf16' else HfApi().model_info(repo).sha
            # Pin before downloading, and reuse the same immutable revision after interruption.
            revision_file = experiment / 'provenance' / f'{name}_download.json'
            if revision_file.exists():
                from common import read
                previous = read(revision_file)
                if previous['repository'] != repo:
                    raise ValueError(f'Repository mismatch: {revision_file}')
                revision = previous['revision']
            pin(revision_file, dict(repository=repo, revision=revision, path=str(paths[name])))
            snapshot_download(repo_id=repo, revision=revision, local_dir=paths[name],
                              allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.md', '*.txt', '*.model'])
        print(f'Hashing {name}: {paths[name]}', flush=True)
        record = provenance(paths[name], REPOS[name], released_student=name == 'released_bf16')
        pin(experiment / 'provenance' / f'{name}.json', record)
        card = paths[name] / 'README.md'
        if card.exists():
            target = experiment / 'provenance' / f'{name}_model_card.md'
            if target.exists() and target.read_bytes() != card.read_bytes():
                raise ValueError(f'Model card changed: {card}')
            target.write_bytes(card.read_bytes())
    print('Checkpoint provenance recorded. No models allocated or GPUs used.')


if __name__ == '__main__':
    main()
