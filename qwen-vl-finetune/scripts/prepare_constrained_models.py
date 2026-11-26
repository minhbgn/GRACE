#!/usr/bin/env python3
"""Record local checkpoint provenance, optionally acquire immutable author releases."""
import argparse
from pathlib import Path
from constrained_experiment import Experiment, pin, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Download the three author releases; network required')
    parser.add_argument('--config', required=True, type=Path, help='Experiment model matrix JSON')
    parser.add_argument('--models', nargs='+', help='Reference result IDs from the configuration')
    args = parser.parse_args()
    spec = Experiment(args.config)
    experiment = spec.directory
    paths = spec.paths
    references = [m for m, v in spec.models.items() if v.get('repository')]
    args.models = args.models or references
    if any(m not in references for m in args.models):
        parser.error('Choose reference models: ' + ', '.join(references))
    for name in args.models:
        model = spec.models[name]
        if args.download and model.get('download'):
            from huggingface_hub import HfApi, snapshot_download
            repo = model['repository']
            # Pin before downloading, and reuse the same immutable revision after interruption.
            revision_file = experiment / 'provenance' / f'{name}_download.json'
            if revision_file.exists():
                previous = read(revision_file)
                if previous['repository'] != repo:
                    raise ValueError(f'Repository mismatch: {revision_file}')
                revision = previous['revision']
            else:
                revision = model.get('revision') or HfApi().model_info(repo).sha
            pin(revision_file, dict(repository=repo, revision=revision, path=str(paths[name])))
            snapshot_download(repo_id=repo, revision=revision, local_dir=paths[name],
                              allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.md', '*.txt', '*.model'])
        print(f'Hashing {name}: {paths[name]}', flush=True)
        record = spec.provenance(name)
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
