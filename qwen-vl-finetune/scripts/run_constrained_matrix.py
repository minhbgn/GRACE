#!/usr/bin/env python3
"""Run a configured W4/W8 QAT/GRACE matrix through the shared constrained launcher."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from constrained_experiment import Experiment, PROJECT, check_processor_compatibility, pin, read, sha
import run_constrained as shared


def prepare(spec, c):
    experiment = spec.directory
    with shared.lock(experiment / '.prepare.lock'):
        manifest = shared.prepare_subset(c, experiment / 'subset.json')
        old_path = shared.ROOT / spec.spec['previous_experiment'] / 'subset.json'
        comparison = dict(previous_manifest=str(old_path), records_sha256=manifest['records_sha256'])
        if old_path.exists():
            old = read(old_path)
            previous = {r['index']: r for r in old['records']}
            current = {r['index']: r for r in manifest['records']}
            common = previous.keys() & current.keys()
            comparison.update(previous_records_sha256=old['records_sha256'],
                              removed=sorted(previous.keys() - current.keys()),
                              added=sorted(current.keys() - previous.keys()),
                              changed_conversations=[i for i in sorted(common)
                                  if previous[i]['sample']['conversations'] != current[i]['sample']['conversations']],
                              changed_token_hashes=[i for i in sorted(common)
                                  if previous[i]['token_sha256'] != current[i]['token_sha256']],
                              same_order=[r['index'] for r in old['records']] == [r['index'] for r in manifest['records']])
        pin(experiment / 'subset_comparison.json', comparison)
        for bits in spec.bits:
            target = experiment / f'w{bits}g128/subset.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                if sha(target) != sha(experiment / 'subset.json'):
                    raise ValueError(f'Mismatched shared subset: {target}')
            else:
                target.symlink_to('../subset.json')
        return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path, help='Experiment model matrix JSON')
    p.add_argument('method', choices=('qat', 'grace', 'all'))
    p.add_argument('--bits', choices=('4', '8', 'all'), default='all')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--profile', action='store_true', help='Default: stress profile, separate from full runs')
    modes.add_argument('--train', action='store_true')
    modes.add_argument('--prepare-only', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='No reads of weights, writes, preparation or GPU access')
    p.add_argument('--resume', action='store_true', help='Only resume native checkpoints from this exact recipe')
    args = p.parse_args()
    if args.resume and (args.prepare_only or args.dry_run):
        p.error('--resume requires a profile or training execution')
    spec = Experiment(args.config)
    experiment = spec.directory
    paths = spec.paths
    c = shared.resolved_config()
    c['model_path'] = str(paths[spec.student])
    c['teacher_path'] = str(paths[spec.teacher])
    c['student_weights_sha256'] = spec.models[spec.student]['weights_sha256']
    bits_set = spec.bits if args.bits == 'all' else (int(args.bits),)
    if any(bits not in spec.bits for bits in bits_set):
        p.error('Selected bit width is not configured in this experiment')
    methods = ('grace', 'qat') if args.method == 'all' else (args.method,)
    profile = not args.train
    for bits in bits_set:
        config = dict(c, qat_bits=bits)
        for method in methods:
            command, _ = shared.build_command(method, config, experiment / f'w{bits}g128', profile, args.resume)
            print(f'[{method} W{bits}G128] ' + shlex.join(command), flush=True)
    if args.dry_run:
        return
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    print('Verifying released student bytes and processor compatibility (CPU only)...', flush=True)
    student = spec.provenance(spec.student)
    pin(experiment / 'provenance' / f'{spec.student}.json', student)
    # Validate teacher vocabulary/template even for QAT so both methods share a valid recipe.
    check_processor_compatibility([paths[spec.student], paths[spec.teacher]])
    pin(experiment / 'provenance' / f'{spec.teacher}.json', spec.provenance(spec.teacher))
    manifest = prepare(spec, c)
    base_recipe = shared.comparison_recipe(c, manifest)
    base_recipe['implementation'].update({f'scripts/{path.name}': sha(path) for path in
        (Path(__file__), Path(__file__).with_name('constrained_experiment.py'))})
    for bits in bits_set:
        group = experiment / f'w{bits}g128'
        config = dict(c, qat_bits=bits)
        recipe = dict(base_recipe, config=config,
                      experiment_config_sha256=sha(spec.config_path),
                      student_provenance_sha256=sha(experiment / 'provenance' / f'{spec.student}.json'),
                      teacher_provenance_sha256=sha(experiment / 'provenance' / f'{spec.teacher}.json'))
        with shared.lock(group / '.prepare.lock'):
            shared.pin_recipe(group / 'comparison_recipe.json', recipe)
        if args.prepare_only:
            continue
        shared.check_gpus()
        for method in methods:
            shared.launch_run(method, config, group, recipe, profile, args.resume, record_invocations=True)
            subprocess.run([sys.executable, str(PROJECT / 'scripts/summarize_constrained_runtime.py'),
                            '--experiment', str(group), '--kind', 'profiles' if profile else 'runs',
                            '--methods', method, '--require-complete'], check=True)
        subprocess.run([sys.executable, str(PROJECT / 'scripts/summarize_constrained_runtime.py'),
                        '--experiment', str(group), '--kind', 'profiles' if profile else 'runs',
                        '--methods', *methods, '--require-complete'], check=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f'ERROR: {exc}')
