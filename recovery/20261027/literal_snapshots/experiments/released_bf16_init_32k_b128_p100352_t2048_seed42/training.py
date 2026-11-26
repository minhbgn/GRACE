#!/usr/bin/env python3
"""Launch independent released-BF16 LoRA QAT/GRACE runs; no changes to old recipes."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from common import (HERE, PROJECT, PREVIOUS, REPOS, BF16_WEIGHTS_SHA256,
                    check_processor_compatibility, model_paths, pin, provenance, read, sha)
sys.path.insert(0, str(PROJECT / 'scripts'))
import run_constrained as shared


def prepare(experiment, c):
    sys.path.insert(0, str(PROJECT))
    from transformers import AutoProcessor
    from qwenvl.data.constrained_data import prepare_manifest
    from qwenvl.data.data_processor import update_processor_pixels
    from qwenvl.train.argument import DataArguments
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    with shared.lock(experiment / '.prepare.lock'):
        processor = AutoProcessor.from_pretrained(c['model_path'], local_files_only=True)
        processor.tokenizer.model_max_length = c['model_max_length']
        update_processor_pixels(processor, DataArguments(max_pixels=c['max_pixels'], min_pixels=c['min_pixels']))
        manifest = prepare_manifest(c['annotation'], c['data_root'], processor, c['model_path'],
                                    c['subset_size'], c['subset_seed'], c['model_max_length'],
                                    c['max_pixels'], c['min_pixels'], experiment / 'subset.json')
        old_path = PREVIOUS / 'subset.json'
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
        for bits in (4, 8):
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
    experiment = Path(os.environ.get('EXPERIMENT_DIR', HERE)).resolve()
    paths = model_paths(experiment)
    c = shared.resolved_config()
    c['model_path'] = str(paths['released_bf16'])
    c['teacher_path'] = str(paths['vanilla_8b_teacher'])
    c['student_weights_sha256'] = BF16_WEIGHTS_SHA256
    bits_set = (4, 8) if args.bits == 'all' else (int(args.bits),)
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
    student = provenance(paths['released_bf16'], REPOS['released_bf16'], released_student=True)
    pin(experiment / 'provenance/released_bf16.json', student)
    # Validate teacher vocabulary/template even for QAT so both methods share a valid recipe.
    check_processor_compatibility([paths['released_bf16'], paths['vanilla_8b_teacher']])
    pin(experiment / 'provenance/vanilla_8b_teacher.json',
        provenance(paths['vanilla_8b_teacher'], REPOS['vanilla_8b_teacher']))
    manifest = prepare(experiment, c)
    import torch
    import transformers
    implementation = shared.implementation_hashes()
    implementation.update({f'released_experiment/{path.name}': sha(path)
                           for path in (HERE / 'training.py', HERE / 'common.py', HERE / 'run_training.sh')})
    for bits in bits_set:
        group = experiment / f'w{bits}g128'
        config = dict(c, qat_bits=bits)
        recipe = dict(config=config, subset_sha256=manifest['records_sha256'], implementation=implementation,
                      student_provenance_sha256=sha(experiment / 'provenance/released_bf16.json'),
                      teacher_provenance_sha256=sha(experiment / 'provenance/vanilla_8b_teacher.json'),
                      environment=dict(torch=torch.__version__, transformers=transformers.__version__))
        with shared.lock(group / '.prepare.lock'):
            shared.pin_recipe(group / 'comparison_recipe.json', recipe)
        if args.prepare_only:
            continue
        shared.check_gpus()
        for method in methods:
            command, output = shared.build_command(method, config, group, profile, args.resume)
            with shared.lock(output / '.run.lock'):
                if (output / 'EXPORT_COMPLETE').exists():
                    raise ValueError(f'Completed output protected: {output}')
                checkpoints = list(output.glob('checkpoint-*'))
                if bool(checkpoints) != args.resume:
                    raise ValueError('Existing native checkpoints require --resume; --resume requires checkpoints')
                run_recipe = dict(comparison=recipe, method=method, profile=profile,
                                  profile_steps=shared.env_int('PROFILE_STEPS', 10) if profile else 0)
                shared.pin_recipe(output / 'launch_recipe.json', run_recipe)
                if args.resume:
                    latest = max(checkpoints, key=lambda path: int(path.name.split('-')[-1]))
                    required = ['launch_recipe.json', 'trainer_state.json', 'rng_state_0.json',
                                'rng_state_1.json', 'scheduler_state.json']
                    if method == 'grace':
                        required.append('grace_ib_state.json')
                    for name in required:
                        if not (latest / name).exists():
                            raise ValueError(f'Missing resumable state: {latest / name}')
                    shared.pin_recipe(latest / 'launch_recipe.json', run_recipe)
                os.environ['GRACE_LAUNCH_RECIPE'] = str(output / 'launch_recipe.json')
                subprocess.run(command, cwd=PROJECT, check=True)
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
