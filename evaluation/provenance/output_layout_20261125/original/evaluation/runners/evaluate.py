#!/usr/bin/env python3
"""Single entry point for the saved 576-cap experiments and fresh reproductions.

The numerical runners remain pinned. This module only selects jobs and outputs.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TOKEN = ROOT / 'experiments/qat_int4_token_reduction_576_seed42'
CONTROL = ROOT / 'experiments/released_bf16_init_32k_b128_p589824_t4096_seed42'
IMAGES = ['hallusionbench', 'mmbench', 'scienceqa', 'ai2d', 'mmmu', 'seedbench_img', 'mmstar']
VIDEOS = ['tempcompass', 'videomme', 'videommmu_perception', 'videommmu_comprehension', 'videommmu_adaptation']
CONFIGS = ['baseline_r100'] + [f'{m}_r{r:03}' for m in ['resolution', 'uniform', 'merge'] for r in [50, 25, 10]]


def choose(value, available):
    values = value.split(',') if value else list(available)
    if len(set(values)) != len(values) or any(v not in available for v in values):
        raise ValueError(f'Choose unique values from {",".join(available)}')
    return values


def command(script, *args):
    return [sys.executable, str(ROOT / 'evaluation' / script), *map(str, args)]


def commands(args):
    output = args.output.resolve()
    if args.workflow == 'timing':
        action = 'prepare' if args.action in ('check', 'plan') else args.action
        result = command('runners/run_videomme_timing_subset.py', action, '--output', output / 'timing', '--seed', 42)
        if args.action == 'run':
            result += ['--gpus', args.gpus]
        if args.configurations:
            result += ['--configurations', ','.join(choose(args.configurations, CONFIGS))]
        return [result]
    if args.workflow == 'controls-images':
        result = command('runners/run_local_development.py', args.action, '--experiment', CONTROL / 'evaluation.json',
            '--output', output / 'controls_images', '--protocol', 'local_native_greedy256_v1',
            '--methods', args.models or 'lora')
        if args.suites:
            result += ['--suites', ','.join(choose(args.suites, IMAGES))]
        if args.action == 'run':
            result += ['--gpu', args.gpus.split(',')[0]]
        if args.smoke:
            result += ['--smoke', str(args.smoke)]
        return [result]
    suites = choose(args.suites, IMAGES+VIDEOS)
    configs = choose(args.configurations, CONFIGS)
    result = []
    common = ['--smoke', str(args.smoke)] if args.smoke else []
    gpu = ['--gpus', args.gpus] if args.action == 'run' else []

    def original(selected_suites, selected_configs, folder):
        if not selected_suites or not selected_configs:
            return
        # A single cartesian matrix for the standard full selection; arbitrary
        # filters use one configuration at a time, sharing the same pinned root.
        batches = [selected_configs] if selected_configs == CONFIGS else [[c] for c in selected_configs]
        for batch in batches:
            methods = list(dict.fromkeys(c.rsplit('_r', 1)[0] for c in batch if c != 'baseline_r100'))
            ratios = list(dict.fromkeys(str(int(c.rsplit('_r', 1)[1])/100) for c in batch))
            result.append(command('runners/run_token_reduction_matrix.py', args.action,
                '--experiment', TOKEN / 'experiment.json', '--output', output / folder,
                '--models', 'qat_w4', '--suites', ','.join(selected_suites),
                '--methods', ','.join(methods or ['resolution']), '--retentions', ','.join(ratios), *gpu, *common))

    original([s for s in suites if s in IMAGES], configs, 'images')
    # Match the published latest matrix exactly, including its three labelled
    # original-policy r10 rows. Fresh adaptive jobs never reuse historical answers.
    for selected_suites, selected_configs in [
            ([s for s in suites if s in VIDEOS[:2]], configs),
            ([s for s in suites if s in VIDEOS[2:]], [c for c in configs if c != 'resolution_r010'])]:
        if selected_suites and selected_configs:
            result.append(command('runners/run_matched_video_evaluation.py', args.action,
                '--output', output / 'videos', '--suites', ','.join(selected_suites),
                '--configurations', ','.join(selected_configs), '--parallel-evaluation', 1,
                '--prefill-chunk-tokens', 512, '--decoder-token-cap', 131072, *gpu, *common))
    if 'resolution_r010' in configs:
        original([s for s in suites if s in VIDEOS[2:]], ['resolution_r010'], 'legacy_video_r010')
    return result


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def compare(args):
    reference = json.loads((ROOT / 'results/merged_results.json').read_text())
    suites = choose(args.suites, IMAGES+VIDEOS)
    configs = choose(args.configurations, CONFIGS)
    records = []
    for ref in reference['rows']:
        if ref['suite'] not in suites or ref['config'] not in configs:
            continue
        folder = 'images' if ref['suite'] in IMAGES else ('legacy_video_r010'
            if ref['suite'] in VIDEOS[2:] and ref['config'] == 'resolution_r010' else 'videos')
        directory = args.output.resolve() / folder / 'full/qat_w4' / ref['suite'] / ref['config']
        row = dict(suite=ref['suite'], config=ref['config'], reference_score=ref['score_percent'], status='missing')
        if (directory / 'COMPLETE.json').exists():
            done = json.loads((directory / 'COMPLETE.json').read_text())
            attempt = directory / done['attempt']
            for name, checksum in done['sha256'].items():
                if sha(attempt / name) != checksum:
                    raise ValueError(f'Corrupt rerun artifact: {attempt / name}')
            validated = json.loads((attempt / 'validated.json').read_text())
            job = json.loads((attempt / 'job.json').read_text())
            if any(done[k] != value for k, value in validated.items()):
                raise ValueError(f'Validation differs: {attempt}')
            if json.loads((attempt / 'exit.json').read_text())['returncode'] != 0:
                raise ValueError(f'Nonzero completed exit: {attempt}')
            same_population = done['sample_identity'] == ref['sample_identity'] and done['count'] == ref['count']
            same_policy = job['policy'] == ref['policy'] and job.get('execution') == ref['execution']
            delta = done['score_percent'] - ref['score_percent']
            row.update(status='similar' if same_population and same_policy and abs(delta) <= args.max_score_drift else 'different',
                       score=done['score_percent'], delta_pp=delta, same_population=same_population, same_policy=same_policy,
                       marker=str(directory / 'COMPLETE.json'))
        records.append(row)
    passed = bool(records) and all(r['status'] == 'similar' for r in records)
    payload = dict(passed=passed, max_score_drift_pp=args.max_score_drift, rows=records,
                   note='Accuracy tolerance is a diagnostic choice; runtime is not required to match.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'comparison_to_frozen.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(dict(passed=passed, compared=len(records), statuses={s:sum(r['status']==s for r in records)
        for s in ('similar', 'different', 'missing')}), indent=2))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'check', 'prepare', 'run', 'summarize', 'verify', 'compare'], nargs='?', default='plan')
    parser.add_argument('--workflow', choices=['token-reduction', 'controls-images', 'timing'], default='token-reduction')
    parser.add_argument('--suites', help='Comma-separated suite IDs; omitted selects all suites')
    parser.add_argument('--configurations', help='Comma-separated configuration IDs; omitted selects all ten')
    parser.add_argument('--models', help='Model selection for controls-images; default lora')
    parser.add_argument('--gpus', help='Physical PCI GPU indices, e.g. 0,2; no implicit GPU selection')
    parser.add_argument('--smoke', type=int, help='Separate N-question output; excluded from merged full results')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/reruns/token_reduction_576')
    parser.add_argument('--max-score-drift', type=float, default=0.5, help='Allowed absolute accuracy drift in pp for compare')
    args = parser.parse_args()
    if args.smoke is not None and args.smoke <= 0:
        parser.error('--smoke must be positive')
    if not math.isfinite(args.max_score_drift) or args.max_score_drift < 0:
        parser.error('--max-score-drift must be nonnegative')
    if args.workflow != 'controls-images' and args.models:
        parser.error('--models applies only to controls-images; token reduction reproduces qat_w4')
    if args.action == 'run' and not args.gpus:
        parser.error('run requires --gpus; inspect GPU ownership before selecting cards')
    if args.gpus and (len(set(args.gpus.split(','))) != len(args.gpus.split(',')) or
                     any(not g.isdigit() for g in args.gpus.split(','))):
        parser.error('--gpus must contain unique nonnegative indices')
    if args.workflow == 'timing' and args.smoke:
        parser.error('Timing uses its fixed 50-question sample; smoke is unavailable')
    if args.workflow == 'controls-images' and args.configurations:
        parser.error('Control checkpoints have no reduction configurations')
    if args.workflow == 'timing' and args.suites:
        parser.error('Timing evaluates VideoMME only; --suites is unavailable')
    if args.action == 'prepare' and args.workflow != 'timing':
        parser.error('Use check for matrix preflight; prepare is a timing action')
    try:
        if args.action in ('summarize', 'verify') and args.workflow == 'token-reduction':
            subprocess.run(command('audits/compile_results.py'), cwd=ROOT, check=True)
            if args.action == 'verify':
                subprocess.run(command('data_tools/consolidate_repository.py', 'verify'), cwd=ROOT, check=True)
            return 0
        if args.action == 'compare':
            if args.workflow != 'token-reduction' or args.smoke:
                parser.error('compare requires full token-reduction results')
            return compare(args)
        if args.action == 'verify':
            parser.error('verify checks the frozen token-reduction catalog')
        # Protect reference runs and their compatibility aliases from accidental writes.
        destination = args.output.resolve()
        fresh = ROOT / 'results/reruns'
        if fresh not in destination.parents and destination != fresh:
            parser.error('--output must be inside results/reruns to protect frozen evidence')
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        for cmd in commands(args):
            print(shlex.join(cmd), flush=True)
            if args.action != 'plan':
                subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        return 0
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    sys.exit(main())
