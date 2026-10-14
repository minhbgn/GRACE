#!/usr/bin/env python3
"""Replay the seven published Python/Slurm recipes, without training caps.

Runs immutable archived author scripts. Script fidelity is not evidence that
these scripts produced Table 6. See README_paper_protocol.md for contradictions.
"""
import argparse
import datetime as dt
from dataclasses import asdict, is_dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from run_lmms_matrix import DEFAULT, ROOT, checkpoint_files, digest, environment, pin, read, selected, sha, verified_completion, write
from run_constrained_matrix_evals import suites, dataset_fingerprint
from constrained_experiment import Experiment
from summarize_constrained_evals import EXPECTED, ID_FIELDS, SUITES
from paper_acceptance import PAPER, SOURCE as PAPER_SOURCE, assess

ARCHIVE = Path(__file__).parent / 'provenance/upstream_20260925'
TOKENS = dict(hallusionbench=8, mmbench=8, scienceqa=8, ai2d=8,
              mmmu=32, seedbench_img=8, mmstar=64)


def source_check():
    manifest = read(ARCHIVE / 'manifest.json')
    for name, checksum in manifest['sha256'].items():
        if sha(ARCHIVE / name) != checksum:
            raise ValueError(f'Archived upstream source changed: {name}')
    return manifest


def command(method, suite, paths, attempt, attn='flash_attention_2', limit=None):
    script, _, arguments = suites()[suite]
    return [sys.executable, str(ARCHIVE / 'evaluation' / script),
            '--model_path', str(paths[method]), '--output', str(attempt / 'results.json'),
            '--attn', attn, '--max_new_tokens', str(TOKENS[suite]),
            *(['--limit', str(limit)] if limit else []), *arguments]


def validate(payload, suite, model_path, limit):
    summary, rows = payload['summary'], payload['results']
    expected = min(limit, EXPECTED[suite]) if limit else EXPECTED[suite]
    if len(rows) != expected or summary['total'] != expected:
        raise ValueError(f'Incomplete {suite}: expected {expected}, got {len(rows)}')
    if Path(summary['model_path']).resolve() != model_path.resolve():
        raise ValueError('Checkpoint path mismatch')
    identities = [tuple(row[key] for key in ID_FIELDS[suite]) for row in rows]
    if len(set(identities)) != expected:
        raise ValueError('Duplicate sample identities')
    if any(row.get('error') for row in rows):
        raise ValueError('Inference errors retained in results; not certifying this run')
    correct = sum(row['correct'] is True for row in rows)
    if correct != summary['correct'] or not math.isclose(
            correct / expected, summary[SUITES[suite][1]], abs_tol=1e-12):
        raise ValueError('Summary does not match sample correctness')
    return {'score_percent': 100 * correct / expected, 'count': expected,
            'sample_identity': digest(sorted([(identity, row['gold']) for identity, row in zip(identities, rows)], key=str))}


def summarize(output, methods, limit):
    rows = []
    for method in methods:
        scores = {}
        for suite in SUITES:
            done = verified_completion(output / method / suite)
            scores[suite] = assess(method, suite, done['score_percent'] if done else None, smoke=bool(limit))
            if done:
                scores[suite]['score_percent'] = done['score_percent']
        rows.append({'method': method, 'suites': scores})
    statuses = [v['status'] for row in rows for v in row['suites'].values()]
    report = {'paper_source': PAPER_SOURCE, 'rows': rows,
              'all_within_preferred_tolerance': bool(statuses) and all(s == 'pass' for s in statuses),
              'all_within_maximum_tolerance': bool(statuses) and all(s in ('pass', 'barely_acceptable') for s in statuses),
              'warning': 'Author-script replay; paper recipe unverified. No judges in these scripts. Smoke scores cannot pass acceptance.'}
    write(output / 'paper_comparison.json', report)
    lines = ['# Author-script replay vs Table 6', '', report['warning'], '',
             '| Model | Suite | Paper % | Local % | Delta pp | Status |',
             '| --- | --- | ---: | ---: | ---: | --- |']
    for row in rows:
        for suite, result in row['suites'].items():
            values = ['—' if result.get(k) is None else f'{result[k]:.2f}' for k in ('paper_percent', 'score_percent', 'delta_pp')]
            lines.append('| ' + ' | '.join([row['method'], suite, *values, result['status']]) + ' |')
    (output / 'paper_comparison.md').write_text('\n'.join(lines) + '\n')
    print(f'Report: {output / "paper_comparison.md"}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan', 'check', 'run', 'summarize'], nargs='?', default='plan')
    p.add_argument('--experiment', type=Path, default=DEFAULT / 'experiment.json')
    p.add_argument('--output', type=Path)
    p.add_argument('--methods', help='Comma-separated model IDs; defaults to the five published references')
    p.add_argument('--suites', help='Comma-separated local suite IDs; defaults to all seven')
    p.add_argument('--gpu', help='One free physical GPU in PCI order; required for run')
    p.add_argument('--smoke', type=int, help='First N rows, isolated from full runs')
    p.add_argument('--attn', choices=['flash_attention_2', 'sdpa', 'eager'], default='flash_attention_2')
    args = p.parse_args()
    if args.smoke is not None and args.smoke <= 0:
        p.error('--smoke must be positive')
    exp = Experiment(args.experiment)
    methods = selected(args.methods or ','.join(PAPER), exp.models)
    chosen = selected(args.suites, SUITES)
    output = (args.output or exp.directory / 'upstream_evaluation') / (f'smoke_{args.smoke}' if args.smoke else 'full')
    if args.action == 'summarize':
        summarize(output, methods, args.smoke)
        return
    source = source_check()
    commands = {m: {s: command(m, s, exp.paths, output / m / s / 'ATTEMPT', args.attn, args.smoke) for s in chosen} for m in methods}
    if args.action == 'plan':
        print(json.dumps({'source_commit': source['commit'], 'commands': commands,
                          'policy': 'Native checkpoint processor + generation config; no pixel override; no judges'}, indent=2))
        return
    processors = {}
    for method in methods:
        checkpoint_files(exp.paths[method])
        if exp.models[method].get('kind') in ('trained', 'export'):
            exp.validate_export(method)
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(exp.paths[method], local_files_only=True)
        image_size = processor.image_processor.size
        processors[method] = {'image_size': asdict(image_size) if is_dataclass(image_size) else dict(image_size),
                              'generation_config': read(exp.paths[method] / 'generation_config.json')}
    for suite in chosen:
        arguments = suites()[suite][2]
        for flag, value in zip(arguments[::2], arguments[1::2]):
            if flag != '--split' and not Path(value).exists():
                raise FileNotFoundError(f'{suite}: {flag} {value}')
    if args.action == 'check':
        print(json.dumps({'source_commit': source['commit'], 'processors': processors,
                          'suites': chosen, 'environment': environment()}, indent=2))
        return
    if not args.gpu or not args.gpu.isdigit():
        p.error('run requires --gpu; first survey GPU ownership')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.matrix.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pin(output / 'protocol.json', {'source': source, 'attention': args.attn, 'limit': args.smoke,
                                      'tokens': TOKENS, 'environment': environment(),
                                      'runner_sha256': sha(__file__),
                                      'helpers_sha256': {name: sha(Path(__file__).with_name(name)) for name in
                                          ('run_lmms_matrix.py', 'run_constrained_matrix_evals.py', 'summarize_constrained_evals.py', 'paper_acceptance.py')},
                                      'datasets': dataset_fingerprint()})
        failures = []
        for method in methods:
            pin(output / method / 'model_provenance.json', exp.provenance(method))
            pin(output / method / 'processor.json', processors[method])
            for suite in chosen:
                job_dir = output / method / suite
                if verified_completion(job_dir):
                    print(f'SKIP {method}/{suite}', flush=True)
                    continue
                attempt = job_dir / dt.datetime.now(dt.timezone.utc).strftime('attempt_%Y%m%dT%H%M%S_%fZ')
                attempt.mkdir(parents=True)
                cmd = command(method, suite, exp.paths, attempt, args.attn, args.smoke)
                write(attempt / 'command.json', cmd)
                # Do not inherit a torchrun rank/world-size from the caller.
                env = {k: v for k, v in os.environ.items() if k not in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')}
                env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=args.gpu,
                           TOKENIZERS_PARALLELISM='false', PYTHONHASHSEED='42')
                print(f'RUN {method}/{suite}: {attempt / "run.log"}', flush=True)
                with (attempt / 'run.log').open('w') as log:
                    code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                try:
                    if code:
                        raise ValueError(f'Worker exited {code}')
                    validated = validate(read(attempt / 'results.json'), suite, exp.paths[method], args.smoke)
                    pin(output / 'sample_identities' / f'{suite}.json', {k: validated[k] for k in ('count', 'sample_identity')})
                    write(attempt / 'validated.json', validated)
                    done = dict(validated, attempt=attempt.name, sha256={name: sha(attempt / name) for name in ('command.json', 'results.json', 'validated.json')})
                    pin(job_dir / 'COMPLETE.json', done)
                    print(f'DONE {method}/{suite}: {validated["score_percent"]:.3f}%', flush=True)
                except (OSError, ValueError, KeyError) as error:
                    failures.append(f'{method}/{suite}: {error}')
                    print(f'FAILED {failures[-1]}', flush=True)
        summarize(output, methods, args.smoke)
        if failures:
            raise SystemExit('\n'.join(failures))


if __name__ == '__main__':
    main()
