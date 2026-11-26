#!/usr/bin/env python3
"""Video benchmark runner with GPU isolation, SIGKILL recovery and allocator fix."""
from __future__ import annotations

import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contextlib import ExitStack
import argparse
import fcntl
import importlib.metadata
import json
import math
import os
import signal

from evaluation.paths import EVALUATION_ROOT, source_file
from evaluation.protocols.video_protocol import DEFAULT_DATA, ROOT, SUITES, digest, load_suite
from evaluation.protocols.video_population import VIDEOMME_SHORT_MEDIUM, select_population
from evaluation.runners.run_video_matrix import (DEFAULT, Experiment, checkpoint_files,
    completed, pin, policy, read, select, sha, summarize as summarize_matrix, validate_attempt, write)
from evaluation.workers.video_runtime import gpu_lock, run_pair
from evaluation.workers.video_reuse import reuse_completed


def summarize(output, methods, datasets, protocol):
    summarize_matrix(output, methods, datasets, protocol)
    population = protocol.get('videomme_population')
    if population and 'videomme' in datasets:
        comparison = output / 'comparison.md'
        label = ('VideoMME population: **short + medium only (1,800 of 2,700 questions); '
                 '900 long questions excluded**. Scores are for this subset. '
                 f'This run selects {len(datasets["videomme"]):,} VideoMME questions '
                 '(including any smoke limit).')
        comparison.write_text(comparison.read_text().replace('\n\n', '\n\n' + label + '\n\n', 1))
        records = read(output / 'comparison.json')
        for record in records:
            if record['suite'] == 'videomme':
                record['population'] = dict(population, evaluation_questions=len(datasets['videomme']))
        write(output / 'comparison.json', records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan', 'check', 'prepare', 'reuse', 'run', 'summarize'], nargs='?', default='plan')
    p.add_argument('--experiment', type=Path, default=DEFAULT / 'experiment.json')
    p.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    p.add_argument('--output', type=Path)
    p.add_argument('--methods')
    p.add_argument('--suites')
    p.add_argument('--smoke', type=int)
    p.add_argument('--skip-videomme-long', action='store_true',
                   help='Exclude the final 900 long VideoMME questions; evaluate 1800 short/medium questions')
    p.add_argument('--reuse-completed-from', type=Path, action='append', default=[], metavar='FULL_DIR',
                   help='Validate and import completed pairs from a previous full/ directory; repeatable')
    p.add_argument('--fps', type=float, default=2)
    p.add_argument('--longest-side', type=int, default=256)
    p.add_argument('--max-new-tokens', type=int, default=4096)
    p.add_argument('--attn', choices=['flash_attention_2', 'sdpa'], default='flash_attention_2')
    p.add_argument('--gpu', help='One physical GPU index in PCI order; must be idle at worker startup')
    p.add_argument('--scratch', type=Path)
    p.add_argument('--sigkill-retries', type=int, default=1,
                   help='Bounded automatic retries per pair; resume validated predictions (default: 1)')
    p.add_argument('--cuda-alloc-conf', default='expandable_segments:True',
                   help='Pinned worker allocator configuration; default fixes long-video fragmentation')
    args = p.parse_args()
    if args.action == 'reuse' and not args.reuse_completed_from:
        p.error('reuse requires --reuse-completed-from FULL_DIR')
    if (not math.isfinite(args.fps) or args.fps < 0 or args.max_new_tokens <= 0
            or (args.smoke is not None and args.smoke <= 0)
            or (args.longest_side != 0 and args.longest_side < 32)
            or args.sigkill_retries < 0):
        p.error('Invalid sampling, smoke, spatial, generation or retry setting')
    if args.action == 'run' and (args.gpu is None or not args.gpu.isdigit()):
        p.error('run requires --gpu with one physical GPU index')
    exp = Experiment(args.experiment)
    methods, suites = select(args.methods, exp.models), select(args.suites, SUITES)
    datasets, reports = {}, {}
    for suite in suites:
        rows, info = load_suite(suite, args.data_root)
        rows, info = select_population(suite, rows, info, args.skip_videomme_long)
        reports[suite] = info
        if info['status'] != 'ready':
            if args.suites:
                raise ValueError(f'Explicitly requested suite is {info["status"]}: {suite}')
            print(f'Excluded {suite}: {info["status"]}', flush=True)
            continue
        datasets[suite] = rows[:args.smoke] if args.smoke else rows
    if not datasets:
        raise ValueError('No complete local video suites')
    protocol = dict(policy(args), version='video_resilient_v3',
                    cuda_allocator=args.cuda_alloc_conf, loading_format='merged BF16 export')
    if args.skip_videomme_long:
        protocol['videomme_population'] = dict(VIDEOMME_SHORT_MEDIUM)
    base = args.output or exp.directory / 'video_evaluation' / (protocol['version'] + '_' + digest(protocol)[:12])
    output = base / (f'smoke_{args.smoke}' if args.smoke else 'full')
    manifest = dict(policy=protocol, experiment=exp.spec, data=reports,
                    model_paths={m: str(exp.paths[m]) for m in methods}, limit=args.smoke,
                    evaluation_questions={s: len(rows) for s, rows in datasets.items()},
                    output=str(output), jobs=len(methods) * len(datasets))
    if args.reuse_completed_from:
        manifest['reuse_candidates'] = [str(root / method / suite)
            for root in args.reuse_completed_from for method in methods for suite in datasets
            if (root / method / suite / 'COMPLETE.json').exists()]
    if args.action == 'plan':
        print(json.dumps(manifest, indent=2))
        return
    if args.action == 'summarize':
        summarize(output, methods, datasets, protocol)
        return
    for method in methods:
        checkpoint_files(exp.paths[method])
        if exp.models[method].get('kind') in ('trained', 'export'):
            exp.validate_export(method)
    if args.action == 'check':
        print(json.dumps(manifest, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((output / '.matrix.lock').open('w'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'run':
            stack.enter_context(gpu_lock(args.gpu))
        files = [source_file(name) for name in ('run_video_matrix.py', 'video_worker.py',
                 'video_protocol.py', 'local_scoring.py', 'paths.py')]
        files += [Path(__file__), EVALUATION_ROOT / 'workers/video_worker_resilient.py',
                  EVALUATION_ROOT / 'workers/video_runtime.py',
                  EVALUATION_ROOT / 'workers/video_reuse.py',
                  EVALUATION_ROOT / 'protocols/video_population.py']
        environment = {}
        for name in ('torch', 'transformers', 'flash-attn', 'decord', 'pandas', 'Pillow'):
            try:
                environment[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                environment[name] = None
        identity = dict(policy=protocol, experiment=exp.spec, limit=args.smoke,
                        code={file.name: sha(file) for file in files}, environment=environment)
        pin(output / 'protocol.json', identity)
        write(output / 'runtime.json', dict(sigkill_retries=args.sigkill_retries,
              gpu=args.gpu, allocator=args.cuda_alloc_conf, resume='validated prefix within this pinned run'))
        for suite, info in reports.items():
            pin(output / 'datasets' / (suite + '.json'), info)
        if args.action == 'prepare':
            from evaluation.workers.video_worker import prepare
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(exp.paths[methods[0]], local_files_only=True)
            context = read(exp.paths[methods[0]] / 'config.json')['text_config']['max_position_embeddings']
            prepare(output, datasets, processor, protocol, context, args.scratch)
            return
        env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=args.gpu,
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                   PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1',
                   PYTORCH_CUDA_ALLOC_CONF=args.cuda_alloc_conf)
        failed = []
        for method in methods:
            model_provenance = exp.provenance(method)
            pin(output / method / 'model_provenance.json', model_provenance)
            for suite, rows in datasets.items():
                job_dir = output / method / suite
                if completed(job_dir, rows, protocol):
                    print(f'Already complete: {method}/{suite}', flush=True)
                    continue
                job = dict(method=method, suite=suite, rows=rows, model_path=str(exp.paths[method]),
                           protocol=protocol, scratch=str(args.scratch) if args.scratch else None,
                           run_identity=digest(identity))
                reused = any(reuse_completed(root, job_dir, job, identity, model_provenance)
                             for root in args.reuse_completed_from)
                if reused or args.action == 'reuse':
                    continue
                attempt, code = run_pair(job_dir, job, env, args.gpu, args.sigkill_retries)
                if code:
                    failed.append(f'{method}/{suite}')
                    print(f'FAILED {method}/{suite}: exit={code}; see {attempt / "exit.json"}', flush=True)
                    if code < 0:
                        summarize(output, methods, datasets, protocol)
                        raise SystemExit(f'{signal.Signals(-code).name}: stopping matrix; rerun the same command to resume')
                    continue
                validated = validate_attempt(attempt, rows, protocol)
                pin(job_dir / 'COMPLETE.json', dict(attempt=attempt.name, summary=validated,
                    sha256={name: sha(attempt / name) for name in
                            ('job.json', 'predictions.jsonl', 'summary.json', 'processor.json')}))
        summarize(output, methods, datasets, protocol)
        if failed:
            raise SystemExit('Incomplete jobs: ' + ', '.join(failed))


if __name__ == '__main__':
    main()
