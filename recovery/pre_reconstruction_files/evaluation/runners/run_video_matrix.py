#!/usr/bin/env python3
"""Evaluation only: offline video benchmark matrix with audited token usage."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from video_protocol import DEFAULT_DATA, ROOT, SUITES, digest, load_suite

sys.path.insert(0, str(ROOT / 'qwen-vl-finetune/scripts'))
from constrained_experiment import Experiment, checkpoint_files, pin, read, sha

DEFAULT = ROOT / 'experiments/released_bf16_init_32k_b128_p589824_t4096_seed42'


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def select(value, available):
    names = value.split(',') if value else list(available)
    if len(set(names)) != len(names) or set(names) - set(available):
        raise ValueError(f'Choose unique names from {list(available)}')
    return names


def policy(args):
    return dict(version='video_unsaturated_v1', fps=args.fps, longest_side=args.longest_side,
                temporal_sampling='uniform over entire clip, both endpoints; no max_frames',
                spatial_policy='source resolution rounded to patch grid' if not args.longest_side else 'fixed longest-side downsample, rounded to patch grid',
                processor_resize=False, processor_frame_sampling=False, tokenizer_truncation=False,
                subtitles=False, audio=False, max_new_tokens=args.max_new_tokens,
                generation='greedy, one beam, repetition penalty 1, tokenizer EOS',
                scoring='deterministic_local_v1; unresolved counts wrong; no judge',
                attention=args.attn, seed=42)


def summary_from_rows(rows):
    from video_protocol import digest
    from local_scoring import choice, generation_record, open_answer
    counts = ['input_visual_tokens', 'input_text_tokens', 'input_tokens', 'output_text_tokens']
    if not rows:
        raise ValueError('Empty prediction population')
    for row in rows:
        if row['question_type'] == 'multiple-choice':
            prediction, _ = choice(row['raw'], row['options'], row['options'])
            correct = prediction == row['gold']
        else:
            prediction, correct = open_answer(row['raw'], row['gold'])
        if prediction != row['prediction'] or correct != row['correct']:
            raise ValueError('Saved score disagrees with raw output')
        generation = row['generation']
        if generation != generation_record(generation['token_ids'], row['eos_token_id'], generation['max_new_tokens']):
            raise ValueError('Invalid generation telemetry')
        tokens = row['tokens']
        if tokens['input_tokens'] != tokens['input_visual_tokens'] + tokens['input_text_tokens']:
            raise ValueError('Invalid input token accounting')
        if tokens['output_text_tokens'] != generation['generated_tokens']:
            raise ValueError('Invalid output token accounting')
    result = dict(count=len(rows), accuracy=sum(r['correct'] for r in rows) / len(rows),
                  unresolved=sum(not r['prediction'] for r in rows),
                  generation_cutoffs=sum(r['generation']['finish_reason'] != 'eos' for r in rows),
                  input_identity=digest([{k: r[k] for k in ('id', 'gold', 'input_identity')} for r in rows]),
                  token_usage={})
    for key in counts:
        values = sorted(r['tokens'][key] for r in rows)
        import math
        result['token_usage'][key] = dict(mean=sum(values) / len(values), total=sum(values),
                                        min=values[0], max=values[-1], p95=values[math.ceil(.95 * len(values)) - 1])
    result['token_usage']['total_text_tokens'] = {
        'mean': sum(r['tokens']['input_text_tokens'] + r['tokens']['output_text_tokens'] for r in rows) / len(rows)}
    categories = sorted({r['category'] for r in rows})
    result['categories'] = {c: dict(count=sum(r['category'] == c for r in rows),
        accuracy=sum(r['correct'] for r in rows if r['category'] == c) / sum(r['category'] == c for r in rows)) for c in categories}
    return result


def validate_attempt(attempt, source_rows, protocol):
    from local_scoring import generation_record
    rows = [json.loads(line) for line in (attempt / 'predictions.jsonl').read_text().splitlines()]
    if [r['id'] for r in rows] != [r['id'] for r in source_rows]:
        raise ValueError('Missing, duplicate, reordered or unexpected predictions')
    for row, source in zip(rows, source_rows):
        if any(row[k] != source[k] for k in ('gold', 'options', 'question_type', 'category')):
            raise ValueError('Prediction source metadata mismatch')
        if row['source_identity'] != digest(source):
            raise ValueError('Prediction source identity mismatch')
        if row['input_identity'] != digest(dict(source=source, sampling=row['sampling'],
                tokens={k: row['tokens'][k] for k in ('input_visual_tokens', 'input_text_tokens', 'input_tokens')},
                input_ids_sha256=row['input_ids_sha256'])):
            raise ValueError('Prediction input identity mismatch')
        if row['generation']['max_new_tokens'] != protocol['max_new_tokens']:
            raise ValueError('Generation cap changed')
        if row['generation'] != generation_record(row['generation']['token_ids'], row['eos_token_id'], protocol['max_new_tokens']):
            raise ValueError('Generation telemetry changed')
        if row['tokens']['input_tokens'] + protocol['max_new_tokens'] > row['context_limit']:
            raise ValueError('Input exceeded context')
    summary = summary_from_rows(rows)
    if summary['generation_cutoffs']:
        raise ValueError('Generation did not finish at EOS; token means are censored')
    if summary != read(attempt / 'summary.json'):
        raise ValueError('Summary disagrees with saved predictions')
    return summary


def completed(job_dir, rows, protocol):
    marker = job_dir / 'COMPLETE.json'
    if not marker.exists():
        return None
    done = read(marker)
    attempt = job_dir / done['attempt']
    for name, checksum in done['sha256'].items():
        if sha(attempt / name) != checksum:
            raise ValueError(f'Completed artifact changed: {attempt / name}')
    validated = validate_attempt(attempt, rows, protocol)
    if validated != done['summary']:
        raise ValueError('Completion summary changed')
    return done


def summarize(output, methods, datasets, protocol):
    if (output / 'protocol.json').exists() and read(output / 'protocol.json')['policy'] != protocol:
        raise ValueError('Summary policy does not match saved protocol')
    comparisons, identities = [], {}
    for method in methods:
        for suite, source in datasets.items():
            done = completed(output / method / suite, source, protocol)
            if done:
                identity = done['summary']['input_identity']
                if suite in identities and identity != identities[suite]:
                    raise ValueError(f'{suite}: models received different inputs')
                identities[suite] = identity
            attempts = sorted((output / method / suite).glob('attempt_*'))
            status = 'complete' if done else 'pending'
            reason = None
            if not done and attempts:
                latest = attempts[-1]
                for name, state in [('blocked.json', 'blocked'), ('failure.json', 'failed')]:
                    if (latest / name).exists():
                        status, reason = state, read(latest / name)
                        break
            comparisons.append(dict(method=method, suite=suite, status=status, reason=reason,
                                    **(done['summary'] if done else {})))
    write(output / 'comparison.json', comparisons)
    lines = ['# Video evaluation', '', 'Local deterministic scoring; question-weighted token means. '
             'Only EOS-complete, fully covered jobs are accepted. Smoke runs are previews.', '',
             '| Model | Suite | Accuracy % | Visual input | Text input | Text output | Status |',
             '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for row in comparisons:
        usage = row.get('token_usage', {})
        means = [f'{usage[k]["mean"]:.2f}' if usage else '—' for k in
                 ('input_visual_tokens', 'input_text_tokens', 'output_text_tokens')]
        score = f'{100 * row["accuracy"]:.2f}' if usage else '—'
        lines.append('| ' + ' | '.join([row['method'], row['suite'], score, *means, row['status']]) + ' |')
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(f'Validated {sum(r["status"] == "complete" for r in comparisons)}/{len(comparisons)} jobs: {output / "comparison.md"}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan', 'check', 'prepare', 'run', 'summarize'], nargs='?', default='plan')
    p.add_argument('--experiment', type=Path, default=DEFAULT / 'experiment.json')
    p.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    p.add_argument('--output', type=Path)
    p.add_argument('--methods', help='Default all ten experiment models')
    p.add_argument('--suites', help='Default all locally available supported suites')
    p.add_argument('--smoke', type=int)
    p.add_argument('--fps', type=float, default=2, help='Target sampling rate; 0 uses every source frame')
    p.add_argument('--longest-side', type=int, help='Optional explicit spatial policy; default source resolution')
    p.add_argument('--max-new-tokens', type=int, default=4096, help='Any non-EOS stop invalidates completion')
    p.add_argument('--attn', choices=['flash_attention_2', 'sdpa'], default='flash_attention_2')
    p.add_argument('--gpu', help='One physical GPU index in PCI order, required for run')
    p.add_argument('--scratch', type=Path, help='Temporary per-video extraction directory')
    args = p.parse_args()
    import math
    if (not math.isfinite(args.fps) or args.fps < 0 or args.max_new_tokens <= 0
            or (args.smoke is not None and args.smoke <= 0)
            or (args.longest_side is not None and args.longest_side < 32)):
        p.error('Invalid sampling, smoke, spatial or generation setting')
    if args.action == 'run' and (args.gpu is None or not args.gpu.isdigit()):
        p.error('run requires --gpu with one physical GPU index')
    exp = Experiment(args.experiment)
    methods = select(args.methods, exp.models)
    suites = select(args.suites, SUITES)
    datasets, reports = {}, {}
    for suite in suites:
        rows, info = load_suite(suite, args.data_root)
        reports[suite] = info
        if info['status'] == 'unavailable':
            if args.suites:
                raise ValueError(f'Explicitly requested suite unavailable: {suite}')
            continue
        if info['status'] == 'incomplete':
            if args.suites:
                raise ValueError(f'{suite}: missing {len(info["missing_videos"])} referenced videos; {info["missing_videos"][:5]}')
            print(f'BLOCKED {suite}: {len(info["missing_videos"])} missing videos; entire suite excluded', file=sys.stderr)
            continue
        datasets[suite] = rows[:args.smoke] if args.smoke else rows
    if not datasets:
        raise ValueError('No complete local video suites')
    protocol = policy(args)
    base = args.output or exp.directory / 'video_evaluation' / (protocol['version'] + '_' + digest(protocol)[:12])
    output = base / (f'smoke_{args.smoke}' if args.smoke else 'full')
    manifest = dict(policy=protocol, experiment=exp.spec, data=reports,
                    model_paths={m: str(exp.paths[m]) for m in methods},
                    limit=args.smoke, output=str(output), jobs=len(methods) * len(datasets),
                    excluded_empty_directories=[str(p) for p in sorted(args.data_root.iterdir())
                        if p.is_dir() and p.name not in {v[0] for v in SUITES.values()} and p.name != 'logs'])
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
    with (output / '.matrix.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = {name: sha(Path(__file__).parent / name) for name in
                ('run_video_matrix.py', 'video_worker.py', 'video_protocol.py', 'local_scoring.py')}
        environment = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'decord', 'pandas', 'Pillow')}
        # Methods are allowed to be run separately into the same comparison directory.
        pin(output / 'protocol.json', dict(policy=protocol, experiment=exp.spec,
                                          limit=args.smoke, code=code, environment=environment))
        for suite, info in reports.items():
            pin(output / 'datasets' / (suite + '.json'), info)
        if args.action == 'prepare':
            from video_worker import prepare
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(exp.paths[methods[0]], local_files_only=True)
            context = read(exp.paths[methods[0]] / 'config.json')['text_config']['max_position_embeddings']
            prepare(output, datasets, processor, protocol, context, args.scratch)
            return
        failed = []
        for method in methods:
            pin(output / method / 'model_provenance.json', exp.provenance(method))
            for suite, rows in datasets.items():
                job_dir = output / method / suite
                if completed(job_dir, rows, protocol):
                    print(f'Already complete: {method}/{suite}', flush=True)
                    continue
                attempt = job_dir / ('attempt_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
                attempt.mkdir(parents=True)
                job = dict(method=method, suite=suite, rows=rows, model_path=str(exp.paths[method]),
                           protocol=protocol, scratch=str(args.scratch) if args.scratch else None)
                write(attempt / 'job.json', job)
                env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=args.gpu,
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
                print(f'Running {method}/{suite}; log: {attempt / "run.log"}', flush=True)
                with (attempt / 'run.log').open('w') as log:
                    result = subprocess.run([sys.executable, str(Path(__file__).with_name('video_worker.py')),
                                             str(attempt / 'job.json')], env=env, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    failed.append(f'{method}/{suite}')
                    print(f'FAILED {method}/{suite}; inspect {attempt / "run.log"}', flush=True)
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
