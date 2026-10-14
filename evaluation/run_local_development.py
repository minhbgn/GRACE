#!/usr/bin/env python3
"""Frozen, native-processor local development matrix; never uses paper tolerances."""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import fcntl
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from run_lmms_matrix import (DEFAULT, ROOT, Experiment, checkpoint_files, digest,
                             environment, pin, read, selected, sha, verified_completion, write)
from run_constrained_matrix_evals import suites, dataset_fingerprint
from summarize_constrained_evals import EXPECTED, ID_FIELDS, SUITES
from local_scoring import (SCORER, METRICS, PINNED, load_metadata, score_rows,
                           corrected_summary, generation_record)

REFERENCES = ('vanilla_2b', 'vanilla_8b_teacher', 'released_bf16', 'released_grace_w4', 'released_grace_w8')
LORA = ('qat_w4', 'grace_w4', 'qat_w8', 'grace_w8')
PROTOCOLS = {
    'local_native_greedy256_v1': {s: 256 for s in SUITES},
    'local_native_greedy_short_v1': dict(hallusionbench=8, mmbench=8, scienceqa=8,
                                       ai2d=8, mmmu=32, seedbench_img=8, mmstar=64),
}


def suite_args(suite):
    args = suites()[suite][2]
    return SimpleNamespace(**{flag[2:]: value for flag, value in zip(args[::2], args[1::2])})


def cap_for(protocol, suite, row=None):
    if protocol.endswith('short_v1') and suite == 'mmmu' and (row is None or row['question_type'].startswith('multi')):
        return 8
    return PROTOCOLS[protocol][suite]


def command(method, suite, paths, attempt, protocol, attn, limit):
    script, _, arguments = suites()[suite]
    return [sys.executable, str(ROOT / 'evaluation' / script),
            '--model_path', str(paths[method]), '--processor_path', str(paths[method]),
            '--output', str(attempt / 'results.json'), '--attn', attn,
            '--max_new_tokens', str(PROTOCOLS[protocol][suite]),
            '--mc_max_new_tokens', str(cap_for(protocol, 'mmmu')),
            '--fixed_generation', '--local_scoring',
            *(['--limit', str(limit)] if limit else []), *arguments]


@lru_cache(maxsize=None)
def metadata(suite):
    return load_metadata(suite, suite_args(suite))


@lru_cache(maxsize=None)
def source_samples(suite):
    """Read IDs/labels without decoding images; validate against actual dataset."""
    import pandas as pd
    args = suite_args(suite)
    if suite in ('mmbench', 'mmstar'):
        frame = pd.read_csv(args.tsv, sep='\t', usecols=['index', 'answer'])
        return [((int(r['index']),), str(r['answer']).strip().upper()) for r in frame.to_dict('records')]
    if suite == 'mmmu':
        return [((key,), str(row['answer']).strip().upper() if str(row['answer']).strip() in 'ABCDEFGHIJ' else str(row['answer']).strip())
                for key, row in metadata(suite).items()]
    if suite == 'scienceqa':
        problems, splits = read(args.problems), read(args.splits)
        return [((key,), problems[key]['answer']) for key in splits[args.split]]
    if suite == 'seedbench_img':
        data = read(args.json_path)
        records = data['questions'] if isinstance(data, dict) else data
        return [(tuple(r[k] for k in ID_FIELDS[suite]), str(r['answer']).strip().upper())
                for r in records if str(r.get('data_type', '')).lower() == 'image']
    results = []
    for path in sorted(Path(args.data_dir).glob('*.parquet')):
        import pyarrow.parquet as pq
        names = pq.ParquetFile(path).schema_arrow.names
        columns = [k for k in names if k != 'image']
        records = pd.read_parquet(path, columns=columns).to_dict('records')
        for row in records:
            if suite == 'hallusionbench':
                row['visual_input'] = str(row.get('visual_input', '0'))
                results.append((tuple(row[k] for k in ID_FIELDS[suite]), str(row['gt_answer']).strip()))
            else:
                idx = int(row.get('question_id', row.get('index', len(results))))
                answer = row['answer']
                answer = int(answer) if str(answer).isdigit() else ord(str(answer).strip().upper()) - ord('A')
                results.append(((idx,), answer))
    return results


def validate(payload, suite, path, protocol, attn, limit, processor):
    rows, summary = payload['results'], payload['summary']
    expected = source_samples(suite)
    if len(expected) != EXPECTED[suite]:
        raise ValueError(f'{suite}: source population changed')
    expected = expected[:limit] if limit else expected
    actual = [(tuple(r[k] for k in ID_FIELDS[suite]), r['gold']) for r in rows]
    canonical = lambda samples: sorted(samples, key=str)
    if canonical(actual) != canonical(expected) or len(set(x[0] for x in actual)) != len(expected):
        raise ValueError(f'{suite}: missing, extra, duplicate or relabeled source samples')
    if any(r.get('error') for r in rows):
        raise ValueError('Inference errors retained; run cannot be marked complete')
    if summary['model_path'] != str(path) or summary['processor_path'] != str(path):
        raise ValueError('Native model/processor path mismatch')
    if summary['image_processor_size'] != processor['image_size']:
        raise ValueError('Native processor size changed')
    expected_decoding = dict(version=1, min_pixels=None, max_pixels=None,
        max_new_tokens=PROTOCOLS[protocol][suite], do_sample=False, num_beams=1,
        repetition_penalty=1.0, context='no truncation; native model context', attention=attn)
    if suite == 'mmmu':
        expected_decoding['multiple_choice_max_new_tokens'] = cap_for(protocol, 'mmmu')
    if summary.get('evaluation_protocol') != expected_decoding:
        raise ValueError('Unexpected decoding settings')
    for row in rows:
        info = row.get('generation')
        if not info or not info.get('token_ids'):
            raise ValueError('Missing generated token IDs/finish reason')
        cap = cap_for(protocol, suite, row)
        if len(info['token_ids']) > cap or info != generation_record(info['token_ids'], processor['eos_token_id'], cap):
            raise ValueError('Generation diagnostics disagree with tokens or protocol cap')
        if info['finish_reason'] == 'other':
            raise ValueError('Unexpected early generation stop')
    rescored = score_rows(suite, copy.deepcopy(rows), metadata(suite))
    for old, new in zip(rows, rescored):
        for key in ('pred', 'correct', 'unresolved', 'extraction_method'):
            if old[key] != new[key]:
                raise ValueError(f'Saved {key} disagrees with deterministic rescoring')
    recomputed = corrected_summary(suite, rescored, metadata(suite), partial=bool(limit))
    # Saved JSON turns integer histogram keys into strings (MMBench rotations).
    # Compare both sides in the serialized representation without changing scores.
    serialized_recomputed = json.loads(json.dumps(recomputed))
    serialized_summary = json.loads(json.dumps(summary))
    if any(serialized_summary.get(k) != v for k, v in serialized_recomputed.items()):
        raise ValueError('Summary disagrees with corrected scoring/diagnostics')
    correct = sum(r['correct'] for r in rows)
    if summary['total'] != len(rows) or summary['correct'] != correct or summary[SUITES[suite][1]] != correct / len(rows):
        raise ValueError('Row-level summary arithmetic mismatch')
    return dict(score_percent=None if recomputed['primary_score'] is None else 100 * recomputed['primary_score'],
                metric=METRICS[suite], count=len(rows), sample_identity=digest(canonical(actual)),
                unresolved=recomputed['unresolved'], truncated=recomputed['truncated'],
                finish_reasons=recomputed['finish_reasons'], generated_tokens=recomputed['generated_tokens'])


def policy(args, exp):
    paths = [Path(__file__), Path(__file__).with_name('local_scoring.py'),
             Path(__file__).with_name('matched_protocol.py'),
             Path(__file__).with_name('rescore_saved_protocols.py'),
             Path(__file__).with_name('run_constrained_matrix_evals.py'),
             Path(__file__).with_name('run_lmms_matrix.py'),
             Path(__file__).with_name('summarize_constrained_evals.py'),
             ROOT / 'qwen-vl-finetune/scripts/constrained_experiment.py',
             PINNED / 'mmmu/utils.py', PINNED / 'mmbench/mmbench_evals.py']
    paths += [ROOT / 'evaluation' / value[0] for value in suites().values()]
    return dict(name=args.protocol, scorer=SCORER, attention=args.attn, limit=args.smoke,
                processor_policy='checkpoint_native_no_pixel_overrides',
                tokens=PROTOCOLS[args.protocol], mmmu_mc_tokens=cap_for(args.protocol, 'mmmu'),
                generation='greedy; one beam; repetition penalty 1; tokenizer EOS; BF16',
                source_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
                experiment_sha256=sha(exp.config_path), datasets=dataset_fingerprint(),
                environment=environment(), metrics=METRICS, judge_calls=0)


def summarize(output, methods, exp, args):
    entries = []
    identities = {}
    for method in methods:
        for suite in SUITES:
            job = output / method / suite
            done = verified_completion(job)
            if done is None:
                entries.append(dict(model=method, suite=suite, status='missing'))
                continue
            if not {'command.json', 'results.json', 'validated.json'} <= set(done['sha256']):
                raise ValueError('Incomplete artifact hashes')
            for key, artifact in [('protocol_sha256', output / 'protocol.json'),
                                  ('processor_sha256', output / method / 'processor.json'),
                                  ('model_provenance_sha256', output / method / 'model_provenance.json')]:
                if done.get(key) != sha(artifact):
                    raise ValueError(f'Changed completion provenance: {artifact}')
            processor = read(output / method / 'processor.json')
            value = validate(read(job / done['attempt'] / 'results.json'), suite, exp.paths[method],
                             args.protocol, args.attn, args.smoke, processor)
            if any(done[k] != v for k, v in value.items()):
                raise ValueError('Completion marker differs from verified results')
            if suite in identities and identities[suite] != value['sample_identity']:
                raise ValueError('Sample coverage differs across models')
            identities[suite] = value['sample_identity']
            entries.append(dict(model=method, suite=suite, status='smoke' if args.smoke else 'complete', **value))
    full = not args.smoke and all(row['status'] == 'complete' for row in entries)
    report = dict(protocol=args.protocol, artifact_ready=full, rows=entries,
        note='Artifact readiness only; unresolved answers count as wrong and capped outputs remain visible. No paper-score gate. No claim of judge-equivalent semantic extraction.')
    write(output / 'local_scores.json', report)
    lines = ['# Local development results', '', report['note'], '',
             '| Model | Suite / metric | Score % | Unresolved | Length stops | Status |',
             '| --- | --- | ---: | ---: | ---: | --- |']
    for row in entries:
        score = row.get('score_percent')
        lines.append('| ' + ' | '.join([row['model'], row['suite'] + ' / ' + METRICS[row['suite']],
            '—' if score is None else f'{score:.3f}', str(row.get('unresolved', '—')),
            str(row.get('truncated', '—')), row['status']]) + ' |')
    (output / 'local_scores.md').write_text('\n'.join(lines) + '\n')
    print(f'Report: {output / "local_scores.md"}; artifact_ready={full}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan', 'check', 'run', 'summarize'], nargs='?', default='plan')
    p.add_argument('--experiment', type=Path, default=DEFAULT / 'experiment.json')
    p.add_argument('--output', type=Path, help='New output base; protocol and full/smoke folders are appended')
    p.add_argument('--protocol', choices=PROTOCOLS, default='local_native_greedy256_v1')
    p.add_argument('--methods', help='references (default), lora, all, or comma-separated experiment IDs')
    p.add_argument('--suites', help='Comma-separated local suite IDs')
    p.add_argument('--gpu', help='One available physical GPU in PCI order; required for run')
    p.add_argument('--smoke', type=int)
    p.add_argument('--attn', choices=['flash_attention_2', 'sdpa', 'eager'], default='flash_attention_2')
    args = p.parse_args()
    if args.smoke is not None and args.smoke <= 0:
        p.error('--smoke must be positive')
    exp = Experiment(args.experiment)
    aliases = {'references': REFERENCES, 'lora': LORA, 'all': REFERENCES + LORA}
    methods = selected(','.join(aliases[args.methods or 'references']) if (args.methods or 'references') in aliases else args.methods, exp.models)
    chosen = selected(args.suites, SUITES)
    output = (args.output or ROOT / 'experiments/reproduction_eval_only/local_development') / args.protocol / (f'smoke_{args.smoke}' if args.smoke else 'full')
    if args.action == 'plan':
        print(json.dumps(dict(output=str(output), protocol=args.protocol, methods=methods,
            commands={m: {s: command(m, s, exp.paths, output / m / s / 'ATTEMPT', args.protocol, args.attn, args.smoke) for s in chosen} for m in methods}), indent=2))
        return
    if args.action == 'run' and (not args.gpu or not args.gpu.isdigit()):
        p.error('run requires --gpu N; first survey GPU ownership')
    print('CPU preflight: hashing datasets and evaluator sources...', flush=True)
    current_policy = policy(args, exp)
    if (output / 'protocol.json').exists() and read(output / 'protocol.json') != current_policy:
        raise ValueError('Protocol/source/environment/datasets changed; use a new --output')
    if args.action == 'summarize':
        if not (output / 'protocol.json').exists():
            raise ValueError('No frozen protocol or completed runs yet')
        with (output / '.matrix.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            summarize(output, methods, exp, args)
        return
    processors, provenance = {}, {}
    from transformers import AutoProcessor
    for method in methods:
        print(f'CPU preflight: validating/hashing {method}...', flush=True)
        checkpoint_files(exp.paths[method])
        if exp.models[method].get('kind') in ('trained', 'export'):
            exp.validate_export(method)
        proc = AutoProcessor.from_pretrained(exp.paths[method], local_files_only=True)
        size = proc.image_processor.size
        processors[method] = dict(image_size=dict(size),
                                  eos_token_id=proc.tokenizer.eos_token_id)
        provenance[method] = exp.provenance(method)
        for filename, value in [('processor.json', processors[method]), ('model_provenance.json', provenance[method])]:
            saved = output / method / filename
            if saved.exists() and read(saved) != value:
                raise ValueError(f'Checkpoint/processor provenance drift: {saved}')
    for suite in chosen:
        for flag, value in vars(suite_args(suite)).items():
            if flag != 'split' and not Path(value).exists():
                raise FileNotFoundError(f'{suite}: {flag} {value}')
        if len(source_samples(suite)) != EXPECTED[suite]:
            raise ValueError(f'{suite}: source population changed')
    if args.action == 'check':
        print(json.dumps(dict(protocol=args.protocol, processors=processors, suites=chosen, status='CPU preflight passed; no inference'), indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    failures = []
    with (output / '.matrix.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pin(output / 'protocol.json', current_policy)
        for method in methods:
            pin(output / method / 'model_provenance.json', provenance[method])
            pin(output / method / 'processor.json', processors[method])
            for suite in chosen:
                job = output / method / suite
                done = verified_completion(job)
                if done:
                    validate(read(job / done['attempt'] / 'results.json'), suite, exp.paths[method], args.protocol, args.attn, args.smoke, processors[method])
                    print(f'SKIP {method}/{suite}', flush=True)
                    continue
                attempt = job / dt.datetime.now(dt.timezone.utc).strftime('attempt_%Y%m%dT%H%M%S_%fZ')
                attempt.mkdir(parents=True)
                cmd = command(method, suite, exp.paths, attempt, args.protocol, args.attn, args.smoke)
                write(attempt / 'command.json', cmd)
                env = {k: v for k, v in os.environ.items() if k not in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')}
                env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=args.gpu,
                           TOKENIZERS_PARALLELISM='false', PYTHONHASHSEED='42', HF_HUB_OFFLINE='1')
                print(f'RUN {method}/{suite}: {attempt / "run.log"}', flush=True)
                with (attempt / 'run.log').open('w') as log:
                    code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                try:
                    if code:
                        raise ValueError(f'Worker exited {code}; see {attempt / "run.log"}')
                    value = validate(read(attempt / 'results.json'), suite, exp.paths[method], args.protocol, args.attn, args.smoke, processors[method])
                    pin(output / 'sample_identities' / f'{suite}.json', {k: value[k] for k in ('count', 'sample_identity')})
                    write(attempt / 'validated.json', value)
                    pin(job / 'COMPLETE.json', dict(value, attempt=attempt.name,
                        protocol_sha256=sha(output / 'protocol.json'),
                        processor_sha256=sha(output / method / 'processor.json'),
                        model_provenance_sha256=sha(output / method / 'model_provenance.json'),
                        sha256={name: sha(attempt / name) for name in ('command.json', 'results.json', 'validated.json')}))
                    print(f'DONE {method}/{suite}: {value["score_percent"]}', flush=True)
                except (OSError, ValueError, KeyError) as error:
                    failures.append(f'{method}/{suite}: {error}')
                    print(f'FAILED {failures[-1]}', flush=True)
        summarize(output, methods, exp, args)
    if failures:
        raise SystemExit('\n'.join(failures))


if __name__ == '__main__':
    main()
