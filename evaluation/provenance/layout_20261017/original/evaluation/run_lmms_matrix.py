#!/usr/bin/env python3
"""Pinned lmms-eval matrix. Planning, checking and preparing never load a model."""
import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import functools
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'qwen-vl-finetune/scripts'))
from constrained_experiment import Experiment, checkpoint_files, pin, read, sha

COMMIT = '88b23e2bfa16a1edbc16e9e238ed82130b3a4f56'
SOURCE = ROOT / 'third_party/lmms-eval-v0.7.1'
DEFAULT = ROOT / 'experiments/released_bf16_init_32k_b128_p100352_t2048_seed42'
TASKS = {
    'scienceqa': ('scienceqa_img', 'scienceqa/scienceqa_img.yaml', 'exact_match,none', 100),
    'ai2d': ('ai2d', 'ai2d/ai2d.yaml', 'exact_match,flexible-extract', 100),
    'mmmu': ('mmmu_val', 'mmmu/mmmu_val.yaml', 'mmmu_acc,none', 100),
    'mmstar': ('mmstar', 'mmstar/mmstar.yaml', 'average,none', 100),
    'seedbench': ('seedbench', 'seedbench/seedbench.yaml', 'seed_image,none', 100),
    'hallusionbench': ('hallusion_bench_image', 'hallusion_bench/hallusion_bench_image.yaml', 'aAcc,none', 1),
    'mmbench': ('mmbench_en_dev', 'mmbench/mmbench_en_dev.yaml', 'gpt_eval_score,none', 1),
}
JUDGED = {'hallusionbench', 'mmbench'}


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + '\n')


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def selected(value, available):
    names = value.split(',') if value else list(available)
    if len(set(names)) != len(names) or set(names) - set(available):
        raise ValueError(f'Choose unique names from {list(available)}; got {names}')
    return names


def source_check():
    commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain', '--untracked-files=all'], text=True)
    if commit != COMMIT or dirty:
        raise ValueError('Harness source must be the clean pinned commit; rerun setup or use a fresh checkout')
    # Editable imports lose to the inherited wheel on this machine. Explicit precedence is required.
    sys.path.insert(0, str(SOURCE))
    import lmms_eval
    if Path(lmms_eval.__file__).resolve().parent != SOURCE / 'lmms_eval':
        raise ValueError(f'Wrong lmms_eval import: {lmms_eval.__file__}')
    return commit


@contextmanager
def local_mmmu_judge_guard(suite):
    """Cover both YAML loading passes; standard MMMU must never call a judge."""
    if suite != 'mmmu':
        yield
        return
    import lmms_eval.llm_judge as judge
    original = judge.get_server
    class DisabledJudge:
        def __getattr__(self, name):
            raise RuntimeError('Judge access forbidden in the standard MMMU task')
    try:
        judge.get_server = lambda *a, **kw: DisabledJudge()
        yield
    finally:
        judge.get_server = original


def mmmu_typed_open_results(doc, results, *, original):
    """Keep the official parser's candidate list and numeric types intact.

    v0.7.1 stringifies candidate[0], then eval_open iterates its characters.
    MC extraction, open normalization and aggregation remain the pinned code.
    """
    payload = original(doc, results)
    if doc['question_type'] != 'multiple-choice':
        candidates = [original.__globals__['parse_open_response'](pred) for pred in results]
        payload['mmmu_acc']['parsed_pred'] = candidates
        payload['mmmu_acc_pass_at_k']['parsed_pred'] = candidates
    return payload


def task_config(suite, protocol=None):
    from lmms_eval.utils import load_yaml_config
    path = str(SOURCE / 'lmms_eval/tasks' / TASKS[suite][1])
    with local_mmmu_judge_guard(suite):
        cfg = load_yaml_config(path)
    if suite == 'mmmu' and cfg['process_results'].__name__ != 'mmmu_process_results':
        raise ValueError('Unexpected MMMU scorer; review judge requirements')
    if cfg['task'] != TASKS[suite][0]:
        raise ValueError(f'Unexpected task: {cfg["task"]}')
    if suite == 'mmmu' and (protocol or {}).get('mmmu_prompt') == 'official_short_answer':
        # The pinned Qwen3 override asks open questions to choose nonexistent
        # options. Use this task's standard MC/open branching instead.
        cfg['lmms_eval_specific_kwargs']['qwen3_vl'] = dict(cfg['lmms_eval_specific_kwargs']['default'])
    if suite == 'mmmu' and (protocol or {}).get('mmmu_open_scoring') == 'typed_candidates':
        cfg['process_results'] = functools.partial(mmmu_typed_open_results, original=cfg['process_results'])
    return cfg


def environment():
    names = ('lmms-eval', 'torch', 'transformers', 'datasets', 'accelerate', 'qwen-vl-utils',
             'decord', 'numpy', 'Pillow', 'huggingface-hub', 'evaluate', 'requests', 'filelock')
    return {'python': sys.version, 'packages': {n: importlib.metadata.version(n) for n in names}}


def validate_result(result, samples, suite, limit):
    task, _, metric, scale = TASKS[suite]
    score = result['results'][task][metric]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError('Primary metric must be finite and numeric')
    if not 0 <= score * scale <= 100:
        raise ValueError('Primary metric outside the declared scale')
    counts = result['n-samples'][task]
    expected = min(limit, counts['original']) if limit else counts['original']
    if expected <= 0 or counts['effective'] != expected or len(samples) != expected:
        raise ValueError('Incomplete sample coverage')
    if result['config']['limit'] != limit:
        raise ValueError('Result limit does not match full/smoke protocol')
    ids = [s['doc_id'] for s in samples]
    if sorted(ids) != list(range(expected)):
        raise ValueError('Missing or duplicate document IDs')
    if any(not s.get('doc_hash') or 'target' not in s or not s.get('resps') for s in samples):
        raise ValueError('Missing prediction provenance')
    identity = digest(sorted([(s['doc_id'], s['doc_hash'], s['target']) for s in samples], key=lambda x: x[0]))
    return {'score_percent': score * scale, 'count': expected, 'sample_identity': identity}


def verified_completion(job_dir):
    marker = job_dir / 'COMPLETE.json'
    if not marker.exists():
        return None
    done = read(marker)
    attempt = job_dir / done['attempt']
    for name, checksum in done['sha256'].items():
        if sha(attempt / name) != checksum:
            raise ValueError(f'Completed output changed: {attempt / name}')
    return done


def summarize(output, methods):
    rows = []
    complete = 0
    for method in methods:
        values = []
        for suite in TASKS:
            done = verified_completion(output / method / suite)
            values.append(done['score_percent'] if done else None)
            complete += done is not None
        rows.append({'method': method, 'scores_percent': dict(zip(TASKS, values)),
                     'mean_percent': sum(values) / len(values) if all(v is not None for v in values) else None})
    summary = {'completed': complete, 'expected': len(methods) * len(TASKS), 'rows': rows,
               'warning': 'Harness protocol; not verified paper reproduction. Means require all seven suites.'}
    write(output / 'comparison.json', summary)
    from paper_acceptance import PAPER, SOURCE as PAPER_SOURCE, assess
    limited = read(output / 'protocol.json').get('limit') if (output / 'protocol.json').exists() else None
    acceptance = {row['method']: {suite: assess(row['method'], suite, score,
                  judged=suite in JUDGED, smoke=limited is not None)
                  for suite, score in row['scores_percent'].items()}
                  for row in rows if row['method'] in PAPER}
    write(output / 'paper_comparison.json', {'paper_source': PAPER_SOURCE, 'models': acceptance,
          'warning': 'Numerical agreement does not establish author-protocol identity; missing suites never pass.'})
    lines = ['# Pinned lmms-eval comparison', '', summary['warning'], '',
             f'Completed {complete}/{summary["expected"]} model/benchmark pairs.', '',
             '| Model | ' + ' | '.join(TASKS) + ' | Mean |',
             '| --- | ' + ' | '.join(['---:'] * (len(TASKS) + 1)) + ' |']
    for row in rows:
        vals = list(row['scores_percent'].values()) + [row['mean_percent']]
        lines.append('| ' + row['method'] + ' | ' + ' | '.join('—' if v is None else f'{v:.2f}' for v in vals) + ' |')
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(f'Validated completion: {complete}/{summary["expected"]}; {output / "comparison.md"}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan', 'check', 'prepare', 'run', 'summarize'], nargs='?', default='plan')
    p.add_argument('--experiment', type=Path, default=DEFAULT / 'experiment.json')
    p.add_argument('--protocol', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--methods', help='Comma-separated experiment model IDs; default all ten')
    p.add_argument('--suites', help='Comma-separated suite IDs; default all seven')
    p.add_argument('--smoke', type=int, metavar='N', help='First N examples, isolated from full runs')
    p.add_argument('--gpu', help='One physical GPU index in PCI_BUS_ID order; required for run')
    p.add_argument('--allow-judge', action='store_true', help='Authorize external paid judging for selected judge tasks')
    args = p.parse_args()
    exp = Experiment(args.experiment)
    protocol = read(args.protocol or exp.directory / 'lmms_protocol.json')
    if protocol['harness_commit'] != COMMIT:
        raise ValueError('Protocol/source commit mismatch')
    if args.smoke is not None and args.smoke <= 0:
        p.error('--smoke must be positive')
    methods = selected(args.methods, exp.models)
    suites = selected(args.suites, TASKS)
    output = (args.output or exp.directory / 'lmms_evaluation') / (f'smoke_{args.smoke}' if args.smoke else 'full')
    if args.action == 'plan':
        print(json.dumps({'jobs': len(methods) * len(suites), 'methods': {n: str(exp.paths[n]) for n in methods},
                          'tasks': {n: TASKS[n][0] for n in suites}, 'output': str(output),
                          'protocol': protocol, 'limit': args.smoke}, indent=2))
        return
    if args.action == 'summarize':
        summarize(output, methods)
        return
    source_check()
    os.environ.update(API_TYPE='openai', MODEL_VERSION=protocol['judges']['mmbench'],
                      OPENAI_API_URL=protocol['judge_endpoint'], NUM_IMAGE='1')
    from lmms_eval.models.simple.qwen3_vl import Qwen3_VL  # CPU import only
    configs = {s: task_config(s, protocol) for s in TASKS}
    for method in methods:
        checkpoint_files(exp.paths[method])
        if exp.models[method].get('kind') in ('trained', 'export'):
            exp.validate_export(method)
    if args.action == 'check':
        print(json.dumps({'source': str(SOURCE), 'commit': COMMIT, 'environment': environment(),
                          'models_checked': methods,
                          'tasks_checked': {s: {'task': c['task'], 'dataset': c['dataset_path'],
                           'split': c.get('test_split', c.get('validation_split')),
                           'max_new_tokens': c.get('generation_kwargs', {}).get('max_new_tokens', 128)}
                           for s, c in configs.items()}}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.matrix.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pin(output / 'protocol.json', {'protocol': protocol, 'limit': args.smoke, 'tasks': {k: list(v) for k, v in TASKS.items()},
                                     'environment': environment(), 'experiment': exp.spec,
                                     'runner_sha256': sha(__file__),
                                     'worker_sha256': sha(Path(__file__).with_name('lmms_worker.py')),
                                     'checkpoint_helper_sha256': sha(ROOT / 'qwen-vl-finetune/scripts/constrained_experiment.py')})
        dataset_lock = output / 'datasets.lock.json'
        if not dataset_lock.exists():
            from huggingface_hub import HfApi
            api = HfApi()
            datasets = {}
            for suite, cfg in configs.items():
                info = api.dataset_info(cfg['dataset_path'])
                datasets[suite] = {'requested_repo': cfg['dataset_path'], 'repo': info.id, 'revision': info.sha}
            pin(dataset_lock, datasets)
        if args.action == 'prepare':
            print(f'Pinned dataset revisions and protocol: {output}; no model loaded or dataset downloaded')
            return
        if not args.gpu or not args.gpu.isdigit():
            p.error('run requires --gpu with one physical index; survey ownership with gpuwho first')
        if set(suites) & JUDGED:
            if not args.allow_judge or os.environ.get('OPENAI_API_KEY', '') in ('', 'YOUR_API_KEY'):
                p.error('Judge tasks require --allow-judge and a configured OPENAI_API_KEY; credentials are never saved')
        failures = []
        for method in methods:
            pin(output / method / 'model_provenance.json', exp.provenance(method))
            for suite in suites:
                job_dir = output / method / suite
                if verified_completion(job_dir):
                    print(f'SKIP complete: {method}/{suite}', flush=True)
                    continue
                attempt = job_dir / dt.datetime.now(dt.timezone.utc).strftime('attempt_%Y%m%dT%H%M%S_%fZ')
                attempt.mkdir(parents=True)
                job = {'method': method, 'suite': suite, 'model_path': str(exp.paths[method]),
                       'protocol': protocol, 'dataset': read(dataset_lock)[suite], 'limit': args.smoke,
                       'gpu': args.gpu, 'output_root': str(output.resolve())}
                write(attempt / 'job.json', job)
                env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=args.gpu,
                           TOKENIZERS_PARALLELISM='false', PYTHONHASHSEED=str(protocol['seed']),
                           PYTHONPATH=str(SOURCE) + os.pathsep + os.environ.get('PYTHONPATH', ''))
                cmd = [sys.executable, str(Path(__file__).with_name('lmms_worker.py')), str(attempt / 'job.json')]
                print(f'RUN {method}/{suite}; log: {attempt / "run.log"}', flush=True)
                with (attempt / 'run.log').open('w') as log:
                    code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                if code or not (attempt / 'validated.json').exists():
                    failures.append(f'{method}/{suite}')
                    print(f'FAILED {method}/{suite}: exit {code}; inspect run.log', flush=True)
                    continue
                done = read(attempt / 'validated.json')
                done.update(attempt=attempt.name, sha256={n: sha(attempt / n) for n in
                            ('job.json', 'results.json', 'samples.jsonl', 'dataset.json', 'judge_audit.json', 'validated.json')})
                pin(job_dir / 'COMPLETE.json', done)
                print(f'DONE {method}/{suite}: {done["score_percent"]:.3f}%', flush=True)
        summarize(output, methods)
        if failures:
            raise SystemExit('Incomplete jobs (rerun to retry): ' + ', '.join(failures))


if __name__ == '__main__':
    main()
