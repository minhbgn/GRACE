#!/usr/bin/env python3
"""Pinned lmms-eval matrix. Planning, checking and preparing never load a model."""
# Support both direct script execution and `python -m evaluation.<group>.<module>`.
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.paths import EVALUATION_ROOT, source_file

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

from evaluation.paths import GRACE_ROOT as ROOT
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


def validate_result(result, samples, suite, limit, tasks=None):
    task, _, metric, scale = (tasks or TASKS)[suite]
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
    from evaluation.protocols.paper_acceptance import PAPER, SOURCE as PAPER_SOURCE, assess
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
                                     'runner_sha256': sha(__file__), 'layout_helper_sha256': sha(source_file('paths.py')),
                                     'worker_sha256': sha(source_file('lmms_worker.py')),
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
                cmd = [sys.executable, str(source_file('lmms_worker.py')), str(attempt / 'job.json')]
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


# Video control lives here alongside the image matrix. The pinned harness remains clean.
VIDEO_PRIORITY = ('qat_w4', 'qat_w8', 'vanilla_2b')


def video_methods(value, available):
    if value:
        return selected(value, available)
    return [name for name in VIDEO_PRIORITY if name in available] + [
        name for name in available if name not in VIDEO_PRIORITY]


VIDEO_TASKS = {
    'tempcompass': ('tempcompass_multi_choice', 'tempcompass/multi-choice.yaml', 'avg_accuracy,none', 1),
    'videomme': ('videomme', 'videomme/videomme.yaml', 'videomme_perception_score,none', 1),
    **{f'videommmu_{part}': (f'video_mmmu_{part}', f'videommmu/{part}.yaml', 'mmmu_acc,none', 100)
       for part in ('perception', 'comprehension', 'adaptation')},
}
VIDEO_NOTES = {
    'tempcompass': 'Local deterministic choice fallback (local_scoring.choice); unresolved answers count wrong. '
                   'Replaces the harness API-judge fallback; not the official judged score.',
    'videomme': 'Pinned lmms-eval local scorer; short + medium only (1800/2700); no subtitles.',
    'videommmu': 'Pinned lmms-eval local MC/open-answer scorer; no API judge.',
    'skipped': {'tempcompass_captioning': 'API judge required; no local captioning scorer found.',
                'tempcompass_yes_no': 'Not in the current local annotation population.',
                'tempcompass_caption_matching': 'Not in the current local annotation population.',
                'longvideobench': 'Long-video benchmark excluded.',
                'videomme_long': '900 long questions excluded.'},
}


def video_local_choice(doc, results):
    from evaluation.protocols.local_scoring import choice
    row = json.loads(doc['_grace_row'])
    prediction, _ = choice(results[0], row['options'], row['options'])
    payload = dict(rating=int(prediction == row['gold']), prediction=prediction,
                   unresolved=not prediction, scoring='local_deterministic_fallback',
                   video_id=doc['video_id'], dim=doc['dim'])
    return {'avg_accuracy': payload, doc['dim'] + '_accuracy': payload}


def video_visual(doc):
    # Media is resolved lazily (including ZIP members) by the local lmms adapter.
    return [json.loads(doc['_grace_row'])['video_id']]


def video_task_config(suite, protocol):
    from lmms_eval.utils import load_yaml_config
    os.environ.setdefault('HF_HOME', str(Path.home() / '.cache/huggingface'))
    cfg = load_yaml_config(str(SOURCE / 'lmms_eval/tasks' / VIDEO_TASKS[suite][1]))
    if cfg['task'] != VIDEO_TASKS[suite][0]:
        raise ValueError('Pinned video task identity changed')
    cfg['doc_to_visual'] = video_visual
    cfg['dataset_kwargs'] = {}  # Local task.download bypasses HF loading and extraction.
    cfg['generation_kwargs'] = dict(max_new_tokens=protocol['max_new_tokens'], temperature=0,
                                     do_sample=False, num_beams=1)
    if suite == 'tempcompass':
        cfg['process_results'] = video_local_choice
    return cfg


def video_documents(suite, rows, data_root):
    """Retain raw harness fields/prompts/scorers; attach the existing local media index."""
    import pyarrow.parquet as pq
    from evaluation.protocols.video_protocol import SUITES
    base = Path(data_root) / SUITES[suite][0] / 'data'
    subset = 'multi-choice' if suite == 'tempcompass' else ('videomme' if suite == 'videomme'
                                                          else suite.split('_')[1].capitalize())
    raw = [doc for path in sorted((base / subset).glob('test-*.parquet'))
           for doc in pq.read_table(path).to_pylist()]
    indexed = {}
    for index, doc in enumerate(raw):
        key = str(index) if suite == 'tempcompass' else str(doc['question_id'] if suite == 'videomme' else doc['id'])
        indexed[key] = doc
    return [dict(indexed[row['id']], _grace_row=json.dumps(row)) for row in rows]


def video_task(suite, protocol, documents):
    from datasets import Dataset, DatasetDict
    from lmms_eval.api.task import ConfigurableTask
    class LocalVideoTask(ConfigurableTask):
        def download(self, dataset_kwargs=None):
            self.dataset = DatasetDict(test=Dataset.from_list(documents))
            self.dataset_no_image = self.dataset
    return LocalVideoTask(config=video_task_config(suite, protocol), model_name='qwen3_vl')


def video_adapter(job, attempt):
    """lmms model interface with the audited native Qwen processor input path."""
    import hashlib
    import time
    import torch
    from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration
    from lmms_eval.api.model import lmms
    from evaluation.protocols.video_protocol import local_video, probe, input_counts, build_inputs
    from evaluation.protocols.local_scoring import generation_record
    policy = job['protocol']
    class VideoModel(lmms):
        def __init__(self):
            super().__init__()
            self.processor = AutoProcessor.from_pretrained(job['model_path'], local_files_only=True)
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
            self.context = read(Path(job['model_path']) / 'config.json')['text_config']['max_position_embeddings']
            self.model = None  # Fail preflight before allocating weights.

        def loglikelihood(self, requests):
            raise NotImplementedError

        def generate_until_multi_round(self, requests):
            raise NotImplementedError

        def generate_until(self, requests):
            outputs = []
            with (attempt / 'telemetry.jsonl').open('a') as stream:
                for request in requests:
                    prompt, _, _, doc_id, task_name, split = request.args
                    doc = self.task_dict[task_name][split][doc_id]
                    row = dict(json.loads(doc['_grace_row']), prompt=prompt)
                    started = time.perf_counter()
                    with local_video(row['asset'], job['scratch']) as path:
                        meta = probe(path, policy, self.processor)
                        counts = input_counts(self.processor, row, meta)
                        if counts['input_tokens'] + policy['max_new_tokens'] > self.context:
                            write(attempt / 'blocked.json', dict(id=row['id'], reason='context_overflow',
                                  tokens=counts, context_limit=self.context))
                            raise ValueError(f'{row["id"]}: input + output reserve exceed native context')
                        inputs, counts = build_inputs(self.processor, row, path, meta)
                    if self.model is None:
                        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                            job['model_path'], local_files_only=True, dtype=torch.bfloat16,
                            attn_implementation=policy['attention']).to('cuda:0').eval()
                        self.model.generation_config = GenerationConfig(
                            do_sample=False, num_beams=1, repetition_penalty=1.0, use_cache=True,
                            eos_token_id=self.processor.tokenizer.eos_token_id,
                            pad_token_id=self.processor.tokenizer.pad_token_id)
                    inputs = inputs.to('cuda:0')
                    input_hash = hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    generate_start = time.perf_counter()
                    with torch.inference_mode():
                        generated = self.model.generate(**inputs, max_new_tokens=policy['max_new_tokens'])
                    torch.cuda.synchronize()
                    generate_seconds = time.perf_counter() - generate_start
                    ids = generated[0, inputs['input_ids'].shape[1]:].tolist()
                    raw = self.processor.tokenizer.decode(ids, skip_special_tokens=True)
                    generation = generation_record(ids, self.processor.tokenizer.eos_token_id, policy['max_new_tokens'])
                    record = dict(id=row['id'], doc_id=doc_id, raw=raw, sampling=meta,
                                  tokens=dict(counts, output_text_tokens=len(ids)), generation=generation,
                                  input_ids_sha256=input_hash, context_limit=self.context,
                                  generate_seconds=generate_seconds, elapsed_seconds=time.perf_counter() - started,
                                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
                    stream.write(json.dumps(record) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    if generation['finish_reason'] != 'eos':
                        raise ValueError('Generation cutoff; cannot certify complete evaluation')
                    outputs.append(raw)
                    self.cache_hook.add_partial('generate_until', (prompt, request.args[1]), raw)
                    del inputs, generated
            return outputs
    return VideoModel()


@contextmanager
def video_no_network():
    """Fail closed on any accidental judge or remote dataset request, even with a key set."""
    from unittest.mock import patch
    with patch('requests.sessions.Session.request', side_effect=RuntimeError('Network forbidden in local video evaluation')):
        yield


def video_worker(job_path):
    from types import SimpleNamespace
    job, attempt = read(job_path), job_path.parent
    source_check()
    from lmms_eval import evaluator
    from lmms_eval.tasks import TaskManager
    from lmms_eval.utils import handle_non_serializable
    suite, protocol = job['suite'], job['protocol']
    docs = video_documents(suite, job['rows'], job['data_root'])
    with video_no_network():
        task = video_task(suite, protocol, docs)
        model = video_adapter(job, attempt)
        result = evaluator.simple_evaluate(model=model, tasks=[task],
            task_manager=TaskManager(model_name='qwen3_vl'), batch_size=1, num_fewshot=0,
            limit=job['limit'], bootstrap_iters=0, log_samples=True,
            random_seed=protocol['seed'], numpy_random_seed=protocol['seed'],
            torch_random_seed=protocol['seed'], fewshot_random_seed=protocol['seed'],
            cli_args=SimpleNamespace(output_path=str(attempt), process_with_media=False))
    samples = result.pop('samples')[VIDEO_TASKS[suite][0]]
    (attempt / 'results.json').write_text(json.dumps(result, indent=2, default=handle_non_serializable) + '\n')
    (attempt / 'samples.jsonl').write_text(''.join(json.dumps(s, default=handle_non_serializable) + '\n' for s in samples))
    validated = validate_result(result, samples, suite, job['limit'], VIDEO_TASKS)
    if validated['count'] != len(job['rows']):
        raise ValueError('Harness coverage differs from selected population')
    telemetry = [json.loads(s) for s in (attempt / 'telemetry.jsonl').read_text().splitlines()]
    if sorted(r['doc_id'] for r in telemetry) != list(range(len(job['rows']))):
        raise ValueError('Missing or duplicate inference telemetry')
    for key in ('input_visual_tokens', 'input_text_tokens', 'output_text_tokens'):
        validated[key + '_mean'] = sum(r['tokens'][key] for r in telemetry) / len(telemetry)
    validated.update(scoring_note=VIDEO_NOTES['videommmu' if suite.startswith('videommmu') else suite],
                     input_identity=digest(sorted((r['doc_id'], r['input_ids_sha256']) for r in telemetry)))
    write(attempt / 'judge_audit.json', dict(calls=0, network='forbidden', notes=VIDEO_NOTES))
    write(attempt / 'validated.json', validated)


def video_summary(output, methods, suites):
    records, identities = [], {}
    for method in methods:
        for suite in suites:
            done = verified_completion(output / method / suite)
            if done:
                identity = done['input_identity']
                if suite in identities and identities[suite] != identity:
                    raise ValueError(f'{suite}: models received different tokenized inputs')
                identities[suite] = identity
            records.append(dict(method=method, suite=suite, status='complete' if done else 'pending', result=done))
    write(output / 'comparison.json', dict(notes=VIDEO_NOTES, rows=records))
    lines = ['# lmms-eval video comparison', '', 'Input policy is pinned in protocol.json. VideoMME excludes long; '
             'smoke runs are previews. TempCompass uses a local scoring fallback.', '',
             '| Model | Suite | Score % | Count | Status |', '| --- | --- | ---: | ---: | --- |']
    for row in records:
        done = row['result']
        lines.append(f'| {row["method"]} | {row["suite"]} | ' +
                     (f'{done["score_percent"]:.2f} | {done["count"]}' if done else '— | —') + f' | {row["status"]} |')
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(f'Validated {sum(r["status"] == "complete" for r in records)}/{len(records)} jobs: {output / "comparison.md"}')


def video_main():
    from evaluation.protocols.video_protocol import DEFAULT_DATA, load_suite
    from evaluation.protocols.video_population import select_population, VIDEOMME_SHORT_MEDIUM
    p = argparse.ArgumentParser(description='Pinned lmms-eval video matrix; offline local scoring, no API judge.')
    p.add_argument('action', choices=['plan', 'check', 'prepare', 'run', 'summarize'], nargs='?', default='plan')
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    p.add_argument('--output', type=Path)
    p.add_argument('--methods', help='Explicit order; default QAT W4, QAT W8, vanilla 2B, then remaining models')
    p.add_argument('--suites')
    p.add_argument('--gpu')
    p.add_argument('--smoke', type=int)
    p.add_argument('--fps', type=float, default=2)
    p.add_argument('--shortest-side', type=int, default=720, help='720p short-edge ceiling; never upscale')
    p.add_argument('--longest-side', type=int, default=1280, help='Additional long-edge ceiling for 720p')
    p.add_argument('--max-new-tokens', type=int, default=4096)
    p.add_argument('--attn', choices=['flash_attention_2', 'sdpa'], default='flash_attention_2')
    p.add_argument('--scratch', type=Path)
    p.add_argument('--skip-videomme-long', action='store_true', default=True,
                   help='Compatibility option; VideoMME long is always excluded')
    args = p.parse_args()
    if (not math.isfinite(args.fps) or args.fps <= 0 or args.shortest_side < 32
            or args.longest_side < args.shortest_side or args.max_new_tokens <= 0
            or (args.smoke is not None and args.smoke <= 0)):
        p.error('Invalid sampling, resolution, smoke or generation setting')
    if args.action == 'run' and (args.gpu is None or not args.gpu.isdigit()):
        p.error('run requires one physical --gpu index')
    exp = Experiment(args.experiment)
    methods, suites = video_methods(args.methods, exp.models), selected(args.suites, VIDEO_TASKS)
    datasets, reports = {}, {}
    for suite in suites:
        rows, info = load_suite(suite, args.data_root)
        rows, info = select_population(suite, rows, info, True)
        reports[suite] = info
        if info['status'] != 'ready':
            if args.suites:
                raise ValueError(f'{suite}: selected population is {info["status"]}')
            continue
        datasets[suite] = rows[:args.smoke] if args.smoke else rows
    if not datasets:
        raise ValueError('No complete local video populations')
    policy = dict(version='lmms_video_v1', harness_commit=COMMIT, fps=args.fps,
                  shortest_side=args.shortest_side, longest_side=args.longest_side,
                  spatial_policy='Downsample only to explicit short/long-edge ceilings, rounded to patch grid',
                  temporal_sampling='Uniform full clip, both endpoints, no frame cap',
                  max_new_tokens=args.max_new_tokens, attention=args.attn, seed=42,
                  subtitles=False, audio=False, truncation=False, network=False,
                  loading_format='merged BF16 export', videomme_population=VIDEOMME_SHORT_MEDIUM,
                  scoring_notes=VIDEO_NOTES)
    output = (args.output or exp.directory / 'video_evaluation' / ('lmms_video_v1_' + digest(policy)[:12])) / (
        f'smoke_{args.smoke}' if args.smoke else 'full')
    manifest = dict(policy=policy, data=reports, output=str(output), methods=methods,
                    evaluation_questions={s: len(r) for s, r in datasets.items()}, jobs=len(methods) * len(datasets))
    if args.action == 'plan':
        print(json.dumps(manifest, indent=2))
        return
    if args.action == 'summarize':
        if (output / 'protocol.json').exists() and read(output / 'protocol.json')['policy'] != policy:
            raise ValueError('Summary policy differs from saved protocol')
        video_summary(output, methods, datasets)
        return
    source_check()
    with video_no_network():
        tasks = {s: video_task(s, policy, video_documents(s, rows, args.data_root)) for s, rows in datasets.items()}
    for method in methods:
        checkpoint_files(exp.paths[method])
        if exp.models[method].get('kind') in ('trained', 'export'):
            exp.validate_export(method)
    if args.action == 'check':
        print(json.dumps(dict(manifest, harness_tasks={s: t.config.task for s, t in tasks.items()}), indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.matrix.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = {name: sha(source_file(name)) for name in ('run_lmms_matrix.py', 'video_protocol.py',
                                                        'video_population.py', 'local_scoring.py', 'paths.py',
                                                        'video_runtime.py', 'video_worker.py')}
        code['checkpoint_helper'] = sha(ROOT / 'qwen-vl-finetune/scripts/constrained_experiment.py')
        pin(output / 'protocol.json', dict(policy=policy, experiment=exp.spec, limit=args.smoke,
                                          code=code, environment=environment()))
        for suite in datasets:
            pin(output / 'datasets' / f'{suite}.json', reports[suite])
        if args.action == 'prepare':
            from transformers import AutoProcessor
            from evaluation.workers.video_worker import prepare
            processor = AutoProcessor.from_pretrained(exp.paths[methods[0]], local_files_only=True)
            context = read(exp.paths[methods[0]] / 'config.json')['text_config']['max_position_embeddings']
            prompted = {}
            for suite, task in tasks.items():
                prompted[suite] = [dict(row, prompt=task.doc_to_text(doc))
                                   for row, doc in zip(datasets[suite], task.eval_docs)]
            report = prepare(output, prompted, processor, policy, context, args.scratch)
            if any(r['errors'] or r['context_overflows'] for r in report.values()):
                raise SystemExit('Preflight found errors/context overflows; inspect preflight/summary.json')
            return
        from evaluation.workers.video_runtime import gpu_lock
        failures = []
        with gpu_lock(args.gpu):
            for method in methods:
                pin(output / method / 'model_provenance.json', exp.provenance(method))
                for suite, rows in datasets.items():
                    job_dir = output / method / suite
                    if verified_completion(job_dir):
                        continue
                    attempt = job_dir / dt.datetime.now(dt.timezone.utc).strftime('attempt_%Y%m%dT%H%M%S_%fZ')
                    attempt.mkdir(parents=True)
                    job = dict(suite=suite, method=method, rows=rows, protocol=policy, limit=args.smoke,
                               model_path=str(exp.paths[method]), data_root=str(args.data_root.resolve()),
                               scratch=str(args.scratch) if args.scratch else None)
                    write(attempt / 'job.json', job)
                    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=args.gpu,
                               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                               PYTHONPATH=str(SOURCE) + os.pathsep + os.environ.get('PYTHONPATH', ''),
                               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                    # Recheck hardware at each launch, including noncooperating jobs.
                    from evaluation.workers.video_runtime import ensure_gpu_idle
                    ensure_gpu_idle(args.gpu)
                    print(f'RUN {method}/{suite}: {attempt / "run.log"}', flush=True)
                    with (attempt / 'run.log').open('w') as log:
                        code = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--video-job',
                                               str(attempt / 'job.json')], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                    if code or not (attempt / 'validated.json').exists():
                        failures.append(f'{method}/{suite}')
                        write(attempt / 'exit.json', dict(returncode=code))
                        continue
                    done = read(attempt / 'validated.json')
                    done.update(attempt=attempt.name, sha256={n: sha(attempt / n) for n in
                        ('job.json', 'results.json', 'samples.jsonl', 'telemetry.jsonl', 'judge_audit.json', 'validated.json')})
                    pin(job_dir / 'COMPLETE.json', done)
        video_summary(output, methods, datasets)
        if failures:
            raise SystemExit('Incomplete jobs: ' + ', '.join(failures))


if __name__ == '__main__':
    if '--video-job' in sys.argv:
        video_worker(Path(sys.argv[sys.argv.index('--video-job') + 1]))
    elif '--video' in sys.argv:
        sys.argv.remove('--video')
        video_main()
    else:
        main()
