# Video control lives here alongside the image matrix. The pinned harness remains clean.
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
    import pandas as pd
    from evaluation.protocols.video_protocol import SUITES
    base = Path(data_root) / SUITES[suite][0] / 'data'
    subset = 'multi-choice' if suite == 'tempcompass' else ('videomme' if suite == 'videomme'
                                                          else suite.split('_')[1].capitalize())
    raw = pd.concat([pd.read_parquet(p) for p in sorted((base / subset).glob('test-*.parquet'))],
                    ignore_index=True).to_dict('records')
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
    lines = ['# lmms-eval video comparison', '', '720p policy is pinned in protocol.json. VideoMME excludes long; '
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
    p.add_argument('--methods')
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
    methods, suites = selected(args.methods, exp.models), selected(args.suites, VIDEO_TASKS)
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
                  spatial_policy='Downsample only; fit 1280x720 envelope (orientation independent), rounded to patch grid',
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
                                                        'video_population.py', 'local_scoring.py', 'paths.py')}
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
                    from evaluation.workers.video_runtime import require_idle_gpu
                    require_idle_gpu(args.gpu)
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

