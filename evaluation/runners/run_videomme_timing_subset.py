#!/usr/bin/env python3
"""Isolated 50-question VideoMME timing matrix; never alters accuracy runs."""
from contextlib import ExitStack
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import patch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.runners import run_token_reduction_video_retry as retry
from evaluation.protocols.video_timing_subset import select_samples, estimate, phase_configs

matrix = retry.matrix
DEFAULT = retry.EXP / 'outputs/reruns/videomme_timing50_seed42'
RECOVERY = retry.EXP / 'outputs/video_recovery_gpu1/full'


def jsonlines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def prepare(output, seed=42):
    output.mkdir(parents=True, exist_ok=True)
    # Reuse the rigorous CPU provenance checks in a separate reference folder.
    exp, policy, sidecar = retry.prepare(output / 'reference', retry.EXECUTION)
    base = RECOVERY / 'qat_w4/videomme/baseline_r100'
    done = matrix.verified_completion(base)
    if done is None or done['count'] != 1800:
        raise ValueError('Need the verified full VideoMME baseline for workload strata')
    attempt = base / done['attempt']
    inputs = jsonlines(attempt / 'video_inputs.jsonl')
    telemetry = jsonlines(attempt / 'telemetry.jsonl')
    samples = {s['doc_id']: s for s in jsonlines(attempt / 'samples.jsonl')}
    rows, info = matrix.load_video('videomme', None)
    if len(inputs) != len(telemetry) or len(inputs) != 1800:
        raise ValueError('Reference workload coverage differs')
    population = []
    for inp, measured in zip(inputs, telemetry, strict=True):
        doc = samples[inp['doc_id']]['doc']
        source = rows[inp['doc_id']]
        if inp['id'] != source['id'] or doc['question_id'] != source['id']:
            raise ValueError('Reference source alignment differs')
        population.append(dict(id=source['id'], video_id=source['video_id'], duration=source['category'],
            domain=doc['domain'], task_type=doc['task_type'],
            duration_seconds=inp['sampling']['duration_seconds'],
            original_visual_tokens=measured['video_adaptation']['original_policy_visual_tokens'],
            adaptive_visual_tokens=measured['video_adaptation']['adaptive_baseline_visual_tokens']))
    selection, strata = select_samples(population, seed=seed)
    chosen = {r['id'] for r in selection}
    reference_model_total = sum(r['model_seconds'] for r in telemetry)
    reference_model_expanded = 36 * sum(t['model_seconds'] for i, t in zip(inputs, telemetry)
                                        if i['id'] in chosen)
    matrix.pin(output / 'selection_diagnostics.json', dict(
        note='Post-selection check only; seed/subset were not optimized against timings. '
             'Historical reference timings may include contention; these are not new measurements.',
        reference_full_model_seconds=reference_model_total,
        reference_expanded_subset_model_seconds=reference_model_expanded,
        reference_model_relative_error=reference_model_expanded / reference_model_total - 1,
        original_visual_token_relative_error=(36 * sum(r['original_visual_tokens'] for r in selection)
                                              / sum(r['original_visual_tokens'] for r in population) - 1)))
    manifest = dict(schema=1, seed=seed, sampling='5 videos/quintile/duration, 1 of 3 questions/video',
        full_questions=1800, subset_questions=50, strata=strata, samples=selection,
        population_identity=matrix.digest(rows),
        reference_marker=str(base / 'COMPLETE.json'), reference_marker_sha256=matrix.sha(base / 'COMPLETE.json'))
    matrix.pin(output / 'subset.json', manifest)
    # No repeated fixed-32-token profiles: all measured questions use greedy EOS.
    # A full untimed first-input warmup is performed once per configuration.
    policy = dict(policy, profile_samples=0)
    files = [Path(__file__), matrix.ROOT / 'evaluation/protocols/video_timing_subset.py']
    protocol = dict(schema=1, model='qat_w4', model_path=str(exp.paths['qat_w4']),
        sidecar_sha256=sidecar, policy=policy, execution=retry.EXECUTION,
        subset_sha256=matrix.sha(output / 'subset.json'),
        reference_protocol_sha256=matrix.sha(output / 'reference/protocol.json'),
        code={str(p.relative_to(matrix.ROOT)): matrix.sha(p) for p in files},
        scheduling='one worker, one exclusive A100, all configurations sequential on same GPU',
        warmup='one full first-input greedy generation per configuration, excluded',
        timing='synchronized model plus separately measured preprocessing/staging; no answer replay',
        uncertainty='4000 paired within-stratum video bootstrap draws; conditional on this run',
        accuracy='Keep historical full-run scores; subset scores are diagnostic only')
    matrix.pin(output / 'protocol.json', protocol)
    return protocol, manifest


def check_protocol(output):
    protocol = matrix.read(output / 'protocol.json')
    if matrix.sha(output / 'subset.json') != protocol['subset_sha256']:
        raise ValueError('Subset changed')
    if matrix.sha(output / 'reference/protocol.json') != protocol['reference_protocol_sha256']:
        raise ValueError('Reference protocol changed')
    reference = matrix.read(output / 'reference/protocol.json')
    for name, checksum in {**reference['source_code'], **reference['repair_code'], **protocol['code']}.items():
        if matrix.sha(matrix.ROOT / name) != checksum:
            raise ValueError(f'Pinned source changed: {name}; use a fresh prepared output')
    return protocol


def worker(job_path):
    job = matrix.read(job_path)
    output = Path(job['output'])
    check_protocol(output)
    subset = matrix.read(output / 'subset.json')
    selected_ids = [r['id'] for r in subset['samples']]
    original_load = matrix.load_video

    def load(suite, limit):
        rows, info = original_load(suite, None)
        if matrix.digest(rows) != subset['population_identity']:
            raise ValueError('Full population changed')
        indexed = {r['id']: r for r in rows}
        selected = [indexed[k] for k in selected_ids]
        return selected, dict(info, questions=50, videos=50, sample_identity=matrix.digest(selected),
                              timing_subset_sha256=job['subset_sha256'])

    retry.install_worker(job['execution'])
    from evaluation.protocols.video_memory_runtime import MemoryMeasuredGenerator
    original_inputs = matrix.video_inputs
    state = {}
    stopped = threading.Event()
    contaminated = []

    def watch():
        from evaluation.workers.video_runtime import gpu_state
        while not stopped.is_set():
            try:
                os.kill(job['controller_pid'], 0)
            except ProcessLookupError:
                os._exit(143)
            if state.get('monitor_gpu'):
                try:
                    observed = gpu_state(job['gpu'])
                    pids = [int(line.split(',')[0]) for line in observed['processes'].splitlines() if line.strip()]
                    if set(pids) - {os.getpid()}:
                        contaminated.append(observed)
                    with (Path(job_path).parent / 'gpu_observations.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),
                                                     **observed)) + '\n')
                except Exception as error:
                    contaminated.append(dict(error=str(error)))
            stopped.wait(5)

    threading.Thread(target=watch, daemon=True).start()

    def inputs(*args, **kwargs):
        started = time.perf_counter()
        batch, meta = original_inputs(*args, **kwargs)
        state.update(id=args[1]['id'], preprocess_seconds=time.perf_counter() - started)
        cls = type(batch)

        class TimedInputs(cls):
            def to(self, device):
                import torch
                torch.cuda.synchronize()
                started = time.perf_counter()
                value = super().to(device)
                torch.cuda.synchronize()
                state['staging_seconds'] = time.perf_counter() - started
                return value

        return TimedInputs(batch), meta

    class Generator(MemoryMeasuredGenerator):
        def __call__(self, *args, **values):
            import torch
            state['monitor_gpu'] = True
            if contaminated:
                raise RuntimeError('Foreign GPU process or monitoring error; reject contaminated timing')
            # Warmup compiles the same kernels/sequence as the first measurement.
            if self.index == 0:
                warm = {k: v for k, v in values.items() if k not in ('max_new_tokens', '_baseline_visual_tokens')}
                started = time.perf_counter()
                self.once(warm, self.job['policy']['max_new_tokens'])
                torch.cuda.synchronize()
                matrix.write(Path(job_path).parent / 'warmup.json', dict(seconds=time.perf_counter() - started))
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = super().__call__(*args, **values)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            with (Path(job_path).parent / 'timing.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(id=state['id'], preprocess_seconds=state['preprocess_seconds'],
                    staging_seconds=state['staging_seconds'], index=self.index - 1, generate_call_seconds=elapsed,
                    request_seconds=state['preprocess_seconds'] + state['staging_seconds'] + elapsed)) + '\n')
                stream.flush()
            print(f'MEASURED {self.index}/50 {state["id"]} {elapsed:.3f}s', flush=True)
            return result

    try:
        with patch.object(matrix, 'load_video', load), patch.object(matrix, 'video_inputs', inputs), \
                patch.object(matrix, 'MeasuredGenerator', Generator):
            matrix.worker(job_path)
        if contaminated:
            raise RuntimeError('Foreign GPU process or monitoring error; reject contaminated timing')
    finally:
        stopped.set()
    retry.finish_adaptation_audit(Path(job_path).parent)
    validate_measurements(Path(job_path).parent, subset)


def validate_measurements(attempt, subset):
    timings = jsonlines(attempt / 'timing.jsonl')
    telemetry = jsonlines(attempt / 'telemetry.jsonl')
    predictions = jsonlines(attempt / 'samples.jsonl')
    expected = [s['id'] for s in subset['samples']]
    predicted = {p['doc_id']: p['doc']['question_id'] for p in predictions}
    if ([t['id'] for t in timings] != expected or [t['index'] for t in telemetry] != list(range(50))
            or [predicted.get(i) for i in range(50)] != expected):
        raise ValueError('Measurement/prediction coverage differs from selected questions')
    if len(telemetry) != 50 or any(t['fused_int4_calls'] <= 0 for t in telemetry):
        raise ValueError('Incomplete INT4 measurements')
    for source, measured in zip(subset['samples'], telemetry, strict=True):
        adaptation = measured['video_adaptation']
        if (adaptation['original_policy_visual_tokens'] != source['original_visual_tokens'] or
                adaptation['adaptive_baseline_visual_tokens'] != source['adaptive_visual_tokens']):
            raise ValueError('Shared baseline workload changed')
    if matrix.read(attempt / 'validated.json')['output_cutoffs']:
        raise ValueError('Generation cutoff: timing is censored')
    return timings, telemetry


def summarize(output):
    subset = matrix.read(output / 'subset.json')
    records, raw = [], {}
    for method, retention in retry.CONFIGS:
        config = matrix.configuration_id(method, retention)
        directory = output / config
        done = matrix.verified_completion(directory)
        if done:
            timings, telemetry = validate_measurements(directory / done['attempt'], subset)
            raw[config] = dict(model={s['id']: t['model_seconds'] for s, t in zip(subset['samples'], telemetry)},
                               request={r['id']: r['request_seconds'] for r in timings})
        record = dict(config=config, status='complete' if done else 'pending',
                      full_accuracy_source='historical full-population comparison; unchanged')
        if done:
            record['subset_score_percent_diagnostic_only'] = done['score_percent']
            for metric in ('model', 'request'):
                baseline = raw.get('baseline_r100', {}).get(metric)
                record[metric] = estimate(subset['samples'], raw[config][metric], baseline)
        records.append(record)
    matrix.write(output / 'extrapolation.json', dict(population=1800, samples=50, rows=records,
        note='Estimates, not measured full runtimes. Long videos excluded. Sampling-only 95% intervals.'))
    lines = ['# VideoMME runtime extrapolation from 50 questions', '',
        '25 short + 25 medium questions, 50 distinct videos; five random videos from each '
        'original-visual-token quintile within each duration. Each question expands to 36. '
        'Target: 900 short + 900 medium questions; long videos excluded.', '',
        'One exclusive A100, batch 1, QAT W4, 2 fps/full duration, shared adaptive baseline '
        '131072-token cap, streamed vision and 512-token prefill, greedy EOS. '
        'One full first-input warmup per configuration is excluded. '
        'Historical full-population accuracy results remain unchanged; the new timing policy '
        'also applies the adaptive baseline to the two historical original-policy resolution rows.', '',
        'Request time includes video preprocessing, input staging and generation. Model time '
        'includes vision, reduction and decoder. Startup, warmup, harness setup/scoring and '
        'unmeasured filesystem context-manager overhead are excluded; this is steady-state '
        'serial runtime, not exact wall time. 95% intervals use paired within-stratum bootstrap '
        'and exclude run-to-run variation, host contention and clock drift.', '',
        '| Configuration | Model h [95% CI] | Request h [95% CI] | Model speedup [95% CI] | Status |',
        '| --- | ---: | ---: | ---: | --- |']
    for r in records:
        cells = []
        for metric in ('model', 'request'):
            if metric in r:
                v = r[metric]; lo, hi = v['full_seconds_ci95']
                cells.append(f'{v["full_seconds"]/3600:.2f} [{lo/3600:.2f}, {hi/3600:.2f}]')
            else:
                cells.append('—')
        v = r.get('model', {})
        speed = f'{v["speedup"]:.2f}x [{v["speedup_ci95"][0]:.2f}, {v["speedup_ci95"][1]:.2f}]' if 'speedup' in v else '—'
        lines.append('| ' + ' | '.join([r['config'], *cells, speed, r['status']]) + ' |')
    (output / 'extrapolation.md').write_text('\n'.join(lines) + '\n')
    return records


def run(output, gpus, wait, configs=None):
    from evaluation.workers.video_runtime import gpu_lock, ensure_gpu_idle, gpu_state
    from evaluation.protocols.gpu_reservation import GPUReservations, run_reserved_worker
    check_protocol(output)
    configs = retry.CONFIGS if configs is None else configs
    selected_names = [matrix.configuration_id(m, r) for m, r in configs]
    hardware = matrix.read(output / 'hardware.json') if (output / 'hardware.json').exists() else None
    if hardware:
        if hardware['gpu'] not in gpus:
            raise ValueError('Resume must include the original GPU to keep hardware matched')
        gpus = [hardware['gpu']]
    if all(matrix.verified_completion(output / matrix.configuration_id(m, r)) for m, r in configs):
        summarize(output)
        matrix.write(output / 'status.json', dict(state='phase_complete', configurations=selected_names, pid=os.getpid()))
        return
    with ExitStack() as stack:
        gpu = None
        while gpu is None:
            for candidate in gpus:
                resources = ExitStack()
                try:
                    resources.enter_context(gpu_lock(candidate))
                    ensure_gpu_idle(candidate)
                except RuntimeError:
                    resources.close()
                    continue
                gpu = candidate
                stack.enter_context(resources)
                break
            if gpu is None:
                if not wait:
                    raise RuntimeError('No idle GPU; use --wait to queue without stopping jobs')
                matrix.write(output / 'status.json', dict(state='waiting_for_idle_gpu', candidates=gpus,
                    pid=os.getpid(), updated=dt.datetime.now(dt.timezone.utc).isoformat()))
                print('WAIT: no idle authorized A100; retry in 60 seconds', flush=True)
                time.sleep(60)
        protocol = check_protocol(output)
        # The queue may have waited days; revalidate the actual checkpoint and
        # environment now, before accepting any newly measured timings.
        retry.prepare(output / 'reference', retry.EXECUTION)
        env = {k: v for k, v in os.environ.items() if k not in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')}
        env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', VIDEO_MEMORY_TEST_DEVICE='cuda',
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', TRITON_CACHE_DIR=str(retry.EXP / 'outputs/cache/triton'))
        current_hardware = gpu_state(gpu)
        if hardware:
            if hardware['state']['device'].split(',')[1:3] != current_hardware['device'].split(',')[1:3]:
                raise ValueError('GPU identity/model changed')
        else:
            matrix.pin(output / 'hardware.json', dict(gpu=gpu, state=current_hardware))
        matrix.write(output / 'status.json', dict(state='gpu_validation', gpu=gpu, pid=os.getpid()))
        for label, command in [('int4', [sys.executable, str(matrix.ROOT / 'evaluation/tests/check_int4_kernel.py')]),
                ('memory', [sys.executable, '-m', 'unittest', 'discover', '-s',
                 str(matrix.ROOT / 'evaluation/tests'), '-p', 'test_video_memory_runtime.py', '-v'])]:
            ensure_gpu_idle(gpu)
            with (output / f'{label}_gate.log').open('w') as stream:
                subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        reservations = GPUReservations()
        stack.callback(reservations.close)
        for method, retention in configs:
            check_protocol(output)
            config = matrix.configuration_id(method, retention)
            directory = output / config
            if matrix.verified_completion(directory):
                continue
            reservations.acquire(gpu)
            attempt = directory / dt.datetime.now(dt.timezone.utc).strftime('attempt_%Y%m%dT%H%M%S_%fZ')
            attempt.mkdir(parents=True)
            job = dict(model='qat_w4', suite='videomme', method=method, retention=retention,
                model_path=protocol['model_path'], policy=protocol['policy'], execution=protocol['execution'],
                sidecar_sha256=protocol['sidecar_sha256'], subset_sha256=protocol['subset_sha256'],
                packed_cache=str(retry.EXP / 'outputs/cache/packed_qat_w4'), limit=None,
                reserve_gpus=True, controller_pid=os.getpid(), gpu=gpu, output=str(output))
            matrix.write(attempt / 'job.json', job)
            matrix.write(output / 'status.json', dict(state='running', config=config, gpu=gpu,
                attempt=str(attempt), pid=os.getpid()))
            print(f'RUN GPU {gpu}: {config}: {attempt}', flush=True)
            started = time.perf_counter()
            with (attempt / 'run.log').open('w') as stream:
                rc = run_reserved_worker([sys.executable, str(Path(__file__).resolve()), '--worker',
                    str(attempt / 'job.json')], env, stream, attempt, gpu, reservations)
            matrix.write(attempt / 'exit.json', dict(returncode=rc, wall_seconds=time.perf_counter()-started))
            if rc:
                raise RuntimeError(f'Timing worker failed: {attempt / "run.log"}')
            validated = matrix.read(attempt / 'validated.json')
            validated.update(attempt=attempt.name, sha256={p.name: matrix.sha(p) for p in attempt.iterdir()
                if p.is_file() and p.name != 'run.log'})
            matrix.pin(directory / 'COMPLETE.json', validated)
            summarize(output)
        matrix.write(output / 'status.json', dict(state='complete' if len(configs) == len(retry.CONFIGS) else 'phase_complete',
            configurations=selected_names, gpu=gpu, pid=os.getpid()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'summarize', 'check-projection'], nargs='?', default='prepare')
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--gpus', default='0,1,2,4', help='Candidates; first idle card runs all jobs serially')
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--configurations', help='Comma-separated configuration IDs; only these jobs run')
    parser.add_argument('--seed', type=int, default=42, help='Subset selection seed for prepare; new seed needs new output')
    args = parser.parse_args()
    output = args.output.resolve()
    from evaluation.protocols.gpu_job_queue import gpu_indices
    gpus = gpu_indices(args.gpus)
    if set(gpus) - {'0', '1', '2', '4'}:
        parser.error('Only the four A100 PCI indices 0,1,2,4 are allowed')
    try:
        configs = phase_configs(args.configurations, retry.CONFIGS)
    except ValueError as error:
        parser.error(str(error))
    if args.action == 'check-projection':
        check_protocol(output)
        from evaluation.audits.check_videomme_timing_projection import check
        check(output)
        return
    if args.action == 'summarize':
        check_protocol(output)
        summarize(output)
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'prepare':
            _, subset = prepare(output, args.seed)
            summarize(output)
            print(f'Prepared {len(subset["samples"])} questions; no GPU allocated: {output}', flush=True)
        else:
            try:
                run(output, gpus, args.wait, configs)
            except BaseException as error:
                matrix.write(output / 'status.json', dict(state='stopped', error=str(error), pid=os.getpid()))
                raise


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker(Path(sys.argv[sys.argv.index('--worker') + 1]))
    else:
        main()
