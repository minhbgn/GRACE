#!/usr/bin/env python3
"""Memory-aware QAT video recovery, preserving the original pinned computation."""
import argparse
from contextlib import ExitStack
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.runners import run_token_reduction_video_retry as retry
from evaluation.protocols.gpu_job_queue import gpu_indices
from evaluation.protocols.parallel_evaluation import GIB, run_pool

matrix = retry.matrix
EXTENSION_FILES = ['evaluation/runners/run_parallel_video_evaluation.py',
    'evaluation/protocols/parallel_evaluation.py', 'evaluation/protocols/evaluation_checkpoints.py',
    'evaluation/workers/parallel_video_worker.py']


def summarize(output):
    entries = []
    smoke = output.name.startswith('smoke_')
    lines = ['# QAT video recovery', '',
        'Parallel-controller timings can include GPU contention or checkpoint replay. '
        'No standalone speedups are inferred from these rows.' +
        (' Smoke results are previews, separate from full populations.' if smoke else ''), '',
        '| Suite | Configuration | Score % | Questions | Origin | Status |',
        '| --- | --- | ---: | ---: | --- | --- |']
    for suite in retry.SUITES:
        for method, retention in retry.CONFIGS:
            config = matrix.configuration_id(method, retention)
            directory = output / 'qat_w4' / suite / config
            if smoke:
                done = matrix.verified_completion(directory)
                origin = 'smoke' if done else None
            else:
                done, origin, directory = retry.resolved(output, suite, config)
            status = 'complete' if done else ('failed/incomplete' if list(
                (output / 'qat_w4' / suite / config).glob('attempt_*')) else 'pending retry')
            entries.append(dict(model='qat_w4', suite=suite, method=method, retention=retention,
                status=status, origin=origin, source_directory=str(directory) if done else None,
                fixed_output_speedup=None, result=done))
            lines.append('| ' + ' | '.join(map(str, [suite, config,
                f"{done['score_percent']:.3f}" if done else '—',
                done['count'] if done else '—', origin or '—', status])) + ' |')
    matrix.write(output / 'comparison.json', dict(rows=entries,
        note='Extension timing is contention/replay affected; no standalone speedup claims.'))
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(output / 'comparison.md', flush=True)


def memory_estimate(job, output, fallback):
    config = matrix.configuration_id(job['method'], job['retention'])
    done, _, _ = retry.resolved(output, job['suite'], config)
    # Use only full-population evidence for this exact suite/configuration.
    # Partial peaks are lower bounds and cannot safely size future long clips.
    if done:
        peak = done['telemetry']['peak_gpu_allocated_bytes']['max']
        return max(2 * GIB, int(peak * 1.25 + GIB))
    original, _, _ = retry.resolved(retry.DEFAULT_OUTPUT / 'full', job['suite'], config)
    if original:
        return max(2 * GIB, int(original['telemetry']['peak_gpu_allocated_bytes']['max'] * 1.25 + GIB))
    return int(fallback * GIB)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'prepare', 'check', 'run', 'summarize'], nargs='?', default='plan')
    gpu = parser.add_mutually_exclusive_group()
    gpu.add_argument('--gpu', default=None)
    gpu.add_argument('--gpus', default=None)
    parser.add_argument('--parallel-evaluation', type=int, default=1,
                        help='Maximum simultaneous evaluations PER GPU; 1 gives each worker exclusive GPU use (default: 1)')
    parser.add_argument('--gpu-pool-file', type=Path,
                        help='Live JSON list of authorized physical GPUs; removals drain current work')
    parser.add_argument('--evaluation-memory-gib', type=float, default=18,
                        help='Shared-GPU fallback peak estimate; ignored with --parallel-evaluation 1. '
                             'Scheduling estimate only, not a CUDA allocation limit (default: 18)')
    parser.add_argument('--memory-headroom-gib', type=float, default=2,
                        help='Scheduling headroom per GPU; sequential budget is total GPU memory minus this (default: 2)')
    parser.add_argument('--suites')
    parser.add_argument('--configurations', help='Optional comma-separated configuration IDs')
    parser.add_argument('--output', type=Path, default=retry.DEFAULT_OUTPUT)
    parser.add_argument('--prefill-chunk-tokens', type=int, default=512)
    parser.add_argument('--decoder-token-cap', type=int, default=131072)
    parser.add_argument('--smoke', type=int, help='Isolated N-question validation; never enters full results')
    args = parser.parse_args()
    if args.parallel_evaluation < 1 or args.prefill_chunk_tokens < 1:
        parser.error('Parallel evaluations and prefill chunk tokens must be positive')
    if (not math.isfinite(args.evaluation_memory_gib) or args.evaluation_memory_gib <= 0 or
        not math.isfinite(args.memory_headroom_gib) or args.memory_headroom_gib < 0):
        parser.error('Memory budget must be positive and headroom nonnegative, both finite')
    if args.smoke is not None and args.smoke < 1:
        parser.error('--smoke must be positive')
    try:
        gpus = gpu_indices(args.gpus or args.gpu or '1')
    except ValueError as error:
        parser.error(str(error))
    native = matrix.read(retry.SOURCE / 'qat_w4/model_provenance.json')['config']['text_config']['max_position_embeddings']
    if not 257 <= args.decoder_token_cap <= native:
        parser.error(f'--decoder-token-cap must be between 257 and {native}')
    output = args.output.resolve() / (f'smoke_{args.smoke}' if args.smoke else 'full')
    if output == retry.SOURCE or retry.SOURCE in output.parents or output in retry.SOURCE.parents:
        parser.error('Recovery output must be separate from original results')
    execution = dict(retry.EXECUTION, prefill_chunk_tokens=args.prefill_chunk_tokens,
                     adaptive_baseline_decoder_tokens=args.decoder_token_cap)
    suites = matrix.selected(args.suites, retry.SUITES)
    jobs, reused, completed = retry.inventory(output, suites)
    configs = [matrix.configuration_id(method, ratio) for method, ratio in retry.CONFIGS]
    selected = matrix.selected(args.configurations, configs)
    jobs = [job for job in jobs if matrix.configuration_id(job['method'], job['retention']) in selected]
    memory_policy = 'exclusive_gpu' if args.parallel_evaluation == 1 else 'shared_gpu'
    if args.smoke:
        # Full original completions are references, never copied into smoke output.
        jobs = [dict(model='qat_w4', suite=suite, method=method, retention=ratio)
                for suite in suites for method, ratio in retry.CONFIGS
                if matrix.configuration_id(method, ratio) in selected and not
                matrix.verified_completion(output / 'qat_w4' / suite / matrix.configuration_id(method, ratio))]
    if args.action == 'plan':
        print(json.dumps(dict(gpus=gpus, gpu_pool_file=str(args.gpu_pool_file) if args.gpu_pool_file else None,
            parallel_evaluation=args.parallel_evaluation, output=str(output), run_count=len(jobs),
            memory_policy=memory_policy, memory_headroom_gib=args.memory_headroom_gib,
            sequential_budget='total GPU memory minus headroom, resolved at dispatch' if args.parallel_evaluation == 1 else None,
            reused=len(reused), completed=len(completed), execution=execution,
            jobs=[dict(job, estimated_peak_gib=memory_estimate(job, output, args.evaluation_memory_gib)/GIB
                       if args.parallel_evaluation > 1 else None)
                  for job in jobs]), indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((output / '.matrix.lock').open('a'))
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f'Another controller is writing {output}; let its current evaluation finish before switching')
        if args.action == 'summarize':
            summarize(output)
            return
        exp, policy, sidecar = retry.prepare(output, execution)
        extension_code = {name: matrix.sha(matrix.ROOT / name) for name in EXTENSION_FILES}
        matrix.pin(output / 'parallel_implementation.json', dict(schema=1, code=extension_code, limit=args.smoke))
        if args.action in ('prepare', 'check'):
            print(f'CPU preflight passed; {len(jobs)} pending evaluations. No GPU allocated.', flush=True)
            return
        if not jobs:
            summarize(output)
            print('No remaining evaluations. No GPU allocated.', flush=True)
            return
        from evaluation.protocols.gpu_reservation import GPUReservations
        from evaluation.workers.video_runtime import gpu_lock, ensure_gpu_idle
        reservations = GPUReservations()
        stack.callback(reservations.close)
        provenance = dict(protocol=matrix.sha(output / 'protocol.json'),
                          model=matrix.sha(output / 'qat_w4/model_provenance.json'),
                          datasets={suite: matrix.sha(output / 'datasets' / f'{suite}.json') for suite in suites})

        def warn(message, level='WARNING'):
            message = f'{dt.datetime.now(dt.timezone.utc).isoformat()} {level} {message}'
            print(message, flush=True)
            with (output / 'parallel_evaluation.log').open('a') as stream:
                stream.write(message + '\n')

        gpu_resources = {}

        def acquire(gpu):
            # Keep a rejected new GPU's lock out of the lifetime stack.
            resources = ExitStack()
            try:
                resources.enter_context(gpu_lock(gpu))
                ensure_gpu_idle(gpu)
                reservations.acquire(gpu)
                # Run the unchanged numerical gates independently on each added GPU.
                reservations.release(gpu)
                ensure_gpu_idle(gpu)
                env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu,
                    VIDEO_MEMORY_TEST_DEVICE='cuda', TOKENIZERS_PARALLELISM='false',
                    TRITON_CACHE_DIR=str(retry.EXP / 'outputs/cache/triton'))
                checks = [([sys.executable, str(matrix.ROOT / 'evaluation/tests/check_int4_kernel.py')], 'int4'),
                    ([sys.executable, '-m', 'unittest', 'discover', '-s', str(matrix.ROOT / 'evaluation/tests'),
                      '-p', 'test_video_memory_runtime.py', '-v'], 'memory')]
                for command, name in checks:
                    warn(f'GPU {gpu} {name} numerical gate', level='INFO')
                    with (output / f'parallel_{name}_gate_gpu{gpu}.log').open('w') as stream:
                        subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
                reservations.acquire(gpu)
                resources.callback(reservations.release, gpu)
            except BaseException:
                reservations.release(gpu)
                resources.close()
                raise
            owned = resources.pop_all()
            gpu_resources[gpu] = owned
            stack.callback(owned.close)

        def start(job, gpu, budget):
            config = matrix.configuration_id(job['method'], job['retention'])
            directory = output / 'qat_w4' / job['suite'] / config
            attempt = directory / dt.datetime.now(dt.timezone.utc).strftime('attempt_%Y%m%dT%H%M%S_%fZ')
            attempt.mkdir(parents=True)
            payload = dict(job, model_path=str(exp.paths['qat_w4']), policy=policy, execution=execution,
                limit=args.smoke, packed_cache=str(retry.EXP / 'outputs/cache/packed_qat_w4'),
                sidecar_sha256=sidecar, reserve_gpus=True, controller_pid=os.getpid(), gpu=gpu,
                extension_code=extension_code, resume_provenance=dict(protocol=provenance['protocol'],
                    model=provenance['model'], dataset=provenance['datasets'][job['suite']]),
                scheduling=dict(parallel_evaluation=args.parallel_evaluation, gpu=gpu,
                                memory_policy=memory_policy, memory_headroom_gib=args.memory_headroom_gib,
                                estimated_peak_gib=budget/GIB, standalone_timing=False))
            matrix.write(attempt / 'job.json', payload)
            env = {key: value for key, value in os.environ.items() if key not in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')}
            env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                TRITON_CACHE_DIR=str(retry.EXP / 'outputs/cache/triton'))
            warn(f'GPU {gpu} RUN {job["suite"]}/{config}, memory_policy={memory_policy}, '
                 f'budget={budget/GIB:.2f} GiB: {attempt / "run.log"}', level='INFO')
            log = (attempt / 'run.log').open('w')
            try:
                process = subprocess.Popen([sys.executable,
                    str(matrix.ROOT / 'evaluation/workers/parallel_video_worker.py'), str(attempt / 'job.json')],
                    env=env, stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            return dict(process=process, attempt=attempt, log=log)

        def finish(item, rc):
            attempt = item['attempt']
            matrix.write(attempt / 'exit.json', dict(returncode=rc))
            if rc or not (attempt / 'validated.json').exists():
                warn(f'GPU {item["gpu"]} incomplete: {attempt}; returncode={rc}')
                return str(attempt.parent)
            done = matrix.read(attempt / 'validated.json')
            payload = matrix.read(attempt / 'job.json')
            done.update(attempt=attempt.name, execution=execution, scheduling=payload['scheduling'],
                sha256={f.name: matrix.sha(f) for f in attempt.iterdir() if f.is_file() and f.name != 'run.log'})
            matrix.pin(attempt.parent / 'COMPLETE.json', done)
            summarize(output)
            return None

        warn('Full-card holds are reacquired only with zero sibling workers; shared-GPU timings are contention affected.', level='INFO')
        estimates = {(job['suite'], job['method'], job['retention']):
                     memory_estimate(job, output, args.evaluation_memory_gib)
                     if args.parallel_evaluation > 1 else 0 for job in jobs}
        failed = run_pool(jobs, gpus, args.parallel_evaluation, args.memory_headroom_gib * GIB,
            reservations, acquire, start, finish,
            lambda job: estimates[(job['suite'], job['method'], job['retention'])], warn,
            pool_file=args.gpu_pool_file, release_gpu=lambda gpu: gpu_resources.pop(gpu).close())
        summarize(output)
        if failed:
            raise SystemExit('Incomplete evaluations (rerun to resume durable answers): ' + ', '.join(failed))


if __name__ == '__main__':
    main()
