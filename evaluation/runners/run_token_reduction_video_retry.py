#!/usr/bin/env python3
"""GPU-1-only QAT video recovery, preserving verified original completions."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import datetime as dt
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.runners import run_lmms_matrix as lmms
from evaluation.runners import run_token_reduction_matrix as matrix
from evaluation.protocols.video_memory_runtime import (
    MemoryMeasuredGenerator, adaptive_video_inputs, install_streamed_vision,
)

from evaluation.protocols.artifact_layout import verify_source

ROOT = matrix.ROOT
EXP = ROOT / 'experiments/qat_int4_token_reduction_576_seed42'
SOURCE = EXP / 'outputs/original/full'
DEFAULT_OUTPUT = EXP / 'outputs/reruns/video_recovery'
SUITES = list(matrix.VIDEO_TASKS)
CONFIGS = matrix.configurations(['resolution', 'uniform', 'merge'], [1., .5, .25, .1])
EXECUTION = dict(vision_temporal_group_chunk=1, prefill_chunk_tokens=512,
                 adaptive_baseline_decoder_tokens=131072,
                 adaptation='shared per-clip spatial baseline; preserve full duration and 2 fps',
                 video_pixels='CPU, staged per vision chunk', decoder_features='CPU, staged per prefill chunk',
                 cache='full GPU KV; no eviction', timing_protocol='streamed_video_v1')
YAML_RELATIVE = 'lmms_eval/tasks/tempcompass/tempcompass_mc.yaml'


def install_task_mapping():
    """Use the MC YAML already shipped with the clean pinned upstream checkout."""
    lmms.VIDEO_TASKS = dict(lmms.VIDEO_TASKS,
        tempcompass=('tempcompass_multi_choice', 'tempcompass/tempcompass_mc.yaml', 'avg_accuracy,none', 1))
    matrix.VIDEO_TASKS = lmms.VIDEO_TASKS


def install_worker(execution):
    install_task_mapping()
    original_load = matrix.load_model

    def load(*args, **kwargs):
        model, audit = original_load(*args, **kwargs)
        install_streamed_vision(model, execution['vision_temporal_group_chunk'])
        audit['video_execution'] = execution
        return model, audit

    def inputs(*args, **kwargs):
        return adaptive_video_inputs(*args, **kwargs,
                                     decoder_token_cap=execution['adaptive_baseline_decoder_tokens'])

    matrix.load_model = load
    matrix.video_inputs = inputs
    matrix.MeasuredGenerator = MemoryMeasuredGenerator


def finish_adaptation_audit(attempt):
    """Make per-question geometry changes visible in the completion summary."""
    rows = [json.loads(line) for line in (attempt / 'telemetry.jsonl').read_text().splitlines()]
    adaptation = dict(questions=len(rows), adapted_questions=sum(r['video_adaptation']['applied'] for r in rows),
        max_original_policy_visual_tokens=max(r['video_adaptation']['original_policy_visual_tokens'] for r in rows),
        max_adaptive_baseline_visual_tokens=max(r['video_adaptation']['adaptive_baseline_visual_tokens'] for r in rows),
        max_actual_decoder_input_tokens=max(r['input_visual_tokens'] + r['input_text_tokens'] for r in rows),
        original_policy_retention=sum(r['input_visual_tokens'] for r in rows) /
            sum(r['video_adaptation']['original_policy_visual_tokens'] for r in rows),
        frame_policy='unchanged 2 fps, full duration')
    validated = matrix.read(attempt / 'validated.json')
    validated['video_adaptation'] = adaptation
    matrix.write(attempt / 'validated.json', validated)


def resolved(output, suite, config):
    directory = output / 'qat_w4' / suite / config
    if (done := matrix.verified_completion(directory)) is not None:
        return done, 'retry', directory
    directory = SOURCE / 'qat_w4' / suite / config
    if (done := matrix.verified_completion(directory)) is not None:
        return done, 'original', directory
    return None, None, directory


def inventory(output, suites):
    jobs, reuse, completed = [], [], []
    for suite in suites:
        for method, retention in CONFIGS:
            config = matrix.configuration_id(method, retention)
            done, origin, directory = resolved(output, suite, config)
            if origin == 'original':
                reuse.append(dict(suite=suite, config=config, marker=str(directory / 'COMPLETE.json'),
                    marker_sha256=matrix.sha(directory / 'COMPLETE.json'),
                    attempt=done['attempt'], count=done['count'], score_percent=done['score_percent']))
            elif origin == 'retry':
                completed.append(dict(suite=suite, config=config))
            else:
                # This is recovery of failed jobs, not permission to expand the matrix.
                attempts = sorted(directory.glob('attempt_*/exit.json'))
                if not attempts or not any(matrix.read(path)['returncode'] != 0 for path in attempts):
                    raise ValueError(f'No failed original job to retry: {directory}')
                jobs.append(dict(model='qat_w4', suite=suite, method=method, retention=retention))
    return jobs, reuse, completed


def summarize(output, suites):
    entries = []
    lines = ['# QAT W4 video recovery', '',
        'Original verified passes are reused by reference, with their original runtime, spatial policy and timings. '
        'Repaired jobs use a declared adaptive spatial baseline for oversized clips (2 fps/full duration preserved). '
        'Do not compare their accuracy deltas to original-policy rows as if inputs were identical. '
        'Speedups are reported only between jobs completed under the repaired execution policy. '
        'No cross-runtime speedup is inferred.', '',
        '| Suite | Configuration | Score % | Questions | Adapted clips | Origin | Speedup | Status |',
        '| --- | --- | ---: | ---: | ---: | --- | ---: | --- |']
    for suite in suites:
        baseline, base_origin, _ = resolved(output, suite, 'baseline_r100')
        for method, retention in CONFIGS:
            config = matrix.configuration_id(method, retention)
            done, origin, directory = resolved(output, suite, config)
            attempts = list((output / 'qat_w4' / suite / config).glob('attempt_*'))
            status = 'complete' if done else ('failed/incomplete' if attempts else 'pending retry')
            speedup = None
            if done and baseline and origin == base_origin == 'retry':
                if baseline['sample_identity'] != done['sample_identity']:
                    raise ValueError('Cannot compare different question populations')
                b = baseline['telemetry']['fixed_output_profile']['mean_seconds']
                v = done['telemetry']['fixed_output_profile']['mean_seconds']
                speedup = b / v if b and v else None
            entries.append(dict(model='qat_w4', suite=suite, method=method, retention=retention,
                status=status, origin=origin, source_directory=str(directory) if done else None,
                fixed_output_speedup=speedup, result=done))
            values = [suite, config, f"{done['score_percent']:.3f}" if done else '—',
                      str(done['count']) if done else '—',
                      str(done.get('video_adaptation', {}).get('adapted_questions', '—')) if done else '—', origin or '—',
                      f'{speedup:.3f}x' if speedup else '—', status]
            lines.append('| ' + ' | '.join(values) + ' |')
    matrix.write(output / 'comparison.json', dict(rows=entries,
        note='Explicit cross-runtime reuse; only same-retry-runtime timing comparisons.'))
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(output / 'comparison.md', flush=True)


def prepare(output, execution):
    """CPU-only, validate and pin everything before a future GPU launch."""
    exp = matrix.Experiment(EXP / 'experiment.json')
    original = matrix.read(SOURCE / 'protocol.json')
    if exp.spec != original['experiment']:
        raise ValueError('Original experiment changed')
    for name, checksum in original['code'].items():
        verify_source(name, checksum)
    env = lmms.environment()
    if env != original['environment'] or importlib.metadata.version('triton') != original['triton_version']:
        raise ValueError('Use the original pinned evaluation Python/packages')
    lmms.source_check()
    install_task_mapping()
    with lmms.video_no_network():
        cfg = lmms.video_task_config('tempcompass', original['policy'])
    if cfg['task'] != 'tempcompass_multi_choice' or cfg['process_results'] is not lmms.video_local_choice:
        raise ValueError('TempCompass local scoring adapter changed')
    datasets = {}
    for suite in SUITES:
        _, info = matrix.load_video(suite, None)
        if info != matrix.read(SOURCE / 'datasets' / f'{suite}.json'):
            raise ValueError(f'Original population changed: {suite}')
        datasets[suite] = info
    exp.validate_export('qat_w4')
    provenance = exp.provenance('qat_w4')
    sidecar = matrix.sha(exp.paths['qat_w4'] / 'qat_quantized_weights.bin')
    provenance['qat_sidecar_sha256'] = sidecar
    if provenance != matrix.read(SOURCE / 'qat_w4/model_provenance.json'):
        raise ValueError('QAT checkpoint provenance changed')
    own_files = [Path(__file__), ROOT / 'evaluation/protocols/video_memory_runtime.py',
                 ROOT / 'evaluation/tests/test_video_memory_runtime.py',
                 ROOT / 'evaluation/protocols/artifact_layout.py',
                 ROOT / 'evaluation/provenance/output_layout_20261125/paths.json',
                 ROOT / 'evaluation/provenance/output_layout_20261125/sources.json']
    protocol = dict(schema=1, source_protocol=str(SOURCE / 'protocol.json'),
        source_protocol_sha256=matrix.sha(SOURCE / 'protocol.json'),
        policy=original['policy'], execution=execution, model='qat_w4', gpu='1',
        environment=env, original_source_code=original['code'],
        source_code={name: matrix.sha(ROOT / name) for name in original['code']},
        repair_code={str(path.relative_to(ROOT)): matrix.sha(path) for path in own_files},
        task_files={name: matrix.sha(lmms.SOURCE / name) for name in (
            YAML_RELATIVE, 'lmms_eval/tasks/tempcompass/_default_template_yaml',
            'lmms_eval/tasks/tempcompass/utils.py')})
    matrix.pin(output / 'protocol.json', protocol)
    for suite, info in datasets.items():
        matrix.pin(output / 'datasets' / f'{suite}.json', info)
    matrix.pin(output / 'qat_w4/model_provenance.json', provenance)
    # Reuse only original passes; additions to this recovery output do not alter the manifest.
    _, reused, _ = inventory(output, SUITES)
    matrix.pin(output / 'reuse_manifest.json', dict(source_protocol_sha256=protocol['source_protocol_sha256'],
                                                  jobs=reused))
    return exp, original['policy'], sidecar


def run_gates(output, reservations, execution):
    """On the user's future launch, gate GPU execution before any benchmark retry."""
    from evaluation.workers.video_runtime import ensure_gpu_idle

    reservations.release('1')
    ensure_gpu_idle('1')
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='1',
               VIDEO_MEMORY_TEST_DEVICE='cuda', TOKENIZERS_PARALLELISM='false',
               TRITON_CACHE_DIR=str(EXP / 'outputs/cache/triton'))
    checks = [([sys.executable, str(ROOT / 'evaluation/tests/check_int4_kernel.py')],
               'int4_kernel_validation_gpu1.log'),
              ([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'evaluation/tests'),
                '-p', 'test_video_memory_runtime.py', '-v'], 'memory_equivalence_gpu1.log')]
    for command, log_name in checks:
        print(f'GPU 1 validation gate: {output / log_name}', flush=True)
        with (output / log_name).open('w') as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    reservations.acquire('1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'prepare', 'check', 'run', 'summarize'], nargs='?', default='plan')
    parser.add_argument('--gpu', choices=['1'], default='1', help='Physical GPU 1 only')
    parser.add_argument('--suites', help='Optional subset of the five video suites')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Results base; /full is appended')
    parser.add_argument('--prefill-chunk-tokens', type=int, default=512)
    parser.add_argument('--decoder-token-cap', type=int, default=131072,
                        help='Adaptive per-clip baseline input + output reserve; original frames/fps preserved')
    args = parser.parse_args()
    if args.prefill_chunk_tokens < 1:
        parser.error('--prefill-chunk-tokens must be positive')
    native_context = matrix.read(SOURCE / 'qat_w4/model_provenance.json')['config']['text_config']['max_position_embeddings']
    if not 257 <= args.decoder_token_cap <= native_context:
        parser.error(f'--decoder-token-cap must be between 257 and native context {native_context}')
    output = args.output.resolve() / 'full'
    if output == SOURCE or SOURCE in output.parents or output in SOURCE.parents:
        parser.error('Recovery output must be separate from the original results')
    execution = dict(EXECUTION, prefill_chunk_tokens=args.prefill_chunk_tokens,
                     adaptive_baseline_decoder_tokens=args.decoder_token_cap)
    suites = matrix.selected(args.suites, SUITES)
    jobs, reused, completed = inventory(output, suites)
    if args.action == 'plan':
        print(json.dumps(dict(gpu='1', model='qat_w4', suites=suites, execution=execution,
            source=str(SOURCE), output=str(output), run_count=len(jobs),
            reuse_count=len(reused), retry_completed_count=len(completed),
            remaining_jobs=jobs, reused_jobs=reused), indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.matrix.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'summarize':
            summarize(output, suites)
            return
        exp, policy, sidecar = prepare(output, execution)
        if args.action in ('prepare', 'check'):
            summarize(output, SUITES)
            print(f'CPU preflight passed; {len(jobs)} retries, {len(reused)} verified original passes. No GPU allocated.')
            return
        if not jobs:
            summarize(output, SUITES)
            print('No remaining failed jobs; no GPU allocated.')
            return
        from evaluation.protocols.gpu_reservation import GPUReservations, run_reserved_worker
        from evaluation.workers.video_runtime import gpu_lock, ensure_gpu_idle
        failures = []
        with ExitStack() as stack:
            stack.enter_context(gpu_lock('1'))
            ensure_gpu_idle('1')
            reservations = GPUReservations()
            stack.callback(reservations.close)
            reservations.acquire('1')
            run_gates(output, reservations, execution)
            for job in jobs:
                config = matrix.configuration_id(job['method'], job['retention'])
                if resolved(output, job['suite'], config)[0] is not None:
                    continue
                directory = output / 'qat_w4' / job['suite'] / config
                attempt = directory / dt.datetime.now(dt.timezone.utc).strftime('attempt_%Y%m%dT%H%M%S_%fZ')
                attempt.mkdir(parents=True)
                job.update(model_path=str(exp.paths['qat_w4']), policy=policy, execution=execution,
                    limit=None, packed_cache=str(EXP / 'outputs/cache/packed_qat_w4'),
                    sidecar_sha256=sidecar, reserve_gpus=True, controller_pid=os.getpid(), gpu='1')
                matrix.write(attempt / 'job.json', job)
                env = {k: v for k, v in os.environ.items() if k not in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')}
                env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='1', HF_HUB_OFFLINE='1',
                    TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                    TRITON_CACHE_DIR=str(EXP / 'outputs/cache/triton'))
                print(f'GPU 1 RUN qat_w4/{job["suite"]}/{config}: {attempt / "run.log"}', flush=True)
                with (attempt / 'run.log').open('w') as stream:
                    rc = run_reserved_worker([sys.executable, str(Path(__file__).resolve()), '--worker',
                        str(attempt / 'job.json')], env, stream, attempt, '1', reservations)
                matrix.write(attempt / 'exit.json', dict(returncode=rc))
                if rc or not (attempt / 'validated.json').exists():
                    failures.append(str(directory))
                else:
                    done = matrix.read(attempt / 'validated.json')
                    done.update(attempt=attempt.name, execution=execution,
                        sha256={f.name: matrix.sha(f) for f in attempt.iterdir() if f.is_file() and f.name != 'run.log'})
                    matrix.pin(directory / 'COMPLETE.json', done)
                summarize(output, SUITES)
                reservations.acquire('1')
        if failures:
            raise SystemExit('Incomplete jobs (rerun this launcher to retry): ' + ', '.join(failures))


if __name__ == '__main__':
    if '--worker' in sys.argv:
        path = Path(sys.argv[sys.argv.index('--worker') + 1])
        job = matrix.read(path)
        if job['gpu'] != '1' or job['model'] != 'qat_w4' or job['suite'] not in SUITES:
            raise ValueError('Recovery workers are restricted to QAT video on physical GPU 1')
        install_worker(job['execution'])
        matrix.worker(path)
        finish_adaptation_audit(path.parent)
    else:
        main()
