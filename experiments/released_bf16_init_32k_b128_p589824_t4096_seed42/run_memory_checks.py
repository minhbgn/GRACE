#!/usr/bin/env python3
"""Run/reuse all four isolated memory checks and rebuild their current reports.

Reuse requires the same resolved recipe, source/model/data/environment hashes,
and intact completion evidence. Missing, failed or stale checks are rerun in
fresh attempt directories. This never prepares or launches production training.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'qwen-vl-finetune/scripts'))
import run_constrained as shared
from constrained_experiment import Experiment, sha, read


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def valid_cached(path, key, method, bits):
    try:
        data = read(path)
        if (data['fingerprint'], data['method'], data['bits'], data['complete']) != (key, method, bits, True):
            return None
        if not data['evidence_sha256']:
            return None
        for name, digest in data['evidence_sha256'].items():
            if sha(path.parent / name) != digest:
                return None
        out = path.parent / 'profiles' / method
        summary = read(out / 'run_summary.json')
        envelope = read(path.parent / 'input_envelope.json')
        if not summary['complete'] or not summary['profile'] or summary['global_step'] != 3:
            return None
        if not (out / 'EXPORT_COMPLETE').is_file() or envelope != data['envelope']:
            return None
        if not list(out.glob('model*.safetensors')):
            return None
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None


def collect(work, key, method, bits, current):
    out = work / 'profiles' / method
    summary, envelope = read(out / 'run_summary.json'), read(work / 'input_envelope.json')
    if not summary['complete'] or not summary['profile'] or summary['global_step'] != 3:
        raise ValueError('Profile did not complete all three optimizer updates')
    if not (out / 'EXPORT_COMPLETE').is_file() or not list(out.glob('model*.safetensors')):
        raise ValueError('Profile export did not finish')
    c = current['config']
    if (envelope['sequence_tokens'] < c['model_max_length'] - 2 or
            envelope['visual_tokens'] != c['max_pixels'] // 1024 or
            envelope['accumulation'] != current['diagnostic_accumulation']):
        raise ValueError('Profile did not exercise the requested envelope')
    if summary['peak_reserved_gib'] > c['peak_memory_gib']:
        raise ValueError('Profile exceeded the configured memory guard')
    steps = [json.loads(line) for line in (out / 'step_metrics.jsonl').read_text().splitlines()]
    if method == 'grace' and (steps[-1]['rcka_scale'] != 1 or steps[-1]['dkd_scale'] != 1):
        raise ValueError('Full distillation losses were not exercised')
    invocations = [json.loads(line) for line in (out / 'invocations.jsonl').read_text().splitlines()]
    if len(invocations) != 1 or invocations[0]['exit_code'] != 0:
        raise ValueError('Profile invocation did not exit successfully')
    peaks = {}
    with (work / 'device_memory.csv').open() as handle:
        for row in csv.DictReader(handle, skipinitialspace=True):
            gpu = row['index'].strip()
            peaks[gpu] = max(peaks.get(gpu, 0), float(row['memory.used [MiB]'].split()[0]) / 1024)
    names = ['input_envelope.json', 'comparison_recipe.json', 'device_memory.csv',
             'student_provenance.json', 'teacher_provenance.json']
    names += [f'profiles/{method}/{name}' for name in ['run_summary.json', 'EXPORT_COMPLETE',
              'step_metrics.jsonl', 'invocations.jsonl', 'launch_recipe.json', 'config.json']]
    return dict(fingerprint=key, method=method, bits=bits, complete=True, updates=3,
                artifact=str(work.relative_to(HERE)), envelope=envelope,
                peak_allocated_gib=max(row['peak_allocated_gib'] for row in steps),
                peak_reserved_gib=summary['peak_reserved_gib'], device_peak_used_gib=peaks,
                launch_wall_seconds=invocations[0]['wall_seconds'],
                evidence_sha256={name: sha(work / name) for name in names})


def reports(key, current, rows, status, failure=None):
    report = dict(schema=2, fingerprint=key, status=status, current_recipe=current,
                  production_training_started=False, rows=rows, failure=failure,
                  profile_launch_gpu_hours=sum(row['launch_wall_seconds'] for row in rows) * 2 / 3600)
    write(HERE / 'profile_memory.json', dict(fingerprint=key, status=status, **current))
    write(HERE / 'memory_results.json', report)
    c = current['config']
    lines = ['# Memory feasibility for the current recipe', '',
             f'**Status: {status}.** Fingerprint: `{key}`.', '',
             f"Requested envelope: **{c['max_pixels']//1024} visual / {c['model_max_length']} total tokens**. "
             f"Reserved-memory guard: **{c['peak_memory_gib']} GiB/GPU**.", '',
             '| Arm | Peak allocated GiB | Peak reserved GiB | Device sampled peak GiB | Evidence |',
             '| --- | ---: | ---: | ---: | --- |']
    for row in rows:
        lines.append(f"| {row['method'].upper()} W{row['bits']} | {row['peak_allocated_gib']:.2f} | "
                     f"{row['peak_reserved_gib']:.2f} | {max(row['device_peak_used_gib'].values()):.2f} | "
                     f"[run]({row['artifact']}/profiles/{row['method']}/run_summary.json) |")
    if failure:
        lines += ['', f'Failure: {failure}']
    lines += ['',
        f"These are three-update synthetic long-answer checks, with accumulation **{current['diagnostic_accumulation']}** "
        f"versus production **{c['grad_accum']}**, RCKA warmup **1** versus **{c['rcka_warmup_steps']}**, "
        'and forced maximum image size. They test optimizer updates, full GDKD/RCKA and checkpoint/export memory. '
        'They are not production training or quality measurements and do not prove that every production batch fits.', '',
        'Device memory is sampled every 0.5 seconds and may miss brief spikes. Training peaks are rank-max PyTorch counters. '
        'The guard checks at optimizer-update boundaries and cannot prevent every transient OOM.', '',
        'Rerun `run_memory_checks.sh` to regenerate missing reports and reuse only matching, validated completed checks. '
        'Recipe/code/model/data/environment changes select a new fingerprint; failed attempts remain separate. '
        'Older report snapshots are retained under `memory_checks/report_history/`; the original 2,048-token screen '
        'is also preserved under `memory_checks/legacy_2048_reports/`.', '',
        'Full dataset preparation, training and evaluation are separate commands in [README.md](README.md).']
    temp = HERE / 'memory_report.md.tmp'
    temp.write_text('\n'.join(lines) + '\n')
    temp.replace(HERE / 'memory_report.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accumulation', type=int, default=2, help='Diagnostic accumulation only; production is unchanged')
    args = parser.parse_args()
    if args.accumulation < 1:
        parser.error('--accumulation must be positive')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    for name, value in dict(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
                            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True').items():
        os.environ.setdefault(name, value)
    with shared.lock(HERE / 'memory_checks/.matrix.lock'):
        print('CPU preflight: hashing current recipe, models, data and code...', flush=True)
        spec = Experiment(HERE / 'experiment.json')
        c = shared.resolved_config()
        c.update(model_path=str(spec.paths[spec.student]), teacher_path=str(spec.paths[spec.teacher]),
                 student_weights_sha256=spec.models[spec.student]['weights_sha256'])
        import torch, transformers
        current = dict(config=c, diagnostic_accumulation=args.accumulation,
            experiment_sha256=sha(spec.config_path), implementation=shared.implementation_hashes(),
            profiler_sha256=sha(HERE / 'profile_memory.py'), runner_sha256=sha(Path(__file__)),
            previous_subset_sha256=sha(ROOT / spec.spec['previous_experiment'] / 'subset.json'),
            models={name: spec.provenance(name) for name in (spec.student, spec.teacher)},
            environment=dict(torch=torch.__version__, transformers=transformers.__version__,
                deepspeed=__import__('importlib.metadata', fromlist=['version']).version('deepspeed'),
                allocator=os.environ['PYTORCH_CUDA_ALLOC_CONF']))
        key = fingerprint(current)
        existing = HERE / 'memory_results.json'
        if existing.exists() and read(existing).get('fingerprint') != key:
            history = HERE / 'memory_checks/report_history' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            history.mkdir(parents=True)
            for name in ['memory_results.json', 'memory_report.md', 'profile_memory.json']:
                if (HERE / name).exists():
                    shutil.copy2(HERE / name, history / name)
        rows = []
        reports(key, current, rows, 'checking')
        try:
            # GRACE first: it includes the teacher and both distillation losses.
            for method in ('grace', 'qat'):
                for bits in spec.bits:
                    group = HERE / 'memory_checks' / key / f'{method}_w{bits}'
                    reused = next((value for path in sorted(group.glob('*/check_result.json'), reverse=True)
                                   if (value := valid_cached(path, key, method, bits))), None)
                    if reused:
                        rows.append(reused)
                        print(f'SKIP verified {method} W{bits}: {reused["artifact"]}', flush=True)
                    else:
                        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                        work = group / f'attempt_{stamp}'
                        group.mkdir(parents=True, exist_ok=True)
                        logpath = group / f'attempt_{stamp}.log'
                        cmd = [sys.executable, str(HERE / 'profile_memory.py'), '--method', method,
                               '--bits', str(bits), '--accumulation', str(args.accumulation), '--output', str(work)]
                        print(f'RUN {method} W{bits}: {logpath}', flush=True)
                        with logpath.open('w') as log:
                            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
                        if code:
                            raise RuntimeError(f'{method} W{bits} exited {code}; see {logpath}')
                        value = collect(work, key, method, bits, current)
                        write(work / 'check_result.json', value)
                        rows.append(value)
                        print(f'PASS {method} W{bits}: {value["peak_reserved_gib"]:.2f} GiB reserved', flush=True)
                    reports(key, current, rows, 'checking')
        except Exception as error:
            reports(key, current, rows, 'failed', str(error))
            raise
        reports(key, current, rows, 'passed')
        print(f'All {len(rows)} checks passed. Reports: {HERE / "memory_results.json"}', flush=True)


if __name__ == '__main__':
    main()
