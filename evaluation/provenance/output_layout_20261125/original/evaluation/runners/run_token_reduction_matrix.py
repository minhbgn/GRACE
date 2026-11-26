#!/usr/bin/env python3
"""Image/video retention matrix with physically compacted tokens and strict fused INT4."""
from __future__ import annotations
import argparse
import copy
from contextlib import ExitStack, contextmanager
import datetime as dt
import fcntl
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.paths import GRACE_ROOT as ROOT
sys.path.insert(0, str(ROOT / 'qwen-vl-finetune'))
from evaluation.runners.run_lmms_matrix import (Experiment, read, write, digest, pin, sha,
    selected, verified_completion, VIDEO_TASKS, video_task, video_documents,
    source_check, video_no_network, validate_result)
from evaluation.runners.run_local_development import command as image_command, source_samples, metadata
from evaluation.runners.run_constrained_matrix_evals import dataset_fingerprint
from evaluation.audits.summarize_constrained_evals import SUITES as IMAGE_SUITES, EXPECTED, ID_FIELDS
from evaluation.protocols.video_protocol import DEFAULT_DATA, load_suite, local_video
from evaluation.protocols.video_population import select_population
from evaluation.protocols.local_scoring import score_rows, corrected_summary, METRICS
from evaluation.protocols.token_reduction_runtime import (BudgetProcessor, MeasuredGenerator,
    load_model, video_inputs)
from evaluation.protocols.gpu_job_queue import gpu_indices, run_gpu_queue
DEFAULT = ROOT / 'experiments/qat_int4_token_reduction_576_seed42/experiment.json'
SUITES = {**{s:'image' for s in IMAGE_SUITES}, **{s:'video' for s in VIDEO_TASKS}}


def configurations(methods, ratios):
    result = [('baseline', 1.)] if 1. in ratios else []
    result += [(method, ratio) for method in methods for ratio in ratios if ratio != 1.]
    return result


def configuration_id(method, ratio):
    return f'{method}_r{round(ratio * 100):03d}'


def load_video(suite, limit):
    rows, info = load_suite(suite, DEFAULT_DATA)
    rows, info = select_population(suite, rows, info, True)
    if info['status'] != 'ready':
        raise ValueError(f'{suite}: {info["status"]}')
    return rows[:limit] if limit else rows, info


def image_validation(attempt, suite, limit):
    payload = read(attempt / 'results.json')
    rows = payload['results']
    source = source_samples(suite)
    if len(source) != EXPECTED[suite]:
        raise ValueError('Image source population changed')
    source = source[:limit] if limit else source
    actual = [(tuple(r[k] for k in ID_FIELDS[suite]), r['gold']) for r in rows]
    if sorted(source, key=str) != sorted(actual, key=str) or any(r.get('error') for r in rows):
        raise ValueError('Incomplete/relabelled/failed image population')
    scored = score_rows(suite, copy.deepcopy(rows), metadata(suite))
    summary = corrected_summary(suite, scored, metadata(suite), partial=bool(limit))
    if any(a['correct'] != b['correct'] for a,b in zip(payload['results'],scored)):
        raise ValueError('Image scoring mismatch')
    return dict(count=len(rows), score_percent=100 * summary['primary_score'] if summary['primary_score'] is not None else None,
                metric=METRICS[suite], sample_identity=digest(sorted(actual, key=str)), scoring=summary)


def worker(job_path):
    job_path = Path(job_path)
    job, attempt = read(job_path), job_path.parent
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    import torch
    torch.manual_seed(job['policy']['seed'])
    torch.set_grad_enabled(False)
    before_device = None
    if job.get('reserve_gpus'):
        from evaluation.protocols.gpu_reservation import await_gpu_handoff
        before_device = lambda: await_gpu_handoff(attempt, job['controller_pid'])
    model, audit = load_model(job['model_path'], job['model'] == 'qat_w4', job['policy']['attention'],
        job.get('packed_cache'), job.get('sidecar_sha256'), before_device=before_device)
    generator = MeasuredGenerator(model, job, attempt, audit)
    model.generate = generator
    processor = AutoProcessor.from_pretrained(job['model_path'], local_files_only=True)
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    processor = BudgetProcessor(processor, job['method'], job['retention'], job['policy']['image_max_pixels'])
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.,
        eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id,
        use_cache=True)
    try:
        if SUITES[job['suite']] == 'image':
            cmd = image_command(job['model'], job['suite'], {job['model']:Path(job['model_path'])},
                attempt, 'local_native_greedy256_v1', job['policy']['attention'], job['limit'])
            # Reuse exact dataset iteration, prompt builders and deterministic local scoring.
            sys.argv = cmd[1:]
            with patch.object(Qwen3VLForConditionalGeneration, 'from_pretrained', return_value=model), \
                 patch.object(AutoProcessor, 'from_pretrained', return_value=processor), video_no_network():
                runpy.run_path(cmd[1], run_name='__main__')
            validated = image_validation(attempt, job['suite'], job['limit'])
        else:
            source_check()
            from lmms_eval import evaluator
            from lmms_eval.api.model import lmms
            from lmms_eval.tasks import TaskManager
            from lmms_eval.utils import handle_non_serializable
            rows, info = load_video(job['suite'], None)
            docs = video_documents(job['suite'], rows, DEFAULT_DATA)
            # Keep full task population; lmms applies the smoke limit after construction.
            with video_no_network():
                task = video_task(job['suite'], job['policy'], docs)
            class Adapter(lmms):
                def loglikelihood(self, requests):
                    raise NotImplementedError
                def generate_until_multi_round(self, requests):
                    raise NotImplementedError
                def generate_until(self, requests):
                    answers = []
                    for req in requests:
                        prompt, _, _, doc_id, task_name, split = req.args
                        doc = self.task_dict[task_name][split][doc_id]
                        row = dict(json.loads(doc['_grace_row']), prompt=prompt)
                        with local_video(row['asset']) as path:
                            inputs, meta = video_inputs(processor, row, path, job['policy'], job['method'], job['retention'])
                        baseline = inputs.pop('_baseline_visual_tokens')
                        inputs = inputs.to('cuda:0')
                        inputs['_baseline_visual_tokens'] = baseline
                        generated = model.generate(**inputs, max_new_tokens=job['policy']['max_new_tokens'])
                        answers.append(processor.tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True))
                        with (attempt / 'video_inputs.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(id=row['id'], doc_id=doc_id, sampling=meta)) + '\n')
                    return answers
            with video_no_network():
                result = evaluator.simple_evaluate(model=Adapter(), tasks=[task],
                    task_manager=TaskManager(model_name='qwen3_vl'), batch_size=1, num_fewshot=0,
                    limit=job['limit'], bootstrap_iters=0, log_samples=True,
                    random_seed=42, numpy_random_seed=42, torch_random_seed=42, fewshot_random_seed=42,
                    cli_args=SimpleNamespace(output_path=str(attempt), process_with_media=False))
            samples = result.pop('samples')[VIDEO_TASKS[job['suite']][0]]
            (attempt / 'results.json').write_text(json.dumps(result, default=handle_non_serializable, indent=2)+'\n')
            (attempt / 'samples.jsonl').write_text(''.join(json.dumps(s, default=handle_non_serializable)+'\n' for s in samples))
            validated = validate_result(result, samples, job['suite'], job['limit'], VIDEO_TASKS)
            write(attempt / 'dataset.json', info)
        telemetry = [json.loads(line) for line in (attempt / 'telemetry.jsonl').read_text().splitlines()]
        if len(telemetry) != validated['count'] or [t['index'] for t in telemetry] != list(range(len(telemetry))):
            raise ValueError('Telemetry coverage mismatch')
        eos = processor.tokenizer.eos_token_id
        eos = eos if isinstance(eos,list) else [eos]
        validated['output_cutoffs'] = sum(t['generated_token_ids'][-1] not in eos for t in telemetry)
        if job['model'] == 'qat_w4' and any(t['fused_int4_calls'] <= 0 for t in telemetry):
            raise ValueError('Missing strict INT4 kernel evidence')
        validated['telemetry'] = aggregate_telemetry(telemetry)
        write(attempt / 'validated.json', validated)
    finally:
        write(attempt / 'runtime.json', audit)


def aggregate_telemetry(rows):
    keys = ('original_input_visual_tokens','input_visual_tokens','input_text_tokens','output_text_tokens',
            'vision_and_embedding_seconds','reduction_seconds','prefill_seconds','decode_seconds',
            'model_seconds','peak_gpu_allocated_bytes')
    result = {key:dict(mean=sum(r[key] for r in rows)/len(rows), total=sum(r[key] for r in rows),
                      max=max(r[key] for r in rows)) for key in keys}
    profiles = [sum(r['fixed_output_profile_seconds']) / len(r['fixed_output_profile_seconds'])
                for r in rows if r['fixed_output_profile_seconds']]
    result['fixed_output_profile'] = dict(samples=len(profiles), mean_seconds=sum(profiles)/len(profiles) if profiles else None)
    original=sum(r['original_input_visual_tokens'] for r in rows)
    result['actual_retention']=sum(r['input_visual_tokens'] for r in rows)/original if original else None
    return result


def summary(output, models, suites, configs):
    entries=[]
    for model in models:
        for suite in suites:
            baseline = verified_completion(output / model / suite / 'baseline_r100')
            for method, ratio in configs:
                done = verified_completion(output / model / suite / configuration_id(method,ratio))
                speedup=None
                if done and baseline:
                    if done['sample_identity'] != baseline['sample_identity']:
                        raise ValueError('Cannot compare different question populations')
                    base=baseline['telemetry']['fixed_output_profile']['mean_seconds']
                    val=done['telemetry']['fixed_output_profile']['mean_seconds']
                    if base and val:
                        speedup=base/val
                entries.append(dict(model=model,suite=suite,method=method,retention=ratio,
                    status='complete' if done else ('failed' if list((output/model/suite/configuration_id(method,ratio)).glob('attempt_*/exit.json')) else 'pending'), fixed_output_speedup=speedup,result=done))
    write(output/'comparison.json',dict(rows=entries, note='Within-model paired population speedup; fixed-output profiles. No means across unlike benchmarks.'))
    lines=['# Token reduction matrix','','| Model | Suite | Method | Target retained | Score % | Visual tokens | Text in / out | Model seconds | Fixed-output speedup | Cutoffs | Status |',
           '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for e in entries:
        d=e['result']; t=d['telemetry'] if d else None
        score=d['score_percent'] if d else None
        values=[e['model'],e['suite'],e['method'],str(e['retention']),
            f'{score:.3f}' if score is not None else '—',
            f'{t["input_visual_tokens"]["mean"]:.1f}' if t else '—',
            f'{t["input_text_tokens"]["mean"]:.1f} / {t["output_text_tokens"]["mean"]:.1f}' if t else '—',
            f'{t["model_seconds"]["mean"]:.4f}' if t else '—',
            f'{e["fixed_output_speedup"]:.3f}x' if e['fixed_output_speedup'] else '—',
            str(d['output_cutoffs']) if d else '—',e['status']]
        lines.append('| '+' | '.join(values)+' |')
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(output/'comparison.md')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','check','kernel-check','run','summarize'],nargs='?',default='plan')
    p.add_argument('--experiment',type=Path,default=DEFAULT)
    p.add_argument('--output',type=Path)
    p.add_argument('--models')
    p.add_argument('--suites')
    p.add_argument('--methods',default='resolution,uniform,merge')
    p.add_argument('--retentions',default='1,.5,.25,.1')
    p.add_argument('--smoke',type=int)
    gpu_options=p.add_mutually_exclusive_group()
    gpu_options.add_argument('--gpu', help='One physical GPU index (legacy single-GPU mode)')
    gpu_options.add_argument('--gpus', help='Physical GPU indices, e.g. 0,1; one concurrent job per GPU')
    p.add_argument('--reserve-gpus', action=argparse.BooleanOptionalAction, default=True, help='Hold VRAM during CPU preflight and loading; release just before CUDA allocation')
    p.add_argument('--attn',choices=['flash_attention_2','sdpa'],default='flash_attention_2')
    p.add_argument('--profile-samples',type=int)
    p.add_argument('--profile-repeats',type=int)
    p.add_argument('--profile-tokens',type=int)
    args=p.parse_args()
    try:
        gpu_value=args.gpus if args.gpus is not None else args.gpu
        gpus=gpu_indices(gpu_value) if gpu_value is not None else []
    except ValueError as error:
        p.error(str(error))
    exp=Experiment(args.experiment)
    models=selected(args.models,exp.models); suites=selected(args.suites,SUITES)
    methods=selected(args.methods,['resolution','uniform','merge'])
    ratios=[float(x) for x in args.retentions.split(',')]
    if len(set(ratios))!=len(ratios) or any(r not in (1.,.5,.25,.1) for r in ratios):
        p.error('Choose unique retentions from 1,.5,.25,.1')
    if args.smoke is not None and args.smoke<=0:
        p.error('--smoke must be positive')
    policy=dict(exp.spec['design'], attention=args.attn)
    for key in ('profile_samples','profile_repeats','profile_tokens'):
        if getattr(args,key) is not None:
            policy[key]=getattr(args,key)
    if policy['profile_samples']<0 or min(policy['profile_repeats'],policy['profile_tokens'])<1:
        p.error('Invalid profiling settings')
    configs=configurations(methods,ratios)
    output=(args.output or exp.directory/'results')/('smoke_'+str(args.smoke) if args.smoke else 'full')
    jobs=[dict(model=m,suite=s,method=method,retention=r,model_path=str(exp.paths[m]),
               policy=policy,limit=args.smoke) for m in models for s in suites for method,r in configs]
    if args.action=='plan':
        print(json.dumps(dict(jobs=len(jobs),models={m:str(exp.paths[m]) for m in models},suites=suites,
            configurations=configs,policy=policy,output=str(output),gpus=gpus),indent=2)); return
    if args.action=='kernel-check':
        if not gpus:
            p.error('kernel-check requires --gpu N or --gpus N,M')
        from evaluation.workers.video_runtime import gpu_lock,ensure_gpu_idle
        with ExitStack() as stack:
            for gpu in sorted(gpus, key=int):
                stack.enter_context(gpu_lock(gpu))
                ensure_gpu_idle(gpu)
            for gpu in gpus:
                env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=gpu,
                         TRITON_CACHE_DIR=str(exp.directory/'runtime_cache/triton'))
                print(f'INT4 numerical gate: GPU {gpu}',flush=True)
                subprocess.run([sys.executable,str(ROOT/'evaluation/tests/check_int4_kernel.py')],env=env,check=True)
        return
    if args.action=='summarize':
        if (output/'protocol.json').exists():
            frozen=read(output/'protocol.json')
            if frozen['policy']!=policy or frozen['experiment']!=exp.spec or frozen['limit']!=args.smoke:
                raise ValueError('Summary settings differ from saved protocol')
        summary(output,models,suites,configs); return
    if args.action=='run' and not gpus:
        p.error('run requires --gpu N or --gpus N,M; inspect ownership first')
    with ExitStack() as resources:
        reservations=None
        if args.action=='run':
            from evaluation.workers.video_runtime import gpu_lock, ensure_gpu_idle
            for gpu in sorted(gpus,key=int):
                resources.enter_context(gpu_lock(gpu))
                ensure_gpu_idle(gpu)
            if args.reserve_gpus:
                from evaluation.protocols.gpu_reservation import GPUReservations
                reservations=GPUReservations()
                resources.callback(reservations.close)
                for gpu in gpus:
                    reservations.acquire(gpu)
        # No weights/GPU are loaded by check. Validate real local source populations.
        datasets={}
        for suite in suites:
            if SUITES[suite]=='image':
                source=source_samples(suite)
                if len(source)!=EXPECTED[suite]:
                    raise ValueError(f'{suite}: unexpected image population')
                datasets[suite]=dict(questions=len(source),sample_identity=digest(source))
            else:
                _,datasets[suite]=load_video(suite,None)
        if any(SUITES[s]=='image' for s in suites):
            datasets['image_assets']=dataset_fingerprint()
        from constrained_experiment import checkpoint_files
        for model in models:
            checkpoint_files(exp.paths[model])
            if exp.models[model].get('kind') in ('trained','export'):
                exp.validate_export(model)
        if 'qat_w4' in models:
            if not (exp.paths['qat_w4']/'qat_quantized_weights.bin').is_file():
                raise FileNotFoundError('QAT sidecar missing; cannot preserve trained INT4 codes/scales')
            from evaluation.protocols.int4_gemm import int4_matmul  # Import/JIT support only; no CUDA allocation.
        source_check()
        if args.attn=='flash_attention_2' and importlib.util.find_spec('flash_attn') is None:
            raise SystemExit('flash_attn is unavailable; select --attn sdpa in a separate protocol/output')
        if args.action=='check':
            print(json.dumps(dict(status='CPU checks passed; fused CUDA execution still requires GPU smoke',datasets=datasets),indent=2)); return
        output.mkdir(parents=True,exist_ok=True)
        from evaluation.workers.video_runtime import gpu_lock, ensure_gpu_idle
        from evaluation.runners.run_lmms_matrix import environment
        own_files=[Path(__file__),ROOT/'evaluation/protocols/gpu_job_queue.py',ROOT/'evaluation/protocols/gpu_reservation.py',ROOT/'evaluation/tests/check_int4_kernel.py',ROOT/'evaluation/protocols/token_reduction_runtime.py',
            ROOT/'evaluation/protocols/int4_gemm.py',ROOT/'evaluation/protocols/video_protocol.py',ROOT/'evaluation/protocols/video_population.py',
            ROOT/'evaluation/protocols/local_scoring.py',ROOT/'evaluation/runners/run_lmms_matrix.py',
            ROOT/'evaluation/runners/run_local_development.py',ROOT/'evaluation/protocols/matched_protocol.py',ROOT/'evaluation/paths.py',
            ROOT/'evaluation/runners/run_constrained_matrix_evals.py',
            ROOT/'evaluation/audits/summarize_constrained_evals.py',ROOT/'qwen-vl-finetune/scripts/constrained_experiment.py',
            ROOT/'qwen-vl-finetune/qwenvl/quantize/qat_to_awq.py',ROOT/'qwen-vl-finetune/qwenvl/quantize/awq_linear.py']
        from evaluation.runners.run_constrained_matrix_evals import suites as image_scripts
        own_files += [ROOT/'evaluation'/item[0] for item in image_scripts().values()]
        frozen=dict(policy=policy,experiment=exp.spec,limit=args.smoke,environment=environment(),triton_version=importlib.metadata.version('triton'),
                    dataset_policy='Per-suite locked annotations/assets; subset runs share the same protocol',
                    code={str(f.relative_to(ROOT)):sha(f) for f in own_files})
        with (output/'.matrix.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            pin(output/'protocol.json',frozen)
            for suite,info in datasets.items():
                pin(output/'datasets'/f'{suite}.json',info)
            sidecar_sha256=None
            for model in models:
                provenance=exp.provenance(model)
                if model=='qat_w4':
                    sidecar_sha256=sha(exp.paths[model]/'qat_quantized_weights.bin')
                    provenance['qat_sidecar_sha256']=sidecar_sha256
                pin(output/model/'model_provenance.json',provenance)
            if 'qat_w4' in models:
                for gpu in gpus:
                    if reservations is None:
                        ensure_gpu_idle(gpu)
                    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=gpu,
                             TRITON_CACHE_DIR=str(exp.directory/'runtime_cache/triton'))
                    gate_log=output/f'int4_kernel_validation_gpu{gpu}.log'
                    print(f'INT4 numerical gate: GPU {gpu}: {gate_log}',flush=True)
                    with gate_log.open('w') as log:
                        command=[sys.executable,str(ROOT/'evaluation/tests/check_int4_kernel.py')]
                        if reservations is not None:
                            from tempfile import TemporaryDirectory
                            from evaluation.protocols.gpu_reservation import run_reserved_worker
                            with TemporaryDirectory(prefix=f'kernel_handoff_gpu{gpu}_',dir=output) as gate_dir:
                                command += ['--handoff-dir',gate_dir,'--controller-pid',str(os.getpid())]
                                rc=run_reserved_worker(command,env,log,Path(gate_dir),gpu,reservations)
                            if rc:
                                raise subprocess.CalledProcessError(rc,command)
                        else:
                            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    if reservations is not None:
                        reservations.acquire(gpu)
            def run_job(job, gpu):
                if job['model']=='qat_w4':
                    job.update(packed_cache=str(exp.directory/'runtime_cache/packed_qat_w4'), sidecar_sha256=sidecar_sha256)
                directory=output/job['model']/job['suite']/configuration_id(job['method'],job['retention'])
                if verified_completion(directory):
                    return None
                if reservations is None:
                    ensure_gpu_idle(gpu)
                job.update(reserve_gpus=reservations is not None, controller_pid=os.getpid())
                attempt=directory/dt.datetime.now(dt.timezone.utc).strftime('attempt_%Y%m%dT%H%M%S_%fZ')
                attempt.mkdir(parents=True); write(attempt/'job.json',dict(job,gpu=gpu))
                env={k:v for k,v in os.environ.items() if k not in ('RANK','LOCAL_RANK','WORLD_SIZE')}
                env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=gpu,HF_HUB_OFFLINE='1',
                    TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                    TRITON_CACHE_DIR=str(exp.directory/'runtime_cache/triton'))
                print(f'GPU {gpu} RUN {job["model"]}/{job["suite"]}/{directory.name}: {attempt/"run.log"}',flush=True)
                with (attempt/'run.log').open('w') as log:
                    command=[sys.executable,str(Path(__file__).resolve()),'--worker',str(attempt/'job.json')]
                    if reservations is not None:
                        from evaluation.protocols.gpu_reservation import run_reserved_worker
                        rc=run_reserved_worker(command,env,log,attempt,gpu,reservations)
                    else:
                        rc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
                if reservations is not None and gpu not in reservations.processes:
                    reservations.acquire(gpu)
                write(attempt/'exit.json',dict(returncode=rc))
                if rc or not (attempt/'validated.json').exists():
                    return str(directory)
                done=read(attempt/'validated.json')
                artifacts=[f for f in attempt.iterdir() if f.is_file() and f.name!='run.log']
                done.update(attempt=attempt.name,sha256={f.name:sha(f) for f in artifacts})
                pin(directory/'COMPLETE.json',done)
            failed=run_gpu_queue(jobs,gpus,run_job)
            summary(output,models,suites,configs)
        if failed:
            raise SystemExit('Incomplete jobs: '+', '.join(failed))


if __name__=='__main__':
    if '--worker' in sys.argv:
        worker(sys.argv[sys.argv.index('--worker')+1])
    else:
        main()
