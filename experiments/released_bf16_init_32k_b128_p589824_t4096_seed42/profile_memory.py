#!/usr/bin/env python3
"""Bounded synthetic memory check; never writes production runs or their subset.

Uses a real square image and a synthetic long answer at the sequence ceiling.
Three optimizer updates with accumulation=2 by default exercise optimizer state and full
RCKA (warmup=1). This is a feasibility screen, not a throughput/quality estimate
or proof that every full-training batch fits. Use run_training.sh --profile for
an additional check on the actual prepared data and batch-128 recipe.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROJECT = ROOT / 'qwen-vl-finetune'
sys.path[:0] = [str(PROJECT / 'scripts'), str(PROJECT)]
import run_constrained as shared
from constrained_experiment import Experiment, check_processor_compatibility, pin, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=('qat', 'grace'), default='grace')
    p.add_argument('--bits', type=int, choices=(4, 8), default=4)
    p.add_argument('--output', type=Path, help='New diagnostic directory selected by the matrix runner')
    p.add_argument('--accumulation', type=int, default=2)
    args = p.parse_args()
    if args.accumulation < 1:
        p.error('--accumulation must be positive')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['PROFILE_STEPS'] = '3'
    for key, value in dict(OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1',
                          TOKENIZERS_PARALLELISM='false', TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
                          PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True').items():
        os.environ.setdefault(key, value)
    survey = subprocess.check_output(['gpuwho', '--json'], text=True,
        env=dict(os.environ, GPU_GUARD_MINE=str(ROOT.parents[1])))
    shared.check_gpus()
    spec = Experiment(HERE / 'experiment.json')
    c = shared.resolved_config()
    # These changes apply ONLY to the synthetic feasibility screen.
    c.update(model_path=str(spec.paths[spec.student]), teacher_path=str(spec.paths[spec.teacher]),
             student_weights_sha256=spec.models[spec.student]['weights_sha256'],
             qat_bits=args.bits, subset_size=8 * args.accumulation, global_batch=2 * args.accumulation, grad_accum=args.accumulation,
             max_steps=4, rcka_warmup_steps=1, min_pixels=c['max_pixels'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    work = args.output.resolve() if args.output else HERE / 'memory_checks' / f'{args.method}_w{args.bits}_{stamp}'
    work.mkdir(parents=True)
    (work / 'gpu_survey.json').write_text(survey)
    pin(work / 'student_provenance.json', spec.provenance(spec.student))
    pin(work / 'teacher_provenance.json', spec.provenance(spec.teacher))
    check_processor_compatibility([spec.paths[spec.student], spec.paths[spec.teacher]])

    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from qwenvl.data.constrained_data import preprocess, token_hash, json_hash, image_path
    from qwenvl.data.data_processor import update_processor_pixels, preprocess_qwen_visual
    from qwenvl.train.argument import DataArguments
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(c['model_path'], local_files_only=True)
    processor.tokenizer.model_max_length = c['model_max_length']
    update_processor_pixels(processor, DataArguments(max_pixels=c['max_pixels'], min_pixels=c['min_pixels']))
    old = json.loads((ROOT / spec.spec['previous_experiment'] / 'subset.json').read_text())
    for record in old['records']:
        sample = deepcopy(record['sample'])
        path = Path(sample['data_path']) / image_path(sample)
        with Image.open(path) as im:
            if im.width == im.height:
                break
    else:
        raise ValueError('No real square image available for the maximum-grid stress input')
    sample['conversations'] = [dict(**{'from': 'human'}, value='<image>\nDescribe this image.'),
                               dict(**{'from': 'gpt'}, value='')]
    lo, hi = 1, c['model_max_length'] * 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        sample['conversations'][1]['value'] = ' detail' * mid
        length = preprocess_qwen_visual([sample], processor, truncate=False)['input_ids'].shape[1]
        if length <= c['model_max_length']:
            lo = mid
        else:
            hi = mid - 1
    sample['conversations'][1]['value'] = ' detail' * lo
    item = preprocess(sample, processor, c['model_max_length'], c['max_pixels'])
    visual = int(item['image_grid_thw'].prod()) // processor.image_processor.merge_size**2
    if visual != c['max_pixels'] // 1024 or item['input_ids'].shape[1] < c['model_max_length'] - 2:
        raise ValueError('Stress input did not reach the requested token envelope')
    stat = path.stat()
    records = [dict(index=i, source='synthetic_memory_only', sample=sample,
                    length=int(item['input_ids'].shape[1]), token_sha256=token_hash(item),
                    image_size=stat.st_size, image_mtime_ns=stat.st_mtime_ns) for i in range(c['subset_size'])]
    manifest = dict(records=records, records_sha256=json_hash(records),
                    purpose='Synthetic feasibility only; not the training subset')
    pin(work / 'subset.json', manifest)
    envelope = dict(visual_tokens=visual, sequence_tokens=records[0]['length'],
                    supervised_tokens=int((item['labels'][:, 1:] != -100).sum()),
                    grid=item['image_grid_thw'].tolist(), accumulation=args.accumulation, updates=3,
                    rcka_warmup_steps=1, synthetic_answer=True)
    pin(work / 'input_envelope.json', envelope)
    recipe = shared.comparison_recipe(c, manifest)
    recipe['profile_script_sha256'] = sha(Path(__file__))
    pin(work / 'comparison_recipe.json', recipe)
    print(json.dumps(dict(output=str(work), **envelope)), flush=True)
    with (work / 'device_memory.csv').open('w') as log:
        monitor = subprocess.Popen(['nvidia-smi', '--id=' + os.environ['CUDA_VISIBLE_DEVICES'],
            '--query-gpu=timestamp,index,memory.used,memory.total', '--format=csv', '-lms', '500'], stdout=log)
        try:
            shared.launch_run(args.method, c, work, recipe, profile=True, record_invocations=True)
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
    output = work / 'profiles' / args.method
    summary = json.loads((output / 'run_summary.json').read_text())
    if not summary['complete'] or not (output / 'EXPORT_COMPLETE').exists():
        raise RuntimeError(f'Memory check did not complete within the guard: {summary}')
    print(json.dumps(dict(output=str(work), **summary), indent=2), flush=True)


if __name__ == '__main__':
    main()
