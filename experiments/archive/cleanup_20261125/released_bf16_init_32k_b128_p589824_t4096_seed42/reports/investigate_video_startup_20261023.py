#!/usr/bin/env python3
"""Isolated diagnostic; never writes to the pinned evaluation or checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
RETRY = EXPERIMENT / 'video_evaluation/parallel_retry_20261019T034141Z_kbsu66'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def child(args):
    sys.path.insert(0, str(ROOT))
    import random
    import numpy as np
    import torch
    import transformers
    from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration
    from evaluation.protocols.video_protocol import build_inputs, local_video, probe

    out = args.output
    job = json.loads(args.job.read_text())
    row = next(r for r in job['rows'] if r['id'] == args.question)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    started = time.monotonic()

    def event(stage, **extra):
        data = dict(stage=stage, elapsed_s=time.monotonic() - started,
                    torch=torch.__version__, transformers=transformers.__version__, **extra)
        if torch.cuda.is_initialized():
            data.update(allocated=torch.cuda.memory_allocated(),
                        reserved=torch.cuda.memory_reserved(),
                        peak_allocated=torch.cuda.max_memory_allocated(),
                        free_total=torch.cuda.mem_get_info())
        with (out / 'events.jsonl').open('a') as f:
            f.write(json.dumps(data) + '\n')
        print(json.dumps(data), flush=True)

    try:
        event('start', model=job['method'], question=row['id'])
        processor = AutoProcessor.from_pretrained(job['model_path'], local_files_only=True)
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        with local_video(row['asset'], job['scratch']) as path:
            meta = probe(path, job['protocol'], processor)
            inputs, counts = build_inputs(processor, row, path, meta)
        event('inputs_cpu', counts=counts)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            job['model_path'], local_files_only=True, dtype=torch.bfloat16,
            attn_implementation=job['protocol']['attention'])
        event('weights_cpu', parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
              embeddings_tied=model.get_input_embeddings().weight is model.get_output_embeddings().weight,
              parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
              module_types=sorted({type(m).__name__ for m in model.modules()}))
        model = model.to('cuda:0').eval()
        event('weights_cuda')
        model.generation_config = GenerationConfig(
            do_sample=False, num_beams=1, repetition_penalty=1.0,
            eos_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id, use_cache=True)
        inputs = inputs.to('cuda:0')
        input_hash = hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest()
        event('inputs_cuda', input_ids_sha256=input_hash)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=job['protocol']['max_new_tokens'])
        ids = generated[0, inputs['input_ids'].shape[1]:].tolist()
        event('generation_complete', generated_ids=ids,
              raw=processor.tokenizer.decode(ids, skip_special_tokens=True), counts=counts)
    except BaseException:
        event('exception', traceback=traceback.format_exc())
        raise


def parent(args):
    # PCI numbering matches the matrix. Use only a currently empty A100.
    inventory = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,name,memory.used,memory.total',
        '--format=csv,noheader,nounits'], text=True)
    gpu = next(line.split(', ') for line in inventory.splitlines()
               if line.split(', ')[0] == str(args.gpu))
    if 'A100' not in gpu[1] or int(gpu[2]) > 100:
        raise RuntimeError(f'Refusing to share an occupied/non-A100 GPU: {gpu}')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'gpu_before.txt').write_text(inventory)
    dump(args.output / 'diagnostic_config.json', dict(gpu=args.gpu, allocator=args.allocator,
         long_only=args.long_only, long_methods=args.long_methods,
         compare_allocators=args.compare_allocators,
         inherited_allocator=os.environ.get('PYTORCH_CUDA_ALLOC_CONF')))
    jobs = []
    for method in ([] if args.long_only else ['vanilla_2b', 'qat_w4', 'grace_w4', 'qat_w8', 'grace_w8']):
        group = 'gpu2' if method == 'vanilla_2b' else 'gpu4'
        source = sorted((RETRY / group / 'full' / method / 'tempcompass').glob('attempt_*/job.json'))[-1]
        jobs.append((method + '_tempcompass', source, '0'))
    if args.long_probe or args.long_only:
        for method in args.long_methods.split(','):
            if method not in ('vanilla_2b', 'qat_w4'):
                raise ValueError(method)
            group = 'gpu2' if method == 'vanilla_2b' else 'gpu4'
            source = sorted((RETRY / group / 'full' / method / 'videomme').glob('attempt_*/job.json'))[-1]
            jobs.append((method + '_videomme_618-1_fresh_process', source, '618-1'))
    results = []
    for name, source, question in jobs:
        output = args.output / name
        output.mkdir()
        env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(args.gpu),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                   PYTHONUNBUFFERED='1')
        if args.allocator:
            env['PYTORCH_CUDA_ALLOC_CONF'] = args.allocator
        if args.compare_allocators:
            if not args.long_only:
                raise ValueError('--compare-allocators requires --long-only')
            if name.startswith('qat_w4_'):
                env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            else:
                env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
        started = time.monotonic()
        print(f'Running {name}', flush=True)
        with (output / 'run.log').open('w') as log:
            try:
                process = subprocess.run([
                    sys.executable, str(Path(__file__).resolve()), '--child', '--job', str(source),
                    '--question', question, '--output', str(output)], env=env,
                    stdout=log, stderr=subprocess.STDOUT, timeout=1800)
                result = dict(name=name, returncode=process.returncode,
                              signal=-process.returncode if process.returncode < 0 else None)
            except subprocess.TimeoutExpired:
                result = dict(name=name, diagnostic_timeout_s=1800)
        result.update(elapsed_s=time.monotonic() - started, source_job=str(source),
                      allocator=env.get('PYTORCH_CUDA_ALLOC_CONF'))
        dump(output / 'exit.json', result)
        results.append(result)
        dump(args.output / 'results.json', results)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--long-probe', action='store_true')
    parser.add_argument('--long-only', action='store_true')
    parser.add_argument('--long-methods', default='qat_w4')
    parser.add_argument('--allocator')
    parser.add_argument('--compare-allocators', action='store_true')
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--job', type=Path)
    parser.add_argument('--question')
    arguments = parser.parse_args()
    (child if arguments.child else parent)(arguments)
