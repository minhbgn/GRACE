#!/usr/bin/env python3
"""One isolated checkpoint/video suite. Never imports training code."""
from __future__ import annotations

# Support both direct script execution and `python -m evaluation.<group>.<module>`.
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.paths import EVALUATION_ROOT, source_file

import json
import hashlib
from pathlib import Path
import random
import sys
import time

from evaluation.protocols.video_protocol import build_inputs, digest, input_counts, local_video, probe
from evaluation.runners.run_video_matrix import read, summary_from_rows, write


def prepare(output, datasets, processor, protocol, context, scratch):
    """CPU geometry and exact tokenizer preflight, no weight loading or CUDA."""
    cache, report = {}, {}
    for suite, rows in datasets.items():
        records = []
        path = output / 'preflight' / (suite + '.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as stream:
            for index, row in enumerate(rows):
                key = digest(row['asset'])
                try:
                    if key not in cache:
                        with local_video(row['asset'], scratch) as video:
                            cache[key] = probe(video, protocol, processor)
                    meta = cache[key]
                    counts = input_counts(processor, row, meta)
                    item = dict(id=row['id'], **counts, sampled_frames=meta['sampled_frames'],
                                source_height=meta['source_height'], source_width=meta['source_width'],
                                processed_height=meta['height'], processed_width=meta['width'],
                                fits_context=counts['input_tokens'] + protocol['max_new_tokens'] <= context)
                except Exception as error:
                    item = dict(id=row['id'], error=f'{type(error).__name__}: {error}', fits_context=False)
                stream.write(json.dumps(item) + '\n')
                stream.flush()
                records.append(item)
                if index % 100 == 0:
                    print(f'CPU preflight {suite}: {index + 1}/{len(rows)}', flush=True)
        valid = [r for r in records if 'error' not in r]
        report[suite] = dict(count=len(rows), context_limit=context,
            errors=sum('error' in r for r in records), context_overflows=sum(not r['fits_context'] for r in valid),
            input_visual_tokens_mean=sum(r['input_visual_tokens'] for r in valid) / len(valid) if valid else None,
            input_visual_tokens_max=max((r['input_visual_tokens'] for r in valid), default=None),
            note='Means describe all probed questions, including context overflows; no model inference')
        print(json.dumps({suite: report[suite]}), flush=True)
    write(output / 'preflight' / 'summary.json', report)
    return report


def main():
    job_path = Path(sys.argv[1])
    job, attempt = read(job_path), job_path.parent
    protocol = job['protocol']
    import numpy as np
    import torch
    from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration
    from evaluation.protocols.local_scoring import choice, generation_record, open_answer
    random.seed(protocol['seed'])
    np.random.seed(protocol['seed'])
    torch.manual_seed(protocol['seed'])
    processor = AutoProcessor.from_pretrained(job['model_path'], local_files_only=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    config = read(Path(job['model_path']) / 'config.json')
    context = config['text_config']['max_position_embeddings']
    native_video = processor.video_processor.to_dict()
    write(attempt / 'processor.json', dict(native_video=native_video, effective_policy=protocol,
          context_limit=context, note='Native video resize/sampling bounds bypassed; sample-specific total pixel bound is above actual input'))
    # Load lazily: context failures are detected before allocating model weights.
    model = None
    results = []
    with (attempt / 'predictions.jsonl').open('w') as stream:
        for index, row in enumerate(job['rows']):
            started = time.perf_counter()
            try:
                with local_video(row['asset'], job['scratch']) as path:
                    meta = probe(path, protocol, processor)
                    expected = input_counts(processor, row, meta)
                    if expected['input_tokens'] + protocol['max_new_tokens'] > context:
                        write(attempt / 'blocked.json', dict(id=row['id'], reason='context_overflow',
                              tokens=expected, context_limit=context, sampling=meta,
                              note='No frame clipping, resolution reduction, or text truncation applied'))
                        raise ValueError(f'{row["id"]}: {expected["input_tokens"]} input tokens + '
                                         f'{protocol["max_new_tokens"]} output reserve exceed context {context}')
                    inputs, counts = build_inputs(processor, row, path, meta)
                if model is None:
                    model = Qwen3VLForConditionalGeneration.from_pretrained(
                        job['model_path'], local_files_only=True, dtype=torch.bfloat16,
                        attn_implementation=protocol['attention']).to('cuda:0').eval()
                    model.generation_config = GenerationConfig(
                        do_sample=False, num_beams=1, repetition_penalty=1.0,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        bos_token_id=processor.tokenizer.bos_token_id,
                        pad_token_id=processor.tokenizer.pad_token_id, use_cache=True)
                inputs = inputs.to('cuda:0')
                input_ids_sha256 = hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest()
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=protocol['max_new_tokens'])
                ids = generated[0, inputs['input_ids'].shape[1]:].tolist()
                raw = processor.tokenizer.decode(ids, skip_special_tokens=True)
                generation = generation_record(ids, processor.tokenizer.eos_token_id, protocol['max_new_tokens'])
                if row['question_type'] == 'multiple-choice':
                    prediction, extraction = choice(raw, row['options'], row['options'])
                    correct = prediction == row['gold']
                else:
                    prediction, correct = open_answer(raw, row['gold'])
                    extraction = 'mmmu_open_typed_candidates'
                record = {k: row[k] for k in ('id', 'gold', 'options', 'question_type', 'category')}
                record.update(raw=raw, prediction=prediction, correct=correct, extraction=extraction,
                              generation=generation, eos_token_id=processor.tokenizer.eos_token_id,
                              context_limit=context, tokens={**counts, 'output_text_tokens': len(ids)},
                              source_identity=digest(row), sampling=meta,
                              input_ids_sha256=input_ids_sha256,
                              input_identity=digest(dict(source=row, sampling=meta, tokens=counts,
                                                         input_ids_sha256=input_ids_sha256)),
                              elapsed_seconds=time.perf_counter() - started,
                              peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
                stream.write(json.dumps(record) + '\n')
                stream.flush()
                results.append(record)
                print(f'{index + 1}/{len(job["rows"])} {row["id"]}: visual={counts["input_visual_tokens"]} '
                      f'text_in={counts["input_text_tokens"]} text_out={len(ids)} finish={generation["finish_reason"]}', flush=True)
                del inputs, generated
            except Exception as error:
                write(attempt / 'failure.json', dict(id=row['id'], error=f'{type(error).__name__}: {error}'))
                raise
    summary = summary_from_rows(results)
    write(attempt / 'summary.json', summary)
    if summary['generation_cutoffs']:
        raise RuntimeError('At least one generation stopped without EOS; increase output allowance in a fresh protocol')


if __name__ == '__main__':
    main()
