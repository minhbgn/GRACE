#!/usr/bin/env python3
"""Targeted QAT W4 VideoMME 618-1 regression through the resilient worker."""
import json
import os
from pathlib import Path
import sys

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.protocols.video_protocol import digest
from evaluation.runners.run_video_matrix import read, validate_attempt, write
from evaluation.workers.video_runtime import gpu_lock, run_pair

out = EXP / 'reports/resilient_runtime_validation_20261023/long_case'
out.mkdir(exist_ok=False)
historical = EXP / 'video_evaluation/parallel_retry_20261019T034141Z_kbsu66/gpu4/full/qat_w4/videomme/attempt_20261019T035547.718615Z/job.json'
job = read(historical)
job['rows'] = [r for r in job['rows'] if r['id'] == '618-1']
identity = read(out.parent / 'smoke/smoke_3/protocol.json')
job['protocol'] = identity['policy']
job['run_identity'] = digest(dict(diagnostic='one-question-regression', sources=identity['code']))
write(out / 'diagnostic.json', dict(note='One-question regression, not a benchmark score', job=job))
env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='0',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
           PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1', PYTORCH_CUDA_ALLOC_CONF=job['protocol']['cuda_allocator'])
with gpu_lock('0'):
    attempt, code = run_pair(out / 'qat_w4/videomme', job, env, '0')
assert code == 0, f'Regression failed: {attempt}'
summary = validate_attempt(attempt, job['rows'], job['protocol'])
prediction = json.loads((attempt / 'predictions.jsonl').read_text())
assert prediction['input_ids_sha256'] == '836c0a976dc48785bbdb111ad502cb938115dec10bad0d217ac9893b1b3bf405'
assert prediction['tokens']['input_tokens'] == 151218
assert prediction['generation']['finish_reason'] == 'eos'
write(out / 'validation.json', dict(passed=True, attempt=str(attempt), summary=summary,
      peak_gpu_allocated_bytes=prediction['peak_gpu_allocated_bytes'], input_ids_sha256=prediction['input_ids_sha256']))
print(json.dumps(dict(passed=True, peak_GiB=prediction['peak_gpu_allocated_bytes']/2**30, tokens=prediction['tokens'])), flush=True)
