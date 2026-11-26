"""Actual subprocess death/recovery, resume integrity and GPU collision checks."""
import copy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.protocols.video_protocol import digest
from evaluation.protocols.local_scoring import generation_record
from evaluation.workers.video_runtime import (ensure_gpu_idle, gpu_lock, prediction_prefix,
    run_pair, seed_attempt, validate_prefix)


def fixture():
    sources, records = [], []
    for uid in ('one', 'two'):
        source = dict(id=uid, gold='B', options={'A': 'left', 'B': 'right'},
                      question_type='multiple-choice', category='action', prompt='Choose')
        counts = dict(input_visual_tokens=10, input_text_tokens=5, input_tokens=15)
        row = dict(source, raw='B', prediction='B', correct=True, extraction='test',
                   generation=generation_record([2, 99], 99, 4096), eos_token_id=99,
                   context_limit=262144, tokens=dict(counts, output_text_tokens=2),
                   sampling={'grid': [1, 4, 10]}, input_ids_sha256='test-input',
                   source_identity=digest(source))
        row['input_identity'] = digest(dict(source=source, sampling=row['sampling'],
                                           tokens=counts, input_ids_sha256='test-input'))
        sources.append(source)
        records.append(row)
    return dict(method='qat_w4', suite='tempcompass', model_path='/test/checkpoint',
                protocol={'max_new_tokens': 4096}, rows=sources, run_identity='pinned-code-env-model',
                test_records=records)


class RecoveryTests(unittest.TestCase):
    def test_sigkill_mid_write_resumes_once_without_duplicate_or_missing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            worker = base / 'worker.py'
            worker.write_text('''import json, os, signal, sys
from pathlib import Path
p = Path(sys.argv[1]).parent
job = json.loads((p / 'job.json').read_text())
n = json.loads((p / 'resume.json').read_text())['predictions']
with (p / 'predictions.jsonl').open('a') as f:
    if n == 0:
        f.write(json.dumps(job['test_records'][0]) + '\\n')
        f.write('{"id":')
        f.flush()
        os.fsync(f.fileno())
        os.kill(os.getpid(), signal.SIGKILL)
    for r in job['test_records'][n:]:
        f.write(json.dumps(r) + '\\n')
''')
            job = fixture()
            env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            with patch('evaluation.workers.video_runtime.ensure_gpu_idle', return_value={'idle': True}):
                attempt, code = run_pair(base / 'pair', job, env, '0', worker=worker)
            self.assertEqual(code, 0)
            attempts = sorted((base / 'pair').glob('attempt_*'))
            self.assertEqual(len(attempts), 2)
            self.assertEqual(json.loads((attempts[0] / 'exit.json').read_text())['signal'], 'SIGKILL')
            self.assertEqual(json.loads((attempts[0] / 'failure.json').read_text())['returncode'], -9)
            self.assertTrue((attempts[0] / 'predictions.jsonl').read_text().endswith('{"id":'))
            self.assertEqual(json.loads((attempt / 'resume.json').read_text())['predictions'], 1)
            records = validate_prefix(prediction_prefix(attempt / 'predictions.jsonl'), job)
            self.assertEqual([r['id'] for r in records], ['one', 'two'])

    def test_repeated_sigkill_has_bounded_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            worker = base / 'worker.py'
            worker.write_text('import os, signal; os.kill(os.getpid(), signal.SIGKILL)\n')
            with patch('evaluation.workers.video_runtime.ensure_gpu_idle', return_value={}):
                _, code = run_pair(base / 'pair', fixture(),
                    dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'),
                    '0', sigkill_retries=1, worker=worker)
            self.assertEqual(code, -signal.SIGKILL)
            self.assertEqual(len(list((base / 'pair').glob('attempt_*'))), 2)

    def test_resume_rejects_changed_run_and_corrupt_rows(self):
        job = fixture()
        for mutation in ('order', 'score', 'source', 'cutoff', 'allowance'):
            with self.subTest(mutation=mutation):
                records = copy.deepcopy(job['test_records'])
                if mutation == 'order': records.reverse()
                if mutation == 'score': records[0]['correct'] = False
                if mutation == 'source': records[0]['source_identity'] = 'changed'
                if mutation == 'cutoff':
                    records[0]['generation'] = generation_record([2, 2], 99, 2)
                if mutation == 'allowance': records[0]['generation']['max_new_tokens'] = 2048
                with self.assertRaises(ValueError): validate_prefix(records, job)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            previous, current = base / 'attempt_1', base / 'attempt_2'
            previous.mkdir(); current.mkdir()
            old = dict(job, run_identity='other-source-version')
            (previous / 'job.json').write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError, 'provenance'):
                seed_attempt(base, current, job)

    def test_complete_malformed_line_is_not_treated_as_an_interrupted_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'predictions.jsonl'
            path.write_text('{broken}\n')
            with self.assertRaisesRegex(ValueError, 'Corrupt prediction'):
                prediction_prefix(path)


class IsolationTests(unittest.TestCase):
    def test_busy_gpu_is_rejected_before_starting_a_worker(self):
        with patch('evaluation.workers.video_runtime.gpu_state', return_value=dict(
                device='0, GPU-test, NVIDIA A100-SXM4-40GB, 10000, 40960',
                processes='123, unrelated_job, 10000')):
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                ensure_gpu_idle('0')

    def test_gpu_lock_applies_across_output_directories(self):
        gpu = f'test-{os.getpid()}'
        with gpu_lock(gpu):
            with self.assertRaisesRegex(RuntimeError, 'holds GPU'):
                with gpu_lock(gpu): self.fail('Acquired a second lock')


if __name__ == '__main__':
    unittest.main()
