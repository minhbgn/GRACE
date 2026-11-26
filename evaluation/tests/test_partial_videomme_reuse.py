"""Certifying a complete selected prefix from an unfinished source run."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.runners.reuse_partial_videomme import import_candidate, read_prefix, validate_candidate
from evaluation.runners.run_video_matrix import completed, write
from evaluation.tests.test_video_runtime import fixture


class PartialReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source = self.base / 'old' / 'full'
        self.job = fixture()
        self.records = self.job.pop('test_records')
        self.job.update(suite='videomme')
        self.job['protocol'].update(version='video_resilient_v3', fps=2)
        self.identity = dict(limit=None, experiment={'name': 'test'}, policy=self.job['protocol'],
            environment={'torch': 'test'}, code={k: 'same' for k in ('run_video_matrix.py',
            'video_protocol.py', 'video_worker.py', 'local_scoring.py', 'paths.py')})
        self.provenance = {'weights_sha256': 'weights'}
        self.attempt = self.source / self.job['method'] / 'videomme' / 'attempt_1'
        self.target = self.base / 'new' / self.job['method'] / 'videomme'
        old_job = copy.deepcopy(self.job)
        old_job['protocol']['version'] = 'video_unsaturated_v2'
        write(self.source / 'protocol.json', dict(self.identity, policy=old_job['protocol']))
        write(self.source / self.job['method'] / 'model_provenance.json', self.provenance)
        write(self.attempt / 'job.json', old_job)
        write(self.attempt / 'processor.json', dict(effective_policy=old_job['protocol'], context_limit=262144))
        self.save_records()

    def save_records(self, suffix=''):
        (self.attempt / 'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.records)+suffix)

    def validate(self):
        return validate_candidate(self.source, self.attempt, self.job, self.identity, self.provenance)

    def test_exact_population_count_is_sufficient_without_summary_or_marker(self):
        before = {p: p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        self.assertFalse((self.attempt / 'summary.json').exists())
        with patch('subprocess.Popen', side_effect=AssertionError('Inference launched')):
            records, processor, audit = self.validate()
            summary = import_candidate(self.target, self.job, records, processor, audit)
        self.assertEqual(summary['count'], 2)
        self.assertEqual(completed(self.target, self.job['rows'], self.job['protocol'])['summary'], summary)
        self.assertTrue(audit['source_was_partial'])
        self.assertEqual({p: p.read_bytes() for p in before}, before)

    def test_broken_excluded_tail_does_not_affect_selected_prefix(self):
        self.save_records(suffix='{unfinished-long-question')
        records, _, _ = self.validate()
        self.assertEqual(records, self.records)

    def test_incomplete_selected_population_and_interrupted_line_rejected(self):
        path = self.attempt / 'predictions.jsonl'
        path.write_text(json.dumps(self.records[0])+'\n')
        with self.assertRaisesRegex(ValueError, 'Only 1'):
            self.validate()
        path.write_text(json.dumps(self.records[0])+'\n'+json.dumps(self.records[1]))
        with self.assertRaisesRegex(ValueError, 'Only 1'):
            self.validate()

    def test_bad_selected_records_rejected_even_if_line_count_is_sufficient(self):
        baseline = copy.deepcopy(self.records)
        for mutation in ('duplicate', 'order', 'score', 'source', 'input', 'cutoff', 'allowance', 'context'):
            self.records = copy.deepcopy(baseline)
            if mutation == 'duplicate': self.records[1] = copy.deepcopy(self.records[0])
            if mutation == 'order': self.records.reverse()
            if mutation == 'score': self.records[0]['correct'] = False
            if mutation == 'source': self.records[0]['source_identity'] = 'changed'
            if mutation == 'input': self.records[0]['input_identity'] = 'changed'
            if mutation == 'cutoff': self.records[0]['generation']['finish_reason'] = 'length'
            if mutation == 'allowance': self.records[0]['generation']['max_new_tokens'] = 10
            if mutation == 'context': self.records[0]['context_limit'] = 4096
            self.save_records()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate()

    def test_incompatible_source_provenance_rejected(self):
        path = self.source / 'protocol.json'
        baseline = json.loads(path.read_text())
        for mutation in ('smoke', 'policy', 'code', 'environment', 'experiment', 'model'):
            original = copy.deepcopy(baseline)
            provenance = self.provenance
            if mutation == 'smoke': original['limit'] = 1
            if mutation == 'policy': original['policy']['fps'] = 1
            if mutation == 'code': original['code']['video_worker.py'] = 'changed'
            if mutation == 'environment': original['environment']['torch'] = 'changed'
            if mutation == 'experiment': original['experiment'] = {'name': 'other'}
            if mutation == 'model': provenance = {'weights_sha256': 'changed'}
            write(path, original)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_candidate(self.source, self.attempt, self.job, self.identity, provenance)

    def test_malformed_record_inside_selected_population_is_rejected(self):
        path = self.attempt / 'predictions.jsonl'
        path.write_text(json.dumps(self.records[0])+'\n{broken}\n')
        with self.assertRaises(ValueError):
            read_prefix(path, 2)


if __name__ == '__main__':
    unittest.main()
