"""Reuse real completion artifacts without inference or weakened integrity."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.tests.test_video_runtime import fixture
from evaluation.runners.run_video_matrix import completed, sha, summary_from_rows, write
from evaluation.workers.video_reuse import reuse_completed


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source = self.base / 'old' / 'full'
        self.target = self.base / 'new' / 'full'
        self.job = fixture()
        self.records = self.job.pop('test_records')
        self.job['protocol'].update(version='video_resilient_v3', fps=2)
        self.identity = dict(limit=None, experiment={'name': 'test'},
            policy=self.job['protocol'], environment={'torch': 'test'},
            code={k: 'same' for k in ('run_video_matrix.py', 'video_protocol.py',
                  'video_worker.py', 'local_scoring.py', 'paths.py')})
        self.provenance = {'weights_sha256': 'weights'}
        self.source_dir = self.source / self.job['method'] / self.job['suite']
        self.target_dir = self.target / self.job['method'] / self.job['suite']
        self.attempt = self.source_dir / 'attempt_1'
        old_job = copy.deepcopy(self.job)
        old_job['protocol'] = dict(max_new_tokens=4096, version='video_unsaturated_v2', fps=2)
        old_identity = dict(self.identity, policy=old_job['protocol'])
        write(self.source / 'protocol.json', old_identity)
        write(self.source_dir.parent / 'model_provenance.json', self.provenance)
        write(self.attempt / 'job.json', old_job)
        (self.attempt / 'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.records))
        write(self.attempt / 'summary.json', summary_from_rows(self.records))
        write(self.attempt / 'processor.json', {'effective_policy': old_job['protocol']})
        write(self.source_dir / 'COMPLETE.json', dict(attempt='attempt_1',
              summary=summary_from_rows(self.records), sha256={k: sha(self.attempt / k)
              for k in ('job.json', 'predictions.jsonl', 'summary.json', 'processor.json')}))

    def reuse(self):
        return reuse_completed(self.source, self.target_dir, self.job, self.identity, self.provenance)

    def test_copies_validated_results_and_preserves_source(self):
        before = {p: p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        with patch('subprocess.Popen', side_effect=AssertionError('Inference launched')):
            self.assertTrue(self.reuse())
        done = completed(self.target_dir, self.job['rows'], self.job['protocol'])
        self.assertEqual(done['summary']['count'], 2)
        self.assertIn('reuse.json', done['sha256'])
        self.assertEqual({p: p.read_bytes() for p in before}, before)

    def test_subsets_saved_predictions_and_recomputes_metrics(self):
        self.job['rows'] = self.job['rows'][1:]
        self.job['protocol']['videomme_population'] = {'name': 'short_medium'}
        self.assertTrue(self.reuse())
        done = completed(self.target_dir, self.job['rows'], self.job['protocol'])
        self.assertEqual(done['summary']['count'], 1)
        records = (self.target_dir / done['attempt'] / 'predictions.jsonl').read_text().splitlines()
        self.assertEqual([json.loads(line)['id'] for line in records], ['two'])

    def test_absent_completion_is_pending(self):
        (self.source_dir / 'COMPLETE.json').unlink()
        self.assertFalse(self.reuse())
        self.assertFalse(self.target_dir.exists())

    def test_rejects_changed_policy_model_code_environment_and_smoke(self):
        baseline = read_json(self.source / 'protocol.json')
        for mutation in ('policy', 'model', 'code', 'environment', 'smoke', 'source'):
            old = copy.deepcopy(baseline)
            provenance = self.provenance
            rows = copy.deepcopy(self.job['rows'])
            if mutation == 'policy': old['policy']['fps'] = 1
            if mutation == 'model': provenance = {'weights_sha256': 'changed'}
            if mutation == 'code': old['code']['video_worker.py'] = 'changed'
            if mutation == 'environment': old['environment']['torch'] = 'changed'
            if mutation == 'smoke': old['limit'] = 1
            if mutation == 'source': self.job['rows'][0]['gold'] = 'A'
            write(self.source / 'protocol.json', old)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                reuse_completed(self.source, self.target_dir, self.job, self.identity, provenance)
            self.job['rows'] = rows
        write(self.source / 'protocol.json', baseline)

    def test_tampered_prediction_does_not_create_target_completion(self):
        with (self.attempt / 'predictions.jsonl').open('a') as stream:
            stream.write('{}\n')
        with self.assertRaises(ValueError):
            self.reuse()
        self.assertFalse((self.target_dir / 'COMPLETE.json').exists())


def read_json(path):
    return json.loads(path.read_text())


if __name__ == '__main__':
    unittest.main()
