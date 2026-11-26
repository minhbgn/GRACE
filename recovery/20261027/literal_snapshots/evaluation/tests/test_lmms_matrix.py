"""Guard against plausible-looking incomplete or mis-scaled evaluation reports."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_lmms_matrix import TASKS, sha, summarize, validate_result, verified_completion, write


def fixture(suite='scienceqa', limit=None):
    task, _, metric, scale = TASKS[suite]
    result = {'results': {task: {metric: 50 / scale}},
              'n-samples': {task: {'original': 2, 'effective': 2}}, 'config': {'limit': limit}}
    samples = [{'doc_id': i, 'doc_hash': f'hash{i}', 'target': 'A', 'resps': [['A']]} for i in range(2)]
    return result, samples


class CompletionTests(unittest.TestCase):
    def test_normalize_all_primary_metrics(self):
        for suite in TASKS:
            result, samples = fixture(suite)
            self.assertEqual(validate_result(result, samples, suite, None)['score_percent'], 50)

    def test_reject_incomplete_duplicate_wrong_task_and_nan(self):
        result, samples = fixture()
        bad_result = copy.deepcopy(result)
        bad_result['results']['scienceqa_img']['exact_match,none'] = float('nan')
        for r, s in [(result, samples[:1]), (result, samples[:1] * 2), (bad_result, samples)]:
            with self.assertRaises(ValueError):
                validate_result(r, s, 'scienceqa', None)
        with self.assertRaises(KeyError):
            validate_result(result, samples, 'mmmu', None)

    def test_smoke_cannot_be_accepted_as_full(self):
        result, samples = fixture(limit=2)
        with self.assertRaises(ValueError):
            validate_result(result, samples, 'scienceqa', None)
        self.assertEqual(validate_result(result, samples, 'scienceqa', 2)['count'], 2)

    def test_primary_metric_scale_guard(self):
        result, samples = fixture()
        result['results']['scienceqa_img']['exact_match,none'] = 75
        with self.assertRaises(ValueError):
            validate_result(result, samples, 'scienceqa', None)

    def test_sample_identity_changes_with_targets(self):
        result, samples = fixture()
        identity = validate_result(result, samples, 'scienceqa', None)['sample_identity']
        samples[0]['target'] = 'B'
        self.assertNotEqual(identity, validate_result(result, samples, 'scienceqa', None)['sample_identity'])

    def test_resume_rejects_modified_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            write(path / 'attempt/results.json', {'score': 1})
            write(path / 'COMPLETE.json', {'attempt': 'attempt', 'sha256': {'results.json': sha(path / 'attempt/results.json')}})
            self.assertIsNotNone(verified_completion(path))
            write(path / 'attempt/results.json', {'score': 2})
            with self.assertRaises(ValueError):
                verified_completion(path)

    def test_partial_matrix_has_no_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            summarize(path, ['model'])
            result = json.loads((path / 'comparison.json').read_text())
            self.assertEqual(result['completed'], 0)
            self.assertIsNone(result['rows'][0]['mean_percent'])


if __name__ == '__main__':
    unittest.main()
