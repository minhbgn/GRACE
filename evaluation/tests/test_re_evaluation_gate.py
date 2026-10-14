"""Stage two must not launch on missing, tampered or out-of-tolerance references."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_re_evaluation_gate import REFERENCES, check_gate
from paper_acceptance import ORDER, PAPER
from run_lmms_matrix import sha, write
from run_upstream_matrix import EXPECTED, SUITES, TOKENS, source_check, validate
from summarize_constrained_evals import ID_FIELDS


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = {m: self.root / 'models' / m for m in REFERENCES}
        self.patch_counts = patch.dict(EXPECTED, {s: 1 for s in SUITES})
        self.patch_counts.start()
        self.addCleanup(self.patch_counts.stop)
        self.patch_targets = patch.dict(PAPER, {m: {s: 100 for s in ORDER} for m in REFERENCES})
        self.patch_targets.start()
        self.addCleanup(self.patch_targets.stop)
        write(self.root / 'full/protocol.json', {'limit': None, 'source': source_check(), 'tokens': TOKENS})
        for method in REFERENCES:
            for suite in SUITES:
                job = self.root / 'full' / method / suite
                attempt = job / 'attempt_test'
                row = {k: 'id0' for k in ID_FIELDS[suite]}
                row.update(gold='A', correct=True, error=None)
                payload = {'summary': {'model_path': str(self.paths[method]), 'total': 1,
                           'correct': 1, SUITES[suite][1]: 1.0}, 'results': [row]}
                result = validate(payload, suite, self.paths[method], None)
                write(attempt / 'results.json', payload)
                write(attempt / 'command.json', ['synthetic'])
                write(attempt / 'validated.json', result)
                write(job / 'COMPLETE.json', {**result, 'attempt': attempt.name,
                      'sha256': {name: sha(attempt / name) for name in ('results.json', 'command.json', 'validated.json')}})

    def test_complete_reference_matrix_passes(self):
        report = check_gate(self.root, self.paths)
        self.assertTrue(report['passed'])
        self.assertEqual(report['completed_pairs'], 35)

    def test_missing_result_blocks(self):
        (self.root / 'full/vanilla_2b/ai2d/COMPLETE.json').unlink()
        report = check_gate(self.root, self.paths)
        self.assertFalse(report['passed'])
        self.assertEqual(report['completed_pairs'], 34)

    def test_barely_requires_explicit_option_and_cannot_allow_large_gap(self):
        PAPER['vanilla_2b']['ai2d'] = 96
        self.assertFalse(check_gate(self.root, self.paths)['passed'])
        self.assertTrue(check_gate(self.root, self.paths, accept_barely=True)['passed'])
        PAPER['vanilla_2b']['ai2d'] = 94
        self.assertFalse(check_gate(self.root, self.paths, accept_barely=True)['passed'])

    def test_tampered_prediction_rejected(self):
        path = self.root / 'full/vanilla_2b/ai2d/attempt_test/results.json'
        path.write_text('{}')
        with self.assertRaises(ValueError): check_gate(self.root, self.paths)

    def test_altered_completion_score_rejected(self):
        path = self.root / 'full/vanilla_2b/ai2d/COMPLETE.json'
        data = json.loads(path.read_text())
        data['score_percent'] = 99
        write(path, data)
        with self.assertRaises(ValueError): check_gate(self.root, self.paths)

    def test_smoke_cannot_unlock_stage_two(self):
        path = self.root / 'full/protocol.json'
        data = json.loads(path.read_text())
        data['limit'] = 1
        write(path, data)
        with self.assertRaises(ValueError): check_gate(self.root, self.paths)


if __name__ == '__main__':
    unittest.main()
