"""Matched reruns must recompute historical passes and resume only local results."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.runners import run_matched_video_evaluation as matched


class MatchedVideoTests(unittest.TestCase):
    def test_historical_pass_is_recomputed_and_local_pass_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / 'source', Path(temporary) / 'output'
            for root in (source, output):
                if root == output:
                    jobs, reused, completed = matched.inventory(output, ['videomme'])
                    self.assertEqual(len(jobs), 1)
                    self.assertEqual(reused, [])
                    self.assertEqual(completed, [])
                directory = root / 'qat_w4/videomme/resolution_r025'
                attempt = directory / 'attempt_test'
                attempt.mkdir(parents=True)
                artifact = attempt / 'validated.json'
                artifact.write_text('{}')
                (directory / 'COMPLETE.json').write_text(json.dumps(dict(
                    attempt=attempt.name, sha256={'validated.json': matched.matrix.sha(artifact)})))
            jobs, reused, completed = matched.inventory(output, ['videomme'])
            self.assertEqual(jobs, [])
            self.assertEqual(reused, [])
            self.assertEqual(completed, [dict(suite='videomme', config='resolution_r025')])
            artifact.write_text('{"tampered":true}')
            with self.assertRaisesRegex(ValueError, 'Completed output changed'):
                matched.inventory(output, ['videomme'])

    def test_existing_recovery_output_is_rejected(self):
        with patch.object(sys, 'argv', ['matched', 'plan', '--output', str(matched.retry.DEFAULT_OUTPUT)]):
            with self.assertRaises(SystemExit):
                matched.main()

    def setUp(self):
        self.configs = patch.object(matched.retry, 'CONFIGS', [('resolution', .25)])
        self.configs.start()
        self.addCleanup(self.configs.stop)


if __name__ == '__main__':
    unittest.main()
