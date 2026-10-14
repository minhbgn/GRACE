import copy
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_acceptance import assess
from run_upstream_matrix import command, validate, source_check

class PaperProtocolTests(unittest.TestCase):
    def test_tolerances_and_missing_results(self):
        for delta, status in [(3, 'pass'), (-3, 'pass'), (3.1, 'barely_acceptable'), (-5, 'barely_acceptable'), (5.1, 'fail')]:
            self.assertEqual(assess('vanilla_2b', 'ai2d', 76.9 + delta)['status'], status)
        self.assertEqual(assess('vanilla_2b', 'mmbench', 86.4, judged=True)['status'], 'pass')
        self.assertEqual(assess('vanilla_2b', 'mmbench', 86.5, judged=True)['status'], 'fail')
        self.assertEqual(assess('vanilla_2b', 'ai2d', None)['status'], 'missing')
        self.assertEqual(assess('vanilla_2b', 'ai2d', 76.9, smoke=True)['status'], 'smoke_only')

    def test_source_replay_has_no_training_overrides(self):
        source_check()
        paths = {'base': Path('/tmp/model')}
        cmd = command('base', 'mmstar', paths, Path('/tmp/result'))
        self.assertEqual(cmd[cmd.index('--max_new_tokens') + 1], '64')
        self.assertNotIn('--max_pixels', cmd)
        self.assertNotIn('--fixed_generation', cmd)
        self.assertNotIn('--processor_path', cmd)

    def test_reject_errors_duplicates_and_partial(self):
        payload = {'summary': {'model_path': '/tmp/model', 'total': 2, 'correct': 1, 'overall_acc': .5},
                   'results': [{'idx': 0, 'gold': 1, 'correct': True, 'error': None},
                               {'idx': 1, 'gold': 0, 'correct': False, 'error': None}]}
        self.assertEqual(validate(payload, 'ai2d', Path('/tmp/model'), 2)['score_percent'], 50)
        for mutation in ('error', 'duplicate', 'partial'):
            bad = copy.deepcopy(payload)
            if mutation == 'error': bad['results'][0]['error'] = 'OOM'
            if mutation == 'duplicate': bad['results'][1]['idx'] = 0
            if mutation == 'partial': bad['results'].pop()
            with self.assertRaises(ValueError): validate(bad, 'ai2d', Path('/tmp/model'), 2)

if __name__ == '__main__':
    unittest.main()
