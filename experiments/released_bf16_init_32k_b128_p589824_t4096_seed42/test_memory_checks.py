"""CPU regressions for recipe-aware cache reuse; no GPU/model imports needed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import run_memory_checks as runner


class MemoryCacheTests(unittest.TestCase):
    def test_fingerprint_tracks_sequence_budget_and_accumulation(self):
        old = dict(config=dict(model_max_length=2048, max_pixels=589824), diagnostic_accumulation=2)
        new = copy.deepcopy(old)
        new['config']['model_max_length'] = 4096
        self.assertNotEqual(runner.fingerprint(old), runner.fingerprint(new))
        new = copy.deepcopy(old)
        new['diagnostic_accumulation'] = 1
        self.assertNotEqual(runner.fingerprint(old), runner.fingerprint(new))
        self.assertEqual(runner.fingerprint(old), runner.fingerprint(copy.deepcopy(old)))

    def fixture(self, work):
        out = work / 'profiles/grace'
        out.mkdir(parents=True)
        runner.write(out / 'run_summary.json', dict(complete=True, profile=True, global_step=3))
        runner.write(work / 'input_envelope.json', dict(sequence_tokens=4096, visual_tokens=576))
        (out / 'EXPORT_COMPLETE').write_text('complete\n')
        (out / 'model.safetensors').write_bytes(b'fixture')
        names = ['profiles/grace/run_summary.json', 'profiles/grace/EXPORT_COMPLETE', 'input_envelope.json']
        data = dict(fingerprint='current', method='grace', bits=4, complete=True,
                    envelope=runner.read(work / 'input_envelope.json'),
                    evidence_sha256={name: runner.sha(work / name) for name in names})
        path = work / 'check_result.json'
        runner.write(path, data)
        return path

    def test_reuse_only_matching_complete_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.fixture(Path(folder))
            self.assertIsNotNone(runner.valid_cached(path, 'current', 'grace', 4))
            for key, method, bits in [('old', 'grace', 4), ('current', 'qat', 4), ('current', 'grace', 8)]:
                self.assertIsNone(runner.valid_cached(path, key, method, bits))
            (path.parent / 'profiles/grace/EXPORT_COMPLETE').unlink()
            self.assertIsNone(runner.valid_cached(path, 'current', 'grace', 4))

    def test_tampering_and_missing_weights_invalidate(self):
        for change in ('summary', 'weights', 'failed'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                path = self.fixture(Path(folder))
                if change == 'summary':
                    runner.write(path.parent / 'profiles/grace/run_summary.json', dict(complete=False, profile=True, global_step=1))
                elif change == 'weights':
                    (path.parent / 'profiles/grace/model.safetensors').unlink()
                else:
                    data = runner.read(path)
                    data['complete'] = False
                    runner.write(path, data)
                self.assertIsNone(runner.valid_cached(path, 'current', 'grace', 4))


if __name__ == '__main__':
    unittest.main()
