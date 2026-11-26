"""CPU checks for reservation lifetime and launcher handoff."""
from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'qwen-vl-finetune/scripts'))
import gpu_preparation_hold as hold
import run_constrained_matrix as matrix


class GPUHoldTests(unittest.TestCase):
    def test_disabled_never_starts_worker(self):
        with patch.object(hold.subprocess, 'Popen') as spawn:
            with hold.PreparationGPUHold():
                pass
        spawn.assert_not_called()

    def test_worker_is_reaped_on_success_and_preparation_failure(self):
        real_popen = subprocess.Popen
        for fail in (False, True):
            with self.subTest(fail=fail):
                # A real pipe-based child exercises EOF cleanup without CUDA.
                def spawn(*args, **kwargs):
                    return real_popen([sys.executable, '-c',
                                       'import sys; print("READY", flush=True); sys.stdin.read()'],
                                      **kwargs)
                reservation = hold.PreparationGPUHold(True)
                with patch.object(hold.subprocess, 'Popen', side_effect=spawn), redirect_stdout(StringIO()):
                    try:
                        with reservation:
                            child = reservation.process
                            self.assertIsNone(child.poll())
                            if fail:
                                raise RuntimeError('preparation failed')
                            reservation.release()
                    except RuntimeError:
                        self.assertTrue(fail)
                self.assertIsNotNone(child.poll())
                self.assertIsNone(reservation.process)

    def test_failed_start_is_cleaned_up(self):
        real_popen = subprocess.Popen
        children = []
        def spawn(*args, **kwargs):
            child = real_popen([sys.executable, '-c', 'raise SystemExit(1)'], **kwargs)
            children.append(child)
            return child
        with patch.object(hold.subprocess, 'Popen', side_effect=spawn), redirect_stdout(StringIO()):
            with self.assertRaisesRegex(ValueError, 'failed to become ready'):
                with hold.PreparationGPUHold(True):
                    self.fail('failed worker was accepted')
        self.assertIsNotNone(children[0].poll())

    def test_dry_run_does_not_reserve_or_check_gpus(self):
        config = ROOT / 'experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/experiment.json'
        with patch.object(sys, 'argv', ['matrix', '--config', str(config), 'all',
                                       '--train', '--reserve-gpus', '--dry-run']), \
                patch.object(matrix, 'PreparationGPUHold') as reservation, \
                patch.object(matrix.shared, 'check_gpus') as check, redirect_stdout(StringIO()):
            matrix.main()
        reservation.assert_not_called()
        check.assert_not_called()

    def test_release_precedes_training_and_prepare_only_releases_on_exit(self):
        config = ROOT / 'experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/experiment.json'
        for mode in ('--train', '--prepare-only'):
            events = []
            class FakeHold:
                def __init__(self, enabled):
                    assert enabled
                def __enter__(self):
                    events.append('reserve')
                    return self
                def release(self):
                    events.append('release')
                def __exit__(self, *exc):
                    self.release()
            def launch(*args, **kwargs):
                self.assertIn('release', events)
                events.append('launch')
            def prepare(*args):
                self.assertEqual(events, ['reserve'])
                events.append('prepare')
                return {'records_sha256': 'fixture'}
            with tempfile.TemporaryDirectory() as tmp, \
                    patch.dict(os.environ, {'EXPERIMENT_DIR': tmp}), \
                    patch.object(sys, 'argv', ['matrix', '--config', str(config), 'all', mode, '--reserve-gpus']), \
                    patch.object(matrix, 'PreparationGPUHold', FakeHold), \
                    patch.object(matrix.shared, 'check_gpus'), \
                    patch.object(matrix.Experiment, 'provenance', return_value={'fixture': True}), \
                    patch.object(matrix, 'check_processor_compatibility'), \
                    patch.object(matrix, 'prepare', side_effect=prepare), \
                    patch.object(matrix.shared, 'comparison_recipe', return_value={'implementation': {}}), \
                    patch.object(matrix.shared, 'launch_run', side_effect=launch), \
                    patch.object(matrix.subprocess, 'run'), redirect_stdout(StringIO()):
                matrix.main()
            self.assertEqual(events[-1], 'release')
            self.assertEqual(events.count('launch'), 4 if mode == '--train' else 0)


if __name__ == '__main__':
    unittest.main()
