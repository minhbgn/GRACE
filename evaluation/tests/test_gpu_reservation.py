"""CPU checks for reservation cleanup and inference-worker handoff."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.protocols.gpu_reservation import GPUReservations, await_gpu_handoff, run_reserved_worker


class ReservationTests(unittest.TestCase):
    def test_release_closes_pipe_and_waits_for_context_exit(self):
        holds = GPUReservations()
        child = Mock()
        holds.processes['2'] = child
        holds.release('2')
        child.stdin.close.assert_called_once()
        child.wait.assert_called_once_with(timeout=10)
        self.assertEqual(holds.processes, {})
        holds.release('2')  # Idempotent cleanup.

    def test_ready_worker_releases_before_go(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            (attempt / 'GPU_READY').touch()
            child = Mock()
            child.poll.side_effect = [None, 0]
            child.wait.return_value = 0
            holds = Mock()
            holds.release.side_effect = lambda gpu: self.assertFalse((attempt / 'GPU_GO').exists())
            with patch('evaluation.protocols.gpu_reservation.subprocess.Popen', return_value=child), \
                 patch('evaluation.workers.video_runtime.ensure_gpu_idle') as idle:
                self.assertEqual(run_reserved_worker(['worker'], {}, None, attempt, '2', holds), 0)
            holds.release.assert_called_once_with('2')
            idle.assert_called_once_with('2')
            self.assertTrue((attempt / 'GPU_GO').exists())

    def test_failed_cpu_worker_does_not_release_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            child = Mock()
            child.poll.return_value = 1
            child.returncode = 1
            holds = Mock()
            with patch('evaluation.protocols.gpu_reservation.subprocess.Popen', return_value=child):
                self.assertEqual(run_reserved_worker(['worker'], {}, None, Path(directory), '0', holds), 1)
            holds.release.assert_not_called()

    def test_worker_detects_dead_controller(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('evaluation.protocols.gpu_reservation.os.kill', side_effect=ProcessLookupError):
            with self.assertRaises(ProcessLookupError):
                await_gpu_handoff(directory, 12345)
            self.assertTrue((Path(directory) / 'GPU_READY').exists())


if __name__ == '__main__':
    unittest.main()
