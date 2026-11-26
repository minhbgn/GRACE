"""CPU-only checks of parallel dispatch, GPU isolation and failure collection."""
import sys
from pathlib import Path
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.protocols.gpu_job_queue import gpu_indices, run_gpu_queue


class GPUQueueTests(unittest.TestCase):
    def test_indices(self):
        self.assertEqual(gpu_indices('0,1'), ['0', '1'])
        self.assertEqual(gpu_indices('02'), ['2'])
        for value in ('', '0,', '-1', '0,0', '0,00', '0,x'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                gpu_indices(value)

    def test_concurrent_exactly_once_and_serial_per_gpu(self):
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        active = set()
        seen = []

        def run(job, gpu):
            with lock:
                self.assertNotIn(gpu, active)
                active.add(gpu)
            # Both GPU workers must be executing at once to pass.
            barrier.wait(timeout=5)
            with lock:
                seen.append((job, gpu))
                active.remove(gpu)
            return str(job) if job == 3 else None

        self.assertEqual(run_gpu_queue(range(6), ['0', '1'], run), ['3'])
        self.assertEqual(sorted(job for job, _ in seen), list(range(6)))
        self.assertEqual({gpu for _, gpu in seen}, {'0', '1'})

    def test_single_gpu_and_exception(self):
        seen = []
        def run(job, gpu):
            seen.append((job, gpu))
        self.assertEqual(run_gpu_queue(range(3), ['0'], run), [])
        self.assertEqual(seen, [(0, '0'), (1, '0'), (2, '0')])
        def broken(job, gpu):
            raise RuntimeError('GPU unavailable')
        with self.assertRaisesRegex(RuntimeError, 'GPU unavailable'):
            run_gpu_queue(range(3), ['0', '1'], broken)


if __name__ == '__main__':
    unittest.main()
