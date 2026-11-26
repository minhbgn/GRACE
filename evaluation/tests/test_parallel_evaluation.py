"""CPU-only scheduler lifecycle, admission, hot-add and crash recovery tests."""
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.protocols.evaluation_checkpoints import AnswerCheckpoint, append, records
from evaluation.protocols.parallel_evaluation import GIB, available_memory, run_pool
from evaluation.workers.parallel_video_worker import last_record


class CheckpointTests(unittest.TestCase):
    def test_restart_replays_answer_telemetry_and_sampling_then_continues(self):
        with tempfile.TemporaryDirectory() as root:
            old, new = [Path(root) / f'attempt_{i}' for i in range(2)]
            old.mkdir(); new.mkdir()
            checkpoint = AnswerCheckpoint(old, 'provenance')
            checkpoint.save('request0', '', dict(index=0), dict(id='a'))
            with (old / 'answers.checkpoint.jsonl').open('a') as stream:
                stream.write('{"torn":')
            restarted = AnswerCheckpoint(new, 'provenance')
            self.assertEqual(restarted.replay('request0'), '')
            self.assertIsNone(restarted.replay('request1'))
            restarted.save('request1', 'B', dict(index=1), dict(id='b'))
            self.assertEqual(len(records(new / 'answers.checkpoint.jsonl')), 2)
            self.assertEqual(last_record(new / 'telemetry.jsonl'), dict(index=0))
            with self.assertRaisesRegex(ValueError, 'provenance'):
                AnswerCheckpoint(new, 'different')

    def test_tampered_or_reordered_answers_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            old, new = Path(root) / 'attempt_0', Path(root) / 'attempt_1'
            old.mkdir(); new.mkdir()
            AnswerCheckpoint(old, 'same').save('question0', 'A', dict(index=0), {})
            restarted = AnswerCheckpoint(new, 'same')
            with self.assertRaisesRegex(ValueError, 'identity/order'):
                restarted.replay('question1')
            path = old / 'answers.checkpoint.jsonl'
            path.write_text(path.read_text().replace('question0', 'tampered'))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                records(path)

    def test_large_last_record(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'telemetry'
            row = dict(payload='x' * 30000)
            append(path, {'previous': True}); append(path, row)
            self.assertEqual(last_record(path), row)


class Simulation:
    def __init__(self, root, oom=False, pool=None, fail=False):
        self.root, self.oom, self.pool, self.fail = Path(root), oom, pool, fail
        self.tick, self.pid, self.peak = 0, 100, Counter()
        self.children, self.events, self.warnings, self.finished = [], [], [], []
        self.counts = Counter()
        self.processes = {}  # holder interface
        self.budgets = []

    def acquire(self, gpu):
        assert not any(p.gpu == gpu and p.returncode is None and p.admitted for p in self.children)
        self.events.append(('hold', gpu))
        self.processes[gpu] = type('Holder', (), dict(pid=1))()

    def release(self, gpu):
        self.events.append(('release', gpu))
        self.processes.pop(gpu, None)

    def add_gpu(self, gpu):
        self.acquire(gpu)

    def start(self, job, gpu, budget):
        self.budgets.append((gpu, budget))
        self.pid += 1
        self.counts[job] += 1
        attempt = self.root / f'{job}_{self.pid}'
        attempt.mkdir()
        (attempt / 'GPU_READY').touch()
        log = (attempt / 'run.log').open('w')
        sim = self

        class Process:
            pid = sim.pid
            returncode = None
            admitted = False

            def poll(self):
                if (attempt / 'GPU_GO').exists() and not self.admitted:
                    self.admitted = True
                    self.deadline = sim.tick + 16
                if self.admitted and sim.tick >= self.deadline and self.returncode is None:
                    self.returncode = 1 if sim.oom and job == '0' and sim.counts[job] == 1 else 0
                    if self.returncode:
                        log.write('torch.OutOfMemoryError: CUDA out of memory\n'); log.flush()
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

        process = Process()
        process.gpu, process.attempt = gpu, attempt
        self.children.append(process)
        self.events.append(('start', gpu, job))
        return dict(process=process, attempt=attempt, log=log)

    def finish(self, item, rc):
        self.finished.append((item['job'], rc))
        return item['job'] if rc else None

    def query(self, gpu):
        self.tick += 1
        if self.pool and self.tick == 7:
            self.pool.write_text('[0, 2]')
        live = [p for p in self.children if p.gpu == gpu and p.returncode is None and
                (p.attempt / 'GPU_GO').exists()]
        self.peak[gpu] = max(self.peak[gpu], len(live))
        if self.fail and self.tick > 8:
            raise KeyboardInterrupt
        if gpu in self.processes:
            return dict(total=40*GIB, free=GIB//2, processes={1:39*GIB})
        return dict(total=40*GIB, free=(40-4*len(live))*GIB,
                    processes={p.pid:4*GIB for p in live})

    def warn(self, message, **kwargs):
        self.warnings.append(message)

    def run(self, jobs=7, budget=10, parallel=3):
        return run_pool(list(map(str, range(jobs))), ['0'], parallel, 2*GIB,
            self, self.add_gpu, self.start, self.finish, lambda job:budget*GIB,
            self.warn, pool_file=self.pool, query=self.query, interval=0)


class SchedulerTests(unittest.TestCase):
    def test_sequential_worker_gets_full_gpu_budget_and_holds_release(self):
        with tempfile.TemporaryDirectory() as root:
            sim = Simulation(root)
            self.assertEqual(sim.run(jobs=3, budget=18, parallel=1), [])
            self.assertEqual(sim.peak['0'], 1)
            self.assertEqual(sim.budgets, [('0', 38*GIB)] * 3)
            self.assertTrue(all(rc == 0 for _, rc in sim.finished))
            self.assertEqual(sum(event == ('release', '0') for event in sim.events), 3)

    def test_sequential_handoff_tolerates_driver_memory_overhead(self):
        with tempfile.TemporaryDirectory() as root:
            sim = Simulation(root)
            original = sim.query

            def query(gpu):
                state = original(gpu)
                state['free'] -= GIB
                return state

            sim.query = query
            self.assertEqual(sim.run(jobs=2, parallel=1), [])
            self.assertEqual(sim.budgets, [('0', 38*GIB)] * 2)
            self.assertFalse(any('serial job budget exceeds' in message for message in sim.warnings))

    def test_sequential_budget_uses_each_gpu_capacity_and_headroom(self):
        with tempfile.TemporaryDirectory() as root:
            sim = Simulation(root)
            original = sim.query

            def query(gpu):
                state = original(gpu)
                if gpu == '2':
                    state['total'] = 80*GIB
                    state['free'] += 40*GIB
                return state

            failed = run_pool(list(map(str, range(6))), ['0', '2'], 1, 8*GIB,
                sim, sim.add_gpu, sim.start, sim.finish,
                lambda job: self.fail('Sequential mode must not consult shared-worker estimates'),
                sim.warn, query=query, interval=0)
            self.assertEqual(failed, [])
            self.assertEqual(set(sim.budgets), {('0', 32*GIB), ('2', 72*GIB)})
            self.assertEqual(sim.peak['0'], 1)
            self.assertEqual(sim.peak['2'], 1)

    def test_three_workers_share_one_gpu_without_reacquiring_sibling_hold(self):
        with tempfile.TemporaryDirectory() as root:
            sim = Simulation(root)
            self.assertEqual(sim.run(), [])
            self.assertEqual(sim.peak['0'], 3)
            self.assertTrue(all(n == 1 for n in sim.counts.values()))
            self.assertEqual(len(sim.finished), 7)

    def test_memory_budget_limits_actual_concurrency_and_warns(self):
        with tempfile.TemporaryDirectory() as root:
            sim = Simulation(root)
            self.assertEqual(sim.run(budget=18), [])
            self.assertEqual(sim.peak['0'], 2)
            self.assertTrue(all(budget == 18*GIB for _, budget in sim.budgets))
            self.assertTrue(any('memory admission limited to 2/3' in w for w in sim.warnings))

    def test_cuda_oom_reduces_limit_and_requeues(self):
        with tempfile.TemporaryDirectory() as root:
            sim = Simulation(root, oom=True)
            self.assertEqual(sim.run(), [])
            self.assertEqual(sim.counts['0'], 2)
            self.assertTrue(any('CUDA OOM' in w and '3 -> 2' in w for w in sim.warnings))

    def test_pool_file_adds_gpu_without_stopping_existing_workers(self):
        with tempfile.TemporaryDirectory() as root:
            pool = Path(root) / 'pool.json'
            pool.write_text('[0]')
            sim = Simulation(root, pool=pool)
            self.assertEqual(sim.run(jobs=12), [])
            self.assertGreater(sim.peak['0'], 0)
            self.assertGreater(sim.peak['2'], 0)
            self.assertTrue(all(rc == 0 for _, rc in sim.finished))

    def test_interrupt_terminates_all_owned_children(self):
        with tempfile.TemporaryDirectory() as root:
            sim = Simulation(root, fail=True)
            with self.assertRaises(KeyboardInterrupt):
                sim.run()
            self.assertTrue(all(p.returncode is not None for p in sim.children))

    def test_serial_default_and_non_cuda_failures_are_not_retried(self):
        with tempfile.TemporaryDirectory() as root:
            sim = Simulation(root)
            original = sim.finish

            def fail_one(item, rc):
                original(item, rc)
                return item['job'] if item['job'] == '0' else None

            sim.finish = fail_one
            self.assertEqual(sim.run(jobs=3, parallel=1), ['0'])
            self.assertEqual(sim.peak['0'], 1)
            self.assertTrue(all(n == 1 for n in sim.counts.values()))

    def test_invalid_pool_update_keeps_workers_running(self):
        with tempfile.TemporaryDirectory() as root:
            pool = Path(root) / 'pool.json'
            pool.write_text('[0]')
            sim = Simulation(root)
            original = sim.query

            def query(gpu):
                state = original(gpu)
                if sim.tick == 5:
                    pool.write_text('{broken')
                return state

            failed = run_pool(list(map(str, range(5))), ['0'], 3, 2*GIB,
                sim, sim.add_gpu, sim.start, sim.finish, lambda job:10*GIB,
                sim.warn, pool_file=pool, query=query, interval=0)
            self.assertEqual(failed, [])
            self.assertEqual(sum('Invalid GPU pool' in w for w in sim.warnings), 1)

    def test_removing_gpu_drains_workers_before_releasing_lock(self):
        with tempfile.TemporaryDirectory() as root:
            pool = Path(root) / 'pool.json'
            pool.write_text('[0,2]')
            sim = Simulation(root)
            original = sim.query
            released = []

            def query(gpu):
                state = original(gpu)
                if sim.tick == 18:
                    pool.write_text('[0]')
                return state

            def release(gpu):
                self.assertFalse(any(p.gpu == gpu and p.returncode is None for p in sim.children))
                self.assertNotIn(gpu, sim.processes)
                released.append(gpu)

            failed = run_pool(list(map(str, range(15))), ['0'], 3, 2*GIB,
                sim, sim.add_gpu, sim.start, sim.finish, lambda job:10*GIB,
                sim.warn, pool_file=pool, query=query, interval=0, release_gpu=release)
            self.assertEqual(failed, [])
            self.assertEqual(released, ['2'])
            self.assertGreater(sim.peak['2'], 0)

    def test_budget_accounts_for_future_peaks(self):
        child = type('Child', (), dict(pid=123))()
        state = dict(free=30*GIB, processes={123:2*GIB})
        self.assertEqual(available_memory(state, [dict(process=child, admitted=True, estimate=10*GIB)]), 22*GIB)


if __name__ == '__main__':
    unittest.main()
