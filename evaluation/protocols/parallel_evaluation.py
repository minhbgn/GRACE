"""One controller owns GPU locks and holders, with bounded per-GPU admission."""
from collections import deque
import json
import os
from pathlib import Path
import signal
import subprocess
import time

GIB = 1024 ** 3


def memory_state(gpu):
    def query(flag):
        return subprocess.check_output(['nvidia-smi', '-i', str(gpu), flag,
            '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
    total, free = map(int, query('--query-gpu=memory.total,memory.free').split(','))
    processes = {}
    for line in query('--query-compute-apps=pid,used_memory').splitlines():
        pid, used = map(int, line.split(','))
        processes[pid] = used * 1024 ** 2
    return dict(total=total * 1024 ** 2, free=free * 1024 ** 2, processes=processes)


def available_memory(state, active):
    """Budget for future peaks, not just the instantaneous nvidia-smi usage."""
    return state['free'] - sum(max(0, item['estimate'] -
        state['processes'].get(item['process'].pid, 0)) for item in active if item['admitted'])


def cuda_oom(path):
    with Path(path).open('rb') as stream:
        stream.seek(max(0, os.fstat(stream.fileno()).st_size - 131072))
        tail = stream.read().decode(errors='replace').lower()
    return 'cuda out of memory' in tail or 'cuda error: out of memory' in tail or 'torch.outofmemoryerror' in tail


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_pool(jobs, initial_gpus, parallel, headroom, reservations, acquire_gpu,
             start, finish, estimate, warn, pool_file=None, query=memory_state, interval=.2,
             release_gpu=None):
    """Callbacks do protocol-specific work; this loop alone mutates holders/queues.

    start returns a dict with process/attempt/log, finish persists the result and
    returns a failure name or None. New pool-file GPUs only receive pending jobs.
    With parallel=1, the sole worker receives the card's total memory minus
    headroom as its scheduling budget. These budgets never cap CUDA allocations.
    """
    pending = deque(jobs)
    devices, active, failed = {}, [], []
    last_pool, last_retry, last_pool_error = None, 0, None
    deferred_addition = False
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise KeyboardInterrupt('Evaluation controller terminated')

    signal.signal(signal.SIGTERM, terminate)
    try:
        while pending or active:
            desired = list(initial_gpus)
            if pool_file:
                try:
                    desired = json.loads(Path(pool_file).read_text())
                    if not isinstance(desired, list) or not desired or any(
                            isinstance(gpu, bool) or not str(gpu).isdigit() for gpu in desired):
                        raise ValueError('Expected a nonempty JSON list of physical GPU indices')
                    desired = [str(int(gpu)) for gpu in desired]
                    if len(set(desired)) != len(desired):
                        raise ValueError('GPU indices must be unique')
                    last_pool_error = None
                except (OSError, ValueError, TypeError) as error:
                    if str(error) != last_pool_error:
                        warn(f'Invalid GPU pool file; retaining previous pool: {error}')
                        last_pool_error = str(error)
                    desired = last_pool or list(initial_gpus)
                # Removing a GPU drains it; it never interrupts its workers.
            # Numerical gates may compile for minutes. Complete any existing
            # CPU/CUDA handoff before gating a newly added GPU, so the ready
            # worker does not hit the pinned handoff's 120-second deadline.
            handoff_pending = any(not item['admitted'] for item in active)
            if not handoff_pending and (desired != last_pool or deferred_addition or
                                        time.monotonic() - last_retry > 30):
                for gpu in desired:
                    if gpu not in devices and pending:
                        try:
                            acquire_gpu(gpu)
                        except (RuntimeError, BlockingIOError, subprocess.SubprocessError) as error:
                            warn(f'GPU {gpu} unavailable; continuing current workers: {error}')
                            continue
                        devices[gpu] = dict(limit=parallel, last_warning=0)
                        warn(f'GPU {gpu} joined pool; requested evaluations={parallel}', level='INFO')
                deferred_addition = False
                last_retry = time.monotonic()
            elif desired != last_pool:
                deferred_addition = any(gpu not in devices for gpu in desired)
            last_pool = desired
            if not devices and not active:
                raise RuntimeError('No requested GPU is available; no evaluation was started')

            for item in list(active):
                process, gpu = item['process'], item['gpu']
                rc = process.poll()
                if rc is None:
                    continue
                active.remove(item)
                item['log'].close()
                failure = finish(item, rc)
                if failure:
                    limit = devices[gpu]['limit']
                    if rc and cuda_oom(item['attempt'] / 'run.log') and limit > 1:
                        devices[gpu]['limit'] = limit - 1
                        pending.appendleft(item['job'])
                        warn(f'GPU {gpu} CUDA OOM: parallel evaluations {limit} -> {limit-1}; '
                             f'requeue {failure} using durable answer checkpoints')
                    else:
                        failed.append(failure)

            for gpu, device in list(devices.items()):
                siblings = [item for item in active if item['gpu'] == gpu]
                if gpu not in desired and not siblings:
                    reservations.release(gpu)
                    if release_gpu is not None:
                        release_gpu(gpu)
                    del devices[gpu]
                    warn(f'GPU {gpu} drained and released from pool', level='INFO')
                    continue
                # Refill the full-card hold only when ALL siblings have exited.
                if not siblings and pending and gpu in desired and gpu not in reservations.processes:
                    reservations.acquire(gpu)
                state = query(gpu)
                allowed = {item['process'].pid for item in siblings}
                if gpu in reservations.processes:
                    allowed.add(reservations.processes[gpu].pid)
                foreign = set(state['processes']) - allowed
                if foreign:
                    raise RuntimeError(f'GPU {gpu}: unexpected GPU processes {sorted(foreign)}; stopping owned workers')
                running = [item for item in siblings if item['admitted']]
                loading = [item for item in siblings if not item['admitted']]
                free = available_memory(state, running)
                if running and free < headroom and device['limit'] > len(running):
                    old = device['limit']
                    device['limit'] = max(1, len(running))
                    warn(f'GPU {gpu} memory pressure: parallel evaluations {old} -> {device["limit"]}; '
                         f'projected free={free/GIB:.2f} GiB, headroom={headroom/GIB:.2f} GiB')
                for item in loading:
                    if (item['attempt'] / 'GPU_READY').exists():
                        held = gpu in reservations.processes
                        if held:
                            reservations.release(gpu)
                            state = query(gpu)
                            if state['processes']:
                                raise RuntimeError(f'GPU {gpu} occupied during first handoff: {state}')
                            free = available_memory(state, running)
                        if free < item['estimate'] + headroom and running:
                            # Avoid the worker's 120s handoff timeout: defer CPU loading
                            # until running siblings drain. No answer progress is lost.
                            stop_process(item['process'])
                            item['log'].close()
                            active.remove(item)
                            finish(item, item['process'].returncode)
                            pending.appendleft(item['job'])
                            device['limit'] = max(1, len(running))
                            warn(f'GPU {gpu} handoff deferred for memory; parallel evaluations -> {device["limit"]}')
                            continue
                        # Exclusive mode budgets the whole device. Driver/context
                        # usage can make free < total without any worker contention.
                        if parallel > 1 and free < item['estimate'] + headroom:
                            warn(f'GPU {gpu} serial job budget exceeds free memory; trying one worker, '
                                 f'budget={item["estimate"]/GIB:.2f} GiB free={free/GIB:.2f} GiB')
                        item['admitted'] = True
                        (item['attempt'] / 'GPU_GO').touch()
                    elif time.monotonic() - item['started'] > 1800:
                        raise RuntimeError(f'GPU {gpu} CPU-loading timeout: {item["attempt"]}')
                siblings = [item for item in active if item['gpu'] == gpu]
                if not pending or gpu not in desired or len(siblings) >= device['limit'] or loading:
                    continue
                held = gpu in reservations.processes
                free = available_memory(state, siblings)
                # A large pending configuration must not block smaller jobs that fit.
                choice = next((index for index, job in enumerate(pending)
                    if not siblings or estimate(job) + headroom <= free), None)
                job = pending[0 if choice is None else choice]
                # A sequential worker owns this GPU; shared-worker estimates
                # are irrelevant even if the fallback is smaller than its peak.
                budget = max(0, state['total'] - headroom) if parallel == 1 else estimate(job)
                if choice is None:
                    message = (f'GPU {gpu} memory admission limited to {len(siblings)}/{parallel}: '
                               f'next budget={budget/GIB:.2f} GiB, projected free={free/GIB:.2f} GiB, '
                               f'headroom={headroom/GIB:.2f} GiB; pending job waits')
                    if time.monotonic() - device['last_warning'] > 30 or not device['last_warning']:
                        warn(message)
                        device['last_warning'] = time.monotonic()
                    continue
                if not held and not siblings:
                    raise RuntimeError(f'GPU {gpu}: idle dispatch requires its preparation hold')
                del pending[choice]
                item = start(job, gpu, budget)
                item.update(gpu=gpu, job=job, estimate=budget, admitted=False, started=time.monotonic())
                active.append(item)
            time.sleep(interval)
    finally:
        # Signal all children first so they stop promptly, then wait for CUDA exit.
        for item in active:
            if item['process'].poll() is None:
                item['process'].terminate()
        for item in active:
            stop_process(item['process'])
            item['log'].close()
        signal.signal(signal.SIGTERM, previous_handler)
    return failed
