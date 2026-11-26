"""Dispatch independent jobs to one serial worker per physical GPU."""
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from threading import Event


def gpu_indices(value):
    indices = value.split(',')
    if any(not index.isdigit() for index in indices):
        raise ValueError('GPU indices must be comma-separated nonnegative integers')
    indices = [str(int(index)) for index in indices]
    if len(set(indices)) != len(indices):
        raise ValueError('GPU indices must be unique')
    return indices


def run_gpu_queue(jobs, gpus, run_job):
    """Return failed job names; stop dispatching after an unexpected exception."""
    pending = Queue()
    for job in jobs:
        pending.put(job)
    stop = Event()

    def consume(gpu):
        failed = []
        try:
            while not stop.is_set():
                try:
                    job = pending.get_nowait()
                except Empty:
                    break
                failure = run_job(job, gpu)
                if failure is not None:
                    failed.append(failure)
        except BaseException:
            stop.set()
            raise
        return failed

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        try:
            results = list(pool.map(consume, gpus))
        except BaseException:
            stop.set()
            raise
    return [failure for failures in results for failure in failures]
