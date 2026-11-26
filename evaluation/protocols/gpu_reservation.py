"""Best-effort VRAM reservations with a CPU-load-to-CUDA handoff."""
import os
from pathlib import Path
import select
import subprocess
import sys
import time


class GPUReservations:
    def __init__(self):
        self.processes = {}

    def acquire(self, gpu):
        from evaluation.workers.video_runtime import ensure_gpu_idle
        ensure_gpu_idle(gpu)
        env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu)
        for key in ('PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF'):
            env.pop(key, None)
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--holder'],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
        self.processes[gpu] = process
        try:
            ready, _, _ = select.select([process.stdout], [], [], 60)
            if not ready or process.stdout.readline().strip() != 'READY':
                raise RuntimeError(f'GPU {gpu} reservation failed')
        except BaseException:
            self.release(gpu)
            raise
        print(f'GPU-HOLD {gpu}: reserved during CPU setup (PID {process.pid})', flush=True)

    def release(self, gpu):
        process = self.processes.pop(gpu, None)
        if process is None:
            return
        try:
            process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        finally:
            process.stdout.close()
        print(f'GPU-HOLD {gpu}: released', flush=True)

    def close(self):
        for gpu in list(self.processes):
            self.release(gpu)


def await_gpu_handoff(attempt, parent_pid):
    """Called by the inference worker after CPU loading/packing, before model.to."""
    attempt = Path(attempt)
    (attempt / 'GPU_READY').touch()
    deadline = time.monotonic() + 120
    while not (attempt / 'GPU_GO').exists():
        os.kill(parent_pid, 0)  # Fail if the controlling process disappeared.
        if time.monotonic() > deadline:
            raise RuntimeError('Timed out waiting for GPU reservation handoff')
        time.sleep(.05)


def run_reserved_worker(command, env, log, attempt, gpu, reservations):
    """Keep VRAM occupied until the worker is ready to move its model to CUDA."""
    from evaluation.workers.video_runtime import ensure_gpu_idle
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        while process.poll() is None:
            if (attempt / 'GPU_READY').exists():
                reservations.release(gpu)
                ensure_gpu_idle(gpu)
                (attempt / 'GPU_GO').touch()
                return process.wait()
            time.sleep(.05)
        return process.returncode
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def holder():
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Reservation worker expects one visible GPU')
    allocations = []
    free, _ = torch.cuda.mem_get_info()
    remaining = free - 512 * 1024**2
    if remaining <= 0:
        raise RuntimeError('Insufficient free memory for GPU reservation')
    while remaining:
        size = min(remaining, 256 * 1024**2)
        allocations.append(torch.empty(size, dtype=torch.uint8, device='cuda:0'))
        remaining -= size
    torch.cuda.synchronize()
    print('READY', flush=True)
    sys.stdin.read()  # EOF on controller exit releases VRAM, including on a crash.


if __name__ == '__main__':
    if sys.argv[1:] != ['--holder']:
        raise SystemExit('Use run_experiment.sh run --reserve-gpus')
    holder()
