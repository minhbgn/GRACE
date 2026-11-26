"""Temporary VRAM holder; the parent owns its lifetime through a stdin pipe."""
import os
from pathlib import Path
import select
import subprocess
import sys


class PreparationGPUHold:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.process = None

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), '--worker'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                env=dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID'),
            )
            ready, _, _ = select.select([self.process.stdout], [], [], 60)
            if not ready or self.process.stdout.readline().strip() != 'READY':
                raise ValueError('GPU preparation holder failed to become ready within 60 seconds')
            print(f'[GPU-HOLD] Reserved GPUs {os.environ.get("CUDA_VISIBLE_DEVICES")} '
                  f'during CPU preparation (PID {self.process.pid}).', flush=True)
            return self
        except BaseException:
            self.release()
            raise

    def release(self):
        if self.process is None:
            return
        process, self.process = self.process, None
        try:
            # EOF releases the worker even if the parent exits unexpectedly.
            process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        finally:
            process.stdout.close()
        print('[GPU-HOLD] Released GPU memory.', flush=True)

    def __exit__(self, *exc):
        self.release()


def worker():
    import torch

    allocations = []
    if torch.cuda.device_count() != 2:
        raise ValueError('GPU holder requires two visible CUDA devices')
    for device in range(2):
        with torch.cuda.device(device):
            free, _ = torch.cuda.mem_get_info()
            remaining = free - 512 * 1024**2
            if remaining <= 0:
                raise ValueError(f'Insufficient free memory on visible GPU {device}')
            while remaining:
                size = min(remaining, 256 * 1024**2)
                allocations.append(torch.empty(size, dtype=torch.uint8, device=device))
                remaining -= size
            torch.cuda.synchronize(device)
    print('READY', flush=True)
    # No compute loop. Closing the parent's pipe ends the reservation.
    sys.stdin.read()


if __name__ == '__main__':
    if sys.argv[1:] != ['--worker']:
        raise SystemExit('Use run_training.sh --reserve-gpus to manage this worker')
    worker()
