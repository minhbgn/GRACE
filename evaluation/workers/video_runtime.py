"""CPU-only supervision and durable progress for video inference workers."""
from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from evaluation.protocols.video_protocol import digest
from evaluation.runners.run_video_matrix import read, summary_from_rows, write


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def host_memory():
    """The child inherits these cgroups; retain counters before and after death."""
    result = {}
    try:
        result['meminfo'] = Path('/proc/meminfo').read_text()
        groups = Path('/proc/self/cgroup').read_text()
        result['cgroups'] = groups
        for line in groups.splitlines():
            _, controllers, path = line.split(':', 2)
            if 'memory' in controllers.split(','):
                base = Path('/sys/fs/cgroup/memory') / path.lstrip('/')
                names = ('memory.oom_control', 'memory.failcnt', 'memory.limit_in_bytes',
                         'memory.usage_in_bytes', 'memory.max_usage_in_bytes')
            elif line.startswith('0::'):
                base = Path('/sys/fs/cgroup') / path.lstrip('/')
                names = ('memory.events', 'memory.max', 'memory.current', 'memory.peak')
            else:
                continue
            for name in names:
                file = base / name
                if file.exists():
                    result[str(file)] = file.read_text()
    except OSError as error:
        result['error'] = str(error)
    return result


def stage(attempt, name, question=None, **details):
    """Persist before each expensive phase, including before importing Torch."""
    data = dict(time=now(), pid=os.getpid(), stage=name, id=question, **details)
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        data['cuda'] = dict(allocated=torch.cuda.memory_allocated(),
                            reserved=torch.cuda.memory_reserved(),
                            peak_allocated=torch.cuda.max_memory_allocated())
    write(attempt / 'status.json', data)
    with (attempt / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(data) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(f'STAGE {name} id={question}', flush=True)


def gpu_state(gpu):
    def query(fields):
        return subprocess.check_output(
            ['nvidia-smi', '-i', str(gpu), *fields, '--format=csv,noheader,nounits'],
            text=True, stderr=subprocess.STDOUT, timeout=15).strip()
    return dict(device=query(['--query-gpu=index,uuid,name,memory.used,memory.total']),
                processes=query(['--query-compute-apps=pid,process_name,used_memory']))


def ensure_gpu_idle(gpu):
    state = gpu_state(gpu)
    fields = [x.strip() for x in state['device'].split(',')]
    if len(fields) != 5 or 'Display' in fields[2]:
        raise RuntimeError(f'Expected one compute GPU in PCI order: {state}')
    if state['processes'] or int(fields[3]) > 256:
        raise RuntimeError(f'GPU {gpu} is occupied; choose a free GPU. {state}')
    return state


@contextmanager
def gpu_lock(gpu):
    """Cooperating evaluator processes cannot race across output directories."""
    path = Path('/tmp') / f'grace-video-gpu-{os.getuid()}-{gpu}.lock'
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another resilient video matrix holds GPU {gpu}: {path}') from error
        yield


def prediction_prefix(path):
    """Keep complete rows; only an incomplete final write may be discarded."""
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith('\n'):
                break
            raise ValueError(f'Corrupt prediction record {index + 1}: {path}')
    return rows


def validate_prefix(rows, job):
    if len(rows) > len(job['rows']):
        raise ValueError('Resume population exceeds job population')
    for row, source in zip(rows, job['rows']):
        if row['id'] != source['id'] or any(row[k] != source[k] for k in
                ('gold', 'options', 'question_type', 'category')):
            raise ValueError('Resume is not an ordered prefix of this population')
        if row['source_identity'] != digest(source):
            raise ValueError('Resume source identity changed')
        counts = {k: row['tokens'][k] for k in
                  ('input_visual_tokens', 'input_text_tokens', 'input_tokens')}
        if row['input_identity'] != digest(dict(source=source, sampling=row['sampling'],
                tokens=counts, input_ids_sha256=row['input_ids_sha256'])):
            raise ValueError('Resume input identity changed')
        if row['generation']['max_new_tokens'] != job['protocol']['max_new_tokens']:
            raise ValueError('Resume generation allowance changed')
        if row['tokens']['input_tokens'] + job['protocol']['max_new_tokens'] > row['context_limit']:
            raise ValueError('Resume context overflow')
    if rows and summary_from_rows(rows)['generation_cutoffs']:
        raise ValueError('Resume contains a non-EOS generation')
    return rows


def seed_attempt(job_dir, attempt, job):
    """Only reuse rows from this pinned run, never import a historical protocol."""
    previous = sorted(p for p in job_dir.glob('attempt_*') if p != attempt)
    rows, source = [], None
    for candidate in reversed(previous):
        other = read(candidate / 'job.json')
        if any(other.get(k) != job.get(k) for k in
               ('method', 'suite', 'model_path', 'protocol', 'rows', 'run_identity')):
            raise ValueError(f'Resume provenance mismatch: {candidate}')
        records = validate_prefix(prediction_prefix(candidate / 'predictions.jsonl'), job)
        if len(records) > len(rows):
            rows, source = records, candidate
    with (attempt / 'predictions.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    write(attempt / 'resume.json', dict(predictions=len(rows), source=str(source) if source else None))
    return len(rows)


def supervise(attempt, command, env):
    before = host_memory()
    started = now()
    with (attempt / 'run.log').open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        write(attempt / 'worker.json', dict(pid=process.pid, started=started,
              allocator=env['PYTORCH_CUDA_ALLOC_CONF'], host_memory=before))
        def terminate_requested(signum, frame):
            raise SystemExit(128 + signum)

        old_term = signal.signal(signal.SIGTERM, terminate_requested)
        try:
            process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            signal.signal(signal.SIGTERM, old_term)
            code = process.returncode
            status = read(attempt / 'status.json') if (attempt / 'status.json').exists() else None
            write(attempt / 'exit.json', dict(pid=process.pid, started=started, ended=now(),
                  returncode=code, signal=signal.Signals(-code).name if code is not None and code < 0 else None,
                  last_status=status, host_memory_before=before, host_memory_after=host_memory()))
    if code and not (attempt / 'failure.json').exists():
        sig = signal.Signals(-code).name if code < 0 else None
        write(attempt / 'failure.json', dict(id=status.get('id') if status else None,
              error=f'Worker terminated: returncode={code}, signal={sig}; see exit.json',
              returncode=code, signal=sig))
    return code


def run_pair(job_dir, job, env, gpu, sigkill_retries=1, worker=None):
    """Bounded SIGKILL recovery; preserve the successful prefix in a new attempt."""
    worker = worker or Path(__file__).with_name('video_worker_resilient.py')
    for retry in range(sigkill_retries + 1):
        state = ensure_gpu_idle(gpu)
        attempt = job_dir / ('attempt_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
        attempt.mkdir(parents=True)
        write(attempt / 'job.json', job)
        write(attempt / 'gpu_before.json', state)
        resumed = seed_attempt(job_dir, attempt, job)
        print(f'Running {job["method"]}/{job["suite"]}; resume={resumed}; log: {attempt / "run.log"}', flush=True)
        code = supervise(attempt, [sys.executable, str(worker), str(attempt / 'job.json')], env)
        if code != -signal.SIGKILL or retry == sigkill_retries:
            return attempt, code
        print(f'SIGKILL: saved exit.json; retry {retry + 1}/{sigkill_retries} from validated progress', flush=True)
