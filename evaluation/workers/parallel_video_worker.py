"""Add durable answer replay around the unchanged, pinned video adapter."""
import json
import os
import threading
import time
from pathlib import Path
import sys
from unittest.mock import patch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.protocols.evaluation_checkpoints import AnswerCheckpoint, digest


def last_record(path):
    # Telemetry can be large; avoid rereading the entire prefix on every question.
    with path.open('rb') as stream:
        end = stream.seek(0, 2)
        size = min(end, 8192)
        while size:
            stream.seek(end - size)
            lines = stream.read(size).splitlines()
            if len(lines) > 1 or size == end:
                return json.loads(lines[-1])
            size = min(end, size * 2)
    raise ValueError(f'Empty per-question record: {path}')


def checkpoint_adapter(adapter, checkpoint, generator, attempt):
    """Replay answers before video decoding, while the harness still scores all rows."""
    generate = adapter.generate_until

    def generate_until(requests):
        answers = []
        for req in requests:
            prompt, _, _, doc_id, task_name, split = req.args
            doc = adapter.task_dict[task_name][split][doc_id]
            key = digest(dict(prompt=prompt, doc_id=doc_id, task=task_name,
                              split=split, source=doc['_grace_row']))
            answer = checkpoint.replay(key)
            if answer is not None:
                generator.index += 1
            else:
                answer = generate([req])[0]
                telemetry = last_record(attempt / 'telemetry.jsonl')
                video_input = last_record(attempt / 'video_inputs.jsonl')
                checkpoint.save(key, answer, telemetry, video_input)
            answers.append(answer)
        return answers

    adapter.generate_until = generate_until


def run(job_path):
    from evaluation.runners import run_token_reduction_video_retry as retry
    from evaluation.runners import run_token_reduction_matrix as matrix
    from evaluation.protocols.video_memory_runtime import MemoryMeasuredGenerator
    job_path = Path(job_path)
    job = matrix.read(job_path)
    # Check sources again in the child before trusting saved answers.
    for filename, checksum in job['extension_code'].items():
        if matrix.sha(matrix.ROOT / filename) != checksum:
            raise ValueError(f'Parallel evaluation source changed: {filename}')
    identity = digest({key: job[key] for key in (
        'model', 'model_path', 'suite', 'method', 'retention', 'policy', 'execution',
        'sidecar_sha256', 'limit', 'resume_provenance', 'extension_code')})
    def watch_parent():
        while True:
            try:
                os.kill(job['controller_pid'], 0)
            except ProcessLookupError:
                os._exit(143)  # Release CUDA even if the controller was SIGKILLed.
            time.sleep(1)

    threading.Thread(target=watch_parent, daemon=True).start()
    checkpoint = AnswerCheckpoint(job_path.parent, identity)
    retry.install_worker(job['execution'])
    state = {}

    class Generator(MemoryMeasuredGenerator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            state['generator'] = self

    matrix.MeasuredGenerator = Generator
    # source_check establishes the pinned harness's import precedence.
    matrix.source_check()
    from lmms_eval import evaluator
    evaluate = evaluator.simple_evaluate

    def resumable(*args, **kwargs):
        checkpoint_adapter(kwargs['model'], checkpoint, state['generator'], job_path.parent)
        return evaluate(*args, **kwargs)

    with patch.object(evaluator, 'simple_evaluate', resumable):
        matrix.worker(job_path)
    retry.finish_adaptation_audit(job_path.parent)


if __name__ == '__main__':
    run(sys.argv[1])
