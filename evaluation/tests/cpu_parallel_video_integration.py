"""Real pinned VideoMME harness: interrupt after one answer, resume and rescore."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.runners import run_token_reduction_matrix as matrix
from evaluation.protocols.evaluation_checkpoints import AnswerCheckpoint, append
from evaluation.workers.parallel_video_worker import checkpoint_adapter

matrix.source_check()
from lmms_eval import evaluator
from lmms_eval.api.model import lmms
from lmms_eval.tasks import TaskManager

rows, _ = matrix.load_video('videomme', None)
documents = matrix.video_documents('videomme', rows, matrix.DEFAULT_DATA)


class Dummy(lmms):
    def __init__(self, attempt, fail_after=None):
        super().__init__()
        self.attempt, self.fail_after, self.calls = attempt, fail_after, 0
        self.generator = SimpleNamespace(index=0)

    def loglikelihood(self, requests):
        raise NotImplementedError

    def generate_until_multi_round(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        answers = []
        for request in requests:
            if self.calls == self.fail_after:
                raise RuntimeError('simulated interruption')
            _, _, _, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]
            answer = doc['answer']
            answers.append(str(answer[0] if isinstance(answer, list) else answer))
            append(self.attempt / 'telemetry.jsonl', dict(index=self.generator.index))
            append(self.attempt / 'video_inputs.jsonl', dict(doc_id=doc_id))
            self.generator.index += 1
            self.calls += 1
        return answers


with tempfile.TemporaryDirectory(prefix='grace-parallel-harness-') as root:
    for number in (0, 1):
        attempt = Path(root) / f'attempt_{number}'
        attempt.mkdir()
        dummy = Dummy(attempt, fail_after=1 if number == 0 else None)
        checkpoint = AnswerCheckpoint(attempt, 'real-videomme-fixture')
        checkpoint_adapter(dummy, checkpoint, dummy.generator, attempt)
        with matrix.video_no_network():
            task = matrix.video_task('videomme', {'max_new_tokens': 256}, documents)
            try:
                result = evaluator.simple_evaluate(model=dummy, tasks=[task],
                    task_manager=TaskManager(model_name='qwen3_vl'), batch_size=1,
                    num_fewshot=0, limit=2, bootstrap_iters=0, log_samples=True,
                    random_seed=42, numpy_random_seed=42, torch_random_seed=42, fewshot_random_seed=42,
                    cli_args=SimpleNamespace(output_path=str(attempt), process_with_media=False))
            except RuntimeError as error:
                assert number == 0 and str(error) == 'simulated interruption', error
                assert checkpoint.index == 1
                continue
        assert number == 1
        samples = result['samples'][matrix.VIDEO_TASKS['videomme'][0]]
        validated = matrix.validate_result(result, samples, 'videomme', 2, matrix.VIDEO_TASKS)
        assert dummy.calls == 1, dummy.calls  # Replayed question 0 never reached inference.
        assert dummy.generator.index == checkpoint.index == 2
        assert validated['count'] == 2 and validated['score_percent'] == 100, validated
        assert [json.loads(line)['index'] for line in (attempt / 'telemetry.jsonl').read_text().splitlines()] == [0, 1]
        print('PASS interrupted VideoMME harness resumed one answer, generated one, rescored both:', validated)
