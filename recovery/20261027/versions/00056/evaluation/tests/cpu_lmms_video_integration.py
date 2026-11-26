"""Pinned harness integration on real local annotations, dummy answers, no GPU/API."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from types import SimpleNamespace
from evaluation.runners.run_lmms_matrix import (VIDEO_TASKS, source_check, video_documents,
    video_task, video_no_network, validate_result)
from evaluation.protocols.video_protocol import DEFAULT_DATA, load_suite
from evaluation.protocols.video_population import select_population
source_check()
from lmms_eval import evaluator
from lmms_eval.api.model import lmms
from lmms_eval.tasks import TaskManager

class Dummy(lmms):
    def loglikelihood(self, requests):
        raise NotImplementedError
    def generate_until_multi_round(self, requests):
        raise NotImplementedError
    def generate_until(self, requests):
        outputs = []
        for request in requests:
            _, _, _, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]
            answer = doc['answer']
            outputs.append(str(answer[0] if isinstance(answer, list) else answer))
        return outputs

for suite in VIDEO_TASKS:
    rows, info = load_suite(suite, DEFAULT_DATA)
    rows, info = select_population(suite, rows, info, True)
    assert info['status'] == 'ready', info
    documents = video_documents(suite, rows, DEFAULT_DATA)
    with video_no_network():
        task = video_task(suite, {'max_new_tokens': 4096}, documents)
        result = evaluator.simple_evaluate(model=Dummy(), tasks=[task],
            task_manager=TaskManager(model_name='qwen3_vl'), batch_size=1, num_fewshot=0,
            limit=2, bootstrap_iters=0, log_samples=True,
            cli_args=SimpleNamespace(output_path='/tmp/grace_lmms_video_cpu', process_with_media=False))
    samples = result['samples'][VIDEO_TASKS[suite][0]]
    validated = validate_result(result, samples, suite, 2, VIDEO_TASKS)
    assert validated['count'] == 2, validated
    assert validated['score_percent'] == 100, validated
    print('PASS', suite, validated)
