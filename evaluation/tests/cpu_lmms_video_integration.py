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

# Exercise the real adapter and video processor with CPU inputs and a fake decoder.
# No checkpoint weights are loaded and every CUDA operation is mocked.
import json
import tempfile
import torch
from unittest.mock import patch
from evaluation.runners.run_lmms_matrix import video_adapter
from evaluation.protocols import video_protocol
rows, _ = load_suite('tempcompass', DEFAULT_DATA)
rows = rows[:1]
checkpoint = Path(__file__).resolve().parents[2] / 'experiments/released_bf16_init_32k_b128_p589824_t4096_seed42'
from evaluation.runners.run_lmms_matrix import Experiment
exp = Experiment(checkpoint / 'experiment.json')
policy = dict(fps=2, shortest_side=720, longest_side=1280, max_new_tokens=4096, attention='sdpa')
original_build = video_protocol.build_inputs
class CPUInputs(dict):
    def to(self, device):
        return self

def cpu_build(*args):
    inputs, counts = original_build(*args)
    return CPUInputs(inputs), counts

class Decoder:
    def to(self, device):
        return self
    def eval(self):
        return self
    def generate(self, input_ids, **kwargs):
        return torch.cat([input_ids, torch.tensor([[eos]])], dim=1)

with tempfile.TemporaryDirectory(prefix='grace-lmms-adapter-') as directory:
    attempt = Path(directory)
    job = dict(model_path=str(exp.paths['vanilla_2b']), protocol=policy, scratch=None)
    with patch('evaluation.protocols.video_protocol.build_inputs', side_effect=cpu_build), \
         patch('transformers.Qwen3VLForConditionalGeneration.from_pretrained', return_value=Decoder()), \
         patch('torch.cuda.synchronize'), patch('torch.cuda.reset_peak_memory_stats'), \
         patch('torch.cuda.max_memory_allocated', return_value=0), video_no_network():
        model = video_adapter(job, attempt)
        eos = model.processor.tokenizer.eos_token_id
        task = video_task('tempcompass', policy, video_documents('tempcompass', rows, DEFAULT_DATA))
        result = evaluator.simple_evaluate(model=model, tasks=[task],
            task_manager=TaskManager(model_name='qwen3_vl'), batch_size=1, num_fewshot=0,
            bootstrap_iters=0, log_samples=True,
            cli_args=SimpleNamespace(output_path=directory, process_with_media=False))
    record = json.loads((attempt / 'telemetry.jsonl').read_text())
    assert record['generation']['finish_reason'] == 'eos', record
    assert record['tokens']['input_visual_tokens'] > 0, record
    assert record['tokens']['input_tokens'] == record['tokens']['input_visual_tokens'] + record['tokens']['input_text_tokens']
    print('PASS real video adapter/processor with fake CPU decoder:', record['tokens'])
