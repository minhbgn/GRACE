"""Run explicitly with .venv-lmms-eval; uses synthetic data and no GPU/judge."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_lmms_matrix import source_check, task_config, validate_result
source_check()
from unittest.mock import patch
from types import SimpleNamespace
import datasets
from lmms_eval import evaluator
from lmms_eval.api.model import lmms
from lmms_eval.tasks import TaskManager

class Dummy(lmms):
    def loglikelihood(self, requests):
        raise NotImplementedError
    def generate_until(self, requests):
        return ['A' for _ in requests]
    def generate_until_multi_round(self, requests):
        raise NotImplementedError

data = datasets.DatasetDict(test=datasets.Dataset.from_dict({
    'question': ['Which?', 'Which one?'], 'hint': ['', ''],
    'choices': [['one', 'two'], ['one', 'two']], 'answer': [0, 1], 'image': [None, None]}))
cfg = task_config('scienceqa')
with patch('datasets.load_dataset', return_value=data):
    result = evaluator.simple_evaluate(model=Dummy(), tasks=[cfg],
        task_manager=TaskManager(model_name='qwen3_vl'), batch_size=1,
        num_fewshot=0, bootstrap_iters=10, log_samples=True,
        cli_args=SimpleNamespace(output_path='/tmp/grace_lmms_cpu_integration', process_with_media=False))
validated = validate_result(result, result['samples']['scienceqa_img'], 'scienceqa', None)
assert validated['score_percent'] == 50.0, validated
print('PASS: actual pinned harness task loading, prompt resolution, scoring, sample logging and completion validation on a synthetic 2-row dataset; no GPU or judge calls.')
