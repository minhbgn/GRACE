"""Exercise both real task-loading passes, MC/open prompts and scoring without credentials."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_lmms_matrix import local_mmmu_judge_guard, source_check, task_config, validate_result
source_check()
import datasets
from lmms_eval import evaluator
from lmms_eval.api.model import lmms
from lmms_eval.tasks import TaskManager

class Dummy(lmms):
    def loglikelihood(self, requests):
        raise NotImplementedError
    def generate_until(self, requests):
        answers = []
        for req in requests:
            prompt = req.args[0]
            if 'Compute' in prompt:
                assert 'Options:' not in prompt and 'options above' not in prompt, prompt
                assert 'single word or phrase' in prompt, prompt
                answers.append('42')
            else:
                assert 'option' in prompt and 'A.' in prompt, prompt
                answers.append('A')
        return answers
    def generate_until_multi_round(self, requests):
        raise NotImplementedError

data = datasets.DatasetDict(validation=datasets.Dataset.from_dict({
    'id': ['validation_Math_1', 'validation_Math_2'],
    'question': ['Which value?', 'Compute 6 * 7.'],
    'options': ["['1', '2']", '[]'], 'answer': ['A', '42'],
    'question_type': ['multiple-choice', 'open']}))
with patch.dict(os.environ, {'OPENAI_API_KEY': '', 'OPENAI_ADMIN_KEY': ''}):
    cfg = task_config('mmmu', {'mmmu_prompt': 'official_short_answer', 'mmmu_open_scoring': 'typed_candidates'})
    scorer = cfg['process_results']
    helpers = scorer.keywords['original'].__globals__
    for gold, response, expected in [('blue whale', 'blue whale', True),
                                      ('42', '42', True), ('42', '41', False),
                                      (['cat', 'feline'], 'feline', True),
                                      ('cat', 'dog', False)]:
        doc = {'id': 'validation_Math_3', 'question_type': 'open', 'answer': gold}
        candidates = scorer(doc, [response])['mmmu_acc']['parsed_pred'][0]
        assert isinstance(candidates, list), candidates
        assert helpers['eval_open'](gold, candidates) is expected, (gold, response, candidates)
    with local_mmmu_judge_guard('mmmu'), patch('datasets.load_dataset', return_value=data):
        result = evaluator.simple_evaluate(model=Dummy(), tasks=[cfg],
            task_manager=TaskManager(model_name='qwen3_vl'), num_fewshot=0,
            batch_size=1, bootstrap_iters=0, log_samples=True, force_simple=True,
            cli_args=SimpleNamespace(output_path='/tmp/grace_mmmu_cpu_integration', process_with_media=False))
validated = validate_result(result, result['samples']['mmmu_val'], 'mmmu', None)
assert validated['score_percent'] == 100.0, validated
print('PASS: real MMMU TaskManager reload, MC/open prompts, exact scoring, sample logging; no credentials or judge calls.')
