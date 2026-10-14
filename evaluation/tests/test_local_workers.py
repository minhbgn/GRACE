"""Exercise all seven real worker mains with fake generation, entirely on CPU."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_scoring import _row_metadata
from run_constrained_matrix_evals import suites


class WorkerIntegration(unittest.TestCase):
    def test_all_seven_worker_mains(self):
        import pandas as pd
        import torch
        from PIL import Image
        evaluation = Path(__file__).resolve().parents[1]
        for suite, (script, _, arguments) in suites().items():
            with self.subTest(suite=suite), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                image = directory / 'sample.png'
                Image.new('RGB', (28, 28)).save(image)
                spec = importlib.util.spec_from_file_location('worker_' + suite, evaluation / script)
                worker = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(worker)
                raw = 'The answer is B' if suite != 'hallusionbench' else 'No'
                class FakeModel:
                    def to(self, device): return self
                    def eval(self): return None
                    def generate(self, **kwargs): return torch.tensor([[1, 7, 9]])
                model = FakeModel()
                tokenizer = SimpleNamespace(pad_token_id=9, eos_token_id=9, bos_token_id=1,
                                            decode=lambda *a, **kw: raw)
                processor = SimpleNamespace(tokenizer=tokenizer, image_processor=SimpleNamespace(size={'shortest_edge':4096, 'longest_edge':50176}))
                row = dict(index=0, question_id=0, id='validation_Test_0', question='Which?',
                    options=['first', 'second'], choices=['first', 'second'], answer='B',
                    question_type='multiple-choice', image=None, category='test', subcategory='test',
                    set_id=0, figure_id=0, visual_input='0', gt_answer='0', A='first', B='second',
                    C=float('nan'), D=float('nan'), __subject='Test', pid='0',
                    data_type='image', data_id='sample.png', question_type_id=1)
                if suite in ('ai2d', 'scienceqa'):
                    row['answer'] = 1
                if suite == 'seedbench_img':
                    row.update(choice_a='first', choice_b='second', choice_c='third', choice_d='fourth')
                data = [row] if suite == 'scienceqa' else ([row], {1:'test'}) if suite == 'seedbench_img' else pd.DataFrame([row])
                meta = {0: row} if suite == 'mmbench' else {row['id']: row} if suite == 'mmmu' else {}
                output = directory / 'results.json'
                argv = [script, '--model_path', '/fake', '--processor_path', '/fake', '--output', str(output),
                        '--attn', 'sdpa', '--limit', '1', '--max_new_tokens', '256', '--mc_max_new_tokens', '256',
                        '--fixed_generation', '--local_scoring', *arguments]
                _row_metadata.cache_clear()
                from contextlib import ExitStack, redirect_stdout
                import io
                with ExitStack() as stack:
                    stack.enter_context(patch.object(sys, 'argv', argv))
                    stack.enter_context(patch.dict('os.environ', {'RANK':'0', 'LOCAL_RANK':'0', 'WORLD_SIZE':'1'}))
                    stack.enter_context(patch.object(worker.torch.cuda, 'set_device'))
                    stack.enter_context(patch.object(worker.Qwen3VLForConditionalGeneration, 'from_pretrained', return_value=model))
                    stack.enter_context(patch.object(worker.AutoProcessor, 'from_pretrained', return_value=processor))
                    stack.enter_context(patch.object(worker, 'load_data', return_value=data))
                    stack.enter_context(patch.object(worker, 'build_inputs', return_value={'input_ids':torch.tensor([[1]])}))
                    stack.enter_context(patch('local_scoring.load_metadata', return_value=meta))
                    if hasattr(worker, '_decode_image_b64'):
                        stack.enter_context(patch.object(worker, '_decode_image_b64', return_value=Image.new('RGB', (28, 28))))
                    if hasattr(worker, '_decode_image'):
                        stack.enter_context(patch.object(worker, '_decode_image', return_value=Image.new('RGB', (28, 28))))
                    if hasattr(worker, 'resolve_image_path'):
                        stack.enter_context(patch.object(worker, 'resolve_image_path', return_value=str(image)))
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    worker.main()
                payload = json.loads(output.read_text())
                result = payload['results'][0]
                self.assertIsNone(result['error'])
                self.assertTrue(result['correct'])
                self.assertEqual(result['generation']['finish_reason'], 'eos')
                self.assertEqual(payload['summary']['scorer'], 'deterministic_local_v1')
                shard = json.loads(Path(str(output) + '.rank0.jsonl').read_text())
                self.assertEqual(shard['pred'], result['pred'])
                self.assertEqual(shard['correct'], result['correct'])
                self.assertIn('extraction_method', shard)
                if suite == 'mmbench':
                    self.assertIsNone(payload['summary']['primary_score'])
        _row_metadata.cache_clear()


if __name__ == '__main__':
    unittest.main()
