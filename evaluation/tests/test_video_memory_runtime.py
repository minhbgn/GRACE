"""CPU equivalence and recovery invariants; repeated on GPU 1 before real retries."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from evaluation.protocols.token_reduction_runtime import prepare_features, compact, decoder_greedy
from evaluation.protocols.video_memory_runtime import (
    HostVideoInputs, adaptive_video_geometry, decoder_greedy_chunked, host_features,
    install_streamed_vision, stream_video_features,
)
from evaluation.runners import run_token_reduction_video_retry as retry


class MemoryEquivalence(unittest.TestCase):
    def setUp(self):
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

        device = os.environ.get('VIDEO_MEMORY_TEST_DEVICE', 'cpu')
        cfg = Qwen3VLConfig(text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=1024, rope_parameters=dict(rope_type='default', rope_theta=10000,
                                                              mrope_section=[1, 1, 2])),
            vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
                patch_size=16, temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=32,
                deepstack_visual_indexes=[0, 1]),
            image_token_id=120, video_token_id=121, vision_start_token_id=122, vision_end_token_id=123)
        cfg._attn_implementation = 'flash_attention_2' if device == 'cuda' else 'sdpa'
        torch.manual_seed(42)
        self.model = Qwen3VLForConditionalGeneration(cfg).eval()
        self.model = self.model.to(device=device, dtype=torch.bfloat16 if device == 'cuda' else torch.float32)
        self.tolerance = 0.02 if device == 'cuda' else 1e-5

    def video_inputs(self):
        device = next(self.model.parameters()).device
        ids = torch.tensor([[1, 122, 121, 121, 121, 121, 123, 5,
                             122, 121, 121, 121, 121, 123, 6,
                             122, 121, 121, 121, 121, 123, 4]], device=device)
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
            mm_token_type_ids=(ids == 121).long() * 2,
            video_grid_thw=torch.tensor([[3, 4, 4]], device=device),
            pixel_values_videos=torch.randn(48, 3 * 2 * 16 * 16, device=device,
                                           dtype=next(self.model.parameters()).dtype))

    def test_streamed_vision_preserves_multiple_clips_and_deepstack(self):
        device = next(self.model.parameters()).device
        grid = torch.tensor([[3, 4, 4], [2, 2, 4]], device=device)
        pixels = torch.randn(64, 3 * 2 * 16 * 16, device=device, dtype=next(self.model.parameters()).dtype)
        getter = self.model.model.get_video_features
        with torch.inference_mode():
            expected = getter(pixels, grid, return_dict=True)
            for size in (1, 2):
                actual = stream_video_features(getter, self.model.model.visual, pixels.cpu(), grid, size)
                self.assertEqual([len(p) for p in actual.pooler_output], [12, 4])
                for left, right in zip(expected.pooler_output, actual.pooler_output, strict=True):
                    torch.testing.assert_close(left, right, atol=self.tolerance, rtol=self.tolerance)
                for left, right in zip(expected.deepstack_features, actual.deepstack_features, strict=True):
                    torch.testing.assert_close(left, right, atol=self.tolerance, rtol=self.tolerance)

    def test_prefill_chunks_preserve_greedy_answers_and_cache(self):
        with torch.inference_mode():
            features = prepare_features(self.model, self.video_inputs())
            for method, ratio in (('baseline', 1.), ('uniform', .5), ('merge', .25)):
                reduced = compact(*features, ratio, method)
                expected, original_stats = decoder_greedy(self.model, reduced, 5, set(), fixed=True)
                for size in (1, 5, 8, 512):
                    actual, stats = decoder_greedy_chunked(self.model, host_features(reduced),
                        5, set(), fixed=True, chunk_size=size)
                    self.assertTrue(torch.equal(expected, actual), (method, size, expected, actual))
                    self.assertEqual(stats['final_kv_tokens'], original_stats['final_kv_tokens'])
                eos = int(expected[0, 0])
                early, stats = decoder_greedy_chunked(self.model, host_features(reduced), 5, {eos}, chunk_size=5)
                self.assertEqual(early.shape[1], 1)
                self.assertEqual(stats['final_kv_tokens'], reduced[0].shape[1])

    def test_streamed_end_to_end_features_and_reduction(self):
        with torch.inference_mode():
            inputs = self.video_inputs()
            expected = prepare_features(self.model, inputs)
            inputs['pixel_values_videos'] = inputs['pixel_values_videos'].cpu()
            install_streamed_vision(self.model)
            actual = prepare_features(self.model, inputs)
            for method in ('uniform', 'merge'):
                left = compact(*expected, .25, method)
                right = compact(*actual, .25, method)
                for a, b in zip(left[:3], right[:3], strict=True):
                    torch.testing.assert_close(a, b, atol=self.tolerance, rtol=self.tolerance)
                for a, b in zip(left[3], right[3], strict=True):
                    torch.testing.assert_close(a, b, atol=self.tolerance, rtol=self.tolerance)

    def test_pixels_stay_on_host_and_no_context_truncation(self):
        pixels = torch.randn(16, 4)
        batch = HostVideoInputs(pixel_values_videos=pixels, input_ids=torch.ones(1, 4, dtype=torch.long))
        moved = batch.to('meta')
        self.assertIs(moved['pixel_values_videos'], pixels)
        self.assertEqual(moved['input_ids'].device.type, 'meta')
        with torch.inference_mode():
            features = prepare_features(self.model, self.video_inputs())
            self.model.config.text_config.max_position_embeddings = features[0].shape[1]
            with self.assertRaisesRegex(ValueError, 'native context'):
                decoder_greedy_chunked(self.model, host_features(features), 1, set())


class RetryBookkeeping(unittest.TestCase):
    def test_adaptive_geometry_preserves_frames_and_shares_baseline_across_methods(self):
        from types import SimpleNamespace
        processor = SimpleNamespace(video_processor=SimpleNamespace(patch_size=16, merge_size=2, temporal_patch_size=2))
        meta = dict(source_height=1080, source_width=1920, sampled_frames=4000,
                    frame_indices=list(range(4000)), source_fps=2., duration_seconds=2000.)
        policy = dict(image_max_pixels=589824, max_new_tokens=256)
        baseline = None
        with patch('evaluation.protocols.video_protocol.input_counts', return_value={'input_text_tokens': 21000}):
            for method, ratio in [('baseline', 1.), ('resolution', .25), ('uniform', .5), ('merge', .1)]:
                selected, adaptation = adaptive_video_geometry(processor, {}, meta, policy, method, ratio, 131072)
                self.assertTrue(adaptation['applied'])
                self.assertIs(selected['frame_indices'], meta['frame_indices'])
                self.assertEqual(selected['sampled_frames'], 4000)
                self.assertEqual(selected['grid'][0], 2000)
                self.assertLessEqual(selected['visual_tokens'] + 21000 + 256, 131072)
                if baseline is None:
                    baseline = adaptation['adaptive_baseline_visual_tokens']
                self.assertEqual(adaptation['adaptive_baseline_visual_tokens'], baseline)
                self.assertGreater(adaptation['original_policy_visual_tokens'], 1_000_000)
            with self.assertRaisesRegex(ValueError, 'All temporal groups'):
                adaptive_video_geometry(processor, {}, meta, policy, 'baseline', 1., 22000)
        meta.update(sampled_frames=2, frame_indices=[0, 1])
        with patch('evaluation.protocols.video_protocol.input_counts', return_value={'input_text_tokens': 100}):
            selected, adaptation = adaptive_video_geometry(processor, {}, meta, policy, 'baseline', 1., 131072)
            self.assertFalse(adaptation['applied'])
            self.assertEqual(adaptation['original_policy_visual_tokens'], selected['visual_tokens'])

    def completion(self, directory):
        directory.mkdir(parents=True)
        attempt = directory / 'attempt_test'
        attempt.mkdir()
        artifact = attempt / 'validated.json'
        artifact.write_text('{}')
        done = dict(attempt=attempt.name, sha256={'validated.json': retry.matrix.sha(artifact)},
                    count=300, score_percent=50.)
        (directory / 'COMPLETE.json').write_text(json.dumps(done))
        return done

    def test_original_pass_skipped_and_corruption_blocks_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / 'source', Path(temporary) / 'output'
            passed = source / 'qat_w4/videomme/resolution_r025'
            self.completion(passed)
            with patch.object(retry, 'SOURCE', source), patch.object(retry, 'CONFIGS', [('resolution', .25)]):
                jobs, reused, completed = retry.inventory(output, ['videomme'])
                self.assertEqual(jobs, [])
                self.assertEqual(len(reused), 1)
                (passed / 'attempt_test/validated.json').write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError, 'Completed output changed'):
                    retry.inventory(output, ['videomme'])

    def test_only_failed_jobs_enter_retry_and_new_pass_skips_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / 'source', Path(temporary) / 'output'
            failed = source / 'qat_w4/videomme/baseline_r100/attempt_failed'
            failed.mkdir(parents=True)
            (failed / 'exit.json').write_text('{"returncode":1}')
            with patch.object(retry, 'SOURCE', source), patch.object(retry, 'CONFIGS', [('baseline', 1.)]):
                jobs, reused, completed = retry.inventory(output, ['videomme'])
                self.assertEqual(len(jobs), 1)
                self.completion(output / 'qat_w4/videomme/baseline_r100')
                jobs, reused, completed = retry.inventory(output, ['videomme'])
                self.assertEqual(jobs, [])
                self.assertEqual(len(completed), 1)
                (failed / 'exit.json').unlink()
                with patch.object(retry, 'resolved', return_value=(None, None, failed.parent)):
                    with self.assertRaisesRegex(ValueError, 'No failed original job'):
                        retry.inventory(output, ['videomme'])

    def test_correct_yaml_and_local_scorer_are_loaded_offline(self):
        retry.lmms.source_check()
        retry.install_task_mapping()
        with retry.lmms.video_no_network():
            config = retry.lmms.video_task_config('tempcompass', {'max_new_tokens': 256})
        self.assertEqual(config['task'], 'tempcompass_multi_choice')
        self.assertIs(config['process_results'], retry.lmms.video_local_choice)
        payload = dict(video_id='v', dim='action', _grace_row=json.dumps({
            'options': {'A': 'left', 'B': 'right'}, 'gold': 'B'}))
        good = config['process_results'](payload, ['B'])
        unresolved = config['process_results'](payload, ['Unknown'])
        self.assertEqual(good['avg_accuracy']['rating'], 1)
        self.assertEqual(unresolved['avg_accuracy']['rating'], 0)
        aggregate = next(m['aggregation'] for m in config['metric_list'] if m['metric'] == 'avg_accuracy')
        self.assertEqual(aggregate([good['avg_accuracy'], unresolved['avg_accuracy']], None), 50)

    def test_real_tempcompass_task_scores_gold_answers_without_network(self):
        from types import SimpleNamespace

        retry.lmms.source_check()
        retry.install_task_mapping()
        from lmms_eval import evaluator
        from lmms_eval.api.model import lmms
        from lmms_eval.tasks import TaskManager

        class GoldModel(lmms):
            def loglikelihood(self, requests):
                raise NotImplementedError

            def generate_until_multi_round(self, requests):
                raise NotImplementedError

            def generate_until(self, requests):
                answers = []
                for request in requests:
                    _, _, _, doc_id, task, split = request.args
                    doc = self.task_dict[task][split][doc_id]
                    answers.append(json.loads(doc['_grace_row'])['gold'])
                return answers

        rows, _ = retry.matrix.load_video('tempcompass', None)
        documents = retry.lmms.video_documents('tempcompass', rows[:2], retry.matrix.DEFAULT_DATA)
        with tempfile.TemporaryDirectory() as directory, retry.lmms.video_no_network():
            task = retry.lmms.video_task('tempcompass', {'max_new_tokens': 256}, documents)
            result = evaluator.simple_evaluate(model=GoldModel(), tasks=[task],
                task_manager=TaskManager(model_name='qwen3_vl'), batch_size=1, num_fewshot=0,
                limit=2, bootstrap_iters=0, log_samples=True,
                cli_args=SimpleNamespace(output_path=directory, process_with_media=False))
        samples = result['samples']['tempcompass_multi_choice']
        validated = retry.lmms.validate_result(result, samples, 'tempcompass', 2, retry.matrix.VIDEO_TASKS)
        self.assertEqual(validated['count'], 2)
        self.assertEqual(validated['score_percent'], 100.)


if __name__ == '__main__':
    unittest.main()
