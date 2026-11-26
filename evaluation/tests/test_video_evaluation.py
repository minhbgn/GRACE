"""Input preservation, actual processor accounting, offline assets and integrity."""
# Support both direct script execution and `python -m evaluation.<group>.<module>`.
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.paths import EVALUATION_ROOT, source_file

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from evaluation.protocols.video_protocol import ROOT, geometry, input_counts, inventory, local_video, sampling
from evaluation.runners.run_video_matrix import summary_from_rows
from evaluation.protocols.local_scoring import generation_record


class VideoPolicyTests(unittest.TestCase):
    def test_sampling_keeps_long_clips_and_endpoints(self):
        indices = sampling(18000, 30, 2)
        self.assertEqual(len(indices), 1200)  # exceeds Qwen's native 768-frame cap
        self.assertEqual((indices[0], indices[-1]), (0, 17999))
        self.assertEqual(len(set(indices)), len(indices))
        self.assertEqual(sampling(3, 24, 0), [0, 1, 2])
        self.assertEqual(sampling(1, 24, 2), [0])

    def test_archive_is_used_without_bulk_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(root / 'chunk.zip', 'w') as archive:
                archive.writestr('data/example.mp4', b'video payload')
                archive.writestr('data/unused.mp4', b'not needed')
            assets = inventory(root)
            with local_video(assets['example'], root) as path:
                self.assertEqual(Path(path).read_bytes(), b'video payload')
                extracted = Path(path)
            self.assertFalse(extracted.exists())
            self.assertFalse((root / 'data').exists())

    def test_source_geometry_is_independent_of_duration_budgets(self):
        short = geometry(14, 316, 600, 16, 2, 2)
        long = geometry(1400, 316, 600, 16, 2, 2)
        self.assertEqual((short['height'], short['width']), (long['height'], long['width']))
        self.assertEqual(long['visual_tokens'], short['visual_tokens'] * 100)


class IntegrityTests(unittest.TestCase):
    def record(self):
        return dict(id='one', gold='B', raw='B', options={'A': 'left', 'B': 'right'},
                    prediction='B', correct=True, question_type='multiple-choice', category='action',
                    input_identity='input hash', eos_token_id=99,
                    generation=generation_record([2, 99], 99, 4096),
                    tokens=dict(input_visual_tokens=1000, input_text_tokens=80,
                                input_tokens=1080, output_text_tokens=2))

    def test_scores_and_token_totals_recomputed(self):
        row = self.record()
        result = summary_from_rows([row])
        self.assertEqual(result['accuracy'], 1)
        self.assertEqual(result['token_usage']['total_text_tokens']['mean'], 82)
        row['correct'] = False
        with self.assertRaisesRegex(ValueError, 'score'):
            summary_from_rows([row])

    def test_censored_outputs_are_visible(self):
        row = self.record()
        row['generation'] = generation_record([2, 3], 99, 2)
        self.assertEqual(summary_from_rows([row])['generation_cutoffs'], 1)
        row['tokens']['input_tokens'] = 1079
        with self.assertRaisesRegex(ValueError, 'accounting'):
            summary_from_rows([row])


class TokenRateAuditTests(unittest.TestCase):
    def setUp(self):
        from evaluation.tests.test_video_runtime import fixture
        from evaluation.protocols.video_protocol import digest
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.job = fixture()
        self.job['protocol']['version'] = 'video_resilient_v3'
        self.records = self.job.pop('test_records')
        for row, source in zip(self.records, self.job['rows']):
            source['video_id'] = source['id']
            row['sampling'].update(duration_seconds=10, source_frames=100, source_fps=10)
            row['elapsed_seconds'] = 2
            row['source_identity'] = digest(source)
            counts = {k: row['tokens'][k] for k in ('input_visual_tokens', 'input_text_tokens', 'input_tokens')}
            row['input_identity'] = digest(dict(source=source, sampling=row['sampling'],
                tokens=counts, input_ids_sha256=row['input_ids_sha256']))

    def matrix(self, name, count, complete=False, records=None):
        from evaluation.protocols.video_protocol import digest
        from evaluation.runners.run_video_matrix import sha, write
        root = self.base / name / 'full'
        write(root / 'protocol.json', dict(limit=None, policy=self.job['protocol'], experiment={}))
        write(root / 'datasets' / (self.job['suite'] + '.json'), dict(status='ready', questions=2,
              sample_identity=digest(self.job['rows'])))
        pair = root / self.job['method'] / self.job['suite']
        attempt = pair / 'attempt_1'
        write(attempt / 'job.json', self.job)
        predictions = self.records[:count] if records is None else records
        (attempt / 'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
        if complete:
            write(attempt / 'summary.json', summary_from_rows(predictions))
            write(attempt / 'processor.json', {})
            write(attempt / 'reuse.json', {'source': 'prior run'})
            write(pair / 'COMPLETE.json', dict(attempt='attempt_1', summary=summary_from_rows(predictions),
                  sha256={k: sha(attempt / k) for k in
                          ('job.json', 'summary.json', 'predictions.jsonl', 'processor.json', 'reuse.json')}))
        return root, pair

    def test_imported_completion_accepts_extra_provenance_hash(self):
        from evaluation.audits.audit_video_token_rates import audit_pair
        _, pair = self.matrix('imported', 2, complete=True)
        entry, measured = audit_pair(pair, self.job['protocol'], 2, partial_over=0)
        self.assertEqual(entry['status'], 'complete')
        self.assertEqual(len(measured), 2)

    def test_concurrent_matrices_choose_completion_without_double_counting(self):
        from evaluation.audits.audit_video_token_rates import collect_runs
        first, _ = self.matrix('early', 1)
        second, _ = self.matrix('late', 2, complete=True)
        with patch('evaluation.audits.audit_video_token_rates.expected_population', return_value=2):
            entries, questions, candidates, _ = collect_runs([first, second], [self.job['method']], 0)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['matrix'], str(second))
        self.assertEqual(len(questions), 2)
        self.assertEqual(len(candidates), 2)

    def test_non_eos_full_coverage_is_reported_without_rates(self):
        from evaluation.audits.audit_video_token_rates import audit_pair, table
        records = copy.deepcopy(self.records)
        records[-1]['generation'] = generation_record([2, 2], 99, 4096)
        _, pair = self.matrix('cutoff', 2, records=records)
        entry, measured = audit_pair(pair, self.job['protocol'], 2, partial_over=0)
        self.assertEqual(entry['coverage_fraction'], 1)
        self.assertEqual(entry['measurement_scope'], 'invalid_partial')
        self.assertEqual(measured, [])
        self.assertIn('non-EOS; rates withheld', '\n'.join(table([entry])))

    def test_declared_subset_has_1800_denominator_and_visible_label(self):
        from evaluation.audits.audit_video_token_rates import expected_population, table
        from evaluation.protocols.video_population import VIDEOMME_SHORT_MEDIUM
        policy = {'videomme_population': VIDEOMME_SHORT_MEDIUM}
        self.assertEqual(expected_population('videomme',
            dict(questions=1800, population=VIDEOMME_SHORT_MEDIUM), policy), 1800)
        with self.assertRaises(ValueError):
            expected_population('videomme', dict(questions=2700, population=VIDEOMME_SHORT_MEDIUM), policy)
        _, pair = self.matrix('partial', 1)
        from evaluation.audits.audit_video_token_rates import audit_pair
        entry, _ = audit_pair(pair, self.job['protocol'], 2, partial_over=0)
        entry.update(suite='videomme', population='short_medium')
        self.assertIn('videomme (short + medium)', '\n'.join(table([entry])))

    def test_interrupted_final_json_record_is_excluded(self):
        from evaluation.audits.audit_video_token_rates import prediction_rows
        path = self.base / 'live.jsonl'
        path.write_text(json.dumps(self.records[0])+'\n'+json.dumps(self.records[1]))
        rows, partial = prediction_rows(path, allow_partial=True)
        self.assertEqual(rows, self.records[:1])
        self.assertTrue(partial)

    def test_incompatible_matrices_and_changed_population_fail(self):
        from evaluation.audits.audit_video_token_rates import collect_runs
        from evaluation.runners.run_video_matrix import write
        first, _ = self.matrix('early', 1)
        second, _ = self.matrix('late', 1)
        write(second / 'protocol.json', dict(limit=None, policy={'different': True}, experiment={}))
        with self.assertRaisesRegex(ValueError, 'share'):
            collect_runs([first, second], [self.job['method']], 0)

    def test_compatible_legacy_partial_preserves_progress(self):
        from evaluation.audits.audit_video_token_rates import collect_runs
        from evaluation.runners.run_video_matrix import write
        first, _ = self.matrix('current', 0)
        second, _ = self.matrix('legacy', 1)
        identity = json.loads((second / 'protocol.json').read_text())
        identity['policy']['version'] = 'video_unsaturated_v2'
        write(second / 'protocol.json', identity)
        old_job = copy.deepcopy(self.job)
        old_job['protocol']['version'] = 'video_unsaturated_v2'
        write(second / self.job['method'] / self.job['suite'] / 'attempt_1/job.json', old_job)
        with patch('evaluation.audits.audit_video_token_rates.expected_population', return_value=2):
            entries, questions, _, _ = collect_runs([first, second], [self.job['method']], 0)
        self.assertEqual(entries[0]['matrix'], str(second))
        self.assertEqual(len(questions), 1)

    def test_original_videomme_population_is_not_mixed_into_subset(self):
        from evaluation.audits.audit_video_token_rates import collect_runs
        from evaluation.protocols.video_population import VIDEOMME_SHORT_MEDIUM
        from evaluation.runners.run_video_matrix import write
        current, legacy = self.base / 'subset/full', self.base / 'original/full'
        subset_policy = dict(self.job['protocol'], videomme_population=VIDEOMME_SHORT_MEDIUM)
        original_policy = dict(self.job['protocol'], version='video_unsaturated_v2')
        for root, policy, count, fingerprint, population in (
                (current, subset_policy, 1800, 'subset', {'population': VIDEOMME_SHORT_MEDIUM}),
                (legacy, original_policy, 2700, 'original', {})):
            write(root / 'protocol.json', dict(limit=None, policy=policy, experiment={}))
            write(root / 'datasets/videomme.json', dict(status='ready', questions=count,
                  sample_identity=fingerprint, **population))
        entries, questions, candidates, _ = collect_runs([current, legacy], ['model'], 0)
        self.assertEqual(entries[0]['expected_questions'], 1800)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(questions, [])

    def test_average_token_counts_are_arithmetic_per_question_means(self):
        from evaluation.audits.audit_video_token_rates import audit_pair, average_table
        from evaluation.protocols.video_protocol import digest
        second = self.records[1]
        second['tokens'].update(input_visual_tokens=30, input_text_tokens=10,
                                input_tokens=40, output_text_tokens=3)
        second['generation'] = generation_record([2, 2, 99], 99, 4096)
        second['sampling'].update(duration_seconds=20, source_frames=200)
        counts = {k: second['tokens'][k] for k in ('input_visual_tokens', 'input_text_tokens', 'input_tokens')}
        second['input_identity'] = digest(dict(source=self.job['rows'][1], sampling=second['sampling'],
                                              tokens=counts, input_ids_sha256=second['input_ids_sha256']))
        _, pair = self.matrix('means', 2, complete=True)
        entry, _ = audit_pair(pair, self.job['protocol'], 2)
        metrics = entry['metrics']
        expected = dict(input_visual_tokens=20, input_text_tokens=7.5, output_text_tokens=2.5,
                        total_text_tokens=10, input_tokens=27.5, total_tokens=30)
        for key, mean in expected.items():
            self.assertEqual(metrics[key + '_mean'], mean)
        rendered = '\n'.join(average_table([entry]))
        self.assertIn('| 20.00 | 7.50 | 2.50 | 10.00 | 27.50 | 30.00 |', rendered)

    def test_video_length_means_label_excluded_long_population(self):
        from evaluation.audits.audit_video_token_rates import aggregate, duration_average_table, measured_rows
        rows = measured_rows(None, self.job, self.records)
        entry = dict(method='test', suite='videomme', population='short_medium', status='incomplete',
                     measurement_scope='partial', categories={'short': aggregate(rows[:1]),
                                                              'medium': aggregate(rows[1:])})
        before = copy.deepcopy(entry)
        rendered = '\n'.join(duration_average_table([entry]))
        self.assertIn('| test | short | 1/900 |', rendered)
        self.assertIn('| test | medium | 1/900 |', rendered)
        self.assertIn('| test | long | excluded | — | — | — | — | — | — | — | excluded from current protocol |', rendered)
        historical = dict(entry, population='full', categories={'long': aggregate(rows)})
        historical_table = '\n'.join(duration_average_table([historical], historical_long=True))
        self.assertIn('| test | long | 2/900 |', historical_table)
        self.assertIn('historical; excluded from current scores', historical_table)
        self.assertEqual(entry, before)

    def test_historical_long_evidence_is_collected_separately(self):
        from evaluation.audits.audit_video_token_rates import collect_historical_videomme
        self.job['suite'] = 'videomme'
        self.job['protocol']['version'] = 'video_unsaturated_v2'
        legacy, _ = self.matrix('legacy', 2, complete=True)
        policy = dict(self.job['protocol'], version='video_resilient_v3',
                      videomme_population={'name': 'short_medium'})
        with patch('evaluation.audits.audit_video_token_rates.expected_population', return_value=2):
            historical = collect_historical_videomme([legacy], [self.job['method']], policy, 0)
        self.assertEqual(len(historical), 1)
        self.assertEqual(historical[0]['report_scope'], 'historical_excluded')
        self.assertEqual(historical[0]['metrics']['questions'], 2)

    def test_question_stem_options_and_task_instructions_are_separate(self):
        from evaluation.audits.audit_video_token_rates import question_payloads
        text = 'What is happening?\nA. left\nB. right'
        prompt = text + '\nAnswer with only the option letter.'
        self.assertEqual(question_payloads('tempcompass', {'question': text}, {'prompt': prompt}),
                         ('What is happening?', text, prompt))
        source = dict(question_type='multiple-choice', options={'A': 'left', 'B': 'right'},
                      prompt='Watch and learn the content.\n' + prompt)
        stem, options, task = question_payloads('videommmu_adaptation', {'question': 'What is happening?'}, source)
        self.assertEqual(stem, 'What is happening?')
        self.assertEqual(options, text)
        self.assertTrue(task.startswith('Watch and learn'))

    def test_question_means_do_not_include_timestamp_or_chat_tokens(self):
        from evaluation.audits.audit_video_token_rates import add_question_metrics, question_length_table
        metrics = dict(suite='tempcompass', category=None, questions=2)
        add_question_metrics(metrics, [dict(question_only_tokens=2, question_with_options_tokens=5, task_prompt_tokens=8),
                                       dict(question_only_tokens=10, question_with_options_tokens=15, task_prompt_tokens=18)])
        metrics.update(full_text_input_mean_reference=1000, video_chat_residual_mean_reference=987)
        self.assertEqual(metrics['question_only_tokens_mean'], 6)
        self.assertEqual(metrics['task_prompt_tokens_mean'], 13)
        rendered = '\n'.join(question_length_table([metrics]))
        self.assertIn('| 6.00 | 10.00 | 13.00 | 1,000.00 | 987.00 |', rendered)


class RealProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = ROOT / 'models/Qwen3-VL-2B-Instruct'
        if not model.is_dir():
            raise unittest.SkipTest('Local processor assets unavailable')
        from transformers import AutoProcessor
        cls.processor = AutoProcessor.from_pretrained(model, local_files_only=True)

    def test_odd_frame_padding_and_exact_prompt_tokens(self):
        import torch
        from transformers.video_utils import VideoMetadata
        from evaluation.protocols.video_protocol import prompt_text
        processor = self.processor
        row = {'prompt': 'What happens in the video? Answer briefly.'}
        for n in (1, 3, 8):
            with self.subTest(frames=n):
                shape = geometry(n, 64, 96, 16, 2, 2)
                meta = dict(frame_indices=list(range(n)), source_fps=10, **shape)
                before = copy.deepcopy(meta)
                expected = input_counts(processor, row, meta)
                self.assertEqual(meta, before)  # HF timestamp padding mutates its argument
                processor.video_processor.size = {'shortest_edge': 1, 'longest_edge': shape['total_pixels'] + 1}
                actual = processor(text=[prompt_text(processor, row)],
                    videos=[torch.zeros(n, 3, 64, 96, dtype=torch.uint8)],
                    videos_kwargs=dict(do_resize=False, do_sample_frames=False,
                        video_metadata=[VideoMetadata(total_num_frames=n, fps=10, frames_indices=list(range(n)))]),
                    truncation=False, return_tensors='pt')
                ids = actual['input_ids'][0]
                visual = int((ids == processor.video_token_id).sum())
                self.assertEqual(expected, dict(input_visual_tokens=visual,
                    input_text_tokens=len(ids) - visual, input_tokens=len(ids)))
                self.assertEqual(actual['video_grid_thw'][0].tolist(), shape['grid'])

    def test_image_text_reconstruction_matches_real_processor(self):
        from PIL import Image
        from evaluation.audits.audit_image_visual_tokens import image_text_tokens
        processor = self.processor
        for count in (0, 1, 2):
            with self.subTest(images=count):
                prompt = 'Question: Which object is left?\n(A) cup\n(B) bowl\nAnswer with only the option letter.'
                content = [{'type': 'image'} for _ in range(count)] + [{'type': 'text', 'text': prompt}]
                rendered = processor.apply_chat_template([{'role': 'user', 'content': content}],
                                                        tokenize=False, add_generation_prompt=True)
                images = [Image.new('RGB', (96, 64)) for _ in range(count)]
                actual = processor(text=[rendered], **({'images': images} if images else {}), return_tensors='pt')
                ids = actual['input_ids'][0]
                visual = int((ids == processor.image_token_id).sum())
                self.assertEqual(image_text_tokens(processor, prompt, count), len(ids) - visual)


class MasterSuiteTokenTableTests(unittest.TestCase):
    def video_report(self):
        rows = []
        for suite, count in (('tempcompass', 1580), ('videomme', 1800),
                             ('videommmu_perception', 300), ('videommmu_comprehension', 300),
                             ('videommmu_adaptation', 300)):
            metrics = dict(questions=count, input_visual_tokens_mean=100, input_text_tokens_mean=20)
            rows.append(dict(suite=suite, method='vanilla_2b', status='complete', metrics=metrics,
                categories={c: dict(metrics, questions=900) for c in ('short', 'medium')},
                matrix='/validated/full', attempt='attempt_1', completion_marker_sha256='verified'))
        return dict(policy=dict(fps=2, longest_side=256), runs=rows)

    def test_master_video_rows_cover_whole_populations_and_exclude_long(self):
        from evaluation.audits.audit_image_visual_tokens import master_video_rows
        rows = master_video_rows(self.video_report())
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(r['questions'] for r in rows), 4280)
        self.assertNotIn('videomme_long', [r['suite'] for r in rows])
        self.assertEqual([r['questions'] for r in rows if r['suite'].startswith('videomme_')], [900, 900])

    def test_master_rejects_partial_prefix_and_different_video_policy(self):
        from evaluation.audits.audit_image_visual_tokens import master_video_rows
        for change in ('partial', 'population', 'policy'):
            report = self.video_report()
            if change == 'partial': report['runs'][0]['status'] = 'incomplete'
            if change == 'population': report['runs'][0]['metrics']['questions'] = 500
            if change == 'policy': report['policy']['longest_side'] = 320
            with self.subTest(change=change), self.assertRaises(ValueError):
                master_video_rows(report)

    def test_mmmu_image_selection_counts_unique_present_references(self):
        import re
        from types import SimpleNamespace
        from evaluation.audits.audit_image_visual_tokens import image_prompt_and_count
        worker = SimpleNamespace(IMAGE_TAG_RE=re.compile(r'<image (\d+)>'), _is_present=lambda x: x is not None,
            _parse_options=lambda x: ['left', 'right'], format_prompt_mc=lambda q, o: 'MC prompt',
            format_prompt_open=lambda q: 'Open prompt')
        row = dict(question='<image 2> <image 2> <image 3>', question_type='multiple-choice', options='test',
                   image_1=b'fallback', image_2=b'referenced', image_3=None)
        self.assertEqual(image_prompt_and_count('mmmu', worker, None, row), ('MC prompt', 1))
        row.update(question='No image tags', question_type='open')
        self.assertEqual(image_prompt_and_count('mmmu', worker, None, row), ('Open prompt', 1))


if __name__ == '__main__':
    unittest.main()
