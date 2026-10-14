"""CPU regression tests for answer interpretation, grouping and run integrity."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_scoring import choice, yesno, open_answer, generation_record, score_rows, corrected_summary, METRICS
from run_local_development import validate, command, PROTOCOLS, cap_for


class ExtractionTests(unittest.TestCase):
    def test_known_parser_regressions(self):
        for text, valid, expected in [
            ('The correct choice is A', 'ABCD', 'A'),
            ('The answer is B', 'ABCDE', 'B'),
            ('C: A forest', 'ABCD', 'C'),
            ('**C**: A forest', 'ABCD', 'C'),
            ('To solve for the missing account balance,', 'ABCDE', ''),
            ('Because the', 'ABCDE', ''),
            ('a forest', 'ABCD', ''), ('A forest is visible', 'ABCD', ''),
            ('(b)', 'ABCD', 'B'), ('d.', 'ABCD', 'D'),
            ('We compare A and B. The final answer is C.', 'ABCD', 'C'),
            ('A or B', 'ABCD', ''), ('Answer is A. Answer is B.', 'ABCD', ''),
            ('Answer is E', 'ABCD', ''), ('', 'ABCD', ''),
            (r'\boxed{B}', 'ABCD', 'B'), ('<answer>C</answer>', 'ABCD', 'C'),
            ('<think>Maybe A', 'ABCD', ''),
            ('<think>A or B</think>\nC', 'ABCD', 'C'),
        ]:
            with self.subTest(text=text):
                self.assertEqual(choice(text, valid)[0], expected)

    def test_option_text_and_gold_independence(self):
        self.assertEqual(choice('Blue whale.', 'AB', {'A': 'Blue whale', 'B': 'Dog'})[0], 'A')
        self.assertEqual(choice('dog', 'AB', {'A': 'dog', 'B': 'dog'})[0], '')
        rows = [dict(raw='The answer is B', gold=g, error=None, n_options=4) for g in ('A', 'B')]
        score_rows('mmstar', rows)
        self.assertEqual([r['pred'] for r in rows], ['B', 'B'])
        self.assertEqual([r['correct'] for r in rows], [False, True])

    def test_yesno_not_copula(self):
        for text, expected in [('Yes.', '1'), ('No, it is not.', '0'), ('The answer is No', '0'),
                               ('It is unclear', ''), ('Is this correct?', ''), ('', '')]:
            self.assertEqual(yesno(text)[0], expected)

    def test_mmmu_open_typed_candidates(self):
        for text, gold, expected in [('42', '42', True), ('41', '42', False),
            ('The answer is a blue whale.', 'blue whale', True),
            ('feline', ['cat', 'feline'], True), ('1,234', '1234', True),
            ('1.2e3', '1200', True), ('-3.14159', '-3.14', True),
            ('The result is 12.346', '12.35', True), ('12.3', '12.35', False),
            ('dog', 'cat', False), ('', '42', False)]:
            with self.subTest(text=text):
                candidates, correct = open_answer(text, gold)
                self.assertIsInstance(candidates, list)
                self.assertEqual(correct, expected)
        self.assertIn(42.0, open_answer('42', '42')[0])

    def test_finish_reason(self):
        self.assertEqual(generation_record([1, 2], 2, 2)['finish_reason'], 'eos')
        self.assertFalse(generation_record([1, 2], 2, 2)['truncated'])
        self.assertTrue(generation_record([1, 3], [2, 4], 2)['truncated'])
        self.assertEqual(generation_record([1, 4], [2, 4], 8)['finish_reason'], 'eos')
        self.assertEqual(generation_record([1], 2, 8)['finish_reason'], 'other')

    def test_mmb_group_completeness_bounds(self):
        from rescore_saved_protocols import circular
        from local_scoring import mmbench_evaluator
        meta = {k: dict(A='cat', B='dog', C=float('nan'), D=float('nan'), answer=g)
                for k, g in [(1, 'A'), (1000001, 'B'), (2, 'A')]}
        rows = [dict(index=k, gold=v['answer'], raw=v['answer'], error=None) for k, v in meta.items()]
        result = circular(rows, meta, mmbench_evaluator())
        self.assertEqual(result['exact_match_circular_percent'], 100)
        rows[1]['raw'] = 'unclear'
        result = circular(rows, meta, mmbench_evaluator())
        self.assertEqual(result['judge_completion_bounds_percent'], [50, 100])
        rows[0]['raw'] = 'B'
        self.assertEqual(circular(rows, meta, mmbench_evaluator())['judge_completion_bounds_percent'], [50, 50])
        with self.assertRaises(ValueError):
            circular(rows[:-1], meta, mmbench_evaluator())
        with self.assertRaises(ValueError):
            circular(rows + rows[:1], meta, mmbench_evaluator())

    def test_smoke_never_reports_partial_circular_score(self):
        row = dict(index=1, raw='A', pred='A', gold='A', correct=True, unresolved=False, error=None)
        result = corrected_summary('mmbench', [row], partial=True)
        self.assertIsNone(result['primary_score'])
        self.assertEqual(result['generation_unknown'], 1)


class IntegrityTests(unittest.TestCase):
    def fixture(self):
        rows = [dict(index=0, raw='The answer is B', gold='B', error=None, n_options=4,
                     generation=generation_record([7, 9], 9, 256))]
        score_rows('mmstar', rows)
        summary = dict(model_path='/model', processor_path='/model', image_processor_size={'longest_edge': 50176},
            total=1, correct=1, overall_acc=1.0,
            evaluation_protocol=dict(version=1, min_pixels=None, max_pixels=None, max_new_tokens=256,
                do_sample=False, num_beams=1, repetition_penalty=1.0,
                context='no truncation; native model context', attention='sdpa'))
        summary.update(corrected_summary('mmstar', rows))
        return dict(summary=summary, results=rows)

    def check(self, payload):
        with patch('run_local_development.source_samples', return_value=[((0,), 'B')]), \
             patch.dict('run_local_development.EXPECTED', mmstar=1), \
             patch('run_local_development.metadata', return_value={}):
            return validate(payload, 'mmstar', Path('/model'), 'local_native_greedy256_v1', 'sdpa', None,
                            {'image_size': {'longest_edge': 50176}, 'eos_token_id': 9})

    def test_validate_and_reject_tampering(self):
        self.assertEqual(self.check(self.fixture())['score_percent'], 100)
        for field, value in [('index', 3), ('gold', 'C'), ('pred', 'C'), ('correct', False),
                             ('generation', None), ('error', 'OOM')]:
            payload = self.fixture()
            payload['results'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(payload)
        payload = self.fixture()
        payload['summary']['primary_score'] = .5
        with self.assertRaises(ValueError):
            self.check(payload)
        payload = self.fixture()
        payload['summary']['image_processor_size']['longest_edge'] = 100352
        with self.assertRaises(ValueError):
            self.check(payload)

    def test_mmbench_json_roundtrip_and_histogram_tampering(self):
        meta = {k: dict(A='cat', B='dog', C=float('nan'), D=float('nan'), answer=g)
                for k, g in [(1, 'A'), (1000001, 'B'), (2, 'A')]}
        rows = [dict(index=k, gold=v['answer'], raw=v['answer'], error=None,
                     generation=generation_record([7, 9], 9, 256)) for k, v in meta.items()]
        score_rows('mmbench', rows, meta)
        summary = self.fixture()['summary']
        summary.update(total=3, correct=3, overall_acc=1.0)
        summary.update(corrected_summary('mmbench', rows, meta))
        payload = json.loads(json.dumps(dict(summary=summary, results=rows)))
        with patch('run_local_development.source_samples', return_value=[((k,), v['answer']) for k, v in meta.items()]), \
             patch.dict('run_local_development.EXPECTED', mmbench=3), \
             patch('run_local_development.metadata', return_value=meta):
            check = lambda data: validate(data, 'mmbench', Path('/model'),
                'local_native_greedy256_v1', 'sdpa', None,
                {'image_size': {'longest_edge': 50176}, 'eos_token_id': 9})
            self.assertEqual(check(payload)['score_percent'], 100)
            self.assertEqual(payload['summary']['circular']['rotation_count_histogram'], {'2': 1, '1': 1})
            payload['summary']['primary_score'] = .5
            with self.assertRaises(ValueError):
                check(payload)
            payload['summary']['primary_score'] = 1.0
            payload['summary']['circular']['rotation_count_histogram']['2'] = 0
            with self.assertRaises(ValueError):
                check(payload)

    def test_commands_preserve_native_processors_and_caps(self):
        for protocol in PROTOCOLS:
            for suite in METRICS:
                cmd = command('m', suite, {'m': Path('/m')}, Path('/out'), protocol, 'sdpa', None)
                self.assertNotIn('--max_pixels', cmd)
                self.assertNotIn('--min_pixels', cmd)
                self.assertEqual(cmd[cmd.index('--processor_path') + 1], '/m')
                self.assertIn('--local_scoring', cmd)
        self.assertEqual(cap_for('local_native_greedy_short_v1', 'mmmu', {'question_type':'open'}), 32)
        self.assertEqual(cap_for('local_native_greedy256_v1', 'mmmu'), 256)


if __name__ == '__main__':
    unittest.main()
