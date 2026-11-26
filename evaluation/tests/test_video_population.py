"""Reduced-population integrity, smoke ordering and report labeling."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.protocols.video_population import VIDEOMME_SHORT_MEDIUM, select_population
from evaluation.protocols.video_protocol import digest
from evaluation.runners import run_video_matrix_resilient as runner


def fixture():
    rows = [dict(id=str(i), video_id=str(i // 3), asset={'size': 1},
                 category=('short' if i < 900 else 'medium' if i < 1800 else 'long'))
            for i in range(2700)]
    info = dict(status='ready', questions=2700, videos=900, missing_videos=[],
                sample_identity=digest(rows), annotation_sha256={'test': 'sha'})
    return rows, info


class PopulationTests(unittest.TestCase):
    def test_exact_suffix_removed_with_original_ids_and_audit(self):
        rows, info = fixture()
        original = copy.deepcopy((rows, info))
        kept, report = select_population('videomme', rows, info, True)
        self.assertEqual(kept, rows[:1800])
        self.assertEqual(report['questions'], 1800)
        self.assertEqual(report['videos'], 600)
        self.assertEqual(report['population'], VIDEOMME_SHORT_MEDIUM)
        self.assertEqual(report['excluded_ids'], [r['id'] for r in rows[1800:]])
        self.assertEqual(report['sample_identity'], digest(kept))
        self.assertEqual(report['source_population'], info)
        self.assertEqual((rows, info), original)

    def test_default_and_other_suites_unchanged(self):
        rows, info = fixture()
        for suite, skip in [('videomme', False), ('tempcompass', True),
                            ('videommmu_perception', True)]:
            kept, report = select_population(suite, rows, info, skip)
            self.assertIs(kept, rows)
            self.assertIs(report, info)
        self.assertEqual(select_population('videomme', [], {'status': 'unavailable'}, True),
                         ([], {'status': 'unavailable'}))

    def test_changed_order_counts_labels_or_duplicate_ids_fail_closed(self):
        for mutation in ('order', 'count', 'label', 'duplicate'):
            rows, info = fixture()
            if mutation == 'order': rows[0], rows[-1] = rows[-1], rows[0]
            if mutation == 'count': rows.pop()
            if mutation == 'label': rows[1800]['category'] = 'medium'
            if mutation == 'duplicate': rows[-1]['id'] = rows[0]['id']
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, '900 long'):
                select_population('videomme', rows, info, True)

    def test_only_selected_missing_assets_block_subset(self):
        rows, info = fixture()
        rows[-1]['asset'] = None
        info.update(status='incomplete', missing_videos=[rows[-1]['video_id']])
        _, report = select_population('videomme', rows, info, True)
        self.assertEqual(report['status'], 'ready')
        rows[0]['asset'] = None
        _, report = select_population('videomme', rows, info, True)
        self.assertEqual(report['status'], 'incomplete')
        self.assertEqual(report['missing_videos'], ['0'])

    def test_runner_all_models_selection_precedes_smoke_and_changes_namespace(self):
        exp = SimpleNamespace(models={'a': {}, 'b': {}}, paths={'a': Path('/a'), 'b': Path('/b')},
                              directory=Path('/test'), spec={})
        manifests = []
        for extra in ([], ['--skip-videomme-long'], ['--skip-videomme-long', '--smoke', '1']):
            stream = io.StringIO()
            with patch.object(runner, 'Experiment', return_value=exp), \
                    patch.object(runner, 'load_suite', side_effect=lambda *a: fixture()), \
                    patch.object(sys, 'argv', ['runner', 'plan', '--suites', 'videomme', *extra]), \
                    contextlib.redirect_stdout(stream):
                runner.main()
            manifests.append(json.loads(stream.getvalue()))
        full, subset, smoke = manifests
        self.assertEqual(subset['jobs'], 2)
        self.assertEqual(list(subset['model_paths']), ['a', 'b'])
        self.assertEqual(full['evaluation_questions']['videomme'], 2700)
        self.assertEqual(subset['evaluation_questions']['videomme'], 1800)
        self.assertEqual(smoke['evaluation_questions']['videomme'], 1)
        self.assertEqual(smoke['data']['videomme']['questions'], 1800)
        self.assertNotEqual(full['output'], subset['output'])
        self.assertNotEqual(subset['output'], smoke['output'])

    def test_comparison_explicitly_labels_subset(self):
        rows, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            runner.summarize(output, ['a', 'b'], {'videomme': rows[:1800], 'tempcompass': []},
                             {'videomme_population': VIDEOMME_SHORT_MEDIUM})
            self.assertIn('900 long questions excluded', (output / 'comparison.md').read_text())
            records = json.loads((output / 'comparison.json').read_text())
            self.assertEqual([r['population']['evaluation_questions'] for r in records
                              if r['suite'] == 'videomme'], [1800, 1800])
            self.assertTrue(all('population' not in r for r in records if r['suite'] == 'tempcompass'))


if __name__ == '__main__':
    unittest.main()
