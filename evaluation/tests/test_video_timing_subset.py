"""Sampling and extrapolation checks; no model or GPU allocation."""
import unittest

from evaluation.protocols.video_timing_subset import select_samples, estimate, phase_configs, historical_fit


def population():
    return [dict(id=f'{duration}-{v}-{q}', video_id=f'{duration}-{v}', duration=duration,
                 original_visual_tokens=(v + 1) * 576)
            for duration in ('short', 'medium') for v in range(300) for q in range(3)]


class TimingSubsetTest(unittest.TestCase):
    def test_resolution_phase_cannot_dispatch_uniform_or_merge(self):
        configs = [('baseline', 1.)] + [(m, r) for m in ('resolution', 'uniform', 'merge') for r in (.5, .25, .1)]
        first = phase_configs('resolution_r050,resolution_r025,resolution_r010', configs)
        self.assertEqual(first, [('resolution', .5), ('resolution', .25), ('resolution', .1)])
        second = phase_configs('baseline_r100,uniform_r050,uniform_r025,uniform_r010,merge_r050,merge_r025,merge_r010', configs)
        self.assertEqual(set(first) | set(second), set(configs))
        self.assertFalse(set(first) & set(second))
        for invalid in ('', 'resolution_r050,resolution_r050', 'resolution_r005'):
            with self.assertRaises(ValueError):
                phase_configs(invalid, configs)

    def test_backtest_checks_duration_errors_even_if_total_cancels(self):
        selected, _ = select_samples(population())
        full = {r['id']: dict(duration=r['duration'], seconds=10) for r in population()}
        self.assertTrue(historical_fit(selected, full, bootstrap=50)['passed'])
        for r in selected:
            full[r['id']]['seconds'] = 12 if r['duration'] == 'short' else 8
        fit = historical_fit(selected, full, bootstrap=50)
        self.assertAlmostEqual(fit['relative_error'], 0)
        self.assertTrue(fit['checks']['total_error'])
        self.assertFalse(fit['checks']['duration_errors'])
        self.assertFalse(fit['passed'])

    def test_sampling_units_weights_and_reproducibility(self):
        selected, strata = select_samples(population())
        self.assertEqual((selected, strata), select_samples(list(reversed(population()))))
        self.assertEqual(len({r['video_id'] for r in selected}), 50)
        self.assertEqual(sum(r['duration'] == 'short' for r in selected), 25)
        self.assertEqual(sum(r['expansion_weight'] for r in selected), 1800)
        for stratum in strata:
            members = [r for r in selected if r['stratum'] == stratum['name']]
            self.assertEqual(len(members), 5)
            self.assertTrue(all(stratum['min_original_visual_tokens'] <= r['original_visual_tokens']
                                <= stratum['max_original_visual_tokens'] for r in members))

    def test_closed_form_expansion_and_paired_uncertainty(self):
        selected, _ = select_samples(population())
        measured = {r['id']: 10 if r['duration'] == 'short' else 40 for r in selected}
        baseline = {k: v * 2 for k, v in measured.items()}
        result = estimate(selected, measured, baseline, bootstrap=100)
        self.assertEqual(result['full_seconds'], 900 * 10 + 900 * 40)
        self.assertEqual(result['full_seconds_ci95'], [45000, 45000])
        self.assertEqual(result['by_duration'], {'short': 9000, 'medium': 36000})
        self.assertEqual(result['speedup_ci95'], [2, 2])

    def test_population_and_timing_mismatches_fail(self):
        rows = population()
        with self.assertRaises(ValueError):
            select_samples(rows[:-1])
        rows[0]['id'] = rows[1]['id']
        with self.assertRaises(ValueError):
            select_samples(rows)
        selected, _ = select_samples(population())
        measured = {r['id']: 1 for r in selected}
        measured[selected[0]['id']] = float('nan')
        with self.assertRaises(ValueError):
            estimate(selected, measured)
        with self.assertRaises(ValueError):
            estimate(selected, {})


if __name__ == '__main__':
    unittest.main()
