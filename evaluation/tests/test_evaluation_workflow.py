"""Job selection and corruption protection for the consolidated entry point."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.runners import evaluate


def options(**values):
    data = dict(action='run', workflow='token-reduction', output=evaluate.TOKEN / 'outputs/reruns/test',
                suites=None, configurations=None, models=None, gpus='0,2', smoke=None, max_score_drift=0.5)
    return argparse.Namespace(**dict(data, **values))


def flag(command, name):
    return command[command.index(name)+1]


class WorkflowTests(unittest.TestCase):
    def test_all_120_jobs_are_selected_once_with_the_recorded_runtime(self):
        jobs = {}
        for command in evaluate.commands(options()):
            suites = flag(command, '--suites').split(',')
            if command[1].endswith('run_token_reduction_matrix.py'):
                methods = flag(command, '--methods').split(',')
                ratios = list(map(float, flag(command, '--retentions').split(',')))
                configs = (['baseline_r100'] if 1 in ratios else []) + [
                    f'{method}_r{round(r*100):03}' for method in methods for r in ratios if r != 1]
                runtime = 'original'
                self.assertEqual(flag(command, '--models'), 'qat_w4')
            else:
                configs = flag(command, '--configurations').split(',')
                runtime = 'adaptive'
                self.assertEqual(flag(command, '--parallel-evaluation'), '1')
                self.assertEqual(flag(command, '--decoder-token-cap'), '131072')
            for suite in suites:
                for config in configs:
                    self.assertNotIn((suite, config), jobs)
                    jobs[suite, config] = runtime
        self.assertEqual(len(jobs), 120)
        for (suite, config), runtime in jobs.items():
            expected = 'original' if suite in evaluate.IMAGES or (
                suite in evaluate.VIDEOS[2:] and config == 'resolution_r010') else 'adaptive'
            self.assertEqual(runtime, expected, (suite, config))

    def test_filters_do_not_expand_configuration_cartesian_product(self):
        cmds = evaluate.commands(options(suites='ai2d', configurations='resolution_r025,merge_r010'))
        self.assertEqual(len(cmds), 2)
        self.assertEqual([(flag(c, '--methods'), flag(c, '--retentions')) for c in cmds],
                         [('resolution', '0.25'), ('merge', '0.1')])

    def test_smoke_outputs_are_kept_separate_in_every_component(self):
        for cmd in evaluate.commands(options(smoke=1)):
            self.assertEqual(flag(cmd, '--smoke'), '1')

    def test_timing_plan_uses_a_supported_prepare_command(self):
        cmd, = evaluate.commands(options(action='plan', workflow='timing'))
        self.assertEqual(cmd[2], 'prepare')
        self.assertEqual(flag(cmd, '--seed'), '42')
        self.assertNotIn('--gpus', cmd)

    def test_plan_runs_no_child_process_and_run_needs_explicit_gpus(self):
        with patch.object(sys, 'argv', ['evaluate', 'plan']), patch.object(evaluate.subprocess, 'run') as run:
            self.assertEqual(evaluate.main(), 0)
            run.assert_not_called()
        with patch.object(sys, 'argv', ['evaluate', 'run']):
            with self.assertRaises(SystemExit):
                evaluate.main()

    def test_compare_rejects_changed_artifacts_and_different_population(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'reports').mkdir()
            ref = dict(suite='ai2d', config='baseline_r100', score_percent=80, sample_identity='full',
                       count=10, policy={'seed':42}, execution=None)
            (root / 'reports/merged_results.json').write_text(json.dumps(dict(rows=[ref])))
            output = root / 'outputs/reruns/test'
            directory = output / 'images/full/qat_w4/ai2d/baseline_r100'
            attempt = directory / 'attempt_test'
            attempt.mkdir(parents=True)
            values = dict(score_percent=80, sample_identity='full', count=10)
            (attempt / 'validated.json').write_text(json.dumps(values))
            (attempt / 'exit.json').write_text(json.dumps(dict(returncode=0)))
            (attempt / 'job.json').write_text(json.dumps(dict(policy={'seed':42})))

            def complete():
                done = dict(values, attempt=attempt.name, sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                                               for p in attempt.iterdir()})
                (directory / 'COMPLETE.json').write_text(json.dumps(done))

            complete()
            args = options(output=output, suites='ai2d', configurations='baseline_r100')
            with patch.object(evaluate, 'ROOT', root):
                self.assertEqual(evaluate.compare(args), 0)
                (attempt / 'validated.json').write_text('{"tampered": true}')
                with self.assertRaisesRegex(ValueError, 'Corrupt rerun artifact'):
                    evaluate.compare(args)
                values['sample_identity'] = 'subset'
                (attempt / 'validated.json').write_text(json.dumps(values))
                complete()
                self.assertEqual(evaluate.compare(args), 1)

    def test_absolute_entry_point_works_outside_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, str(Path(evaluate.__file__)), 'plan'], cwd=temp,
                                    text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count('--suites'), 4)


if __name__ == '__main__':
    unittest.main()
