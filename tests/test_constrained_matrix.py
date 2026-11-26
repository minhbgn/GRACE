"""CPU-only regression tests; no model downloads, GPU allocation or training."""
import copy
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/released_bf16_init_32k_b128_p100352_t2048_seed42'
CONFIG = EXP / 'experiment.json'
sys.path.insert(0, str(ROOT / 'qwen-vl-finetune/scripts'))
sys.path.insert(0, str(ROOT / 'evaluation'))
import constrained_experiment as common
import run_constrained_matrix_evals as evaluate
import summarize_constrained_evals as reports
import run_constrained as launcher
import run_constrained_matrix as matrix
SPEC = common.Experiment(CONFIG)
METHODS = tuple(SPEC.models)
STUDENT_HASH = SPEC.models[SPEC.student]['weights_sha256']


class LauncherTests(unittest.TestCase):
    def run_shell(self, name, *args):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Path(tmp) / 'never-created'
            env = dict(os.environ, PYTHON=sys.executable, EXPERIMENT_DIR=str(experiment))
            result = subprocess.run(['bash', str(EXP / name), *args], env=env,
                                    text=True, capture_output=True, check=True)
            self.assertFalse(experiment.exists(), 'Dry run wrote outputs')
            return result.stdout.splitlines()

    def test_four_fresh_training_routes(self):
        commands = self.run_shell('run_training.sh', 'all', '--train', '--dry-run')
        self.assertEqual(len(commands), 4)
        for line in commands:
            method, bit = line.split(']')[0][1:].split()
            bits = bit[1]
            tokens = shlex.split(line.split('] ', 1)[1])
            def value(flag):
                return tokens[tokens.index('--' + flag) + 1]
            self.assertTrue(value('model_name_or_path').endswith('models/Qwen3-VL-2B-GRACE-BF16'))
            self.assertEqual(value('qat_bits'), bits)
            self.assertTrue(value('output_dir').endswith(f'w{bits}g128/runs/{method}'))
            self.assertEqual(value('max_steps'), '256')
            self.assertEqual(value('gradient_accumulation_steps'), '64')
            self.assertEqual(value('rcka_warmup_steps'), '25')
            self.assertEqual(value('resume_run'), 'False')
            self.assertEqual(value('grace_enable'), str(method == 'grace'))
            self.assertEqual('--teacher_model_path' in tokens, method == 'grace')

    def test_profile_output_is_separate(self):
        line, = self.run_shell('run_training.sh', 'qat', '--bits', '8', '--profile', '--dry-run')
        self.assertIn('/w8g128/profiles/qat', line)
        self.assertIn('--max_steps 10', line)
        self.assertIn('--profile_run True', line)

    def test_evaluation_matrix_and_explicit_policy(self):
        lines = self.run_shell('run_local_evaluation.sh', '--dry-run')
        self.assertEqual(len(lines), 70)
        outputs = set()
        for line in lines:
            tokens = shlex.split(line)
            self.assertIn('--fixed_generation', tokens)
            self.assertEqual(tokens[tokens.index('--max_pixels') + 1], '100352')
            model = tokens[tokens.index('--model_path') + 1]
            self.assertEqual(model, tokens[tokens.index('--processor_path') + 1])
            outputs.add(tokens[tokens.index('--output') + 1])
        self.assertEqual(len(outputs), 70)
        self.assertTrue(any('/evaluation/sft_2b/' in p for p in outputs))
        self.assertTrue(any('/evaluation/vanilla_8b_teacher/' in p for p in outputs))

    def test_old_training_default_stays_w4_vanilla(self):
        sys.path.insert(0, str(common.PROJECT / 'scripts'))
        import run_constrained
        with patch.dict(os.environ, {}, clear=True):
            c = run_constrained.resolved_config()
        self.assertEqual(c['qat_bits'], 4)
        self.assertTrue(c['model_path'].endswith('models/Qwen3-VL-2B-Instruct'))

    def test_weight_identity_rejects_renamed_vanilla(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, data in [('config.json', {'model_type': 'qwen3_vl'}),
                               ('tokenizer.json', {}), ('tokenizer_config.json', {}),
                               ('processor_config.json', {})]:
                (root / name).write_text(json.dumps(data))
            header = json.dumps({'model.test.weight': {'dtype': 'BF16', 'shape': [1], 'data_offsets': [0, 2]}}).encode()
            (root / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\x00\x00')
            with self.assertRaisesRegex(ValueError, 'not the pinned'):
                common.provenance(root, expected_weights_sha256=STUDENT_HASH)

    def test_matrix_delegates_actual_launch_to_shared_launcher(self):
        from contextlib import redirect_stdout
        from io import StringIO
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'EXPERIMENT_DIR': tmp}), \
                patch.object(common.Experiment, 'provenance', return_value={'fixture': True}), \
                patch.object(matrix, 'check_processor_compatibility'), \
                patch.object(matrix, 'prepare', return_value={'records_sha256': 'fixture'}), \
                patch.object(launcher, 'comparison_recipe', return_value={'implementation': {}}), \
                patch.object(launcher, 'check_gpus'), \
                patch.object(launcher, 'launch_run') as launch, \
                patch.object(matrix.subprocess, 'run'), \
                patch.object(sys, 'argv', ['matrix', '--config', str(CONFIG), 'all', '--train']), \
                redirect_stdout(StringIO()):
            matrix.main()
            self.assertEqual(len(launch.call_args_list), 4)
            self.assertEqual([(call.args[0], call.args[1]['qat_bits']) for call in launch.call_args_list],
                             [('grace', 4), ('qat', 4), ('grace', 8), ('qat', 8)])
            for call in launch.call_args_list:
                self.assertTrue(call.args[1]['model_path'].endswith('Qwen3-VL-2B-GRACE-BF16'))
                self.assertFalse(call.args[4])  # full run, not profile

    def test_shared_launcher_rejects_missing_resume_and_completed_output(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            group = Path(tmp)
            config = launcher.resolved_config()
            with patch.object(launcher.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'existing checkpoint'):
                    launcher.launch_run('grace', config, group, {}, resume=True)
                output = group / 'runs/grace'
                (output / 'EXPORT_COMPLETE').write_text('complete')
                with self.assertRaisesRegex(ValueError, 'Completed output'):
                    launcher.launch_run('grace', config, group, {})
                run.assert_not_called()

    def test_matrix_runtime_cli_sums_resumed_invocations(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Path(tmp)
            for bits in (4, 8):
                for method in ('qat', 'grace'):
                    output = experiment / f'w{bits}g128/profiles/{method}'
                    common.pin(output / 'run_summary.json', dict(complete=True, profile=True,
                               global_step=10, metrics={'train_runtime': 40}))
                    common.pin(output / 'launch_recipe.json', {'comparison': {'config': dict(
                        global_batch=128, max_pixels=100352, model_max_length=2048, max_steps=256)}})
                    common.pin(output / 'trainable_parameters.json', {'adapter': {'parameters': 32}})
                    (output / 'EXPORT_COMPLETE').write_text('complete')
                    rows = [dict(step=i, seconds=1, cycle_seconds=1.5, peak_allocated_gib=1,
                                 peak_reserved_gib=2) for i in (1, 2)]
                    (output / 'step_metrics.jsonl').write_text('\n'.join(map(json.dumps, rows)))
                    invocations = [dict(wall_seconds=120, train_runtime_seconds=100),
                                   dict(wall_seconds=60, train_runtime_seconds=40)]
                    (output / 'invocations.jsonl').write_text('\n'.join(map(json.dumps, invocations)))
            subprocess.run([sys.executable, str(common.PROJECT / 'scripts/summarize_constrained_runtime.py'),
                            '--config', str(CONFIG), '--kind', 'profiles'],
                           env=dict(os.environ, EXPERIMENT_DIR=tmp), check=True, capture_output=True, text=True)
            rows = common.read(experiment / 'reports/profile_runtime_memory.json')
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertAlmostEqual(row['process_gpu_hours'], 0.1)
                self.assertAlmostEqual(row['outside_recorded_training_hours'], 40 / 3600)
                self.assertEqual(row['trainable_parameters'], 32)


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.model = Path('/test/released-model')
        self.payload = dict(summary=dict(
            model_path=str(self.model), processor_path=str(self.model),
            image_processor_size=dict(shortest_edge=4096, longest_edge=100352),
            total=2, correct=1, overall_acc=0.5,
            evaluation_protocol=dict(version=1, min_pixels=4096, max_pixels=100352,
                                     max_new_tokens=8, do_sample=False, num_beams=1,
                                     repetition_penalty=1.0, context='no truncation; native model context',
                                     attention='sdpa')),
            results=[dict(index=1, gold='A', correct=True), dict(index=2, gold='B', correct=False, pred='', error=None)])

    def validate(self, data):
        path = self.out / 'mmstar.json'
        path.write_text(json.dumps(data))
        with patch.dict(evaluate.EXPECTED, mmstar=2):
            return evaluate.validate_result(path, 'mmstar', self.model, 'sdpa')

    def test_failed_parse_retained_and_good_result_resumable(self):
        self.assertEqual(self.validate(self.payload)['summary']['overall_acc'], 0.5)

    def test_incomplete_duplicate_wrong_policy_and_wrong_model_rejected(self):
        mutations = [lambda p: p['results'].pop(),
                     lambda p: p['results'][1].update(index=1),
                     lambda p: p['summary']['evaluation_protocol'].update(max_pixels=50176),
                     lambda p: p['summary'].update(model_path='/wrong/model'),
                     lambda p: p['summary'].update(correct=2)]
        for mutation in mutations:
            data = copy.deepcopy(self.payload)
            mutation(data)
            with self.assertRaises(ValueError):
                self.validate(data)

    def test_pin_rejects_recipe_drift(self):
        path = self.out / 'recipe.json'
        common.pin(path, {'bits': 4})
        with self.assertRaisesRegex(ValueError, 'Pinned artifact changed'):
            common.pin(path, {'bits': 8})

    def test_export_rejects_profile_wrong_bits_or_wrong_initialization(self):
        common.pin(self.out / 'run_summary.json', {'complete': True, 'profile': False, 'global_step': 256})
        (self.out / 'EXPORT_COMPLETE').write_text('complete')
        config = dict(max_steps=256, qat_bits=8, model_path=str(self.model),
                      student_weights_sha256=STUDENT_HASH)
        common.pin(self.out / 'launch_recipe.json', {'comparison': {'config': config}})
        common.pin(self.out / 'config.json', {'qat': {'bits': 8, 'group_size': 128, 'num_replaced': 196},
                                             'qat_lora': {'merged': True}})
        with patch.object(common, 'checkpoint_files', return_value=[]):
            common.validate_export(self.out, 8, self.model)
            with self.assertRaisesRegex(ValueError, 'bit width'):
                common.validate_export(self.out, 4, self.model)
            with self.assertRaisesRegex(ValueError, 'initialization'):
                common.validate_export(self.out, 8, Path('/vanilla'))
            (self.out / 'run_summary.json').write_text(json.dumps({'complete': True, 'profile': True, 'global_step': 256}))
            with self.assertRaisesRegex(ValueError, 'full-training export'):
                common.validate_export(self.out, 8, self.model)

    def test_complete_matrix_summary_and_cross_model_gold_validation(self):
        policy = {'attention': 'sdpa', 'comparisons': SPEC.spec['comparisons'], 'models': {m: {'model': str(self.model)} for m in METHODS}}
        common.pin(self.out / 'protocol.json', policy)
        with patch.dict(evaluate.EXPECTED, {s: 2 for s in evaluate.SUITES}):
            for method in METHODS:
                common.pin(self.out / method / 'provenance.json', {'fixture': True})
                for suite, (_, metric) in evaluate.SUITES.items():
                    data = copy.deepcopy(self.payload)
                    data['summary'][metric] = .5
                    data['summary']['evaluation_protocol']['max_new_tokens'] = evaluate.suites()[suite][1]
                    if suite == 'mmmu':
                        data['summary']['evaluation_protocol']['multiple_choice_max_new_tokens'] = 8
                    if suite == 'hallusionbench':
                        data['summary'].update(qAcc=.5, fAcc=.5, qAcc_total_groups=2, fAcc_total_groups=2)
                    for i, row in enumerate(data['results']):
                        row.update({key: i for key in reports.ID_FIELDS[suite]})
                    common.pin(self.out / method / f'{suite}.json', data)
            reports.summarize_matrix(self.out)
            self.assertTrue((self.out / 'comparison.csv').exists())
            self.assertIn('W8: distillation gain', (self.out / 'deltas.csv').read_text())
            path = self.out / 'grace_w8/mmstar.json'
            data = common.read(path)
            data['results'][1]['gold'] = 'C'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'identities/labels differ'):
                reports.summarize_matrix(self.out)


if __name__ == '__main__':
    unittest.main()
