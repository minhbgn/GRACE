"""CPU-only regression checks for recipe compatibility and checkpoint pause."""
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / 'qwen-vl-finetune'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load('recipe', ROOT / 'scripts/grace_resume_recipe.py').prepare
Callback = load('pause', ROOT / 'qwenvl/train/grace_resume.py').CheckpointPauseCallback


class ResumeTests(unittest.TestCase):
    def test_recipe(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            pairs = ['TRAINING_BACKEND=ddp', 'LEARNING_RATE=0.0002']
            prepare(output, 2, 512, '', pairs)
            checkpoint = output / 'checkpoint-50'
            checkpoint.mkdir()
            for name in ('trainer_state.json', 'grace_ib_state.json', 'optimizer.pt', 'scheduler.pt'):
                (checkpoint / name).touch()
            (checkpoint / 'launch_recipe.json').write_text((output / 'launch_recipe.json').read_text())
            prepare(output, 4, 512, '', pairs)
            with self.assertRaisesRegex(SystemExit, 'effective_batch'):
                prepare(output, 4, 1024, '', pairs)
            with self.assertRaisesRegex(SystemExit, 'LEARNING_RATE'):
                prepare(output, 4, 512, '', ['TRAINING_BACKEND=ddp', 'LEARNING_RATE=0.001'])
            (checkpoint / 'optimizer.pt').unlink()
            with self.assertRaisesRegex(SystemExit, 'missing resumable state'):
                prepare(output, 4, 512, '', pairs)

    def test_zero_world_change_rejected(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            config = output / 'zero.json'
            config.write_text('{"zero_optimization": {"stage": 2}}')
            pairs = ['TRAINING_BACKEND=zero2']
            prepare(output, 2, 512, config, pairs)
            checkpoint = output / 'checkpoint-50'
            checkpoint.mkdir()
            (checkpoint / 'launch_recipe.json').write_text((output / 'launch_recipe.json').read_text())
            with self.assertRaisesRegex(SystemExit, 'world_size'):
                prepare(output, 4, 512, config, pairs)

    def test_pause_and_restart(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            args = SimpleNamespace(output_dir=directory)
            state = SimpleNamespace(is_world_process_zero=True, global_step=50)
            control = SimpleNamespace(should_training_stop=False)
            callback = Callback()
            checkpoint = output / 'checkpoint-50'
            checkpoint.mkdir()
            recipe = output / 'launch_recipe.json'
            recipe.write_text('{}')
            with patch.dict(os.environ, GRACE_LAUNCH_RECIPE=str(recipe)):
                callback.on_save(args, state, control)
                self.assertFalse(callback.paused)
                (output / 'STOP_AFTER_CHECKPOINT').touch()
                callback.on_save(args, state, control)
            self.assertTrue(callback.paused)
            self.assertTrue(control.should_training_stop)
            self.assertFalse((output / 'STOP_AFTER_CHECKPOINT').exists())
            self.assertTrue((checkpoint / 'launch_recipe.json').exists())
            self.assertEqual((output / 'PAUSED_AT_CHECKPOINT').read_text().strip(), str(checkpoint))
            Callback().on_train_begin(args, state, control)
            self.assertFalse((output / 'PAUSED_AT_CHECKPOINT').exists())


if __name__ == '__main__':
    unittest.main()
