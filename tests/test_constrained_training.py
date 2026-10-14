"""CPU checks for the matched experiment; no model weights or CUDA required."""
import copy
from contextlib import redirect_stdout
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/grace-constrained-triton")
ROOT = Path(__file__).resolve().parents[1] / "qwen-vl-finetune"
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import TrainingArguments
from transformers.image_utils import SizeDict
from qwenvl.data.constrained_data import allocate_quotas, validate_input, fit_complete_turns, json_hash, ConstrainedDataset
from qwenvl.data.data_processor import update_processor_pixels, IMAGE_TOKEN_INDEX
from qwenvl.train.argument import DataArguments
from qwenvl.train.train_qwen_constrained import install_adapters, ConstrainedTrainer
from qwenvl.train.train_qwen_grace import GraceArguments
from qwenvl.train.grace_profiler import GraceLearningDiagnostics, GraceStepProfiler
from qwenvl.train.grace_modules import AdaptiveIBController
from qwenvl.train.qat_modules import QATLoraLinear, QuantizedLinear, merge_lora_weights_inplace

spec = importlib.util.spec_from_file_location("constrained_launcher", ROOT / "scripts/run_constrained.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = torch.nn.ModuleDict({name: torch.nn.Linear(16, 16)
                                                   for name in ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj")})
        self.visual = torch.nn.ModuleDict({"merger": torch.nn.Linear(16, 16)})
        self.lm_head = torch.nn.Linear(16, 16)

    def forward(self, input_ids=None, labels=None, use_cache=False):
        x = self.visual["merger"](input_ids)
        for layer in self.language_model.values():
            x = torch.tanh(layer(x))
        logits = self.lm_head(x)
        return SimpleNamespace(loss=(logits-labels).square().mean(), logits=logits)


def lora_args():
    return SimpleNamespace(lora_r=2, lora_alpha=4, lora_dropout=0.0,
                           lora_target_modules="q_proj,k_proj,v_proj,o_proj")


def make_model(base, method):
    model = copy.deepcopy(base)
    model.requires_grad_(False)
    model.visual["merger"].requires_grad_(True)
    torch.manual_seed(101)
    install_adapters(model, method, lora_args(), SimpleNamespace(qat_bits=4, qat_group_size=8))
    return model


class MatchedTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_method_specific_verbose_labels(self):
        base = TinyModel()
        for method in ("sft", "qat", "grace"):
            with self.subTest(method=method):
                output = StringIO()
                with redirect_stdout(output):
                    make_model(base, method)
                self.assertIn(f"[{method.upper()}-LoRA]", output.getvalue())

                profiler = GraceStepProfiler(
                    enabled=False,
                    profile_steps=0,
                    output_dir="/tmp/unused-profile",
                    merge_size=2,
                    method=method,
                )
                diagnostics = GraceLearningDiagnostics(
                    enabled=False,
                    max_steps=0,
                    output_dir="/tmp/unused-profile",
                    method=method,
                )
                self.assertEqual(profiler.log_prefix, f"[{method.upper()}-PROFILE]")
                self.assertEqual(diagnostics.log_prefix, f"[{method.upper()}-LEARNING]")

    def test_matching_adapters_and_optimizer_inventory(self):
        base = TinyModel()
        models = {method: make_model(base, method) for method in ("sft", "qat", "grace")}
        trainable = {}
        for method, model in models.items():
            trainable[method] = {n for n,p in model.named_parameters() if p.requires_grad}
            self.assertFalse(model.lm_head.weight.requires_grad)
            self.assertTrue(model.visual["merger"].weight.requires_grad)
        self.assertEqual(trainable["qat"], trainable["grace"])
        self.assertEqual(trainable["sft"], {n for n in trainable["qat"] if not n.endswith("log_scale")})
        for name, value in models["sft"].named_parameters():
            if "lora_" in name:
                for method in ("qat", "grace"):
                    torch.testing.assert_close(value, dict(models[method].named_parameters())[name])

    def test_sft_has_no_quantization_and_merge_is_equivalent(self):
        base = TinyModel().eval()
        model = make_model(base, "sft").eval()
        x, y = torch.randn(2, 16), torch.randn(2, 16)
        with patch.object(QuantizedLinear, "_quantize_weight", side_effect=AssertionError("SFT quantized")):
            torch.testing.assert_close(base(x, y).logits, model(x, y).logits)
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, QATLoraLinear):
                        module.lora_B.normal_(std=0.1)
            before = model(x, y).logits.detach()
            merge_lora_weights_inplace(model)
            torch.testing.assert_close(before, model(x, y).logits, atol=1e-6, rtol=1e-5)

    def test_baseline_ce_gradients_and_parameter_groups_without_teacher(self):
        for method in ("sft", "qat"):
            with self.subTest(method=method), TemporaryDirectory() as directory:
                model = make_model(TinyModel(), method)
                args = TrainingArguments(output_dir=directory, use_cpu=True, report_to="none")
                args.mm_projector_lr = 1e-5
                trainer = ConstrainedTrainer(model=model, args=args, teacher=None,
                                             grace_args=GraceArguments(grace_enable=False),
                                             base_lr=2e-4, scale_lr=3e-4)
                inputs = dict(input_ids=torch.randn(2, 16), labels=torch.randn(2, 16))
                expected = model(**inputs).loss
                actual = trainer.compute_loss(model, inputs)
                torch.testing.assert_close(actual, expected)
                actual.backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for n,p in model.named_parameters() if n.endswith("lora_B")))
                optimizer = trainer.create_optimizer()
                for group in optimizer.param_groups:
                    self.assertTrue(group["params"])
                    for p in group["params"]:
                        names = [n for n,q in model.named_parameters() if p is q]
                        expected_lr = 1e-5 if "merger" in names[0] else 3e-4 if names[0].endswith("log_scale") else 2e-4
                        self.assertEqual(group["lr"], expected_lr)
                if method == "qat":
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                        for n,p in model.named_parameters() if n.endswith("log_scale")))

    def test_rng_json_restores_python_numpy_and_torch(self):
        with TemporaryDirectory() as directory:
            trainer = ConstrainedTrainer(model=TinyModel(), args=TrainingArguments(
                output_dir=directory, use_cpu=True, report_to="none"),
                grace_args=GraceArguments(grace_enable=False))
            trainer._save_rng_state(directory)
            expected = (random.random(), np.random.random(), torch.rand(4))
            trainer._load_rng_state(directory)
            self.assertEqual(expected[0], random.random())
            self.assertEqual(expected[1], np.random.random())
            torch.testing.assert_close(expected[2], torch.rand(4))

    def test_accumulated_cpu_training_and_native_checkpoint(self):
        with TemporaryDirectory() as directory:
            model = make_model(TinyModel(), "qat")
            args = TrainingArguments(output_dir=directory, use_cpu=True, report_to="none",
                                     max_steps=2, per_device_train_batch_size=1,
                                     gradient_accumulation_steps=2, save_strategy="steps", save_steps=1,
                                     disable_tqdm=True, remove_unused_columns=False)
            dataset = [dict(input_ids=torch.randn(16), labels=torch.randn(16)) for _ in range(4)]
            trainer = ConstrainedTrainer(model=model, args=args, train_dataset=dataset,
                                         grace_args=GraceArguments(grace_enable=False),
                                         base_lr=2e-4, scale_lr=2e-4)
            trainer.model_accepts_loss_kwargs = False
            trainer.train()
            self.assertEqual(trainer.state.global_step, 2)
            checkpoint = Path(directory)/"checkpoint-2"
            self.assertTrue((checkpoint/"rng_state_0.json").exists())
            self.assertTrue((checkpoint/"scheduler_state.json").exists())
            self.assertTrue((checkpoint/"model.safetensors").exists())
            self.assertTrue(any(p.abs().sum() > 0 for n,p in model.named_parameters() if n.endswith("lora_B")))

    def test_grace_loss_and_ib_controller_with_tiny_teacher(self):
        class VLM(torch.nn.Module):
            def __init__(self, pen=None):
                super().__init__()
                self.embed = torch.nn.Embedding(32, 16)
                self.proj = torch.nn.Linear(16,16)
                self.head = torch.nn.Linear(16,32)
                self.pen = pen

            def forward(self, input_ids, labels=None, output_hidden_states=False, use_cache=False, **kwargs):
                hidden = torch.tanh(self.proj(self.embed(input_ids % 32)))
                logits = self.head(hidden)
                if self.pen is not None:
                    self.pen["hidden"] = hidden
                loss = None if labels is None else F.cross_entropy(logits[:,:-1].reshape(-1,32), labels[:,1:].reshape(-1))
                return SimpleNamespace(loss=loss, logits=logits, hidden_states=(hidden,hidden))

        with TemporaryDirectory() as directory:
            pen = {}
            student, teacher = VLM(), VLM(pen)
            teacher.requires_grad_(False)
            ga = GraceArguments(grace_enable=True, dkd_chunk_size=2, dkd_checkpoint_chunks=True,
                                ib_tau=0.0, ib_beta_init=0.5)
            ib = AdaptiveIBController(tau=0.0, eta=0.003, beta_init=0.5, beta_min=0.1, beta_max=1.0)
            trainer = ConstrainedTrainer(model=student, teacher=teacher, teacher_pen_buf=pen,
                                         args=TrainingArguments(output_dir=directory, use_cpu=True, report_to="none",
                                                                gradient_accumulation_steps=2),
                                         grace_args=ga, ib_controller=ib)
            ids = torch.tensor([[1,IMAGE_TOKEN_INDEX,IMAGE_TOKEN_INDEX,IMAGE_TOKEN_INDEX,IMAGE_TOKEN_INDEX,7,8]])
            labels = torch.tensor([[-100,-100,-100,-100,-100,7,8]])
            inputs = dict(input_ids=ids, labels=labels, image_grid_thw=torch.tensor([[1,4,4]]))
            initial = ib.beta
            first = trainer.compute_loss(student, inputs)
            self.assertEqual(ib.beta, initial)
            first.backward()
            second = trainer.compute_loss(student, inputs)
            second.backward()
            self.assertGreater(ib.beta, initial)
            self.assertTrue(torch.isfinite(second))
            self.assertGreater(trainer._grace_logs["gdkd"], 0)
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            self.assertIsNotNone(student.proj.weight.grad)


class DataAndLauncherTests(unittest.TestCase):
    def item(self):
        ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX, 7]])
        return dict(input_ids=ids, labels=torch.tensor([[-100,-100,-100,-100,-100,7]]),
                    image_grid_thw=torch.tensor([[1, 4, 4]]), pixel_values=torch.ones(16, 1536))

    def test_input_envelope(self):
        item = self.item()
        validate_input(item, 6, 4096)
        with self.assertRaisesRegex(ValueError, "Sequence shape"):
            validate_input(item, 5, 4096)
        with self.assertRaisesRegex(ValueError, "pixel budget"):
            validate_input(item, 6, 4000)
        item["labels"].fill_(-100)
        with self.assertRaisesRegex(ValueError, "No supervised"):
            validate_input(item, 6, 4096)
        item = self.item()
        item["input_ids"][0, 1] = 3
        with self.assertRaisesRegex(ValueError, "Incomplete image"):
            validate_input(item, 6, 4096)

    def test_sizedict_and_dict_pixel_settings(self):
        for size in ({"shortest_edge": 65536, "longest_edge": 16777216},
                     SizeDict(shortest_edge=65536, longest_edge=16777216)):
            processor = SimpleNamespace(image_processor=SimpleNamespace(size=size))
            update_processor_pixels(processor, DataArguments(min_pixels=4096, max_pixels=50176))
            self.assertEqual(processor.image_processor.size["longest_edge"], 50176)
            self.assertEqual(processor.image_processor.size["shortest_edge"], 4096)

    def test_complete_turn_trimming(self):
        sample = {"conversations": [{"from": who,"value": "x"} for who in ("human","gpt","human","gpt")]}
        def tokenize(samples, processor, truncate):
            self.assertFalse(truncate)
            return {"input_ids": torch.zeros(1, len(samples[0]["conversations"])*10)}
        with patch("qwenvl.data.constrained_data.preprocess_qwen_visual", side_effect=tokenize), \
             patch("qwenvl.data.constrained_data.preprocess", return_value=self.item()):
            fitted, _ = fit_complete_turns(sample, None, 25, 4096)
            self.assertEqual(len(fitted["conversations"]), 2)
            self.assertEqual(len(sample["conversations"]), 4)
            with self.assertRaisesRegex(ValueError, "First complete"):
                fit_complete_turns(sample, None, 10, 4096)

    def test_manifest_tamper_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/"subset.json"
            path.write_text(json.dumps(dict(records=[{"index": 2}], records_sha256=json_hash([{"index": 1}]))))
            with self.assertRaisesRegex(ValueError, "checksum"):
                ConstrainedDataset(path, None, 1024, 50176)

    def test_stratification_and_same_commands(self):
        self.assertEqual(allocate_quotas({"a": 80, "b": 20}, 10), {"a": 8, "b": 2})
        with patch.dict(os.environ, {}, clear=True):
            config = launcher.resolved_config()
            self.assertEqual(config["max_steps"], 512)
            self.assertEqual(config["grad_accum"], 32)
            commands = {}
            for method in ("sft","qat","grace"):
                command, _ = launcher.build_command(method, config, Path("/tmp/comparison"))
                start = command.index("--method")
                commands[method] = dict(zip(command[start::2], command[start+1::2]))
            allowed = {"--method", "--output_dir", "--qat_enable", "--grace_enable", "--teacher_model_path"}
            for method in ("sft", "qat"):
                changed = {key for key in commands[method].keys() | commands["grace"].keys()
                           if commands[method].get(key) != commands["grace"].get(key)}
                self.assertLessEqual(changed, allowed)

    def test_recipe_mismatch_fails_without_overwrite(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/"recipe.json"
            launcher.pin_recipe(path, {"seed": 42})
            with self.assertRaisesRegex(ValueError, "Recipe mismatch"):
                launcher.pin_recipe(path, {"seed": 43})
            self.assertEqual(json.loads(path.read_text()), {"seed": 42})


if __name__ == "__main__":
    unittest.main()
