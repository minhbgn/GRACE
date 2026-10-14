"""Matched SFT/QAT/GRACE LoRA training on a validated, bounded subset."""
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoProcessor, HfArgumentParser, TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint

from qwenvl.data.constrained_data import ConstrainedDataset
from qwenvl.data.data_processor import FlattenedDataCollatorForSupervisedDataset, update_processor_pixels
from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments
from qwenvl.train.grace_modules import AdaptiveIBController
from qwenvl.train.grace_profiler import GraceProfilerCallback, GraceLearningDiagnosticsCallback
from qwenvl.train.grace_resume import CheckpointPauseCallback
from qwenvl.train.qat_modules import (
    DEFAULT_QAT_SKIP_KEYWORDS, QATLoraLinear, replace_linears_with_qat_lora,
    merge_lora_weights_inplace, bake_qat_weights_inplace,
    strip_qat_lora_from_checkpoint_dir, materialize_quantized_state_dict,
)
from qwenvl.train.train_qwen_grace import (
    GraceArguments, GraceTrainer, GraceStateCallback, _build_student, _build_teacher,
    _load_ib_state, _close_distributed_process_group, _is_world_process_zero,
)
from qwenvl.train.train_qwen_qat import (
    QATArguments, SaveProcessorCallback, _qwen3vl_submodule, set_model,
    safe_save_model_for_hf_trainer,
)
from qwenvl.train.trainer import replace_qwen2_vl_attention_class


class BFloatLoraLinear(QATLoraLinear):
    """Same adapter initialization and merge format, with no fake quantization.

    The inherited scale tensor is frozen and never used. Retaining the layout
    allows the same native checkpoint/export code for all three methods.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_scale.requires_grad_(False)

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        delta = F.linear(F.linear(self.lora_dropout(x).to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return base + (delta * self.lora_scaling).to(base.dtype)


def install_adapters(model, method, args, qat_args):
    targets = tuple(name.strip() for name in args.lora_target_modules.split(",") if name.strip())
    common = dict(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    if method != "sft":
        return replace_linears_with_qat_lora(
            model, bits=qat_args.qat_bits, group_size=qat_args.qat_group_size,
            target_modules=targets, method=method, **common,
        )
    adapted = []
    for parent_name, parent in list(model.named_modules()):
        for name, child in list(parent.named_children()):
            full = f"{parent_name}.{name}" if parent_name else name
            if name not in targets or not isinstance(child, torch.nn.Linear):
                continue
            if any(skip in full for skip in DEFAULT_QAT_SKIP_KEYWORDS):
                continue
            setattr(parent, name, BFloatLoraLinear(child, **common))
            adapted.append(full)
    if not adapted:
        raise ValueError("No language-model LoRA targets found")
    if _is_world_process_zero():
        print(
            f"[SFT-LoRA] Adapted {len(adapted)} linears "
            f"(targets={sorted(targets)}, r={args.lora_r}, alpha={args.lora_alpha}).",
            flush=True,
        )
    return [], adapted


@dataclass
class ConstrainedArguments:
    method: str = "grace"
    subset_manifest: str = ""
    profile_run: bool = False
    max_train_hours: float = 5.0
    peak_memory_gib: float = 34.0
    resume_run: bool = False


class ConstrainedTrainer(GraceTrainer):
    """Use JSON for HF RNG/scheduler state, including on the pinned torch 2.5.

    DeepSpeed still owns its native optimizer checkpoints. No Transformers
    pickle-load safety checks are disabled or monkey-patched.
    """
    def _save_rng_state(self, output_dir):
        state = np.random.get_state()
        payload = dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), *state[2:]],
                       cpu=torch.random.get_rng_state().tolist())
        if torch.cuda.is_available():
            payload["cuda"] = [value.tolist() for value in torch.cuda.random.get_rng_state_all()]
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"rng_state_{self.args.process_index}.json").write_text(json.dumps(payload))

    def _load_rng_state(self, checkpoint):
        if checkpoint is None:
            return
        payload = json.loads((Path(checkpoint) / f"rng_state_{self.args.process_index}.json").read_text())
        random.setstate((payload["python"][0], tuple(payload["python"][1]), payload["python"][2]))
        state = payload["numpy"]
        np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))
        torch.random.set_rng_state(torch.tensor(payload["cpu"], dtype=torch.uint8))
        if torch.cuda.is_available():
            torch.cuda.random.set_rng_state_all([torch.tensor(value, dtype=torch.uint8)
                                                for value in payload["cuda"]])

    def _save_optimizer_and_scheduler(self, output_dir):
        super()._save_optimizer_and_scheduler(output_dir)
        if self.args.should_save and self.lr_scheduler is not None:
            (Path(output_dir) / "scheduler_state.json").write_text(json.dumps(self.lr_scheduler.state_dict()))

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None:
            return
        if self.is_deepspeed_enabled:
            from accelerate.utils import DeepSpeedSchedulerWrapper
            # DeepSpeed has already restored its optimizer and any DS scheduler.
            if not isinstance(self.lr_scheduler, DeepSpeedSchedulerWrapper):
                self.lr_scheduler.load_state_dict(json.loads((Path(checkpoint) / "scheduler_state.json").read_text()))
            return
        return super()._load_optimizer_and_scheduler(checkpoint)


class RunMonitor(TrainerCallback):
    """Collect rank-max memory/timing and pause collectively at a safe boundary."""
    def __init__(self, config, trainer):
        self.config, self.trainer = config, trainer
        self.rows = []
        self.reason = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.monotonic()
        self.previous_elapsed = 0.0
        if state.is_world_process_zero:
            path = Path(args.output_dir) / "step_metrics.jsonl"
            # A resumed segment has its own elapsed clock, recorded in every row.
            self.segment_start_step = state.global_step
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_started = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.max_memory_allocated() / 2**30
            reserved = torch.cuda.max_memory_reserved() / 2**30
        else:
            allocated = reserved = 0.0
        elapsed = time.monotonic() - self.started
        values = torch.tensor([time.monotonic() - self.step_started, allocated, reserved, elapsed],
                              device=args.device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.MAX)
        seconds, allocated, reserved, elapsed = values.tolist()
        row = dict(step=state.global_step, seconds=seconds, peak_allocated_gib=allocated,
                   peak_reserved_gib=reserved, elapsed_seconds=elapsed,
                   cycle_seconds=elapsed - self.previous_elapsed,
                   **self.trainer._grace_logs)
        self.previous_elapsed = elapsed
        self.rows.append(row)
        if reserved > self.config.peak_memory_gib:
            self.reason = "memory_headroom_exceeded"
        if elapsed >= self.config.max_train_hours * 3600:
            self.reason = "training_time_budget_reached"
        if self.reason:
            control.should_save = control.should_training_stop = True
        if state.global_step >= state.max_steps:
            # Preserve a native adapter + optimizer checkpoint before final bake.
            control.should_save = True
        if state.is_world_process_zero:
            row["segment_start_step"] = self.segment_start_step
            with open(Path(args.output_dir) / "step_metrics.jsonl", "a") as handle:
                handle.write(json.dumps(row) + "\n")
        return control


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments,
                              QATArguments, GraceArguments, ConstrainedArguments))
    ma, da, ta, qa, ga, ca = parser.parse_args_into_dataclasses()
    if ca.method not in ("sft", "qat", "grace") or not ca.subset_manifest:
        raise ValueError("Specify method=sft/qat/grace and a subset manifest")
    if not ta.lora_enable or ta.per_device_train_batch_size != 1 or da.data_packing:
        raise ValueError("Constrained runs require LoRA, micro-batch 1, no data packing")
    ga.grace_enable = ca.method == "grace"
    qa.qat_enable = ca.method != "sft"
    set_seed(ta.seed)
    output = Path(ta.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = get_last_checkpoint(str(output))
    if checkpoint and not ca.resume_run:
        raise ValueError("Checkpoint exists: use --resume or a new experiment directory")
    if ca.resume_run and not checkpoint:
        raise ValueError("--resume requested but no checkpoint exists")

    processor = AutoProcessor.from_pretrained(ma.model_name_or_path, local_files_only=True)
    processor.tokenizer.model_max_length = ta.model_max_length
    da.model_type = "qwen3vl"
    update_processor_pixels(processor, da)
    dataset = ConstrainedDataset(ca.subset_manifest, processor, ta.model_max_length,
                                 da.max_pixels, ca.profile_run)
    # Validate a record before allocating model weights.
    dataset[0]
    model = _build_student(ma, ta, "flash_attention_2")
    set_model(ma, model)
    for parameter in _qwen3vl_submodule(model, "language_model").parameters():
        parameter.requires_grad_(False)
    model.lm_head.requires_grad_(False)
    quantized, adapted = install_adapters(model, ca.method, ta, qa)
    model.config.qat_lora = dict(enabled=True, merged=False, r=ta.lora_r,
                                 alpha=ta.lora_alpha, dropout=ta.lora_dropout,
                                 target_modules=ta.lora_target_modules.split(","),
                                 num_adapted=len(adapted))
    if qa.qat_enable:
        model.config.qat = dict(bits=qa.qat_bits, group_size=qa.qat_group_size,
                                skip_keywords=list(DEFAULT_QAT_SKIP_KEYWORDS),
                                num_replaced=len(quantized))
    model.config.use_cache = False
    model.enable_input_require_grads()
    replace_qwen2_vl_attention_class()

    teacher, pen = _build_teacher(ga, ta) if ga.grace_enable else (None, None)
    if teacher is not None:
        if teacher.config.text_config.vocab_size != model.config.text_config.vocab_size:
            raise ValueError("Teacher and student vocabulary sizes differ")
    ib = AdaptiveIBController(tau=ga.ib_tau, eta=ga.ib_eta, beta_init=ga.ib_beta_init,
                              beta_min=ga.ib_beta_min, beta_max=ga.ib_beta_max,
                              ema_decay=ga.ib_ema_decay)
    if checkpoint and ga.grace_enable:
        _load_ib_state(ib, str(output))
    trainer = ConstrainedTrainer(model=model, processing_class=processor.tokenizer, args=ta,
                           train_dataset=dataset,
                           data_collator=FlattenedDataCollatorForSupervisedDataset(processor.tokenizer),
                           teacher=teacher, teacher_pen_buf=pen, grace_args=ga,
                           ib_controller=ib, merge_size=processor.image_processor.merge_size,
                           base_lr=ta.learning_rate, scale_lr=qa.qat_scale_lr,
                           method=ca.method)
    # Every method uses a mean loss per example. This custom loss does not use
    # HF's accumulated num_items_in_batch; Trainer must divide by accumulation.
    trainer.model_accepts_loss_kwargs = False
    trainer.add_callback(SaveProcessorCallback(processor))
    if ga.grace_enable:
        trainer.add_callback(GraceStateCallback(ib))
    trainer.add_callback(GraceProfilerCallback(trainer.grace_profiler))
    trainer.add_callback(GraceLearningDiagnosticsCallback(trainer.learning_diagnostics))
    monitor = RunMonitor(ca, trainer)
    trainer.add_callback(monitor)
    pause = CheckpointPauseCallback()
    trainer.add_callback(pause)
    if _is_world_process_zero():
        inventory = {name: dict(shape=list(p.shape), parameters=p.numel())
                     for name, p in model.named_parameters() if p.requires_grad}
        (output / "trainable_parameters.json").write_text(json.dumps(inventory, indent=2) + "\n")
    result = trainer.train(resume_from_checkpoint=checkpoint if ca.resume_run else None)
    trainer.save_state()
    complete = not (monitor.reason or pause.paused) and trainer.state.global_step >= trainer.state.max_steps
    if _is_world_process_zero():
        summary = dict(method=ca.method, complete=complete, profile=ca.profile_run,
                       stop_reason=monitor.reason or ("user_pause" if pause.paused else None),
                       global_step=trainer.state.global_step, metrics=result.metrics,
                       peak_reserved_gib=max((r["peak_reserved_gib"] for r in monitor.rows), default=0),
                       native_checkpoint=get_last_checkpoint(str(output)))
        (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not complete:
        print(f"Paused with native checkpoint: {monitor.reason or 'user request'}", flush=True)
        _close_distributed_process_group()
        return

    # Export profile runs too: test the merge/bake/save memory path before a long run.
    trainer.teacher = teacher = pen = None
    gc.collect()
    torch.cuda.empty_cache()
    merge_lora_weights_inplace(model)
    model.config.qat_lora["merged"] = True
    if qa.qat_enable:
        bake_qat_weights_inplace(model)
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer, str(output))
    if dist.is_initialized():
        dist.barrier()
    if _is_world_process_zero():
        processor.save_pretrained(output)
        strip_qat_lora_from_checkpoint_dir(str(output))
        if qa.qat_enable and qa.qat_save_quantized:
            torch.save(materialize_quantized_state_dict(model), output / "qat_quantized_weights.bin")
        (output / "EXPORT_COMPLETE").write_text(
            "BF16 storage; QAT weights lie on W4/W8 grid. Packed INT4 requires separate export.\n"
        )
    _close_distributed_process_group()


if __name__ == "__main__":
    train()
