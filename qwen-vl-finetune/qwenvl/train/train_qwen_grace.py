"""GRACE training entrypoint for Qwen3-VL.

GRACE = Gated Relational Alignment via Confidence-based Distillation.
A frozen BF16 teacher (Qwen3-VL-8B) supervises a quantization-aware
student (Qwen3-VL-2B) via three losses (see grace_modules.py):

    L_total = L_CE + beta * L_GDKD + omega * L_RCKA

The student's INT4/INT8 QAT is identical to train_qwen_qat.py — GRACE only
adds teacher-driven loss terms. The QAT machinery (qat_modules.py) is reused
unchanged: group-wise LSQ, log-scale optimizer group, weight baking, etc.
"""

from __future__ import annotations

import gc
import os
import json
import logging
import pathlib
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import transformers

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Trainer, TrainerCallback
from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
    set_hf_deepspeed_config,
    unset_hf_deepspeed_config,
)

from qwenvl.train.trainer import replace_qwen2_vl_attention_class
from qwenvl.data.data_processor import (
    make_supervised_data_module,
    IMAGE_TOKEN_INDEX,
    VIDEO_TOKEN_INDEX,
)
from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments
from qwenvl.train.qat_modules import (
    DEFAULT_QAT_SKIP_KEYWORDS,
    bake_qat_weights_inplace,
    collect_qat_param_groups,
    materialize_quantized_state_dict,
    merge_lora_weights_inplace,
    replace_linears_with_qat,
    replace_linears_with_qat_lora,
    strip_log_scale_from_checkpoint_dir,
    strip_qat_lora_from_checkpoint_dir,
)
from qwenvl.train.train_qwen_qat import (
    QATArguments,
    SaveProcessorCallback,
    _qwen3vl_submodule,
    safe_save_model_for_hf_trainer,
    set_model,
)
from qwenvl.train.grace_modules import (
    AdaptiveIBController,
    gated_dkd_loss,
    rcka_loss,
)
from qwenvl.train.grace_resume import CheckpointPauseCallback
from qwenvl.train.grace_profiler import (
    GraceLearningDiagnostics,
    GraceLearningDiagnosticsCallback,
    GraceProfilerCallback,
    GraceStepProfiler,
)

def _is_world_process_zero() -> bool:
    """Rank check that works across old and new Transformers releases."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except ValueError:
        return True


def rank0_print(*args):
    if _is_world_process_zero():
        print(*args)


def _close_distributed_process_group() -> None:
    """Synchronize successful exits and avoid stale NCCL process groups."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ----------------------------------------------------------------------
# Helpers: penultimate-layer capture hook + warmup ramp
# ----------------------------------------------------------------------
def _find_llm_layers(model) -> torch.nn.ModuleList:
    """Locate the LLM decoder ModuleList in a Qwen3VL model.

    Robust to the different wrapping paths the transformers integration has
    used (``model.language_model.layers``, ``model.model.language_model.layers``,
    ...). Skips the vision encoder's blocks (which also expose ``self_attn``).
    """
    candidates = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.ModuleList) or len(mod) == 0:
            continue
        if "visual" in name or "vision" in name:
            continue
        first = mod[0]
        if not any(hasattr(first, a) for a in ("self_attn", "attention", "attn")):
            continue
        candidates.append((name, mod))
    if not candidates:
        raise RuntimeError("Could not locate LLM decoder layers for the RCKA hook.")
    candidates.sort(
        key=lambda nm: (1 if "language_model" in nm[0] else 0, len(nm[1])),
        reverse=True,
    )
    return candidates[0][1]


def _attach_teacher_penult_hook(teacher, rcka_layer: int) -> dict:
    """Forward-hook the teacher's RCKA layer so its forward call does NOT
    need ``output_hidden_states=True`` — which under ``no_grad`` would still
    retain references to every layer's output for the 8B teacher. Caller
    reads ``storage["hidden"]`` after each teacher forward and resets it.
    """
    layers = _find_llm_layers(teacher)
    target = layers[rcka_layer]
    storage: dict = {"hidden": None}

    def hook(_module, _args, output):
        storage["hidden"] = output[0] if isinstance(output, tuple) else output

    target.register_forward_hook(hook)
    return storage


def _warmup_scale(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, max(0.0, step / float(warmup_steps)))


@dataclass
class GraceArguments:
    grace_enable: bool = field(default=True)
    teacher_model_path: str = field(
        default="",
        metadata={"help": "Path to the frozen BF16 teacher (Qwen3-VL-8B)."},
    )
    teacher_attn: str = field(default="flash_attention_2")
    # --- Confidence-Gated Decoupled KD ---
    dkd_temperature: float = field(default=2.0)
    dkd_alpha: float = field(default=1.0, metadata={"help": "TCKD weight."})
    dkd_beta: float = field(default=4.0, metadata={"help": "NCKD weight."})
    dkd_chunk_size: int = field(
        default=2048, metadata={"help": "Valid-token chunk for the KL pass."}
    )
    dkd_checkpoint_chunks: bool = field(
        default=False,
        metadata={"help": "Recompute DKD chunks in backward to bound retained KL intermediates."},
    )
    # --- Relational Centered Kernel Alignment ---
    rcka_weight: float = field(default=1.0, metadata={"help": "omega."})
    rcka_layer: int = field(
        default=-2, metadata={"help": "hidden_states index (penultimate = -2)."}
    )
    # --- Adaptive IB controller ---
    ib_tau: float = field(default=0.35)
    ib_eta: float = field(default=0.0015)
    ib_beta_init: float = field(default=1.0)
    ib_beta_min: float = field(default=0.1)
    ib_beta_max: float = field(default=5.0)
    ib_ema_decay: float = field(default=0.99)
    # --- Loss-term linear warmups (stabilize early training where the
    # student representation is still random-ish and distillation gradients
    # would otherwise dominate the CE signal). ---
    dkd_warmup_steps: int = field(
        default=0,
        metadata={"help": "Linear warmup (in optimizer steps) for the GDKD term."},
    )
    rcka_warmup_steps: int = field(
        default=0,
        metadata={"help": "Linear warmup (in optimizer steps) for the RCKA term."},
    )
    # --- Opt-in diagnostics. Disabled by default so reproduction runs retain
    # their original execution path. ---
    profile_steps: int = field(
        default=0,
        metadata={"help": "Profile the first N optimizer steps; 0 disables profiling."},
    )
    profile_deepspeed_comms: bool = field(
        default=False,
        metadata={"help": "Print the DeepSpeed collective summary at train end."},
    )
    skip_final_export: bool = field(
        default=False,
        metadata={
            "help": "Skip merge/bake/model export after a profiling-only run."
        },
    )
    learning_diagnostics_steps: int = field(
        default=0,
        metadata={
            "help": "Record gradient/update/loss-connectivity checks for N steps."
        },
    )


# ----------------------------------------------------------------------
# IB controller persistence (so beta survives checkpoint resume)
# ----------------------------------------------------------------------
class GraceStateCallback(TrainerCallback):
    def __init__(self, controller: AdaptiveIBController):
        self.controller = controller

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt):
            with open(os.path.join(ckpt, "grace_ib_state.json"), "w") as f:
                json.dump(self.controller.state_dict(), f)


def _load_ib_state(controller: AdaptiveIBController, output_dir: str) -> None:
    ckpts = sorted(
        pathlib.Path(output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not ckpts:
        return
    state_file = ckpts[-1] / "grace_ib_state.json"
    if state_file.is_file():
        with open(state_file) as f:
            controller.load_state_dict(json.load(f))
        rank0_print(f"[GRACE] Restored IB controller state from {state_file}")


# ----------------------------------------------------------------------
# Model builders
# ----------------------------------------------------------------------
@contextmanager
def _without_transformers_zero3_init(training_args):
    """Temporarily keep ``from_pretrained`` out of ``deepspeed.zero.Init``.

    Parsing ``TrainingArguments`` installs its DeepSpeed configuration in a
    process-global weak reference used by Transformers model loading. That is
    desirable for the student, but the frozen teacher is not handed to
    DeepSpeed and must therefore be loaded as an ordinary, replicated module.
    """
    if not is_deepspeed_zero3_enabled():
        yield
        return

    hf_deepspeed_config = getattr(training_args, "hf_deepspeed_config", None)
    if hf_deepspeed_config is None:
        raise RuntimeError(
            "Transformers reports ZeRO-3 initialization is enabled, but "
            "TrainingArguments does not retain its DeepSpeed configuration."
        )

    unset_hf_deepspeed_config()
    try:
        yield
    finally:
        # Restore ZeRO-3 before Trainer construction. Transformers stores only
        # a weak reference here, while TrainingArguments owns the strong one.
        set_hf_deepspeed_config(hf_deepspeed_config)


def _build_student(model_args, training_args, attn_implementation):
    compute_dtype = torch.bfloat16 if training_args.bf16 else None
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        dtype=compute_dtype,
    )
    return model


def _build_teacher(grace_args, training_args):
    """Frozen BF16 teacher. Not wrapped by DeepSpeed, not quantized.

    Returns (teacher, pen_buf) where pen_buf["hidden"] is filled with the
    teacher's RCKA-layer output after each forward — replacing the need for
    ``output_hidden_states=True`` (which would retain every layer's output).
    """
    with _without_transformers_zero3_init(training_args):
        teacher = Qwen3VLForConditionalGeneration.from_pretrained(
            grace_args.teacher_model_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=grace_args.teacher_attn,
            dtype=torch.bfloat16,
        )
    teacher.config.use_cache = False
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    pen_buf = _attach_teacher_penult_hook(teacher, grace_args.rcka_layer)
    return teacher, pen_buf


# ----------------------------------------------------------------------
# GRACE trainer
# ----------------------------------------------------------------------
class GraceTrainer(Trainer):
    """QAT student + frozen teacher; combined GRACE objective."""

    def __init__(self, *args, teacher=None, teacher_pen_buf=None,
                 grace_args=None, ib_controller=None, merge_size=2,
                 base_lr=2e-5, scale_lr=2e-5, method="grace", **kwargs):
        super().__init__(*args, **kwargs)
        self.method = str(method).strip().lower()
        if not self.method:
            raise ValueError("method must be a non-empty name")
        self.teacher = teacher.to(self.args.device) if teacher is not None else None
        if self.teacher is not None:
            self.teacher.eval()
        if grace_args and grace_args.grace_enable and self.teacher is None:
            raise ValueError("GRACE requires a teacher")
        self.teacher_pen_buf = teacher_pen_buf
        self.grace_args = grace_args
        self.ib = ib_controller
        self.merge_size = merge_size
        self._base_lr = base_lr
        self._scale_lr = scale_lr
        self._grace_logs: dict = {}
        # IB controller: GDKD is averaged across the gradient-accumulation
        # cycle and the controller fires once per optimizer step. The previous
        # per-micro-batch update made ema_decay / eta evolve grad_accum times
        # faster than the comments imply.
        self._gdkd_accum: Optional[torch.Tensor] = None
        self._gdkd_count: int = 0
        self._micro_step: int = 0
        self.grace_profiler = GraceStepProfiler(
            enabled=bool(grace_args and grace_args.profile_steps > 0),
            profile_steps=(grace_args.profile_steps if grace_args else 0),
            output_dir=os.path.join(self.args.output_dir, "profiling"),
            merge_size=merge_size,
            method=self.method,
        )
        self.grace_profiler.install_model_hooks(self.model, self.teacher)
        self.learning_diagnostics = GraceLearningDiagnostics(
            enabled=bool(grace_args and grace_args.learning_diagnostics_steps > 0),
            max_steps=(grace_args.learning_diagnostics_steps if grace_args else 0),
            output_dir=os.path.join(self.args.output_dir, "profiling"),
            method=self.method,
        )
        self.learning_diagnostics.install(self.model, self.teacher)

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        started = time.perf_counter()
        result = super().get_batch_samples(epoch_iterator, num_batches, device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.grace_profiler.begin_step(
            step_number=int(getattr(self.state, "global_step", 0)) + 1,
            data_wait_ms=elapsed_ms,
        )
        return result

    def training_step(self, model, inputs, num_items_in_batch=None):
        self.grace_profiler.observe_inputs(inputs)
        result = super().training_step(model, inputs, num_items_in_batch)
        self.grace_profiler.finish_micro_step()
        return result

    # --- optimizer: same QAT param-group split as train_qwen_qat ---
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        opt_cls, opt_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        param_groups = collect_qat_param_groups(
            self.model,
            base_lr=self._base_lr,
            scale_lr=self._scale_lr,
            weight_decay=self.args.weight_decay,
        )
        # LoRA commonly uses a much larger LR than full-rank modules. Keep the
        # visual merger on its explicit projector LR so the LoRA comparison
        # does not silently train that full-rank module 20x faster than the
        # original GRACE setup.
        mm_lr = getattr(self.args, "mm_projector_lr", None)
        if mm_lr is not None:
            merger_params = [
                p for name, p in self.model.named_parameters()
                if p.requires_grad and "merger" in name
            ]
            if merger_params:
                merger_ids = {id(p) for p in merger_params}
                for group in param_groups:
                    group["params"] = [
                        p for p in group["params"] if id(p) not in merger_ids
                    ]
                param_groups.append({
                    "params": merger_params,
                    "lr": mm_lr,
                    "weight_decay": self.args.weight_decay,
                })
        # SFT has no trainable quantization scales. ZeRO must not receive an
        # empty scale group (or an emptied base group after projector splitting).
        param_groups = [group for group in param_groups if group["params"]]
        self.optimizer = opt_cls(param_groups, **opt_kwargs)
        return self.optimizer

    @staticmethod
    def _sanitize_loss(name: str, value: torch.Tensor):
        """Returns (clean, is_finite). Non-finite distill terms are replaced
        by a fresh no-grad zero so a single bad step contributes nothing —
        value AND gradient — instead of poisoning the entire run."""
        if torch.isfinite(value).all():
            return value, True
        rank0_print(f"[GRACE] WARN: {name} non-finite — zeroing this step")
        return value.new_zeros(()), False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ga = self.grace_args

        if not ga.grace_enable:
            outputs = model(**inputs, use_cache=False)
            loss = outputs.loss
            self.learning_diagnostics.observe_term("ce", loss, 1.0)
            self.grace_profiler.arm_backward(loss)
            self._grace_logs = {"ce": float(loss.detach())}
            return (loss, outputs) if return_outputs else loss

        # --- student forward (quantized, grad on) ---
        # Student keeps output_hidden_states=True: under gradient_checkpointing
        # the inter-block activations are already retained by the checkpoint
        # graph, so the extra tuple costs only references. The hook trick
        # used for the teacher does NOT work here (a forward hook on a
        # checkpointed layer captures the inner no-grad tensor).
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        ce_loss = outputs.loss
        student_logits = outputs.logits
        student_hidden = outputs.hidden_states[ga.rcka_layer]

        # --- teacher forward (frozen, no grad, RCKA layer via hook) ---
        teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        if self.teacher_pen_buf is not None:
            self.teacher_pen_buf["hidden"] = None
        with torch.no_grad():
            t_out = self.teacher(**teacher_inputs, use_cache=False)
        teacher_logits = t_out.logits
        teacher_hidden = (
            self.teacher_pen_buf["hidden"] if self.teacher_pen_buf is not None else None
        )
        if self.teacher_pen_buf is not None:
            self.teacher_pen_buf["hidden"] = None  # release ref promptly

        # --- 1) confidence-gated decoupled KD ---
        with self.grace_profiler.cuda_region("gdkd"):
            l_gdkd = gated_dkd_loss(
                student_logits, teacher_logits, inputs["labels"],
                temperature=ga.dkd_temperature,
                alpha=ga.dkd_alpha, beta_dkd=ga.dkd_beta,
                chunk_size=ga.dkd_chunk_size,
                checkpoint_chunks=ga.dkd_checkpoint_chunks,
            )

        # --- 2) relational CKA: per-modality alignment ---
        ids = inputs["input_ids"]
        image_mask = ids == IMAGE_TOKEN_INDEX
        video_mask = ids == VIDEO_TOKEN_INDEX
        if teacher_hidden is None:
            l_rcka = student_hidden.new_zeros(())
        else:
            with self.grace_profiler.cuda_region("rcka"):
                l_rcka = rcka_loss(
                    student_hidden, teacher_hidden,
                    image_mask, video_mask,
                    inputs.get("image_grid_thw"), inputs.get("video_grid_thw"),
                    self.merge_size,
                )

        # --- 3) NaN/Inf guard on distill terms (CE is left alone — a NaN
        # there is a real bug we want the optimizer to surface). ---
        l_gdkd_clean, gdkd_finite = self._sanitize_loss("gdkd", l_gdkd)
        l_rcka_clean, _ = self._sanitize_loss("rcka", l_rcka)

        # --- 4) Adaptive IB controller — averaged GDKD per OPTIMIZER step.
        # Local accumulation (no per-micro-batch collective); one all_reduce
        # at the accumulation-cycle boundary. ---
        if gdkd_finite:
            v = l_gdkd.detach()
            self._gdkd_accum = v if self._gdkd_accum is None else (self._gdkd_accum + v)
            self._gdkd_count += 1
        ga_steps = max(1, int(self.args.gradient_accumulation_steps))
        self._micro_step += 1
        if self._micro_step % ga_steps == 0:
            if self._gdkd_count > 0 and self._gdkd_accum is not None:
                avg = self._gdkd_accum / self._gdkd_count
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(avg, op=dist.ReduceOp.SUM)
                    avg = avg / dist.get_world_size()
                self.ib.update(float(avg.item()))
            self._gdkd_accum = None
            self._gdkd_count = 0
        beta = self.ib.beta

        # --- 5) Linear warmup on distill terms. RCKA in particular is
        # noisy in the first hundreds of steps before the student's
        # representation has shape. ---
        step = int(getattr(self.state, "global_step", 0))
        dkd_scale = _warmup_scale(step, ga.dkd_warmup_steps)
        rcka_scale = _warmup_scale(step, ga.rcka_warmup_steps)

        loss = (
            ce_loss
            + beta * dkd_scale * l_gdkd_clean
            + ga.rcka_weight * rcka_scale * l_rcka_clean
        )
        self.learning_diagnostics.observe_term("ce", ce_loss, 1.0)
        self.learning_diagnostics.observe_term(
            "gdkd", l_gdkd_clean, beta * dkd_scale
        )
        self.learning_diagnostics.observe_term(
            "rcka", l_rcka_clean, ga.rcka_weight * rcka_scale
        )
        self.grace_profiler.arm_backward(loss)

        self._grace_logs = {
            "ce": round(float(ce_loss.detach()), 4),
            "gdkd": round(float(l_gdkd.detach()), 4),
            "rcka": round(float(l_rcka.detach()), 4),
            "beta": round(float(beta), 4),
            "dkd_scale": round(dkd_scale, 4),
            "rcka_scale": round(rcka_scale, 4),
        }
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self._grace_logs:
            logs = {**logs, **self._grace_logs}
        return super().log(logs, *args, **kwargs)


# ----------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------
def train(attn_implementation="flash_attention_2"):
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, QATArguments, GraceArguments)
    )
    model_args, data_args, training_args, qat_args, grace_args = (
        parser.parse_args_into_dataclasses()
    )

    os.makedirs(training_args.output_dir, exist_ok=True)

    if qat_args.qat_enable and qat_args.qat_bits not in (4, 8):
        raise ValueError(f"--qat_bits must be 4 or 8, got {qat_args.qat_bits}")
    if grace_args.grace_enable and not grace_args.teacher_model_path:
        raise ValueError("--teacher_model_path is required when --grace_enable True")
    if training_args.lora_enable and not qat_args.qat_enable:
        raise ValueError("GRACE native LoRA requires --qat_enable True.")
    if training_args.lora_enable and training_args.lora_r <= 0:
        raise ValueError(f"--lora_r must be positive, got {training_args.lora_r}")

    # ---- student ----
    model, model_type = _build_student(model_args, training_args, attn_implementation), "qwen3vl"
    data_args.model_type = model_type
    rank0_print(f"[GRACE] student = {model_args.model_name_or_path}")

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    processor.tokenizer.model_max_length = training_args.model_max_length
    merge_size = getattr(processor.image_processor, "merge_size", 2)

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    set_model(model_args, model)
    if training_args.lora_enable:
        # Keep the vision/merger choices established by set_model, but freeze
        # every language-model base parameter. Targeted linears are changed
        # only through LoRA A/B and their corresponding QAT log_scale.
        language_model = _qwen3vl_submodule(model, "language_model")
        for p in language_model.parameters():
            p.requires_grad_(False)
        lm_head = getattr(model, "lm_head", None) or getattr(model.model, "lm_head", None)
        if lm_head is not None:
            for p in lm_head.parameters():
                p.requires_grad_(False)

    # ---- QAT swap on the student (before the trainer / DeepSpeed) ----
    if qat_args.qat_enable:
        skip = list(DEFAULT_QAT_SKIP_KEYWORDS)
        if qat_args.qat_skip_extra:
            skip.extend([s for s in qat_args.qat_skip_extra.split(",") if s.strip()])
        if training_args.lora_enable:
            lora_targets = [
                name.strip()
                for name in training_args.lora_target_modules.split(",")
                if name.strip()
            ]
            replaced, adapted = replace_linears_with_qat_lora(
                model,
                bits=qat_args.qat_bits,
                group_size=qat_args.qat_group_size,
                target_modules=lora_targets,
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                skip_keywords=skip,
                method="grace",
            )
            model.config.qat_lora = {
                "enabled": True,
                "r": training_args.lora_r,
                "alpha": training_args.lora_alpha,
                "dropout": training_args.lora_dropout,
                "target_modules": lora_targets,
                "num_adapted": len(adapted),
                "merged": False,
            }
        else:
            replaced = replace_linears_with_qat(
                model, bits=qat_args.qat_bits,
                group_size=qat_args.qat_group_size, skip_keywords=skip,
            )
        model.config.qat = {
            "bits": qat_args.qat_bits, "group_size": qat_args.qat_group_size,
            "skip_keywords": skip, "num_replaced": len(replaced),
        }
        rank0_print(f"[QAT] LLM linears quantized: {len(replaced)} layers, "
                    f"W{qat_args.qat_bits}, group={qat_args.qat_group_size}")

    trainable = sum(
        int(getattr(p, "ds_numel", p.numel()))
        for p in model.parameters() if p.requires_grad
    )
    total = sum(int(getattr(p, "ds_numel", p.numel())) for p in model.parameters())
    rank0_print(
        f"[GRACE] trainable student parameters: {trainable:,} / {total:,} "
        f"({100.0 * trainable / max(total, 1):.3f}%)"
    )

    # ---- teacher (frozen, BF16, no QAT) ----
    teacher, teacher_pen_buf = (
        _build_teacher(grace_args, training_args) if grace_args.grace_enable else (None, None)
    )
    # transformers >=4.57 moves LLM-side config fields into a nested
    # ``text_config`` on multimodal configs; older releases kept them flat.
    def _llm_cfg_attr(cfg, name):
        tc = getattr(cfg, "text_config", None)
        if tc is not None and hasattr(tc, name):
            return getattr(tc, name)
        return getattr(cfg, name)
    if teacher is not None:
        t_vocab = _llm_cfg_attr(teacher.config, "vocab_size")
        s_vocab = _llm_cfg_attr(model.config, "vocab_size")
        if t_vocab != s_vocab:
            raise ValueError(
                f"Teacher/student vocab mismatch ({t_vocab} vs {s_vocab}); "
                f"logit distillation requires a shared vocab."
            )
        rank0_print(f"[GRACE] teacher = {grace_args.teacher_model_path} "
                    f"(layers={_llm_cfg_attr(teacher.config, 'num_hidden_layers')}, frozen BF16)")

    # ---- data ----
    data_module = make_supervised_data_module(processor, data_args=data_args)

    # ---- IB controller ----
    ib = AdaptiveIBController(
        tau=grace_args.ib_tau, eta=grace_args.ib_eta,
        beta_init=grace_args.ib_beta_init,
        beta_min=grace_args.ib_beta_min, beta_max=grace_args.ib_beta_max,
        ema_decay=grace_args.ib_ema_decay,
    )
    _load_ib_state(ib, training_args.output_dir)

    base_lr = training_args.learning_rate
    scale_lr = qat_args.qat_scale_lr if qat_args.qat_scale_lr is not None else base_lr

    trainer = GraceTrainer(
        model=model, processing_class=tokenizer, args=training_args,
        teacher=teacher, teacher_pen_buf=teacher_pen_buf,
        grace_args=grace_args, ib_controller=ib,
        merge_size=merge_size, base_lr=base_lr, scale_lr=scale_lr,
        **data_module,
    )
    trainer.add_callback(SaveProcessorCallback(processor))
    trainer.add_callback(GraceStateCallback(ib))
    trainer.add_callback(
        GraceProfilerCallback(
            trainer.grace_profiler,
            profile_deepspeed_comms=grace_args.profile_deepspeed_comms,
        )
    )
    trainer.add_callback(
        GraceLearningDiagnosticsCallback(trainer.learning_diagnostics)
    )

    pause_callback = CheckpointPauseCallback()
    trainer.add_callback(pause_callback)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    if pause_callback.paused:
        rank0_print("[GRACE] Paused after checkpoint; skipping final merged export.")
        _close_distributed_process_group()
        return

    if grace_args.skip_final_export:
        rank0_print(
            "[GRACE-PROFILE] Skipping final merge/bake/export as requested."
        )
        _close_distributed_process_group()
        return

    model.config.use_cache = True

    # The replicated 8B teacher is no longer needed. Releasing it before the
    # merge leaves headroom for the chunked LoRA delta and QAT bake.
    trainer.teacher = None
    teacher = None
    teacher_pen_buf = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    is_zero3 = is_deepspeed_zero3_enabled()

    # ---- merge native LoRA into the dense base before quantization bake ----
    if training_args.lora_enable:
        n_lora = merge_lora_weights_inplace(model, is_zero3=is_zero3)
        model.config.qat_lora["merged"] = True
        rank0_print(f"[GRACE-LoRA] Merged {n_lora} adapters into base weights.")

    # ---- bake fake-quantized weights, then save (same as QAT path) ----
    if qat_args.qat_enable:
        try:
            n = bake_qat_weights_inplace(model, is_zero3=is_zero3)
            rank0_print(f"[QAT] Baked fake-quantized weights into {n} linear layers.")
        except Exception as e:
            if training_args.lora_enable:
                raise RuntimeError(
                    "QAT bake failed after LoRA merge; refusing to save a "
                    "partially exported final model."
                ) from e
            rank0_print(f"[QAT] Weight baking failed: {e}")

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if _is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)

    if qat_args.qat_enable and _is_world_process_zero():
        try:
            if training_args.lora_enable:
                removed = strip_qat_lora_from_checkpoint_dir(training_args.output_dir)
                rank0_print(
                    f"[GRACE-LoRA] Stripped {removed} scale/adapter entries "
                    "from merged checkpoint."
                )
            else:
                removed = strip_log_scale_from_checkpoint_dir(training_args.output_dir)
                rank0_print(f"[QAT] Stripped {removed} log_scale entries from checkpoint.")
        except Exception as e:
            if training_args.lora_enable:
                raise RuntimeError(
                    "Could not remove merged LoRA/QAT auxiliary tensors from "
                    "the final model."
                ) from e
            rank0_print(f"[QAT] log_scale strip failed: {e}")

    if qat_args.qat_enable and qat_args.qat_save_quantized and \
            _is_world_process_zero():
        try:
            qsd = materialize_quantized_state_dict(model)
            out = os.path.join(training_args.output_dir, "qat_quantized_weights.bin")
            torch.save(qsd, out)
            rank0_print(f"[QAT] Saved fake-quantized weights + scales to {out}")
        except Exception as e:
            rank0_print(f"[QAT] Quantized state-dict export failed: {e}")

    _close_distributed_process_group()


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
