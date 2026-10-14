"""Lightweight, opt-in timing for the GRACE training path.

The profiler uses CUDA events and records no tensor values used by the loss.
It is disabled by default so normal reproduction runs retain their original
execution path.  DeepSpeed communication logging is intentionally handled by
a separate profiling run because DeepSpeed synchronizes after every profiled
collective and therefore perturbs throughput substantially.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional

import torch
import torch.distributed as dist
from transformers import TrainerCallback


REGION_NAMES = (
    "student_backbone",
    "student_lm_head",
    "teacher_backbone",
    "teacher_lm_head",
    "gdkd",
    "rcka",
    "backward",
    "optimizer",
    "train_cycle_gpu",
)


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


class GraceStepProfiler:
    """Collect per-optimizer-step timings without changing model outputs."""

    def __init__(self, enabled: bool, profile_steps: int, output_dir: str,
                 merge_size: int, method: str = "grace"):
        self.enabled = bool(enabled and profile_steps > 0 and torch.cuda.is_available())
        self.profile_steps = max(0, int(profile_steps))
        self.output_dir = output_dir
        self.merge_size = int(merge_size)
        method_label = str(method).strip().upper()
        if not method_label:
            raise ValueError("method must be a non-empty name for profiler logging")
        self.log_prefix = f"[{method_label}-PROFILE]"
        self.rank = _rank()
        self.world_size = _world_size()

        self.active = False
        self.step_number = 0
        self.data_wait_ms = 0.0
        self.step_wall_start = 0.0
        self.regions = defaultdict(list)
        self.hook_stacks = defaultdict(list)
        self.backward_start: Optional[torch.cuda.Event] = None
        self.optimizer_start_event: Optional[torch.cuda.Event] = None
        self.cycle_start_event: Optional[torch.cuda.Event] = None
        self.counts = {"samples": 0, "tokens": 0}
        self.supervised_count_tensors = []
        self.image_count_tensors = []
        self.records = []
        self.hook_handles = []

        if self.rank == 0 and self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)
            self.jsonl_path = os.path.join(self.output_dir, "grace_timing_steps.jsonl")
            self.summary_path = os.path.join(self.output_dir, "grace_timing_summary.json")
        else:
            self.jsonl_path = ""
            self.summary_path = ""

    @staticmethod
    def _event() -> torch.cuda.Event:
        return torch.cuda.Event(enable_timing=True)

    def install_model_hooks(self, student, teacher=None) -> None:
        if not self.enabled:
            return
        self._install_pair(student, "student")
        if teacher is not None:
            self._install_pair(teacher, "teacher")

    def _install_pair(self, wrapper, prefix: str) -> None:
        backbone = getattr(wrapper, "model", None)
        lm_head = getattr(wrapper, "lm_head", None)
        if backbone is None or lm_head is None:
            raise RuntimeError(
                f"Profiling could not locate model/lm_head on {type(wrapper).__name__}."
            )
        self._install_region_hooks(backbone, f"{prefix}_backbone")
        self._install_region_hooks(lm_head, f"{prefix}_lm_head")

    def _install_region_hooks(self, module, name: str) -> None:
        def pre_hook(_module, _args):
            if not self.active:
                return
            start = self._event()
            start.record()
            self.hook_stacks[name].append(start)

        def post_hook(_module, _args, _output):
            if not self.active or not self.hook_stacks[name]:
                return
            start = self.hook_stacks[name].pop()
            end = self._event()
            end.record()
            self.regions[name].append((start, end))

        self.hook_handles.append(module.register_forward_pre_hook(pre_hook))
        self.hook_handles.append(module.register_forward_hook(post_hook))

    def begin_step(self, step_number: int, data_wait_ms: float) -> None:
        self.active = self.enabled and step_number <= self.profile_steps
        if not self.active:
            return
        self.step_number = int(step_number)
        self.data_wait_ms = float(data_wait_ms)
        self.step_wall_start = time.perf_counter()
        self.regions = defaultdict(list)
        self.hook_stacks = defaultdict(list)
        self.backward_start = None
        self.optimizer_start_event = None
        self.counts = {"samples": 0, "tokens": 0}
        self.supervised_count_tensors = []
        self.image_count_tensors = []
        torch.cuda.reset_peak_memory_stats()
        self.cycle_start_event = self._event()
        self.cycle_start_event.record()

    def observe_inputs(self, inputs: dict) -> None:
        if not self.active:
            return
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            self.counts["tokens"] += int(input_ids.numel())

        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None and attention_mask.ndim == 1:
            # Flattened FlashAttention batches use cumulative sequence lengths.
            self.counts["samples"] += max(0, int(attention_mask.numel()) - 1)
        elif input_ids is not None:
            self.counts["samples"] += int(input_ids.shape[0])

        labels = inputs.get("labels")
        if labels is not None:
            self.supervised_count_tensors.append(labels.ne(-100).sum().detach())

        for key in ("image_grid_thw", "video_grid_thw"):
            grid = inputs.get(key)
            if grid is not None and grid.numel() > 0:
                visual_tokens = grid.prod(dim=-1).sum() // (self.merge_size * self.merge_size)
                self.image_count_tensors.append(visual_tokens.detach())

    @contextmanager
    def cuda_region(self, name: str):
        if not self.active:
            yield
            return
        start = self._event()
        end = self._event()
        start.record()
        try:
            yield
        finally:
            end.record()
            self.regions[name].append((start, end))

    def arm_backward(self, loss: torch.Tensor) -> None:
        if not self.active or not loss.requires_grad:
            return
        start = self._event()
        self.backward_start = start

        def mark_backward_start(grad):
            start.record()
            return grad

        loss.register_hook(mark_backward_start)

    def finish_micro_step(self) -> None:
        if not self.active or self.backward_start is None:
            return
        end = self._event()
        end.record()
        self.regions["backward"].append((self.backward_start, end))
        self.backward_start = None

    def optimizer_start(self) -> None:
        if not self.active:
            return
        self.optimizer_start_event = self._event()
        self.optimizer_start_event.record()

    def optimizer_end(self) -> None:
        if not self.active or self.optimizer_start_event is None:
            return
        end = self._event()
        end.record()
        self.regions["optimizer"].append((self.optimizer_start_event, end))
        self.optimizer_start_event = None

    @staticmethod
    def _elapsed_ms(pairs) -> float:
        total = 0.0
        for start, end in pairs:
            end.synchronize()
            total += float(start.elapsed_time(end))
        return total

    def _aggregate(self, local_metrics: dict, local_counts: dict) -> dict:
        device = torch.device("cuda", torch.cuda.current_device())
        metric_keys = sorted(local_metrics)
        values = torch.tensor([local_metrics[k] for k in metric_keys], dtype=torch.float64, device=device)
        sums = values.clone()
        maxima = values.clone()
        if self.world_size > 1:
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX)

        count_keys = sorted(local_counts)
        counts = torch.tensor([local_counts[k] for k in count_keys], dtype=torch.float64, device=device)
        if self.world_size > 1:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)

        result = {"step": self.step_number, "world_size": self.world_size}
        for idx, key in enumerate(metric_keys):
            result[f"{key}_rank_mean"] = float((sums[idx] / self.world_size).item())
            result[f"{key}_rank_max"] = float(maxima[idx].item())
        for idx, key in enumerate(count_keys):
            result[f"global_{key}"] = int(counts[idx].item())

        wall_s = result["total_step_wall_ms_rank_max"] / 1000.0
        if wall_s > 0:
            result["global_tokens_per_second"] = result["global_tokens"] / wall_s
            result["global_supervised_tokens_per_second"] = (
                result["global_supervised_tokens"] / wall_s
            )
            result["global_visual_tokens_per_second"] = (
                result["global_visual_tokens"] / wall_s
            )
            result["global_samples_per_second"] = result["global_samples"] / wall_s
        return result

    def finish_step(self) -> None:
        if not self.active:
            return
        cycle_end = self._event()
        cycle_end.record()
        if self.cycle_start_event is not None:
            self.regions["train_cycle_gpu"].append((self.cycle_start_event, cycle_end))

        region_ms = {name: self._elapsed_ms(self.regions.get(name, [])) for name in REGION_NAMES}
        compute_wall_ms = (time.perf_counter() - self.step_wall_start) * 1000.0
        total_wall_ms = self.data_wait_ms + compute_wall_ms
        local_metrics = {
            **{f"{name}_ms": region_ms[name] for name in REGION_NAMES},
            "data_wait_and_preprocess_ms": self.data_wait_ms,
            "compute_and_update_wall_ms": compute_wall_ms,
            "total_step_wall_ms": total_wall_ms,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024 ** 3),
        }

        supervised = sum(int(t.item()) for t in self.supervised_count_tensors)
        visual = sum(int(t.item()) for t in self.image_count_tensors)
        local_counts = {
            **self.counts,
            "supervised_tokens": supervised,
            "visual_tokens": visual,
        }
        result = self._aggregate(local_metrics, local_counts)

        if self.rank == 0:
            self.records.append(result)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, sort_keys=True) + "\n")
            print(
                f"{self.log_prefix} "
                f"step={result['step']} "
                f"wall={result['total_step_wall_ms_rank_max'] / 1000.0:.2f}s "
                f"student={result['student_backbone_ms_rank_max'] / 1000.0:.2f}s "
                f"teacher={result['teacher_backbone_ms_rank_max'] / 1000.0:.2f}s "
                f"gdkd={result['gdkd_ms_rank_max'] / 1000.0:.2f}s "
                f"backward={result['backward_ms_rank_max'] / 1000.0:.2f}s "
                f"tok/s={result['global_tokens_per_second']:.1f} "
                f"peak={result['peak_allocated_gib_rank_max']:.2f}GiB"
            )
        self.active = False

    def close(self) -> None:
        if self.rank == 0 and self.records:
            def means(records):
                numeric = {
                    key
                    for record in records
                    for key, value in record.items()
                    if isinstance(value, (int, float)) and key not in ("step", "world_size")
                }
                return {
                    key: sum(float(record[key]) for record in records if key in record)
                    / sum(1 for record in records if key in record)
                    for key in sorted(numeric)
                }

            summary = {
                "profiled_steps": len(self.records),
                "all_steps_mean": means(self.records),
                "steady_state_excluding_first_mean": means(self.records[1:]) if len(self.records) > 1 else None,
                "steps_file": self.jsonl_path,
            }
            with open(self.summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, sort_keys=True)
            print(f"{self.log_prefix} summary={self.summary_path}")
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()


class GraceProfilerCallback(TrainerCallback):
    def __init__(self, profiler: GraceStepProfiler, profile_deepspeed_comms: bool = False):
        self.profiler = profiler
        self.profile_deepspeed_comms = bool(profile_deepspeed_comms)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.profiler.optimizer_start()
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.profiler.optimizer_end()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self.profiler.finish_step()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.profiler.close()
        if self.profile_deepspeed_comms:
            import deepspeed.comm as ds_dist

            ds_dist.log_summary(show_straggler=False)
        return control


class GraceLearningDiagnostics:
    """Opt-in proof that adapters, QAT scales, and GRACE terms participate.

    Parameter hooks capture backward signal before DeepSpeed can partition or
    clear gradients. Small CPU snapshots then verify that trainable parameter
    values actually change while a representative frozen QAT scale does not.
    This is intentionally diagnostic-only because the extra norm reductions
    perturb timing.
    """

    CATEGORIES = ("lora_A", "lora_B", "qat_scale", "merger", "other")

    def __init__(self, enabled: bool, max_steps: int, output_dir: str,
                 method: str = "grace"):
        self.enabled = bool(enabled and max_steps > 0)
        self.max_steps = max(0, int(max_steps))
        self.output_dir = output_dir
        method_label = str(method).strip().upper()
        if not method_label:
            raise ValueError("method must be a non-empty name for learning diagnostics logging")
        self.log_prefix = f"[{method_label}-LEARNING]"
        self.rank = _rank()
        self.hook_handles = []
        self.grad_sq = {}
        self.grad_tensors = defaultdict(int)
        self.term_stats = defaultdict(lambda: defaultdict(float))
        self.snapshots = {}
        self.records = []
        self.pending = None
        self.inventory = {}
        self.teacher_trainable = 0
        self.runtime_setup_deltas = {}
        self.runtime_baseline_refreshed = False
        self.jsonl_path = os.path.join(output_dir, "learning_diagnostics.jsonl")
        self.summary_path = os.path.join(output_dir, "learning_diagnostics_summary.json")

    @staticmethod
    def _category(name: str) -> str:
        if name.endswith("lora_A"):
            return "lora_A"
        if name.endswith("lora_B"):
            return "lora_B"
        if name.endswith("log_scale"):
            return "qat_scale"
        if "merger" in name:
            return "merger"
        return "other"

    @staticmethod
    def _numel(param) -> int:
        return int(getattr(param, "ds_numel", param.numel()))

    @staticmethod
    def _sample(param, limit: int = 4096) -> torch.Tensor:
        return param.detach().reshape(-1)[:limit].float().cpu().clone()

    def install(self, student, teacher) -> None:
        if not self.enabled:
            return
        if self.rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)

        inventory = {
            category: {"tensors": 0, "parameters": 0}
            for category in self.CATEGORIES
        }
        frozen_scale = None
        for name, param in student.named_parameters():
            if param.requires_grad:
                category = self._category(name)
                inventory[category]["tensors"] += 1
                inventory[category]["parameters"] += self._numel(param)
                if category not in self.snapshots:
                    self.snapshots[category] = {
                        "name": name,
                        "param": param,
                        "initial": self._sample(param),
                    }

                def capture(grad, category=category):
                    if grad is not None:
                        value = grad.detach().float().square().sum()
                        self.grad_sq[category] = self.grad_sq.get(category, 0.0) + value
                        self.grad_tensors[category] += 1
                    return grad

                self.hook_handles.append(param.register_hook(capture))
            elif name.endswith("log_scale") and frozen_scale is None:
                frozen_scale = {
                    "name": name,
                    "param": param,
                    "initial": self._sample(param),
                }

        if frozen_scale is not None:
            self.snapshots["frozen_qat_scale"] = frozen_scale

        teacher_trainable = sum(
            self._numel(param) for param in (teacher.parameters() if teacher is not None else [])
            if param.requires_grad
        )
        self.inventory = inventory
        self.teacher_trainable = teacher_trainable
        if self.rank == 0:
            with open(os.path.join(self.output_dir, "learning_inventory.json"), "w") as f:
                json.dump(
                    {
                        "trainable_inventory": inventory,
                        "teacher_trainable_parameters": teacher_trainable,
                        "snapshot_names": {
                            key: value["name"] for key, value in self.snapshots.items()
                        },
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )

    def refresh_baseline_after_runtime_setup(self) -> None:
        """Exclude framework dtype/device setup from optimizer-update checks.

        DeepSpeed prepares the model after ``install`` and may cast even an
        FP32 frozen parameter to BF16. That deterministic setup conversion is
        not a training update, but comparing against the earlier snapshot
        incorrectly reported that frozen QAT scales changed. ``on_train_begin``
        runs after model/optimizer preparation and before the first batch.
        """
        if not self.enabled or self.runtime_baseline_refreshed:
            return
        for category, snapshot in self.snapshots.items():
            current = self._sample(snapshot["param"])
            self.runtime_setup_deltas[category] = {
                "sample_parameter": snapshot["name"],
                "sample_max_abs_delta_during_setup": float(
                    (current - snapshot["initial"]).abs().max()
                ),
                "runtime_dtype": str(snapshot["param"].dtype),
            }
            snapshot["initial"] = current
        self.runtime_baseline_refreshed = True

    def observe_term(self, name: str, tensor: torch.Tensor, coefficient: float) -> None:
        if not self.enabled:
            return
        stats = self.term_stats[name]
        stats["raw_sum"] += float(tensor.detach())
        stats["coefficient_sum"] += float(coefficient)
        stats["microbatches"] += 1
        stats["requires_grad_count"] += int(tensor.requires_grad)
        if tensor.requires_grad:
            def capture(multiplier, name=name):
                self.term_stats[name]["backward_multiplier_abs_sum"] += float(
                    multiplier.detach().abs()
                )
                return multiplier
            tensor.register_hook(capture)

    def before_optimizer_step(self, step: int) -> None:
        if not self.enabled or step > self.max_steps:
            return
        gradients = {}
        for category in self.CATEGORIES:
            value = self.grad_sq.get(category)
            gradients[category] = {
                "backward_signal_l2": (
                    float(torch.sqrt(value).item()) if isinstance(value, torch.Tensor) else 0.0
                ),
                "gradient_tensors_seen": self.grad_tensors.get(category, 0),
            }
        terms = {}
        for name, stats in self.term_stats.items():
            count = max(1.0, stats["microbatches"])
            terms[name] = {
                "raw_loss_mean": stats["raw_sum"] / count,
                "coefficient_mean": stats["coefficient_sum"] / count,
                "requires_grad_fraction": stats["requires_grad_count"] / count,
                "backward_multiplier_abs_sum": stats[
                    "backward_multiplier_abs_sum"
                ],
            }
        self.pending = {"step": int(step), "gradients": gradients, "terms": terms}
        self.grad_sq = {}
        self.grad_tensors = defaultdict(int)
        self.term_stats = defaultdict(lambda: defaultdict(float))

    def after_optimizer_step(self) -> None:
        if not self.enabled or self.pending is None:
            return
        deltas = {}
        for category, snapshot in self.snapshots.items():
            current = self._sample(snapshot["param"])
            initial = snapshot["initial"]
            deltas[category] = {
                "sample_parameter": snapshot["name"],
                "sample_max_abs_delta_from_start": float((current - initial).abs().max()),
            }
        self.pending["parameter_deltas"] = deltas
        if self.rank == 0:
            self.records.append(self.pending)
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(self.pending, sort_keys=True) + "\n")
            g = self.pending["gradients"]
            t = self.pending["terms"]
            print(
                f"{self.log_prefix} "
                f"step={self.pending['step']} "
                f"lora_A={g['lora_A']['backward_signal_l2']:.3e} "
                f"lora_B={g['lora_B']['backward_signal_l2']:.3e} "
                f"qat_scale={g['qat_scale']['backward_signal_l2']:.3e} "
                f"gdkd_coeff={t.get('gdkd', {}).get('coefficient_mean', 0.0):.3e} "
                f"rcka_coeff={t.get('rcka', {}).get('coefficient_mean', 0.0):.3e}"
            )
        self.pending = None

    def close(self) -> None:
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        if self.rank == 0 and self.records:
            last = self.records[-1]
            final_deltas = {}
            for category, snapshot in self.snapshots.items():
                current = self._sample(snapshot["param"])
                final_deltas[category] = {
                    "sample_parameter": snapshot["name"],
                    "sample_max_abs_delta_from_start": float(
                        (current - snapshot["initial"]).abs().max()
                    ),
                }
            summary = {
                "steps_recorded": len(self.records),
                "trainable_inventory": self.inventory,
                "teacher_trainable_parameters": self.teacher_trainable,
                "runtime_setup_parameter_deltas": self.runtime_setup_deltas,
                "last_step": last,
                "final_parameter_deltas": final_deltas,
                "checks": {
                    "teacher_is_frozen": self.teacher_trainable == 0,
                    "lora_B_received_gradient": any(
                        r["gradients"]["lora_B"]["backward_signal_l2"] > 0
                        for r in self.records
                    ),
                    "qat_scale_received_gradient": any(
                        r["gradients"]["qat_scale"]["backward_signal_l2"] > 0
                        for r in self.records
                    ),
                    "gdkd_connected_to_backward": any(
                        r["terms"].get("gdkd", {}).get(
                            "backward_multiplier_abs_sum", 0.0
                        ) > 0
                        for r in self.records
                    ),
                    "rcka_connected_to_backward_after_warmup": any(
                        r["terms"].get("rcka", {}).get(
                            "backward_multiplier_abs_sum", 0.0
                        ) > 0
                        for r in self.records
                    ),
                    "lora_B_parameter_changed": final_deltas.get(
                        "lora_B", {}
                    ).get("sample_max_abs_delta_from_start", 0.0) > 0.0,
                    "qat_scale_parameter_changed": final_deltas.get(
                        "qat_scale", {}
                    ).get("sample_max_abs_delta_from_start", 0.0) > 0.0,
                    "frozen_qat_scale_unchanged": final_deltas.get(
                        "frozen_qat_scale", {}
                    ).get("sample_max_abs_delta_from_start", 0.0) == 0.0,
                },
            }
            with open(self.summary_path, "w") as f:
                json.dump(summary, f, indent=2, sort_keys=True)
            print(f"{self.log_prefix} summary={self.summary_path}")


class GraceLearningDiagnosticsCallback(TrainerCallback):
    def __init__(self, diagnostics: GraceLearningDiagnostics):
        self.diagnostics = diagnostics

    def on_train_begin(self, args, state, control, **kwargs):
        self.diagnostics.refresh_baseline_after_runtime_setup()
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.diagnostics.before_optimizer_step(int(state.global_step) + 1)
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.diagnostics.after_optimizer_step()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.diagnostics.close()
        return control
