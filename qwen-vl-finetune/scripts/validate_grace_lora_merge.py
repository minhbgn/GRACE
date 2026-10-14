#!/usr/bin/env python3
"""Merge a native GRACE QAT-LoRA checkpoint and validate HF compatibility.

This is an export-only check: it does not create a teacher, dataset, optimizer,
or run another training step.  It intentionally mirrors the final training
export order (LoRA merge -> QAT bake -> save -> auxiliary-key stripping).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

# Accelerate may import DeepSpeed, whose Triton atexit hook writes an autotune
# cache. Keep that incidental cache in a writable location on managed hosts.
_TRITON_CACHE = Path("/tmp") / f"grace-triton-cache-{os.getuid()}"
_TRITON_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TRITON_CACHE_DIR", str(_TRITON_CACHE))

import torch
import torch.nn.functional as F
from accelerate import load_checkpoint_in_model
from safetensors import safe_open
from transformers import AutoConfig, Qwen3VLForConditionalGeneration


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from qwenvl.train.qat_modules import (  # noqa: E402
    QATLoraLinear,
    bake_qat_weights_inplace,
    merge_lora_weights_inplace,
    replace_linears_with_qat_lora,
    strip_qat_lora_from_checkpoint_dir,
)


PROCESSOR_FILES = (
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Merge device. CPU avoids consuming shared GPU memory.",
    )
    parser.add_argument("--sample-modules", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def checkpoint_weight_files(checkpoint: Path) -> list[Path]:
    files = sorted(checkpoint.glob("model*.safetensors"))
    if not files:
        files = sorted(checkpoint.glob("pytorch_model*.bin"))
    return [p for p in files if not p.name.endswith(".index.json")]


def checkpoint_aux_counts(checkpoint: Path) -> dict[str, int]:
    counts = {"lora_A": 0, "lora_B": 0, "log_scale": 0}
    for path in checkpoint_weight_files(checkpoint):
        if path.suffix != ".safetensors":
            continue
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                for fragment in counts:
                    if fragment in key:
                        counts[fragment] += 1
    return counts


def select_modules(model: torch.nn.Module, count: int):
    modules = [(name, module) for name, module in model.named_modules()
               if isinstance(module, QATLoraLinear)]
    if not modules:
        raise RuntimeError("No QATLoraLinear modules found after checkpoint setup.")
    count = min(max(1, count), len(modules))
    if count == 1:
        return [modules[0]], len(modules)
    indices = sorted({round(i * (len(modules) - 1) / (count - 1))
                      for i in range(count)})
    return [modules[i] for i in indices], len(modules)


@torch.inference_mode()
def capture_premerge_samples(samples, seed: int):
    captured = []
    for index, (name, module) in enumerate(samples):
        generator = torch.Generator(device="cpu").manual_seed(seed + index)
        x = torch.randn(1, module.in_features, generator=generator,
                        dtype=torch.float32).to(
                            device=module.weight.device,
                            dtype=module.weight.dtype,
                        )
        y = module(x).float().cpu()
        rows = sorted({0, module.out_features // 2, module.out_features - 1})
        cols = sorted({0, module.in_features // 2, module.in_features - 1})
        expected = {}
        for row in rows:
            for col in cols:
                delta = torch.dot(module.lora_B[row].float(),
                                  module.lora_A[:, col].float())
                value = module.weight[row, col].float() + (
                    delta * module.lora_scaling
                )
                expected[f"{row},{col}"] = float(value.cpu())
        captured.append({
            "name": name,
            "module": module,
            "input": x.cpu(),
            "training_output": y,
            "expected_merged_entries": expected,
        })
    return captured


@torch.inference_mode()
def validate_samples_after_bake(captured) -> tuple[list[dict], float]:
    rows = []
    max_merge_entry_error = 0.0
    for item in captured:
        module = item["module"]
        for coordinate, expected in item["expected_merged_entries"].items():
            row, col = map(int, coordinate.split(","))
            actual = float(module.weight[row, col].float().cpu())
            # The weight has already been QAT-baked at this point, so the
            # entry can differ from the dense merged value by quantization.
            max_merge_entry_error = max(
                max_merge_entry_error, abs(actual - expected)
            )

        x = item["input"].to(
            device=module.weight.device, dtype=module.weight.dtype
        )
        exported = F.linear(x, module.weight, module.bias).float().cpu()
        trained = item["training_output"]
        diff = exported - trained
        denom = trained.square().mean().sqrt().clamp_min(1e-12)
        rel_rmse = float(diff.square().mean().sqrt() / denom)
        cosine = float(F.cosine_similarity(trained, exported, dim=-1).mean())
        rows.append({
            "module": item["name"],
            "max_abs_error": float(diff.abs().max()),
            "relative_rmse": rel_rmse,
            "cosine_similarity": cosine,
        })
    return rows, max_merge_entry_error


def copy_processor_files(checkpoint: Path, output_dir: Path) -> None:
    for name in PROCESSOR_FILES:
        source = checkpoint / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    qat = getattr(config, "qat", None)
    lora = getattr(config, "qat_lora", None)
    if not isinstance(qat, dict) or not isinstance(lora, dict):
        raise RuntimeError("Checkpoint config lacks qat/qat_lora metadata.")
    if lora.get("merged"):
        raise RuntimeError("Checkpoint is already marked as LoRA-merged.")

    device = torch.device(args.device)
    dtype = torch.bfloat16
    print(f"[MERGE-TEST] Loading native checkpoint on {device}: {checkpoint}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint,
        config=config,
        dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    replaced, adapted = replace_linears_with_qat_lora(
        model,
        bits=int(qat["bits"]),
        group_size=int(qat["group_size"]),
        target_modules=lora["target_modules"],
        r=int(lora["r"]),
        lora_alpha=float(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        skip_keywords=qat["skip_keywords"],
        verbose=False,
    )
    # from_pretrained loads all ordinary tensors but intentionally ignores the
    # native auxiliary keys. Reload strictly after installing QAT-LoRA modules.
    load_checkpoint_in_model(
        model,
        checkpoint,
        device_map={"": str(device)},
        dtype=dtype,
        strict=True,
    )
    model.eval()

    samples, adapter_count = select_modules(model, args.sample_modules)
    captured = capture_premerge_samples(samples, args.seed)
    b_nonzero = sum(int(module.lora_B.count_nonzero().cpu())
                    for _, module in model.named_modules()
                    if isinstance(module, QATLoraLinear))
    if b_nonzero == 0:
        raise RuntimeError("All LoRA B tensors are zero; no learned update to merge.")

    print(f"[MERGE-TEST] Merging {adapter_count} adapters")
    merged_count = merge_lora_weights_inplace(model, is_zero3=False)
    if merged_count != adapter_count:
        raise RuntimeError(
            f"Merged {merged_count} adapters, expected {adapter_count}."
        )
    residual_nonzero = sum(int(module.lora_A.count_nonzero().cpu())
                           + int(module.lora_B.count_nonzero().cpu())
                           for _, module in model.named_modules()
                           if isinstance(module, QATLoraLinear))
    if residual_nonzero:
        raise RuntimeError("Adapter tensors were not zeroed after the merge.")

    print(f"[MERGE-TEST] Baking {len(replaced)} QAT linears")
    baked_count = bake_qat_weights_inplace(model, is_zero3=False)
    if baked_count != len(replaced):
        raise RuntimeError(f"Baked {baked_count} QAT linears, expected {len(replaced)}.")
    drift_rows, max_quantized_entry_shift = validate_samples_after_bake(captured)

    model.config.use_cache = True
    model.config.qat_lora["merged"] = True
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    copy_processor_files(checkpoint, output_dir)
    removed = strip_qat_lora_from_checkpoint_dir(str(output_dir))
    output_aux = checkpoint_aux_counts(output_dir)
    if any(output_aux.values()):
        raise RuntimeError(f"Auxiliary tensors remain after stripping: {output_aux}")

    del captured, samples, model
    gc.collect()

    print("[MERGE-TEST] Reloading stripped output with vanilla Qwen3-VL")
    vanilla = Qwen3VLForConditionalGeneration.from_pretrained(
        output_dir,
        dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        logits = vanilla(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
    logits_finite = bool(torch.isfinite(logits).all())
    if not logits_finite:
        raise RuntimeError("Vanilla reload produced non-finite logits.")

    worst_rel_rmse = max(row["relative_rmse"] for row in drift_rows)
    worst_cosine = min(row["cosine_similarity"] for row in drift_rows)
    report = {
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "device": str(device),
        "input_auxiliary_tensors": checkpoint_aux_counts(checkpoint),
        "adapter_count": adapter_count,
        "qat_linear_count": len(replaced),
        "learned_lora_B_nonzero_values": b_nonzero,
        "merged_adapter_count": merged_count,
        "baked_qat_linear_count": baked_count,
        "stripped_auxiliary_tensors": removed,
        "output_auxiliary_tensors": output_aux,
        "vanilla_reload": True,
        "vanilla_forward_logits_shape": list(logits.shape),
        "vanilla_forward_logits_finite": logits_finite,
        "training_to_export_drift": {
            "sampled_modules": len(drift_rows),
            "worst_relative_rmse": worst_rel_rmse,
            "worst_cosine_similarity": worst_cosine,
            "max_sampled_dense_merge_to_baked_entry_shift": max_quantized_entry_shift,
            "per_module": drift_rows,
        },
    }
    report_path = output_dir / "merge_validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"[MERGE-TEST] PASS: {report_path}")


if __name__ == "__main__":
    main()
