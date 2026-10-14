#!/usr/bin/env python3
"""Shared launcher: prepare once, lock the comparison recipe, then torchrun."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent


def env_int(name, default, minimum=1):
    value = int(os.environ.get(name, default))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def resolved_config():
    count = env_int("SUBSET_SIZE", 32768)
    batch = env_int("GLOBAL_BATCH", 64)
    if count % batch or batch % 2:
        raise ValueError("SUBSET_SIZE must divide into full GLOBAL_BATCH updates; GLOBAL_BATCH must be even")
    data_root = Path(os.environ.get("SHAREGPT4V_ROOT", ROOT / "ShareGPT4V")).resolve()
    return dict(
        subset_size=count, seed=env_int("SEED", 42, 0), subset_seed=env_int("SUBSET_SEED", 2026, 0),
        global_batch=batch, grad_accum=batch // 2, max_steps=count // batch,
        model_max_length=env_int("MODEL_MAX_LENGTH", 1024),
        max_pixels=env_int("MAX_PIXELS", 50176), min_pixels=env_int("MIN_PIXELS", 4096),
        lora_r=env_int("LORA_R", 32), lora_alpha=env_int("LORA_ALPHA", 64),
        lora_dropout=float(os.environ.get("LORA_DROPOUT", "0.05")),
        learning_rate=float(os.environ.get("LEARNING_RATE", "2e-4")),
        mm_projector_lr=float(os.environ.get("MM_PROJECTOR_LR", "1e-5")),
        qat_scale_lr=float(os.environ.get("QAT_SCALE_LR", "2e-4")),
        qat_bits=4, qat_group_size=128, lora_targets="q_proj,k_proj,v_proj,o_proj",
        warmup_ratio=0.03, weight_decay=0.0,
        dkd_temperature=2.0, dkd_alpha=1.0, dkd_beta=4.0,
        dkd_chunk_size=128, dkd_checkpoint_chunks=True,
        rcka_weight=3.0, rcka_layer=-2, rcka_warmup_steps=max(1, round((count // batch) * 50 / 512)),
        ib_tau=3.0, ib_eta=0.003, ib_beta_init=0.5, ib_beta_min=0.1, ib_beta_max=1.0,
        save_steps=env_int("SAVE_STEPS", 50), save_total_limit=env_int("SAVE_TOTAL_LIMIT", 2),
        max_train_hours=float(os.environ.get("MAX_TRAIN_HOURS", "5")),
        peak_memory_gib=float(os.environ.get("PEAK_MEMORY_GIB", "34")),
        model_path=str(Path(os.environ.get("BASE_MODEL_DIR", ROOT / "models/Qwen3-VL-2B-Instruct")).resolve()),
        teacher_path=str(Path(os.environ.get("TEACHER_MODEL_PATH", ROOT / "models/Qwen3-VL-8B-Instruct")).resolve()),
        data_root=str(data_root / "data"),
        annotation=str(data_root / "sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json"),
        deepspeed=str(PROJECT / "scripts/zero2_lora_memory_safe.json"),
    )


def build_command(method, config, experiment, profile=False, resume=False):
    c = config
    output = experiment / ("profiles" if profile else "runs") / method
    values = dict(
        method=method, model_name_or_path=c["model_path"],
        subset_manifest=str(experiment / "subset.json"), output_dir=str(output),
        deepspeed=c["deepspeed"], tune_mm_vision=False, tune_mm_mlp=True, tune_mm_llm=True,
        bf16=True, tf32=True, lora_enable=True, lora_r=c["lora_r"], lora_alpha=c["lora_alpha"],
        lora_dropout=c["lora_dropout"], lora_target_modules=c["lora_targets"],
        max_steps=env_int("PROFILE_STEPS", 3) if profile else c["max_steps"],
        num_train_epochs=1, per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=c["grad_accum"], model_max_length=c["model_max_length"],
        max_pixels=c["max_pixels"], min_pixels=c["min_pixels"],
        data_flatten=True, data_packing=False, gradient_checkpointing=True,
        learning_rate=c["learning_rate"], mm_projector_lr=c["mm_projector_lr"],
        weight_decay=c["weight_decay"], max_grad_norm=1.0, warmup_ratio=c["warmup_ratio"],
        lr_scheduler_type="cosine", optim="adamw_torch", eval_strategy="no",
        save_strategy="steps", save_steps=1 if profile else c["save_steps"],
        save_total_limit=c["save_total_limit"], logging_steps=1 if profile else 10,
        dataloader_num_workers=4, dataloader_drop_last=True, remove_unused_columns=False,
        ddp_find_unused_parameters=False, group_by_modality_length=False,
        seed=c["seed"], data_seed=c["seed"], report_to="none",
        qat_enable=method != "sft", qat_bits=c["qat_bits"], qat_group_size=c["qat_group_size"],
        qat_scale_lr=c["qat_scale_lr"], qat_save_quantized=True,
        grace_enable=method == "grace", dkd_temperature=c["dkd_temperature"],
        dkd_alpha=c["dkd_alpha"], dkd_beta=c["dkd_beta"],
        dkd_chunk_size=c["dkd_chunk_size"], dkd_checkpoint_chunks=c["dkd_checkpoint_chunks"],
        rcka_weight=c["rcka_weight"], rcka_layer=c["rcka_layer"],
        rcka_warmup_steps=c["rcka_warmup_steps"], dkd_warmup_steps=0,
        ib_tau=c["ib_tau"], ib_eta=c["ib_eta"], ib_beta_init=c["ib_beta_init"],
        ib_beta_min=c["ib_beta_min"], ib_beta_max=c["ib_beta_max"],
        profile_run=profile, profile_steps=env_int("PROFILE_STEPS", 3) if profile else 0,
        learning_diagnostics_steps=env_int("PROFILE_STEPS", 3) if profile and method == "grace" else 0,
        max_train_hours=c["max_train_hours"], peak_memory_gib=c["peak_memory_gib"], resume_run=resume,
    )
    if method == "grace":
        values["teacher_model_path"] = c["teacher_path"]
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
               "-m", "qwenvl.train.train_qwen_constrained"]
    for key, value in values.items():
        command.extend(["--" + key, str(value)])
    return command, output


def hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_hashes():
    paths = list((PROJECT / "qwenvl/train").glob("*.py")) + list((PROJECT / "qwenvl/data").glob("*.py"))
    paths += [Path(__file__), PROJECT / "scripts/zero2_lora_memory_safe.json"]
    return {str(path.relative_to(PROJECT)): hash_file(path) for path in sorted(paths)}


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"Another process holds {path}; wait for it to finish")
        yield


def pin_recipe(path, recipe):
    if path.exists():
        previous = json.loads(path.read_text())
        changed = sorted(key for key in previous.keys() | recipe.keys() if previous.get(key) != recipe.get(key))
        if changed:
            raise ValueError(f"Recipe mismatch in {path}: {changed}. Use a new EXPERIMENT_DIR for a new comparison.")
    else:
        path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")


def check_gpus():
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 2 or len(set(devices)) != 2 or any(not device.isdigit() for device in devices):
        raise ValueError("Set CUDA_VISIBLE_DEVICES to two available physical GPU IDs, e.g. 0,1")
    result = subprocess.check_output([
        "nvidia-smi", "--id=" + ",".join(devices), "--query-gpu=name,memory.free",
        "--format=csv,noheader,nounits"], text=True)
    for line in result.strip().splitlines():
        name, free = line.rsplit(",", 1)
        if "A100" not in name or float(free) < 30 * 1024:
            raise ValueError(f"Need two available A100s with >=36 GiB free each; observed {line}")
    print(result, flush=True)


def prepare_subset(c, manifest_path):
    """Validate/reuse a CPU subset; caller holds the preparation lock."""
    sys.path.insert(0, str(PROJECT))
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    # Preparation imports the processor only; it never constructs a model.
    from transformers import AutoProcessor
    from qwenvl.data.constrained_data import prepare_manifest
    from qwenvl.data.data_processor import update_processor_pixels
    from qwenvl.train.argument import DataArguments
    import torch
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    processor = AutoProcessor.from_pretrained(c["model_path"], local_files_only=True)
    processor.tokenizer.model_max_length = c["model_max_length"]
    update_processor_pixels(processor, DataArguments(max_pixels=c["max_pixels"], min_pixels=c["min_pixels"]))
    manifest = prepare_manifest(c["annotation"], c["data_root"], processor, c["model_path"],
                                c["subset_size"], c["subset_seed"], c["model_max_length"],
                                c["max_pixels"], c["min_pixels"], manifest_path)
    return manifest


def comparison_recipe(c, manifest):
    import torch, transformers
    return dict(config=c, subset_sha256=manifest["records_sha256"],
                implementation=implementation_hashes(),
                environment=dict(torch=torch.__version__, transformers=transformers.__version__))


def launch_run(method, c, experiment, recipe, profile=False, resume=False, record_invocations=False):
    """Launch the existing trainer with one lock, recipe and native-resume policy."""
    command, output = build_command(method, c, experiment, profile, resume)
    with lock(output / ".run.lock"):
        if (output / "EXPORT_COMPLETE").exists():
            raise ValueError(f"Completed output already exists: {output}; use a new EXPERIMENT_DIR")
        checkpoints = list(output.glob("checkpoint-*"))
        if bool(checkpoints) != resume:
            raise ValueError("Existing checkpoints require --resume; --resume requires an existing checkpoint")
        run_recipe = dict(comparison=recipe, method=method, profile=profile,
                          profile_steps=env_int("PROFILE_STEPS", 3) if profile else 0)
        pin_recipe(output / "launch_recipe.json", run_recipe)
        if resume:
            latest = max(checkpoints, key=lambda path: int(path.name.split("-")[-1]))
            if not (latest / "launch_recipe.json").exists():
                raise ValueError(f"Missing checkpoint recipe: {latest}")
            pin_recipe(latest / "launch_recipe.json", run_recipe)
            if not (latest / "trainer_state.json").exists():
                raise ValueError(f"Incomplete checkpoint: {latest}")
            for filename in ("rng_state_0.json", "rng_state_1.json", "scheduler_state.json"):
                if not (latest / filename).exists():
                    raise ValueError(f"Missing resumable state: {latest / filename}")
            if method == "grace" and not (latest / "grace_ib_state.json").exists():
                raise ValueError(f"Missing IB controller state: {latest}")
        os.environ["GRACE_LAUNCH_RECIPE"] = str(output / "launch_recipe.json")
        if not record_invocations:
            subprocess.run(command, cwd=PROJECT, check=True)
            return
        started = time.time()
        status = None
        try:
            result = subprocess.run(command, cwd=PROJECT)
            status = result.returncode
            result.check_returncode()
        finally:
            invocation = dict(started_unix=started, wall_seconds=time.time() - started,
                              exit_code=status, resume=resume, gpu_count=2)
            summary_path = output / "run_summary.json"
            if summary_path.exists() and summary_path.stat().st_mtime >= started:
                invocation["train_runtime_seconds"] = json.loads(summary_path.read_text())["metrics"]["train_runtime"]
            with (output / "invocations.jsonl").open("a") as handle:
                handle.write(json.dumps(invocation) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("sft", "qat", "grace"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print resolved recipe/command; no writes or GPU access")
    mode.add_argument("--prepare-only", action="store_true", help="Build/reuse the shared subset on CPU")
    mode.add_argument("--profile", action="store_true", help="Stress longest retained examples; separate checkpoints/output")
    parser.add_argument("--resume", action="store_true", help="Restore native optimizer/model/RNG and IB state")
    args = parser.parse_args()
    c = resolved_config()
    if not 0 <= c["lora_dropout"] < 1 or not 0 < c["min_pixels"] <= c["max_pixels"]:
        raise ValueError("Invalid dropout or pixel range")
    if c["max_train_hours"] <= 0 or c["peak_memory_gib"] <= 0:
        raise ValueError("Time and memory budgets must be positive")
    experiment = Path(os.environ.get("EXPERIMENT_DIR", ROOT / "experiments/constrained_32k_seed42")).resolve()
    command, output = build_command(args.method, c, experiment, args.profile, args.resume)
    print(json.dumps(c, indent=2), flush=True)
    print("Command: " + shlex.join(command), flush=True)
    if args.dry_run:
        return
    for path in (c["model_path"], c["annotation"], c["data_root"], c["deepspeed"]):
        if not Path(path).exists():
            raise ValueError(f"Required path missing: {path}")
    if args.method == "grace" and not args.prepare_only and not Path(c["teacher_path"]).exists():
        raise ValueError(f"Teacher missing: {c['teacher_path']}")
    with lock(experiment / ".prepare.lock"):
        manifest = prepare_subset(c, experiment / "subset.json")
        recipe = comparison_recipe(c, manifest)
        pin_recipe(experiment / "comparison_recipe.json", recipe)
    if args.prepare_only:
        print(f"Prepared {len(manifest['records'])} examples: {experiment / 'subset.json'}", flush=True)
        return
    check_gpus()
    launch_run(args.method, c, experiment, recipe, args.profile, args.resume)



if __name__ == "__main__":
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"ERROR: {error}")
