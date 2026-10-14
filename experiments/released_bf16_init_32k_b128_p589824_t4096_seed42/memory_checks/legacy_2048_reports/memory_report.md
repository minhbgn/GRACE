# 576-token memory feasibility — 2026-10-05

**All four bounded checks passed, including native checkpoint saving and merged/baked export. Keep the requested 589,824-pixel / 576-token ceiling; no lower-cap fallback was needed. Full training and evaluation have not started.**

Two free NVIDIA A100-SXM4-40GB GPUs (PCI indices 0 and 4) were surveyed before each run. No other workloads were stopped. Source weights, processor compatibility, implementation hashes, launch recipes and GPU surveys are recorded per check. Machine-readable results: [memory_results.json](memory_results.json).

| Arm | Peak allocated GiB/GPU | Peak reserved GiB/GPU | Highest sampled device use GiB | Updates / export | Evidence |
| --- | ---: | ---: | ---: | --- | --- |
| GRACE W4 | 24.93 | 25.56 | 26.62 | 3 / complete | [run](memory_checks/grace_w4_20261005T072630065075Z/profiles/grace/run_summary.json) |
| QAT W4 | 8.01 | 9.09 | 10.15 | 3 / complete | [run](memory_checks/qat_w4_20261005T073903145713Z/profiles/qat/run_summary.json) |
| GRACE W8 | 24.93 | 25.56 | 26.62 | 3 / complete | [run](memory_checks/grace_w8_20261005T073441840651Z/profiles/grace/run_summary.json) |
| QAT W8 | 8.01 | 9.09 | 10.15 | 3 / complete | [run](memory_checks/qat_w8_20261005T074338504949Z/profiles/qat/run_summary.json) |

Training allocated/reserved values are the maximum across ranks at optimizer-step boundaries (using PyTorch peak counters). Device use is sampled every 0.5 seconds across loading, training, checkpointing and export and includes runtime overhead; a sampled trace can miss brief spikes. The production reserved-memory guard remains **34 GiB**. It checks at optimizer-update boundaries and cannot prevent every transient OOM.

## What was exercised

- A real square image forced to a 768×768 processed grid: `[1,48,48]` patches, exactly **576 merged visual tokens**.
- Exactly **2,048 total tokens**, including **1,458 supervised answer tokens** from a synthetic repeated-word answer. This stresses the long-answer distillation path; losses and accuracy on this artificial example have no research interpretation.
- The actual trainer, released SFT initialization, QAT modules, LoRA rank/targets, merger, optimizer, ZeRO-2, gradient checkpointing, BF16 storage and W4/W8 settings.
- For GRACE: the actual 8B teacher, GDKD and **full-weight RCKA at updates 2–3**.
- Three optimizer updates, native checkpoints, LoRA merge, quantization bake, model/processor save and quantized research export.

## Limits and production handoff

To bound diagnostic cost, this screen used accumulation **1** (global batch 2), RCKA warmup **1** and a forced minimum image area equal to the ceiling. Production retains accumulation **64** (global batch 128), warmup **25**, minimum **4,096**, and **256** updates. The eight repeated synthetic records are isolated under `memory_checks/`; they are never the production subset.

A fixed-envelope screen is evidence of memory feasibility, not an exhaustive test of every aspect ratio, retained conversation, accumulation cycle or later optimization state. For additional assurance, the documented `run_training.sh all --profile` uses the actual freshly prepared data and production accumulation. Training is protected by the existing pause/resume guard; do not change image budgets midway through a pinned experiment.

No production throughput or full-run duration is inferred from these short, checkpoint-heavy runs. The recorded two-GPU launch wall time across all four checks is **0.488 GPU-hours** (including model loading/checkpoint/export within each launcher invocation; excluding CPU provenance/input preparation and idle time between invocations). This is diagnostic cost, not the cost of training the four production models.

## Input distribution and non-GPU validation

The [256-image CPU preview](token_budget_preview.json) increased mean/median visual counts from **89.76/88** to **250.84/260**, with maximum **576**. Small images remain below the ceiling. Measure the new full subset separately once prepared.

All four production dry-run commands retain batch 128, 256 updates, 2,048 total tokens and 25-step RCKA warmup, and point to fresh production outputs. The latest evaluation plan covers four models × seven suites. Shell/Python syntax checks passed. The 11-test local scoring/worker suite passed; after strengthening histogram-tampering coverage, all 10 scoring tests passed again. The MMBench fix changes JSON-key comparison only, not scoring semantics or old completion artifacts.

See [README.md](README.md) for preparation, training, resume and evaluation commands.
