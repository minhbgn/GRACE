# Resilient video evaluation fix — 2026-10-23

The experiment's normal launch script now runs
`evaluation/runners/run_video_matrix_resilient.py` and
`evaluation/workers/video_worker_resilient.py`. The new CPU-only
`evaluation/workers/video_runtime.py` supervises workers and restores progress.
The five v2 evaluator source hashes are unchanged, preserving the code used by
the existing SFT job on GPU 2. New launches write to a v3 namespace.

## Changes

- The allocator defaults to `expandable_segments:True`, is set before importing
  Torch, and is recorded in the pinned protocol. Input sampling, resolution,
  generation allowance, scoring and checkpoint weights are unchanged.
- Predictions are synced after every question. Retries and reruns seed a new
  attempt with a validated ordered prefix. An interrupted final JSON write is
  discarded; complete corrupt records are rejected. Source/run identity,
  scoring, input identity, context and EOS checks remain mandatory. Old v2
  partial outputs are not imported into the new code/protocol namespace.
- The parent saves `worker.json` and `exit.json`, including PID, signal/return
  code, allocator and before/after host/cgroup memory records. Worker stage
  markers distinguish imports, video processing, weight loading, CUDA
  transfers, generation and durable prediction saves.
- One SIGKILL retry is allowed by default. A repeated kill stops that matrix;
  it no longer blindly starts every remaining checkpoint in the same failure
  condition. Other fatal signals also stop the matrix.
- Per-GPU locks coordinate these runners; occupied GPUs are rejected before
  each worker. Existing jobs are never killed by the runner. The parallel
  launcher now uses independent, resumable v3 group outputs rather than
  symlinks into the historical v2 tree.

SIGKILL remains an external termination that Python cannot intercept or block.
The historical sender is still unknown. These changes fix progress loss,
unrecorded exits, uncontrolled failure cascades and the reproduced allocator
OOM; they do not claim to prevent a kernel or another process from killing a
worker. The per-GPU lock is advisory and does not reserve hardware against
unrelated programs.

Inference still uses the merged BF16 QAT/GRACE research exports. Packed INT4/INT8
inference is a separate backend change and was not introduced here.

## Validation

All six new runtime tests passed. They use actual subprocess SIGKILLs to verify
mid-write recovery without duplicate/missing questions, bounded retries, and
rejection of changed/corrupt resume data. They also check busy-GPU rejection
and locking across output directories. All six existing video-policy,
processor/token-accounting and integrity tests passed. Shell launchers parse.

Real-worker validation ran on initially free GPU 0:

| Checkpoint | TempCompass questions | Result |
| --- | ---: | --- |
| QAT W4 | 3 | Complete, EOS |
| GRACE W4 | 3 | Deliberately SIGKILLed after question 1; automatic retry resumed at question 2; complete, EOS |
| QAT W8 | 3 | Complete, EOS |
| GRACE W8 | 3 | Complete, EOS |

The kill targeted only a child created by the validation script, after checking
its PID's command line against that test's job path. Completion checksums and
population coverage were validated for all four checkpoints.

The new supervised worker also passed the exact QAT W4 VideoMME `618-1`
regression: **151,218 identical input tokens**, answer `C` + EOS, peak allocated
**33.3223 GiB**. Its input-ID hash matches the original failed question. This
was a one-question regression, not a full VideoMME accuracy run.

Evidence is in `resilient_runtime_validation_20261023/validation.json`, its
`smoke/smoke_3/*/tempcompass/` attempts, and `long_case/validation.json`. The
source hashes used by the live smoke tests match the final new source files.
`validate_video_runtime_20261023.py` and
`validate_video_long_case_20261023.py` reproduce these checks with new outputs.

An additional layout suite passed six of seven tests. Its existing documentation
link check treats `README_video_evaluation.md#commands` as a filesystem filename,
so it fails on that valid anchor link. No runtime or data-path check failed.

## Run

From the workspace root, choosing an idle PCI GPU:

```bash
EXP=reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42
bash "$EXP/run_video_evaluation.sh" run --gpu 0 \
  --methods qat_w4,grace_w4,qat_w8,grace_w8
```

Rerun the same command to resume this v3 run. Add
`--suites tempcompass --smoke 3` for a small startup check. Use tmux for a full
evaluation that must survive SSH disconnection. `--sigkill-retries 0` disables
automatic retries. Changed source/configuration requires a fresh `--output`.

No full benchmark matrix was started as part of this fix. Historical results,
checkpoint weights and the active legacy SFT inference worker were left intact.
