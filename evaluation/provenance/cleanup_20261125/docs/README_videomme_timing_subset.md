# VideoMME timing subset

For new runs, use the **resolution-first workflow** below. The runner changed on
2026-11-23 to add phase selection; historical output directories retain their old
source hashes and must not be resumed with the new source. The new workflow uses
`results_videomme_timing50_seed42_phased_20261123/`.

Inside a tmux terminal, from the workspace root:

```bash
PHASE=reproduction/GRACE/experiments/qat_int4_token_reduction_576_seed42/run_videomme_timing_phases.sh
bash "$PHASE" prepare && bash "$PHASE" resolution && bash "$PHASE" check
```

This performs CPU preparation, measures only `resolution_r050/r025/r010` on GPU 4,
then writes `projection_check.md/json`. It stops there. No baseline, uniform or
merge evaluation is dispatched. `TIMING_GPUS=2 bash "$PHASE" resolution` can use
GPU 2 before hardware is pinned; all subsequent phases must use that same GPU.

The CPU projection check backtests this exact subset inside each historical
1,800-question resolution run. Pragmatic screening thresholds are ≤10% absolute
relative model-time error overall, ≤15% in each duration group, and ≤20%
historical bootstrap relative half-width. It also compares fresh projected model
time to the serial full resolution-50% reference, with matching input/execution
checks. Old resolution-25%/10% inputs/runtime differ, so those fresh-versus-old
comparisons are explicitly marked non-comparable. Historical backtests for them
still isolate the fixed sample's error within their respective recorded runs.
No request-wall-time calibration is claimed from model-time references.

Read `projection_check.md` and `extrapolation.md` in the new output directory.
Only after review, explicitly start the next phase:

```bash
bash "$PHASE" remaining
```

`remaining` regenerates the checks and refuses to launch if screening fails or
resolution runs are incomplete. When accepted, it runs the baseline and six
uniform/merge configurations. It never changes the sample. The three resolution
measurements stay available for the final paired report. All phases skip verified
completions and restart failed attempts without replaying answers.

If screening fails, first inspect the per-duration errors, interval width and
runtime/input mismatches. Do not repeatedly try random seeds until one passes.
An explicitly revised sample can be prepared with `TIMING_SUBSET_SEED=43` and a
fresh `TIMING_OUTPUT`; keep those overrides identical for every phase. Prefer a
documented coverage improvement or a larger sample if 50 questions are too noisy.

The remaining sections describe the common measurement protocol and the
historical all-configurations launcher.

This experiment estimates serial runtime for the existing **1,800 short/medium
questions** using 50 new measurements per configuration. Full accuracy results
are kept in their existing files. Long videos are outside this population.

The seed-42 sample contains 25 short and 25 medium questions from **50 distinct
videos**. Within each duration, the 300 videos are sorted by original visual
token count and partitioned into five equal workload quintiles. Five videos are
sampled uniformly without replacement from each quintile, then one of each
video's three questions is sampled uniformly. Every question has inclusion
probability 1/36. Selection never uses accuracy or measured time.

All ten QAT W4 configurations use the exact same questions and one exclusive
A100, sequentially, with the TempCompass recovery execution settings: 2 fps,
full duration, shared adaptive baseline capped at 131,072 decoder tokens,
streamed vision, 512-token prefill, batch one, greedy generation to EOS. One full
first-input warmup per configuration is excluded. Fixed-32-token profiling and
answer replay are disabled. GPU occupancy is sampled every five seconds during
measurement; observed foreign processes or monitoring errors reject the attempt.
This cannot detect every transient occupant or remove shared CPU/I/O contention.

The report expands each observed runtime by 36 and sums it. Model time covers
vision, token reduction and decoding. Request time additionally covers video
preprocessing and input staging. Startup, warmup, harness setup/scoring and
filesystem context-manager overhead are excluded. These are **steady-state
serial estimates**, not exact full-process wall-clock predictions.

Paired baseline speedups and 95% intervals use 4,000 within-stratum video
bootstrap draws. The same resampled videos are used for both configurations in
a ratio. These are sampling-only intervals; they do not measure run-to-run
variation. No finite-population correction is applied. Five samples per stratum
give only an approximate interval, particularly for heavy tails.

The adaptive input policy also applies to the two resolution configurations
whose historical accuracy runs used the original input policy. The report
therefore does not claim those historical accuracy scores were measured under
the new timing policy.

From the workspace root:

```bash
EXP=reproduction/GRACE/experiments/qat_int4_token_reduction_576_seed42
bash "$EXP/run_videomme_timing_subset.sh" prepare
bash "$EXP/run_videomme_timing_subset.sh" queue
bash "$EXP/run_videomme_timing_subset.sh" summarize
```

`queue` starts the controller in the detached tmux session `videomme-timing` and
checks physical A100 indices 0,1,2,4 once per minute. Attach with
`tmux attach -t videomme-timing`; detach with Ctrl-b, then d. Closing the IDE/chat
does not stop the tmux session. Repeating `queue` while the session exists
reports the existing session. It waits for an idle GPU and its shared evaluator lock, without stopping
any job or allocating CUDA while waiting. The first available GPU runs the
entire matrix. Resume requires that same GPU. CPU checkpoint/environment checks
are repeated after the wait, followed by the existing CUDA numerical gates.

Results are under `results_videomme_timing50_seed42/` in the experiment directory:

| File | Contents |
| --- | --- |
| `subset.json` | Exact sample IDs, workload strata and expansion weights |
| `selection_diagnostics.json` | Post-selection comparison against historical baseline workload/timings; not new timing results |
| `protocol.json`, `reference/` | Frozen computation, source/checkpoint/data hashes and CPU preflight evidence |
| `status.json`, `queue.log`, `queue.pid` | Live queue/controller status |
| `extrapolation.md`, `extrapolation.json` | Estimates and sampling intervals, refreshed after each configuration |
| `<config>/attempt_*/` | Independent job, timings, telemetry, warmup, GPU observations and predictions |

Only fully validated, EOS-complete 50-question attempts enter estimates.
Completed configurations are skipped on resume. An interrupted configuration
starts a fresh measurement attempt; saved answers are never replayed into timing.
Source changes require a fresh output directory. The early
`results_videomme_timing50_seed42_preparation_draft/` contains CPU-only preparation
artifacts from implementation; it is not the queued experiment.

If `videomme-timing` still exists with a **dead pane**, `queue` will only report
that existing session. Inspect the failed attempt's `run.log` first. The
2026-11-16 attempt stopped after detecting other processes on GPU 4; its partial
timings were rejected. Restart a dead pane from the workspace root with:

```bash
tmux respawn-pane -t videomme-timing:0.0 \
  -c /raid/lvchengtao/Minh/Distillation \
  'exec bash reproduction/GRACE/experiments/qat_int4_token_reduction_576_seed42/run_videomme_timing_subset.sh run --wait --gpus 4'
tmux attach -t videomme-timing
```

The command intentionally omits `-k`, so tmux refuses to replace a live pane.
This output directory's `hardware.json` pins GPU 4; a different GPU requires a
fresh prepared output directory. All ten configurations are included by default,
including the baseline needed for speedup ratios. Preserve the occupancy guard:
shared-GPU measurements cannot establish isolated runtime.

Refresh the projected timings from any validated completions with:

```bash
bash reproduction/GRACE/experiments/qat_int4_token_reduction_576_seed42/run_videomme_timing_subset.sh summarize
```
