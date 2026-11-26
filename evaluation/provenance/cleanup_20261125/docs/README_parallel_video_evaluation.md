# Parallel QAT video evaluations

The QAT recovery launcher `run_remaining_qat_videos_gpu1.sh` now uses
`runners/run_parallel_video_evaluation.py`. It defaults to physical GPU 1 and
`--parallel-evaluation 1`. This feature applies to the QAT video recovery
experiment, including its streamed vision/prefill execution. Other historical
matrix launchers retain their existing behavior. Original pinned Python sources
and existing completion markers are unchanged.

## Interruption behavior

The previous recovery controller skips verified `COMPLETE.json` pairs but
restarts an interrupted pair at question 1. Its telemetry alone is not a saved
harness response stream. Existing in-flight attempts therefore cannot acquire
question-level resume retroactively.

New-controller attempts persist each completed answer, its request identity,
telemetry and video sampling in an fsynced, checksummed answer checkpoint.
Rerunning replays the longest compatible prefix before decoding videos, then
continues inference. The full pinned harness scores every selected question
again and validates coverage before completion. Replay restores telemetry indices
and profiling position. A torn final append is discarded; corrupt complete
records or changed request/provenance identities fail explicitly. At most the
current uncommitted question is repeated. Earlier inference/preprocessing phases
within that question are not checkpointed. Old attempts and artifacts remain
intact. Changing the selected suite list or GPU count does not invalidate an
otherwise compatible answer prefix.

## Run multiple evaluations per GPU

From the workspace root:

```bash
EXP=reproduction/GRACE/experiments/qat_int4_token_reduction_576_seed42
RUN="$EXP/run_remaining_qat_videos_gpu1.sh"
bash "$RUN" plan --parallel-evaluation 3 --suites videomme
bash "$RUN" check --parallel-evaluation 3 --suites videomme
# After the existing controller has released the output/GPU locks:
bash "$RUN" run --parallel-evaluation 3 --suites videomme
```

The maximum is **per GPU**. Evaluations are separate configuration/suite jobs;
this does not split one VideoMME evaluation across GPUs. `--gpus 1,2` enables
both physical GPUs, with the same per-GPU maximum. Use a distinct smoke
namespace for a GPU test:

```bash
bash "$RUN" run --gpu 1 --parallel-evaluation 3 --suites videomme \
  --configurations uniform_r050,uniform_r025,uniform_r010 --smoke 2
```

An output and GPU each retain one exclusive controller lock. A second controller
cannot join the same output directory. For the currently running legacy
controller, let the in-flight pair finish before stopping it and switching;
terminating that pair loses its partial progress. A queued legacy controller
will continue into its next pair unless explicitly stopped. The feature does
not stop/restart any existing process automatically.

## Memory admission and holds

With the default `--parallel-evaluation 1`, each GPU runs one evaluation at a
time. Its scheduling budget is the GPU's total memory minus
`--memory-headroom-gib` (default 2 GiB): about 38 GiB on a 40 GiB GPU.
`--evaluation-memory-gib` and historical peak estimates are ignored in this mode.
The plan reports `memory_policy: exclusive_gpu`; the actual budget is resolved
from the GPU at dispatch and recorded in `job.json` and the controller log.
For a 32 GiB scheduling budget on a 40 GiB card, use `--memory-headroom-gib 8`.

Budgets are scheduling estimates, not PyTorch/CUDA allocation limits. The sole
worker can use the available GPU memory for vision, decoder prefill, and the full
KV cache. Prefill executes on GPU; CPU staging and the existing chunk size and
adaptive token cap retain the declared evaluation protocol. The preparation
holder releases before the worker allocates its model on GPU.

With `--parallel-evaluation` greater than 1, a worker receives a conservative peak
budget, rather than assuming its current GPU usage is its maximum. A fully
completed matching suite/configuration provides
a peak estimate with 25% padding plus 1 GiB. Unknown configurations default to
`--evaluation-memory-gib 18`; partial observed peaks are lower bounds. The
controller additionally leaves `--memory-headroom-gib 2` free. Thus requesting
three unknown workers on a 40 GiB GPU normally admits two. Lower the fallback
budget only after representative profiling; for example
`--evaluation-memory-gib 10` allows three when sufficient memory remains, but is
not a guarantee that long clips fit.

Scheduler source hashes are pinned per output. After this scheduler update, use
a fresh `--output` base for new evaluations; retain existing output protocols and
completion records as they are. The matched-video launcher also inherits this
sequential memory policy.

The recorded VideoMME **baseline** peaked at 16.17 GiB allocated. The completed
50% resolution run peaked at 9.15 GiB. The 50% uniform run's observed peak was
6.09 GiB over its first 351 questions on November 9, 2026; that partial sample
does not establish its full-population peak. CUDA context/allocator reservations
also consume memory beyond allocated tensor peaks.

The scheduler accounts for each active worker's unused peak budget, checks
actual free memory and rejects unexpected GPU processes. Pending jobs that fit
may pass larger pending jobs. Admissions pause when memory is insufficient;
pressure lowers the GPU's concurrency ceiling. CUDA OOM lowers that ceiling
again and requeues the failed job using its answer checkpoint. A serial OOM
remains a failure. No spatial/fps/token policy is reduced by the scheduler.
Memory can still spike between polls, so admission is best effort.

There is **one holder per GPU**, held during initial CPU setup/loading. The
controller releases it at the first worker's CUDA handoff. Sibling workers join
under memory admission without requiring the GPU to be idle. The full-card holder
is reacquired only when **all** sibling workers have exited and work remains.
This prevents the old between-job hold from consuming a sibling's memory.
Locks remain owned while each GPU is in the pool; holders release on controller
pipe EOF. Owned workers terminate on normal controller interruption and detect
controller death. Foreign processes are never killed by this controller.

Warnings include GPU, requested/admitted concurrency, budgets, headroom and OOM
backoff. Controller messages are saved in `full/parallel_evaluation.log`; worker
logs stay in each attempt's `run.log`. Holder messages also appear on stdout.
`job.json` and completion markers record scheduling provenance. Shared-GPU or
replayed timing samples are not standalone latency measurements; the extension's
comparison report suppresses speedup claims. Use the original serial protocol
in an isolated run for publishable standalone latency comparisons.

## Add a GPU while evaluations run

Create a pool file listing only the GPUs authorized for this controller:

```bash
POOL="$EXP/evaluation_gpu_pool.json"
printf '[1]\n' > "$POOL"
bash "$RUN" run --gpu-pool-file "$POOL" --parallel-evaluation 3
```

When physical GPU 2 becomes free, update the same file from another terminal:

```bash
printf '[1,2]\n' > "${POOL}.tmp"
mv "${POOL}.tmp" "$POOL"
```

The controller locks the added GPU, checks it is idle, runs its numerical gates,
and dispatches pending jobs there. Existing workers continue. Busy/locked GPUs
produce a warning and are retried about every 30 seconds. Removing an index
stops new dispatch to it, lets its current workers finish, then releases its
holder and GPU lock. Invalid/missing
pool files retain the previous valid pool with a warning. There is no automatic
claim of unspecified GPUs. A pool update has no effect after all jobs have
finished and the controller has exited.

## Validation

```bash
python3 -m unittest discover -s reproduction/GRACE/evaluation/tests \
  -p test_parallel_evaluation.py -v
/home/air/miniconda3/envs/grace/bin/python \
  reproduction/GRACE/evaluation/tests/cpu_parallel_video_integration.py
```

The unit tests simulate GPU processes without CUDA. The integration exercises
the real pinned VideoMME harness/local annotations with dummy answers, interrupts
after one saved answer, then verifies replay, inference of only the remaining
answer, full rescoring, and ordered telemetry. GPU concurrency/throughput requires
a subsequent smoke run on available cards.


## Failure-only phone alerts

`scripts/watch_eval_health.py` polls a specific controller and its latest worker
attempts every 60 seconds without changing the evaluation or GPU allocations.
For the November 9 two-GPU run, the controller PID was `3175763`:

```bash
python3 scripts/watch_eval_health.py --controller-pid 3175763 \
  --topic fyp-process-health --tmux-session fyp-process-health
```

Run on the host so `/proc` and tmux expose the real processes. Subscribe to
`fyp-process-health` on `ntfy.sh` in the phone app. No startup, healthy, completion,
or memory-admission messages are published. Explicit failed exits notify on the
next poll; a disappeared worker/controller must remain missing for 120 seconds
before alerting. CUDA OOMs notify even if the controller automatically retries.
Each alert includes suite/configuration, GPU, attempt, return code/signal when
available, terminating exception text, saved question count, and log path.
Unknown kill/OOM causes are reported as unconfirmed rather than guessed.
Confirmed deliveries are persisted to avoid repeated alerts across watcher
restarts; failed HTTP requests retry. A network timeout after server acceptance
can still produce a duplicate on retry. Historical controllers are ignored;
new attempts/retries from the selected controller are enrolled automatically.
A new controller PID needs a new watcher invocation. Slow questions alone do
not trigger failure alerts.

State and logs are in `results_video_retry_gpu1/full/health_monitor/`, with filenames
containing the controller PID. The watcher exits quietly when its controller
has ended and all observed attempts are resolved/notified. For example:

```bash
tail -f "$EXP/results_video_retry_gpu1/full/health_monitor/controller_3175763.log"
tmux attach -t fyp-process-health
# Stop only the watcher; evaluations continue:
tmux kill-session -t fyp-process-health
```

The watcher honors `NTFY_TOKEN` if authentication is needed. `--once --dry-run`
checks without publishing. Tests use a fake process tree and mocked HTTP calls;
no test notification is sent to the phone.
