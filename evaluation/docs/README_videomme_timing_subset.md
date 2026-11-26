# VideoMME timing subset

The completed seed-42 sweep has all ten configurations. Read [the current experiment report](../../experiments/qat_int4_token_reduction_576_seed42/REPORT.md) for full accuracy, serial projections and the comparison with the newest matched resolution reruns. [The command reference](../../experiments/README.md) covers the unified launcher.

From `reproduction/GRACE`, prepare a fresh output and run every timing configuration on the same idle A100:

```bash
bash evaluate.sh prepare --workflow timing
bash evaluate.sh run --workflow timing --gpus 0
bash evaluate.sh summarize --workflow timing
```

The launcher writes to `experiments/qat_int4_token_reduction_576_seed42/outputs/reruns/reproduction_v2/timing/`; select another rerun with `--output experiments/qat_int4_token_reduction_576_seed42/outputs/reruns/<name>`. All ten settings run serially; a full-input first-generation warmup is excluded for each. The first selected card is pinned to that output. Use the same GPU on resume. Foreign GPU processes or monitoring errors reject the measurement. Interrupted configurations restart measurement without replaying saved answers; verified completed configurations are skipped.

To measure a selected phase:

```bash
bash evaluate.sh run --workflow timing --gpus 0 \
  --configurations resolution_r050,resolution_r025,resolution_r010
# Explicitly run the remaining configurations when ready:
bash evaluate.sh run --workflow timing --gpus 0 \
  --configurations baseline_r100,uniform_r050,uniform_r025,uniform_r010,merge_r050,merge_r025,merge_r010
```

The original resolution-first wrapper and screening gate are archived under `experiments/archive/cleanup_20261125/`. Its historical projection check predates the two matched full reruns and compared against original-policy r25/r10 workloads; it is not current validation of ±10% prediction error. The latest audit measures **+15.06%/+16.70%** fresh model-time error at resolution r25/r10. Same-run backtesting on those full reruns is within 1%, with most fresh discrepancy in vision/embedding. The consolidated report retains this distinction rather than treating the old phase gate as a present accuracy guarantee.

## Sampling and timing

The sample contains 25 short and 25 medium questions from **50 distinct videos**. Within each duration, the 300 videos are sorted by original visual-token count into five workload quintiles. Five videos are sampled per quintile, then one of the video's three questions. Every question has inclusion probability 1/36; seed 42 and the exact manifest are fixed independently of accuracy/timing. No accuracy extrapolation is performed.

All configurations use batch one, greedy EOS generation, 2 fps, full duration, streamed vision by temporal group, 512-token prefill and a common adaptive baseline with a 131,072-token operational cap. Fixed-32-token profiling and answer replay are disabled. GPU occupancy is sampled every five seconds; monitoring cannot rule out all transient or CPU/I/O contention.

Each observed model/request time is expanded by 36. Model time covers synchronized vision/embedding, reduction, prefill and decode. Request time adds media preprocessing, input staging and measured generation-call overhead. Startup, warmup, harness setup/scoring, durable writes and unmeasured filesystem overhead are excluded. These are steady-state serial estimates rather than observed full-run wall time.

Paired speedup intervals use 4,000 within-stratum video bootstrap draws. The same resampled videos are used in both sides of a ratio; no finite-population correction is applied. Five videos per stratum provide approximate sampling intervals. They do not quantify card/load/run variation, which exceeded the intervals for resolution r25/r10.

## Artifacts

| File | Contents |
| --- | --- |
| `subset.json` | Exact IDs, workload strata and expansion weights |
| `selection_diagnostics.json` | Post-selection check against the historical workload |
| `protocol.json`, `reference/` | Pinned computation, model/data/source hashes and CPU preflight |
| `hardware.json`, `status.json` | Selected GPU and controller state |
| `extrapolation.md/json` | Timing estimates and sampling intervals |
| `<configuration>/attempt_*/` | Job, predictions, telemetry, timings, warmup and GPU observations |

The completed reference is under [selected timing artifacts](../../experiments/qat_int4_token_reduction_576_seed42/outputs/latest/videomme_timing50_seed42/). Rebuild the frozen matrix/timing audit with `bash evaluate.sh summarize`; this reads original artifacts and does not launch inference. Source changes require a fresh output; never rewrite old hashes to resume another computation.
