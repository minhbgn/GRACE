# QAT W4 visual token reduction, seed 42

**Complete: 120/120 configurations across twelve suites**, plus ten VideoMME timing configurations. This is inference-only reduction on the frozen 576-cap CE-only LoRA-QAT checkpoint, using packed Triton W4A16. No reducer or checkpoint weights were trained in this sweep.

Read [REPORT.md](REPORT.md) for findings, [the shared command reference](../README.md) for suite IDs, environment pins and controls, and [the combined report](../REPORT.md) for the training history.

## Layout

```text
README.md / REPORT.md           Current documentation
experiment.json                Frozen experiment design
run_experiment.sh              Unified launcher wrapper
reports/
  metrics/                     Current audited JSON, CSV and tables
  figures/                     Accuracy/runtime plots, PNG and PDF
  history/documents/           Original reports, readmes and recovery note
  history/artifacts/           Earlier consolidated analyses and tables
  provenance/                  Validation evidence
outputs/                       Ignored by Git
  original/                    Original full and smoke matrix
  reserved_gpu02_smoke/         Separate early smoke runs
  video_recovery_gpu1/          First streamed/adaptive recovery
  video_recovery_scheduler/     Completed scheduler/resume recovery
  video_matched*/              Matched attempts; exclusive run supplies latest r25/r10
  videomme_timing50_seed42_*/   Initial, draft and completed phased timing runs
  latest/                      Links to the selected raw artifacts
  reruns/                      New evaluations, isolated from recorded runs
  cache/ / diagnostics/        Runtime caches, fragments and auxiliary evidence
```

Raw predictions, attempts, logs, fragments and caches stay in `outputs/`. Curated report artifacts stay in `reports/` and can be versioned. Only README and REPORT remain as Markdown files at this experiment’s surface. Earlier Markdown documents are preserved under [reports/history/documents/](reports/history/documents/).

## Evaluate and verify

Run from `reproduction/GRACE`; the default rerun is this experiment’s `outputs/reruns/reproduction_v2/`:

```bash
bash evaluate.sh plan
bash evaluate.sh check
bash evaluate.sh verify
bash evaluate.sh run --gpus 0,2 --smoke 1
bash evaluate.sh run --gpus 0,2
bash evaluate.sh compare

# Rebuild the saved metric/timing summaries without inference:
bash evaluate.sh summarize

# Reproduce the fixed 50-question VideoMME timing sweep:
bash evaluate.sh prepare --workflow timing
bash evaluate.sh run --workflow timing --gpus 0
bash evaluate.sh summarize --workflow timing
```

GPU indices are examples; select available cards. `run_experiment.sh` forwards the same arguments to the unified launcher. Fresh runs record current source hashes; use a new run name if code or protocol changes. Historical run records retain their original bytes, paths and hashes, resolved through the explicit relocation map without old surface aliases.

## Results

| Evidence | Location |
| --- | --- |
| Current experiment metrics and tables | [reports/metrics/](reports/metrics/) |
| Plots | [reports/figures/](reports/figures/) |
| Selected predictions, telemetry and markers | [outputs/latest/](outputs/latest/) |
| Explicit completed-run selection | [result_catalog.json](../../evaluation/protocols/result_catalog.json) |
| All-experiment merged results | [JSON](../../reports/merged_results.json), [CSV](../../reports/merged_results.csv), [score table](../../reports/SUMMARY.md) |
| Preserved report history | [Documents](reports/history/documents/), [earlier analyses](reports/history/artifacts/) |
| Relocation paths and source snapshots | [Output layout provenance](../../evaluation/provenance/output_layout_20261125/) |

VideoMME r25/r10 use the latest matched exclusive reruns. Three Video-MMMU resolution-r10 rows retain the original policy and have no paired baseline delta/speedup claim. Vanilla/BF16 matrices remain incomplete. See REPORT before quoting timing projections or speedups.
