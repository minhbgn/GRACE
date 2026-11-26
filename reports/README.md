# Consolidated reports and metrics

[Score table](SUMMARY.md) · [Merged JSON](merged_results.json) · [Merged CSV](merged_results.csv) · [Experiment synthesis](../experiments/REPORT.md) · [Evaluation commands](../experiments/README.md)

The JSON contains 120 artifact-verified full configurations, ten timing configurations and 215 historical report-only scores. The CSV combines score rows with an `evidence` column. Historical predictions absent from this checkout remain labelled report-only.

The token-reduction experiment’s [reports/](../experiments/qat_int4_token_reduction_576_seed42/reports/) contains current metrics, plots and report history. Its [outputs/](../experiments/qat_int4_token_reduction_576_seed42/outputs/) contains named raw runs, predictions, attempts, fragments, diagnostics, caches and new reruns; all are ignored by Git. Selected raw artifacts are linked under its `outputs/latest/`.

`sources/` contains the other experiments’ immutable report documents and the repository-wide [checksum index](sources/index.json). The token-reduction source documents are under its `reports/history/documents/`. `provenance/migration_20261125.json` preserves the first migration’s checksums; `evaluation/provenance/output_layout_20261125/` records the current paths and pre-relocation source snapshots.

From the GRACE root, `bash evaluate.sh summarize` regenerates curated metrics here and in the experiment’s `reports/metrics/`. `bash evaluate.sh verify` additionally rehashes all 3,220 relocated files. Fresh inference goes under the experiment’s `outputs/reruns/`; `bash evaluate.sh compare` checks it against the frozen scores. Saved protocols, prediction bytes and completion markers are never rewritten to accommodate a path move.
