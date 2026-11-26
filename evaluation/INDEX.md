# Evaluation

Start with [the experiment command reference](../experiments/README.md), [the consolidated report](../experiments/REPORT.md), or [the evaluation tool index](docs/README.md).

From `reproduction/GRACE`: `bash evaluate.sh plan`, `check`, `summarize`, or `run --gpus 0,2`. All twelve completed QAT W4 suites share this entry point. Curated metrics live under [reports/](../reports/); raw predictions live under the experiment’s [outputs/](../experiments/qat_int4_token_reduction_576_seed42/outputs/); historical protocols remain documented separately.
