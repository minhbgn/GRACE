# Experiments and evaluation

The latest completed study is **QAT W4 with visual token reduction**: 120 full configurations across seven image and five video suites, plus a separate ten-configuration VideoMME timing sweep. Start with the [combined experiment report](REPORT.md), the [detailed current report](qat_int4_token_reduction_576_seed42/REPORT.md), or the [complete score table](../reports/SUMMARY.md).

Run the commands below from `reproduction/GRACE`. The launcher also works by absolute path from another directory.

```bash
cd reproduction/GRACE
bash evaluate.sh plan          # Print the four component commands; no models or GPUs
bash evaluate.sh check         # CPU preflight: packages, checkpoints, data and source pins
bash evaluate.sh summarize     # Audit saved artifacts; regenerate merged JSON/CSV/tables
bash evaluate.sh verify        # Also verify every file relocated during cleanup
```

`summarize` and `verify` describe the frozen completed runs. To assess a fresh rerun, use `compare --output ...` below. They do not promote new outputs into the frozen catalog.

## Reproduce all twelve suites

```bash
# Select idle physical PCI GPU indices. One evaluation runs per selected GPU.
# Repeating the same command resumes only its own verified output/checkpoints.
bash evaluate.sh run --gpus 0,2 --smoke 1
bash evaluate.sh run --gpus 0,2
bash evaluate.sh compare

# Optional isolated rerun, with an explicit score-drift tolerance in pp:
bash evaluate.sh run --gpus 0,2 --output experiments/qat_int4_token_reduction_576_seed42/outputs/reruns/reproduction_01
bash evaluate.sh compare --output experiments/qat_int4_token_reduction_576_seed42/outputs/reruns/reproduction_01 --max-score-drift 0.5
```

Default fresh output: `experiments/qat_int4_token_reduction_576_seed42/outputs/reruns/reproduction_v2/`. The launcher requires explicit GPU selection and restricts output to `experiments/qat_int4_token_reduction_576_seed42/outputs/reruns/`. It preserves the frozen results. GPU indices above are examples. A full rerun is substantial: 337,000 scored question inferences, with long video contexts. The cleanup itself was validated on CPU; full GPU inference was not repeated.

The default is exactly the completed **QAT W4** matrix, rather than the unfinished vanilla/BF16 matrices. Ten configurations per suite: `baseline_r100` and `resolution`, `uniform`, `merge` at `r050`, `r025`, `r010`. Ratios mean **fraction retained**. All methods share the single baseline. To run one suite or configuration:

```bash
bash evaluate.sh run --gpus 0 --suites ai2d --configurations baseline_r100,resolution_r025
bash evaluate.sh compare --suites ai2d --configurations baseline_r100,resolution_r025

# Seven images only; the five video IDs can be selected the same way.
bash evaluate.sh run --gpus 0,2 \
  --suites hallusionbench,mmbench,scienceqa,ai2d,mmmu,seedbench_img,mmstar
```

| Suite ID | Full population | Scoring / scope |
| --- | ---: | --- |
| `hallusionbench` | 1,129 | Local per-answer aAcc; includes 178 text-only rows |
| `mmbench` | 4,329 rotations / 1,164 base questions | Circular accuracy; all supplied rotations must be correct |
| `scienceqa` | 4,241 | Full test set, image and text questions |
| `ai2d` | 3,088 | Local multiple-choice accuracy |
| `mmmu` | 900 | Validation; multiple-choice and open answers |
| `seedbench_img` | 14,233 | Image questions only |
| `mmstar` | 1,500 | Six equal-sized categories |
| `tempcompass` | 1,580 | MC only; deterministic local fallback, no API judge |
| `videomme` | 1,800 | 900 short + 900 medium; long, audio and subtitles excluded |
| `videommmu_perception` | 300 | Full perception split |
| `videommmu_comprehension` | 300 | Full comprehension split |
| `videommmu_adaptation` | 300 | Full adaptation split |

Images retain the original runtime. Videos use fresh streamed/adaptive execution, **except the three Video-MMMU `resolution_r010` rows**, which use the original runtime to reproduce the existing results. Their baseline deltas and speedups remain withheld. The launcher dispatches 70 image jobs, 47 adaptive video jobs and three original-policy video jobs, without historical answer replay. An entirely adaptive Video-MMMU sweep would be a new experiment.

`compare` verifies completion artifacts, full population identity, policy and execution equality before comparing scores. It writes `comparison_to_frozen.json`, returns nonzero for missing/different rows, and accepts at most 0.5 pp absolute score drift by default. This is a configurable diagnostic threshold, not a demonstrated repeatability interval. Greedy generation and fixed data should keep accuracy close; CUDA kernels/cards can still change answers. Runtime is reported separately and is not required to match.

## VideoMME serial timing sweep

```bash
bash evaluate.sh prepare --workflow timing
bash evaluate.sh run --workflow timing --gpus 0
bash evaluate.sh summarize --workflow timing
```

This reuses the pinned seed-42 selection: 25 short and 25 medium questions from 50 distinct videos, five videos per duration/token quintile. All ten configurations run serially on one idle A100, with a full-input warmup excluded. Do not change cards between configurations. Outputs live in the rerun's `timing/` folder; use `--configurations` for a selected phase. The completed reference timing sweep is linked under `experiments/qat_int4_token_reduction_576_seed42/outputs/latest/videomme_timing50_seed42/`.

Model/request estimates expand the sample by 36 with 4,000 paired bootstrap draws. These are steady-state estimates, not full wall time. Fresh r25/r10 projections differed from the full runs by 15–17%; bootstrap intervals cover sampling variability within an execution, not all hardware/load variation.

## Training controls and older protocols

```bash
# Four unpruned QAT/GRACE W4/W8 controls: seven image suites, native processors,
# BF16-loaded research exports, greedy 256-token generation.
bash evaluate.sh plan --workflow controls-images
bash evaluate.sh check --workflow controls-images
bash evaluate.sh run --workflow controls-images --gpus 0 --smoke 2
bash evaluate.sh run --workflow controls-images --gpus 0
bash evaluate.sh summarize --workflow controls-images

# Include the available references explicitly:
bash evaluate.sh plan --workflow controls-images --models all
```

`evaluation.json` excludes the old local `sft_2b` model. Reference models retain their native processors and do not provide a matched-resolution precision comparison. The historical 28-pair control scores survive in reports, but the corresponding raw predictions are absent from this checkout; the merged file labels them `report_only`. A fresh control evaluation writes to the selected rerun’s `controls_images/` folder.

Other evaluation protocols remain available through the organized runners and their [specialist guides](../evaluation/docs/README.md): [native image controls](../evaluation/docs/README_local_development.md), [lmms harness tasks](../evaluation/docs/README_lmms_constrained.md), [upstream replay](../evaluation/docs/README_paper_protocol.md), and [older video controls](../evaluation/docs/README_video_evaluation.md). They differ in prompts, judging, output caps and/or image/video budgets. The 720p harness default is a different experiment from the completed packed-INT4 sweep. Historical score gates and API-judged protocols are not part of `evaluate.sh run`.

The historical author-replay archive `evaluation/provenance/upstream_20260925/manifest.json` was already absent in this checkout. Upstream replay and its nine archive-dependent tests require restoring that original archive. The twelve-suite workflow above passes CPU preflight without it.

Production training and memory preflight remain in the [576-cap control experiment](released_bf16_init_32k_b128_p589824_t4096_seed42/README.md). Reproducing evaluation does not require retraining completed checkpoints.

## Environment and reproducibility

`evaluate.sh` uses `LMMS_PYTHON` when set; otherwise it tries `.venv-lmms-eval/bin/python`, then the installed `/home/air/miniconda3/envs/grace/bin/python`. On this checkout the latter exists and matches the saved evaluation environment. To select another compatible installation:

```bash
export LMMS_PYTHON=/path/to/compatible/bin/python
bash evaluate.sh check

# For saved-result audits alone, system Python is sufficient:
python3 evaluation/audits/compile_results.py
```

The saved [original protocol](qat_int4_token_reduction_576_seed42/outputs/original/full/protocol.json) records all package versions and code hashes. Important pins: Python 3.11.0, torch 2.5.1+cu121, transformers 5.9.0, triton 3.1.0, lmms-eval 0.7.1 at commit `88b23e2bfa16a1edbc16e9e238ed82130b3a4f56`, qwen-vl-utils 0.0.14, FlashAttention 2, seed 42. The launcher sets HF/Transformers offline mode. [Environment preparation](../evaluation/data_tools/prepare_lmms_eval.sh) is a bootstrap helper; installation alone is not proof of parity. The CPU preflight checks the saved environment before adaptive video runs.

The sweep freezes the 256-step CE-only LoRA-QAT W4 checkpoint and its quantized sidecar. It packs 196 decoder linears as Triton W4A16, group size 128; vision, embeddings and head stay floating point. Batch one, greedy EOS, a 256-output-token cap, 4,096 minimum / 589,824 maximum image pixels and 2 fps video sampling remain unchanged. Adaptive video uses a common 131,072-token operational cap, temporal vision chunks of one and prefill chunks of 512; full duration and temporal groups are retained.

Keep checkpoints, dataset annotations/media, saved processor files and the pinned lmms checkout in place. [Dataset preparation](../evaluation/docs/README_datasets.md) and [download tools](../evaluation/docs/README_download_all_data.md) describe recovery on another machine. Code, model, dataset and protocol mismatch guards remain enabled. Never edit saved hashes to accept a changed environment or computation.

## Files and maintained helpers

| Location | Contents |
| --- | --- |
| [`../evaluate.sh`](../evaluate.sh) | Unified shell entry point; interpreter selection |
| [`evaluation/runners/evaluate.py`](../evaluation/runners/evaluate.py) | Job selection, fresh output routing and rerun comparison |
| [`evaluation/audits/compile_results.py`](../evaluation/audits/compile_results.py) | Combined CPU matrix/timing audit and JSON/CSV/table export |
| [`result_catalog.json`](../evaluation/protocols/result_catalog.json) | Explicit 120-row selection with marker checksums |
| [`historical_results.json`](../evaluation/protocols/historical_results.json) | Rounded historical scores, protocol labels and report checksums |
| [`reports/merged_results.json`](../reports/merged_results.json) | 120 verified rows, ten timing rows, 215 report-only scores and provenance |
| [`reports/merged_results.csv`](../reports/merged_results.csv) | Verified and historical scores together; filter by `evidence` and protocol |
| [Selected raw artifacts](qat_int4_token_reduction_576_seed42/outputs/latest/) | Relative links to selected raw predictions, telemetry, markers and timing evidence |
| [Current experiment metrics](qat_int4_token_reduction_576_seed42/reports/metrics/) | Regenerated detailed matrix and timing tables/JSON/CSV |
| [`reports/sources/`](../reports/sources) | Immutable source documents and checksum index |
| [`reports/provenance/migration_20261125.json`](../reports/provenance/migration_20261125.json) | Every relocated file and its original checksum |
| [`archive/cleanup_20261125/`](archive/cleanup_20261125/) | Superseded one-off experiment helpers; historical source, not launch instructions |

The token-reduction experiment has two surface Markdown files: `README.md` and `REPORT.md`. Its raw runs use descriptive names under `outputs/`: `original`, `video_recovery_gpu1`, `video_recovery_scheduler`, `video_matched_exclusive`, the initial/draft/phased timing runs and new `reruns/`. Predictions, attempts, fragments, logs, diagnostics and caches are ignored by Git. Curated metrics, plots and archived documents live under its `reports/`; repository-wide merged results and historical source indexes live under `GRACE/reports/`.

The former surface `results*` aliases have been removed. Current code uses the new paths. Immutable saved records retain their original paths and checksums; `evaluation/protocols/artifact_layout.py` resolves those paths through the recorded relocation map. Pre-relocation source snapshots and old/new hashes are retained in `evaluation/provenance/output_layout_20261125/`. Fresh evaluations record the current source hashes and must use a fresh output namespace. Checkpoints remain with their training runs. Other experiments retain their evidence links into `GRACE/outputs/`.
