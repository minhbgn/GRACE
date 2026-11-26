# Evaluation tools

Use the experiment's launcher when one is available. Shared tools are grouped by
role; benchmark code and datasets retain their existing locations.

| Directory | Contents |
| --- | --- |
| [runners/](../runners/) | Matrix entry points and the original seven-suite Bash launcher |
| [workers/](../workers/) | Isolated lmms-eval and video inference workers |
| [protocols/](../protocols/) | Scoring, generation policy, acceptance criteria, video adapters |
| [audits/](../audits/) | Saved-result validation, rescoring, summaries, image-token audit |
| [data_tools/](../data_tools/) | Video downloads and pinned lmms-eval environment setup |
| [tests/](../tests/) | CPU unit/integration tests |
| [provenance/](../provenance/) | Immutable upstream sources and pre-layout source snapshot |
| Benchmark directories and `video_benchmarks/` | Existing per-suite code and local assets |

## Current experiment commands

Start with [the experiment README](../../experiments/README.md), [the consolidated report](../../experiments/REPORT.md), and [the merged results](../../reports/merged_results.json). The unified launcher reproduces all twelve completed QAT W4 token-reduction suites, with the recorded original/adaptive runtime choices and fresh outputs under `experiments/qat_int4_token_reduction_576_seed42/outputs/reruns/`.

```bash
# From reproduction/GRACE:
bash evaluate.sh plan
bash evaluate.sh check
bash evaluate.sh summarize
bash evaluate.sh run --gpus 0,2 --smoke 1
bash evaluate.sh run --gpus 0,2
bash evaluate.sh compare
```

The full matrix is complete at 120/120 configurations. The two newest matched VideoMME resolution rows supersede the originals. Three Video-MMMU resolution-r10 rows retain the original protocol and have no paired baseline delta or speedup claim. No API judge is used in this workflow. [Runtime and score caveats](../../experiments/qat_int4_token_reduction_576_seed42/REPORT.md) accompany the results.

## Specialist workflows and historical protocols

| Purpose | Guide | Entry point |
| --- | --- | --- |
| Completed packed-INT4/token-reduction matrix | [Experiment commands](../../experiments/README.md) | [evaluate.py](../runners/evaluate.py) |
| Combined saved-artifact matrix/timing audit | [Merged results](../../reports/SUMMARY.md) | [compile_results.py](../audits/compile_results.py) |
| VideoMME 25-short/25-medium serial timing extrapolation | [Timing subset](README_videomme_timing_subset.md) | [run_videomme_timing_subset.py](../runners/run_videomme_timing_subset.py) |
| Native-processor image controls | [Local development](README_local_development.md) | [run_local_development.py](../runners/run_local_development.py) |
| lmms-eval harness image/video controls | [Harness protocol](README_lmms_constrained.md), [video guide](README_video_evaluation.md) | [run_lmms_matrix.py](../runners/run_lmms_matrix.py) |
| Historical video recovery/concurrency | [Parallel video evaluations](README_parallel_video_evaluation.md) | [run_parallel_video_evaluation.py](../runners/run_parallel_video_evaluation.py) |
| Archived author-script replay | [Paper/replay protocol](README_paper_protocol.md) | [run_upstream_matrix.py](../runners/run_upstream_matrix.py) |
| Earlier constrained comparison | [Vanilla baseline](README_vanilla_baseline.md) | [run_constrained_matrix_evals.py](../runners/run_constrained_matrix_evals.py) |
| Dataset preparation | [Dataset guide](README_datasets.md) | [Video downloader](../data_tools/download_video_benchmarks.sh) |
| Differences from upstream | [Source audit](differences_from_org.md) | Historical analysis |

These protocols are separate. In particular, the newer 720p lmms controls, older 256-pixel local video runs, and completed packed-INT4 sweep have different input/execution policies. Their scores and timings must retain those labels.

Direct scripts also work from outside the checkout; module invocation works from `reproduction/GRACE`:

```bash
python3 -m evaluation.runners.evaluate plan
python3 evaluation/audits/compile_results.py
```

## Supporting code

| File | Purpose |
| --- | --- |
| [lmms_worker.py](../workers/lmms_worker.py) | One harness model/task job, invoked by its matrix runner |
| [video_worker.py](../workers/video_worker.py) | One video job and CPU input preflight |
| [local_scoring.py](../protocols/local_scoring.py) | Deterministic scoring and generation telemetry |
| [matched_protocol.py](../protocols/matched_protocol.py) | Shared image/generation options for benchmark workers |
| [paper_acceptance.py](../protocols/paper_acceptance.py) | Historical numerical paper-score checks |
| [video_protocol.py](../protocols/video_protocol.py) | Local video assets, sampling, processor token accounting |
| [audit_image_visual_tokens.py](../audits/audit_image_visual_tokens.py) | CPU image-token distribution measurement |
| [audit_video_token_rates.py](../audits/audit_video_token_rates.py) | Validate complete saved video runs; report tokens per video second and elapsed second, with incomplete-run coverage |
| [audit_local_scoring.py](../audits/audit_local_scoring.py) | Validate corrected scoring on saved answers |
| [rescore_saved_protocols.py](../audits/rescore_saved_protocols.py) | Rescore archived answers without inference |
| [summarize_constrained_evals.py](../audits/summarize_constrained_evals.py) | Validate and summarize saved constrained results |
| [check_re_evaluation_gate.py](../audits/check_re_evaluation_gate.py) | Historical reference-stage completion/score gate |
| [prepare_lmms_eval.sh](../data_tools/prepare_lmms_eval.sh) | Prepare the pinned harness environment |
| [download_video_benchmarks.py](../data_tools/download_video_benchmarks.py) | Resumable download/extraction implementation |
| [paths.py](../paths.py) | Stable asset roots and source-file lookup used by launchers/hashes |

Measure saved video token rates without inference (run from `reproduction/GRACE`):

```bash
python evaluation/audits/audit_video_token_rates.py \
  --include-partial-over 0.5 \
  --output-prefix experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/reports/video_token_rates_20261023
```

The default selects the current full video matrix. Use `--experiment` and
`--matrix` to select another experiment/matrix. Markdown, JSON and CSV outputs
include complete-run rates, VideoMME duration groups, per-question measurements,
and an attempt ledger. `--include-partial-over 0.5` also measures the latest
attempt of incomplete pairs with strictly more than 50% question coverage.
These observed-prefix rates remain labelled incomplete; retries are never
merged. Omit this flag to measure only completed runs. Other incomplete pairs
have blank rates; historical smoke runs are separate. Primary rates divide
summed tokens by summed source-video seconds.

Standalone benchmark guides: [Video-MME](benchmarks/VideoMME.md),
[RealWorldQA](benchmarks/RealWorldQA.md), [MMMU](benchmarks/mmmu.md),
[MathVision](benchmarks/MathVision.md), [ODinW-13](benchmarks/ODinW-13.md).
These describe their separate upstream-style evaluators, which can have different
dependencies, prompts and judging policies from the matrix protocols.

## Layout migration and experiment hashes

On 2026-10-17, shared code moved out of the evaluation root and project-owned
READMEs moved here. Dataset READMEs and archived upstream/source READMEs remain
inside their immutable data/provenance directories.

Existing checkpoints, predictions, completion markers, protocol manifests and
validation reports were not repinned or rewritten. The previous sources are
preserved byte-for-byte under
[`provenance/layout_20261017/original/`](../provenance/layout_20261017/original/).
The [migration manifest](../provenance/layout_20261017/manifest.json) records old
paths, new paths and original SHA256 checksums. Old saved command lines remain
historical records; current launchers use the new paths.

Moving a hashed source changes its provenance even when numerical evaluation
behavior is unchanged. Do **not** edit old hashes to make a new runner resume
an old experiment. To rerun with the organized code, select a fresh `--output`
(or `OUT_DIR`) and let the runner record new source hashes. Existing mismatch
guards remain enabled. To reproduce an old implementation, restore the sources
from the snapshot into a separate checkout rather than modifying the live tree.
The snapshot alone does not relocate a saved environment or its absolute paths.

The reorganized video runner is `video_unsaturated_v2`. Its subsequent unified
default is **2 fps / 256 px longest side for all suites**, selected by the user;
the policy hash gives this configuration its own output namespace. Higher
resolution must be explicit, and `--longest-side 0` requests source resolution.
The old source-resolution v1/v2 smoke/preflight evidence remains historical.
Image protocols retain their names and require a fresh output base when source
checks reject an old run; the current image launcher's default is
`local_development_layout2/`.

New scripts should use package imports, [paths.py](../paths.py) for source/asset
locations, and the existing experiment mappings. Place new workflow guidance in
this folder and add it to the table above.

## Consolidation — 2026-11-25

Duplicate restored evaluation-root Python/Bash files are archived under `provenance/cleanup_20261125/root/`. Use the organized paths above. Experiment report/progress/timing helpers are consolidated into `audits/compile_results.py`; one-off sources remain in `experiments/archive/cleanup_20261125/`. The token-reduction experiment stores raw named runs under its ignored `outputs/`, curated metrics/figures/history under its `reports/`, and only README/REPORT as surface Markdown. Repository-wide merged files and source indexes live under `GRACE/reports/`; other raw evidence lives under `GRACE/outputs/`. Surface result aliases have been removed from the token-reduction experiment; current code uses the new paths, while the recorded resolver maps immutable old paths. `reports/provenance/migration_20261125.json` records every moved file checksum. The current catalog selects completions explicitly, not by directory timestamps. The original pinned source bytes are preserved in `provenance/output_layout_20261125/original/`; current path revisions are explicitly hashed and fresh outputs pin the current code.
