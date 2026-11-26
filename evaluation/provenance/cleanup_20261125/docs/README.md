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

## Choose a workflow

| Purpose | Guide | Main entry point |
| --- | --- | --- |
| Current video matrix: all ten checkpoints | [Video evaluation](README_video_evaluation.md) | [run_video_matrix.py](../runners/run_video_matrix.py) |
| QAT video recovery with per-GPU concurrency and live GPU additions | [Parallel video evaluations](README_parallel_video_evaluation.md) | [run_parallel_video_evaluation.py](../runners/run_parallel_video_evaluation.py) |
| VideoMME 25-short/25-medium serial runtime extrapolation | [Timing subset](README_videomme_timing_subset.md) | [run_videomme_timing_subset.py](../runners/run_videomme_timing_subset.py) |
| Corrected local image evaluation | [Local development](README_local_development.md) | [run_local_development.py](../runners/run_local_development.py) |
| Pinned standard lmms-eval tasks | [Harness protocol](README_lmms_constrained.md) | [run_lmms_matrix.py](../runners/run_lmms_matrix.py) |
| Archived author-script replay | [Paper/replay protocol](README_paper_protocol.md) | [run_upstream_matrix.py](../runners/run_upstream_matrix.py) |
| Earlier constrained comparison | [Vanilla baseline](README_vanilla_baseline.md) | [run_constrained_matrix_evals.py](../runners/run_constrained_matrix_evals.py), [Bash launcher](../runners/run_all_constrained_evals.sh) |
| Dataset preparation | [Dataset guide](README_datasets.md) | [Video download tool](../data_tools/download_video_benchmarks.sh) |
| Differences from upstream | [Source audit](differences_from_org.md) | Historical analysis |

The current experiment's [video launcher](../../experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/run_video_evaluation.sh)
and [image launcher](../../experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/run_local_evaluation.sh)
point to the organized code. They run only the action requested on the command
line. Start with `plan` and `check`.

All five video tasks now share **2 fps / 256 px longest side** by default,
including TempCompass. A single `run --gpu N` command selects all ten models and
all available tasks; see the [video guide](README_video_evaluation.md#commands).
Higher resolution is selected explicitly with `--longest-side N`.

Direct script invocation works from any working directory. Package invocation
also works from `reproduction/GRACE`:

```bash
python -m evaluation.runners.run_video_matrix plan
python evaluation/runners/run_video_matrix.py plan
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
