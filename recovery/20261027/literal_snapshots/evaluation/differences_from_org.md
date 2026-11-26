# Evaluation differences from the original GRACE repository

Audit date: 2026-09-26. This documents the current source, not a claim that any particular historical result used the current settings.

The comparison baseline is [ForeverBlue816/GRACE at `b73fa9b4d427aa64ed3ad274def641efb3145877`](https://github.com/ForeverBlue816/GRACE/tree/b73fa9b4d427aa64ed3ad274def641efb3145877/evaluation), the local checkout's base commit. It is deliberately pinned rather than compared to a moving `main`. All **67 original files across 13 benchmark directories** were compared against that commit: **7 Python evaluators changed, 60 files unchanged, none deleted**. The [saved upstream manifest](provenance/upstream_20260925/manifest.json) contains 17 files (seven evaluators, seven Slurm launchers, README, requirements and license); every archived file was verified against both its SHA-256 and the corresponding Git blob.

Scope includes all benchmark directories, shared evaluation scripts, experiment evaluation wrappers, protocol configurations, validation and supporting documentation. Downloaded datasets, generated predictions/logs/reports, virtual environments and training implementation changes are not evaluator modifications. The local lmms-eval workflow is an added implementation, not a patch to the seven original evaluators. Documentation-only work for this audit did not run inference or change evaluation code.

## 1. Which entry point uses which protocol?

| Entry point | Implementation and default behavior | Why it exists / effect |
| --- | --- | --- |
| Original suite `eval_*.slurm` | Unchanged cluster launchers; normally four processes, FlashAttention 2, checkpoint processor and generation configuration. In the working tree they now invoke the modified Python files. | Preserves original launch recipes, but using an unchanged launcher alone does not bypass the local Python changes. |
| [run_all_constrained_evals.sh](run_all_constrained_evals.sh) | Local seven-suite scripts, baseline/SFT/QAT/GRACE, two processes by default. Baseline loads its own weights with the SFT processor by default. No explicit pixel override or generation-config reset. | Portable four-model comparison; preprocessing comes from the selected saved processor. |
| [run_constrained_matrix_evals.py](run_constrained_matrix_evals.py) | Local seven-suite scripts; configured model matrix; each model's own processor, forced **4,096–100,352 pixels**, fresh generation config, two processes by default. | Controls visual budget and decoding across vanilla, released and trained models. This changes the original protocol. |
| [run_upstream_matrix.py](run_upstream_matrix.py) | Executes the **archived, unmodified** seven Python scripts. One GPU, native checkpoint processor and generation config, original token limits including MMStar 64. | Replays the supplied script behavior with provenance and validation. It does not establish that these scripts produced the published paper scores. |
| [run_lmms_matrix.py](run_lmms_matrix.py) + [lmms_worker.py](lmms_worker.py) | Pinned lmms-eval Qwen3-VL backend, one GPU, batch size 1, zero-shot, SDPA, **200,704–1,605,632 pixels**, task-specific prompts/scorers/token limits. | Independent harness comparison. Changes question scope and/or scoring for several suites. |
| `run_paper_standard_evaluation.sh` | Same harness runner with `paper_standard_protocol.json` (v2); enables two explicit MMMU corrections. | Separates corrected harness scores from the earlier v1 harness results. |
| `run_re_evaluation.sh` | Upstream replay in two stages, with a reference-result gate before the four trained LoRA exports. | Adds workflow acceptance rules; does not introduce a fourth benchmark scorer. |

The shell runner and Python constrained matrix are **not interchangeable**: only the latter forces pixel bounds and resets the generation configuration. A training context limit of 2,048 is not imposed on inference by these local runners.

## 2. Every change inside the seven original Python evaluators

The following shared edits occur in all seven files listed in section 3.

| Modification | Original behavior | Current behavior and purpose |
| --- | --- | --- |
| Import shared policy | No shared local policy module. | Imports `sys` and `Path`, prepends the parent evaluation directory to `sys.path`, and imports `add_protocol_arguments`, `apply_protocol`, `protocol_metadata` from [matched_protocol.py](matched_protocol.py). Allows direct script execution to locate the helper. |
| `--processor_path` | Always `AutoProcessor.from_pretrained(args.model_path)`. | Optional independent processor path, falling back to the model path. Allows matching preprocessing/chat templates while retaining the selected model's weights. This selects the **whole processor/tokenizer**, not only image dimensions. |
| Pixel CLI options | Uses the loaded processor's settings. | Adds `--min_pixels` and `--max_pixels`, both defaulting to `None`. Requires both together and `0 < min <= max`. Sets `image_processor.size.shortest_edge/longest_edge` and older `min_pixels/max_pixels` attributes. Controls visual input cost and can alter accuracy. Values are pixel budgets, not literal image side lengths. |
| `--fixed_generation` | Keeps checkpoint generation configuration, with the existing explicit `generate` arguments. | Optional fresh Transformers `GenerationConfig`: `do_sample=False`, `num_beams=1`, `repetition_penalty=1.0`, tokenizer BOS/EOS/PAD IDs, `use_cache=True`; other fields take library defaults. Removes inherited checkpoint decoding settings for controlled comparison. |
| Policy application | No override step. | Calls `apply_protocol` after loading the processor and assigning an EOS fallback when PAD is absent. Applies overrides before processing questions. |
| Result metadata | Summary records model path and suite metrics. | Always adds resolved `processor_path` and `image_processor_size`. When `--fixed_generation` is enabled, adds `evaluation_protocol` with version, requested pixel bounds, token cap, greedy/beam/repetition settings, attention backend and no-truncation context policy. MMMU also records its fixed MC cap of 8. Enables validation of result comparability. |

These policy changes are opt-in. Pixel overrides may be used without `--fixed_generation`; in that case the image size is recorded but `evaluation_protocol` is absent. Existing per-suite `generate` calls, prompts, data loading, rank sharding, successful-answer parsing, score aggregation and token defaults were not rewritten by these edits.

### Additional error-path changes

| File | Exact change | What it does |
| --- | --- | --- |
| [AI2D/eval_ai2d.py](AI2D/eval_ai2d.py) | Exception handler previously set both `pred` and `gold` to `-1`; now retains `pred=-1` and recomputes gold with `_normalize_answer(row['answer'], len(row['options']))`. | Preserves the question label for cross-model identity checks after an inference/image error. The row remains incorrect under the existing correctness expression. Normalizing malformed source data can still raise an exception. |
| [MMStar/eval_mmstar.py](MMStar/eval_mmstar.py) | Exception handler previously blanked gold; now stores `str(row.get('answer', '')).strip().upper()`. Empty prediction and empty valid-letter list remain. | Keeps the gold label stable even if one model fails on that question. Failed rows remain incorrect. |
| [SEEDBench/eval_seedbench.py](SEEDBench/eval_seedbench.py) | Same gold-preservation change as MMStar. | Keeps labels available for comparison; no change to the successful choice parser or image-subset selection. |
| [mmmu/eval_mmmu.py](mmmu/eval_mmmu.py) | Error gold previously used `str(gold_raw)`; now strips whitespace and uppercases MC gold, while only stripping open-answer gold. Explicit `correct=False` remains. | Makes failed-row label normalization consistent with ordinary rows; does not fix or replace the local open-answer scorer. |

HallusionBench, MMBench and ScienceQA have only the shared changes above. The four error edits change stored labels, not the intended correctness of failed rows; upstream already marked these failures incorrect.

## 3. Suite-by-suite coverage

Counts below describe the configured full local datasets. They are enforced by the Python matrix/replay validators, not by the basic shell runner. Original `.slurm` files are unchanged for every suite.

| Suite / local evaluator | Original scope retained in modified evaluator | Source modifications | Launcher-level differences |
| --- | --- | --- | --- |
| [HallusionBench](HallusionBench/eval_hallusion.py) | 1,129 image + non-image rows; rule-based yes/no aAcc with qAcc/fAcc group metrics; no judge. | Shared processor/policy/metadata edits only. | 8 output tokens unchanged. Harness selects image-only data and judged metrics instead (section 5). |
| [MMBench](MMBench/eval_mmbench.py) | English dev 20231003 TSV, 4,329 rows; per-row choice accuracy; no judge. | Shared edits only. | 8 tokens unchanged. Harness changes to rolling judged extraction/scoring. |
| [ScienceQA](ScienceQA/eval_scienceqa.py) | Full test: 4,241 image and text-only questions, including modality/category breakdowns. | Shared edits only. | 8 tokens unchanged. Local preflight rejects missing expected images; the inherited evaluator alone can otherwise treat missing images as text-only. |
| [AI2D](AI2D/eval_ai2d.py) | 3,088 test rows; existing custom choice parser and overall accuracy. Reads every parquet in the supplied directory. | Shared edits + error-gold preservation. | 8 tokens unchanged. Directory must contain test shards only; path preflight does not prove split purity. |
| [MMMU](mmmu/eval_mmmu.py) | 900 validation rows across subjects; custom MC parser and simplified numeric/text open-answer matching. | Shared edits + error-gold normalization + MC cap metadata. | MC 8 / open 32 tokens unchanged. Separate inherited `run_mmmu.py` workflow is unchanged. |
| [SEEDBench](SEEDBench/eval_seedbench.py) | Image subset, 14,233 questions; existing letter parser and question accuracy. | Shared edits + error-gold preservation. | 8 tokens unchanged. Local runner ID is `seedbench_img`; harness ID is `seedbench`. |
| [MMStar](MMStar/eval_mmstar.py) | 1,500 TSV questions; existing last-valid-letter parser and per-question/category accuracy. | Shared edits + error-gold preservation. | **Both constrained runners pass 8 tokens versus 64 in the original Slurm recipe.** Python script default remains 64; archived replay restores 64. Shorter outputs can change truncation and the last-letter parser's result. |
| [MathVista](MathVista/eval_mathvista.py) | Existing Python evaluator and Slurm recipe. | **None: both original files unchanged.** | Not included in the added seven-suite matrices. |
| [OCRBench](OCRBench/eval_ocrbench.py) | Existing Python evaluator and Slurm recipe. | **None: both files unchanged.** | Not included in added matrices. |
| [MathVision](MathVision/) | Instruct/think inference and scoring scripts, helpers, requirements and README. | **None: all 10 files unchanged.** | Not included in added matrices. |
| [ODinW-13](ODinW-13/) | Instruct/think scripts, dataset/scoring helpers, requirements and README. | **None: all 9 files unchanged.** | Not included in added matrices. |
| [RealWorldQA](RealWorldQA/) | Instruct/think scripts, helpers, requirements, README and baseline Slurm file. | **None: all 11 files unchanged.** | Not included in added matrices. |
| [VideoMME](VideoMME/) | Instruct/think scripts, helpers, requirements and README. | **None: all 9 files unchanged.** | Not included in added matrices. |

MMMU has 12 original files: only `eval_mmmu.py` changed; its other 11 files, including `run_mmmu.py`, `eval_utils.py`, instruct/think scripts and Slurm launcher, are unchanged. Each other modified-suite directory has two original files: its Python evaluator changed and its Slurm launcher did not.

Inherited character-scanning choice parsers and the simplified local MMMU open scorer remain limitations of both local comparison and archived replay. They should not be described as newly introduced fixes or official leaderboard scoring.

## 4. Added local orchestration, validation and reporting

### Portable shell launcher

[run_all_constrained_evals.sh](run_all_constrained_evals.sh) is entirely new relative to GRACE:

- Replaces cluster modules, hardcoded scratch paths and Slurm submission with paths resolved from the script, configurable `PYTHON`, `DATA_ROOT`, individual dataset variables, model/output roots, `NPROC` and attention implementation. Uses `python -m torch.distributed.run --standalone`; default workers change from four to two. It does not reproduce the Slurm environment setup/offline exports.
- Runs baseline, SFT, QAT and GRACE sequentially over seven suites (28 commands by default). Validates method names/duplicates, positive worker count and supported attention choices. Baseline processor defaults to `<experiment>/runs/sft`; trained methods load their own processors. Baseline weights remain the original checkpoint.
- Adds `--dry-run` and CPU-only `--check`. Preflight checks model weights/configs, saved processors, and trained export completion/non-profile metadata; checks dataset assets and all expected ScienceQA/SEED image paths. No dataset download is performed.
- Refuses existing merged outputs or shards unless `--resume` is given. Resume skips any existing merged JSON and replaces partial rank shards when retrying. **This shell skip is based on existence, not validated contents.**
- Adds `--keep-going` to attempt remaining jobs after failures and return nonzero afterward. Output layout becomes `<OUT_DIR>/<method>/<suite>.json` plus inherited rank JSONL shards.
- Uses the original suite split selections and output caps except MMStar's 64 → 8 change.

### Configured constrained matrix

[run_constrained_matrix_evals.py](run_constrained_matrix_evals.py) adds an experiment-JSON-driven alternative rather than modifying the shell launcher:

- Resolves arbitrary configured model IDs through `Experiment` in [constrained_experiment.py](../qwen-vl-finetune/scripts/constrained_experiment.py). Validates selected exports/provenance and processor compatibility; reuses shell dataset/image preflight.
- Explicitly passes each model's own processor, 4,096–100,352 pixels and `--fixed_generation` for every job. This makes the decoding/image comparison controlled but different from native checkpoints.
- Pins protocol, implementation hashes, experiment-config hash, model provenance, question-asset SHA-256 values and an external-image inventory. Image inventory hashes cover relative paths, sizes and modification times, **not image content bytes**.
- Locks the output directory, rejects protocol/provenance drift and validates existing merged outputs before skipping. Retries invalid/incomplete jobs, collects failures and produces a full comparison only when all configured rows are present.
- Adds `--dry-run`, `--check`, `--summarize`, and model selection; `NPROC`, `ATTN`, `OUT_DIR` and dataset environment overrides remain supported.

[summarize_constrained_evals.py](summarize_constrained_evals.py) is also new. It recomputes counts/accuracy, checks unique question IDs and matching labels across models, emits percentage scores, percentage-point deltas, descriptive seven-suite means, category breakdowns, HallusionBench group scores and error/parse-failure diagnostics. Matrix validation additionally checks exact expected counts, model/processor paths, image size and policy metadata. It writes Markdown/CSV comparisons. Its local validation **reports error rows but does not reject every run containing them**; archived replay has a stricter error rule.

### Archived upstream replay

[run_upstream_matrix.py](run_upstream_matrix.py) adds a separate, auditable execution path:

- Verifies the 17-file source archive before use and executes archived evaluator paths, bypassing `matched_protocol.py` and all seven working-tree edits.
- Defaults to five reference models, all seven suites, one GPU, FlashAttention 2; permits explicitly recorded SDPA/eager alternatives. Uses native checkpoint processors/generation configs, no training pixel/context caps, and MMStar 64 tokens.
- Removes inherited `RANK`, `WORLD_SIZE`, `LOCAL_RANK` and selects a physical GPU in PCI order. Records commands and per-attempt logs. One process replaces the original Slurm data-parallel launch; this is an execution/environment difference, not an intentional scope/scorer change.
- Provides plan/check/run/summarize and isolated `smoke_N` versus `full` directories. A new timestamped attempt preserves previous failed output; OOM/errors do not trigger automatic resolution reduction or sample dropping.
- Pins source/environment/model/dataset evidence and records resolved processor/generation settings. Validates exact unique sample coverage, labels shared across models, model path and recomputed accuracy. Any recorded inference error blocks completion; failed answer parses remain in the denominator.
- Writes hashed `COMPLETE.json` markers and verifies artifact hashes before resume. Produces paper-comparison JSON/Markdown. These markers and reports are local additions, not upstream output formats.

## 5. Added lmms-eval suites and deviations from the original scripts

[prepare_lmms_eval.sh](prepare_lmms_eval.sh) creates a local system-site-packages venv and editable source installation of lmms-eval v0.7.1, pinned to `88b23e2bfa16a1edbc16e9e238ed82130b3a4f56`. It installs `decord==0.6.0`, `setuptools==80.9.0`, `wheel==0.45.1` without dependency replacement, preserving source-only YAML templates. The runner requires a clean pinned checkout and explicitly gives it import precedence over an inherited wheel. These dependencies and installation rules are new local infrastructure.

### Common harness differences

- Uses `qwen3_vl` rather than direct suite-specific Transformers loops, with BF16 inference, a system message (`You are a helpful assistant.`), `interleave_visuals=false`, pinned task prompts/filters/stop strings, SDPA and common 200,704–1,605,632 pixel bounds. The original scripts construct their own user-message prompts and use checkpoint-native processors.
- Runs zero-shot, batch size 1, one GPU, seed 42 for Python/NumPy/Torch/few-shot selection, 1,000 bootstrap iterations, and explicit greedy/one-beam generation. It **does not use the constrained runner's fresh `GenerationConfig`**; unspecified checkpoint settings can remain inherited.
- Loads Hugging Face datasets from task YAMLs instead of local TSV/JSON/parquet arguments. Resolves and locks repository revisions, changes dataset `token=True` to `token=None` for optional cached authentication, and records dataset split sizes/fingerprints and sample identities across models. `prepare` pins metadata without loading a model or downloading the full dataset.
- Executes each model/task in an isolated subprocess and timestamped attempt. Saves job configuration, result JSON, raw samples, dataset evidence and judge audit. Full/sample coverage, metric scale/finiteness, document identities and artifact hashes are checked before issuing completion markers.
- Normalizes metrics to percent: multiplies ScienceQA/AI2D/MMMU/MMStar/SEED fractional scores by 100; HallusionBench/MMBench scores already use percent. Missing suites prevent a seven-suite mean.
- Does not re-quantize exported weights or load packed INT4 kernels. QAT/GRACE grid-projected weights are evaluated in BF16 storage, so this is accuracy evaluation rather than deployment speed/memory measurement.

| Suite | Local/original script | Added harness task and primary metric | Output cap and practical effect |
| --- | --- | --- | --- |
| HallusionBench | Image + non-image (1,129), rule-based yes/no aAcc. | `hallusion_bench_image`: image-only 951 rows; judged `aAcc,none`, also qAcc/fAcc. | 8 → 128. Changes population, prompt and answer assessment; scores are not interchangeable. Judge configured as `gpt-4`. |
| MMBench | English dev TSV; each of 4,329 rows scored independently by local choice parser. | `mmbench_en_dev`: `gpt_eval_score,none`, rolling/grouped evaluation with answer extraction and judge fallback. | 8 → 1,024. Changes extraction and aggregation even for the same benchmark family. Judge configured as `gpt-4o-2024-11-20`. |
| ScienceQA | All 4,241 test questions. | `scienceqa_img`: 2,017 image questions; `exact_match,none`, harness prompt/result processing. | 8 → 16. Removes text-only questions from the primary metric. |
| AI2D | Local test parquet and custom choice scan. | `ai2d`: `exact_match,flexible-extract`, harness prompt and answer filter. | 8 → 16. Replaces parser/prompt/data source; not a patch to local `parse_choice`. |
| MMMU | Validation MC + simplified open-answer scorer. | `mmmu_val`: `mmmu_acc,none`, harness MC extraction/open normalization and aggregation; v2 corrections below. | MC 8 / open 32 → 128 for both. Prompt and open-answer matching differ materially. |
| SEED | Filters image questions before evaluation (14,233). | `seedbench`: evaluates 17,990 image/video-category rows, reports `seed_image,none` from 14,233 image questions. | 8 → backend default 128. Image metric is selected deliberately; `seed_all` is a different metric. |
| MMStar | Per-question accuracy; last-valid-letter parser. | `mmstar`: `average,none`, mean over 18 L2 category accuracies, harness extraction and Qwen3-specific prompt. | Original 64 / constrained 8 → backend default 128. Changes prompt, extraction, cap and aggregation definition. |

The precise prompt/filter implementations come from the [pinned local task source](../third_party/lmms-eval-v0.7.1/lmms_eval/tasks/) and [Qwen3-VL backend](../third_party/lmms-eval-v0.7.1/lmms_eval/models/simple/qwen3_vl.py). Those harness differences are adoption of another evaluator, not edits to the original GRACE suite files.

### MMMU-specific runtime adaptations

These live in `run_lmms_matrix.py` / `lmms_worker.py`; the pinned third-party source is not edited.

| Adaptation | Enabled when | Purpose/effect |
| --- | --- | --- |
| No-judge guard | All local harness MMMU runs. | Temporarily replaces judge-server creation with an object that raises on access. Covers both YAML import passes, including TaskManager during actual evaluation. Allows the selected exact scorer to load without credentials and prevents accidental judge use. Checks that `mmmu_process_results` is the selected scorer. |
| Standard MC/open prompt branching | `mmmu_prompt: official_short_answer` in v2. | Replaces Qwen3-specific task kwargs with the task's default kwargs. Open questions request a short answer instead of choosing from nonexistent options; MC questions keep their options. This changes model inputs. |
| Typed open-answer candidates | `mmmu_open_scoring: typed_candidates` in v2. | Replaces open-question `parsed_pred` values for accuracy/pass-at-k payloads with lists of all normalized candidates, retaining numeric types. Avoids the pinned wrapper's first-candidate string conversion, which causes the downstream open scorer to iterate characters. Keeps pinned normalization, MC processing and aggregation. This changes open-answer scoring. |

The [v1 protocol](../experiments/released_bf16_init_32k_b128_p100352_t2048_seed42/lmms_protocol.json) lacks the two opt-in MMMU flags; the [v2 protocol](../experiments/released_bf16_init_32k_b128_p100352_t2048_seed42/paper_standard_protocol.json) enables them. Historical v1 results are not retroactively corrected.

### Judge handling

The seven original Python scripts make no judge calls. The added harness requires `--allow-judge` and a configured API key for HallusionBench/MMBench. The worker sets the configured endpoint/model IDs, binds the HallusionBench judge into YAML-loaded aggregation functions, audits HTTP errors/empty responses and counts MMBench random extraction fallbacks. It retains outputs but refuses completion after a judge failure or random fallback. Judge IDs are the declared harness settings, not verified identities of the authors' judges.

## 6. Experiment wrappers and acceptance gate

All paths in this table are under `experiments/` and are additions relative to the upstream base.

| File(s) | Added behavior and purpose |
| --- | --- |
| [constrained_32k_b128_p100352_t2048_seed42/run_local_evaluation.sh](../experiments/constrained_32k_b128_p100352_t2048_seed42/run_local_evaluation.sh) | Pins four methods and the experiment/output root, selects the available grace Python, defaults to shell `--resume --keep-going`, then writes comparisons after a successful default run. |
| [released_bf16_init_32k_b128_p100352_t2048_seed42/run_local_evaluation.sh](../experiments/released_bf16_init_32k_b128_p100352_t2048_seed42/run_local_evaluation.sh) | Uses the configured Python constrained matrix and PCI GPU ordering. This wrapper has the forced image/decoding policy, unlike the older four-method wrapper. |
| [run_lmms_evaluation.sh](../experiments/released_bf16_init_32k_b128_p100352_t2048_seed42/run_lmms_evaluation.sh) | Selects the dedicated harness Python/import path and experiment config; uses v1 protocol by default. |
| [run_paper_standard_evaluation.sh](../experiments/released_bf16_init_32k_b128_p100352_t2048_seed42/run_paper_standard_evaluation.sh) | Selects v2, a separate `paper_standard_evaluation` output root and five reference models by default. |
| [run_re_evaluation.sh](../experiments/released_bf16_init_32k_b128_p100352_t2048_seed42/run_re_evaluation.sh) | Runs 35 reference pairs in stage 1 and, only on explicit stage-2 invocation after the gate passes, 28 trained LoRA pairs. Uses released BF16 as the reference rather than the prior local SFT export. Adds plan/check/status modes, logs and a separate `re_evaluation` output root. |
| `lmms_protocol.json`, `paper_standard_protocol.json` | Store declared harness settings and separate v1 from v2. Experiment JSON supplies checkpoint IDs/paths and comparison pairs; expanding the model matrix does not itself change a suite's scoring. |

[paper_acceptance.py](paper_acceptance.py) adds five published reference rows and per-suite numerical acceptance: unjudged absolute gaps ≤3 pp pass, >3 to ≤5 pp are barely acceptable, >5 pp fail; actual judged harness tasks use ≤8 pp. Missing/smoke results cannot pass, and models without published targets are labeled accordingly. `seedbench_img` is mapped to the published SEED key.

[check_re_evaluation_gate.py](check_re_evaluation_gate.py) independently reopens hashed full replay outputs, verifies archived source/token policy, expected artifacts/counts/identities and recomputed scores, and compares all 35 pairs. Stage 2 requires every pair to pass; `--accept-barely` explicitly allows up to 5 pp. All replay suites are unjudged, so the 8 pp allowance never applies there. It writes `stage1_gate.json` / `.md` and returns exit 2 for a blocked gate. These are local acceptance decisions, not benchmark scoring rules or proof of paper reproduction.

## 7. Supporting additions and verification scope

| Added files | What they document or verify |
| --- | --- |
| [README_datasets.md](README_datasets.md) | Dataset sources/layouts and local suite path preparation. Downloaded MMBench/MMStar TSVs are assets, not evaluator patches. |
| [README_vanilla_baseline.md](README_vanilla_baseline.md) | Untuned model baseline and optional shared processor interpretation. |
| [README_lmms_constrained.md](README_lmms_constrained.md) | Harness setup and earlier manual examples. Its historical pixel/token examples are not the current pinned protocol JSON defaults. |
| [README_paper_protocol.md](README_paper_protocol.md) | Source provenance, archived replay versus standard harness, unresolved author-recipe choices and acceptance policy. |
| [provenance/upstream_20260925/](provenance/upstream_20260925/) | Immutable baseline copies and hash manifest; the archived README is the upstream README, not a new evaluator. |
| [tests/test_lmms_matrix.py](tests/test_lmms_matrix.py) | Metric scaling, coverage/identity rejection, smoke/full separation, output tamper detection and incomplete-matrix reporting. |
| [tests/test_paper_protocol.py](tests/test_paper_protocol.py) | Acceptance thresholds, replay commands without training overrides, and invalid replay result rejection. |
| [tests/test_re_evaluation_gate.py](tests/test_re_evaluation_gate.py) | Complete/missing/tampered references, tolerance opt-in and smoke-run exclusion from stage 2. |
| [tests/cpu_lmms_integration.py](tests/cpu_lmms_integration.py), [tests/cpu_mmmu_integration.py](tests/cpu_mmmu_integration.py) | Synthetic CPU harness integration, including MMMU prompt branching and typed open-answer scoring. They validate software behavior, not model quality. |

Audit verification: compared all 67 original evaluation files byte-for-byte against the pinned Git commit; inspected the complete seven-file diff and all added shared runners/helpers/wrappers; verified all 17 archive hashes against Git; checked the harness task/backend definitions for token limits and protocol behavior. Only this Markdown file was added for this request. No GPU evaluation, dataset download, live judging or result regeneration was needed.

For interpretation, keep three questions separate: whether code matches the supplied scripts, whether benchmark scoring follows the chosen standard task, and whether scores numerically match the paper. Native replay, constrained comparisons and v1/v2 harness results answer different versions of those questions and should retain distinct protocol labels.
