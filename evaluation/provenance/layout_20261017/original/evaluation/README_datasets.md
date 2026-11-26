# Evaluation dataset downloads

For the implemented offline video matrix, including all ten checkpoints in
`released_bf16_init_32k_b128_p589824_t4096_seed42`, see
[README_video_evaluation.md](README_video_evaluation.md). It consumes local
TempCompass, Video-MME and Video-MMMU assets, supports ZIP-contained videos, and
audits 2 fps/source-resolution inputs without adaptive processor ceilings.
Missing assets and context overflows remain explicit blockers.

For LLMC+ with temporal token reduction, see the new
[video-reasoning benchmark guide](#video-reasoning-benchmarks-for-llmc-and-token-reduction)
below, including download commands, storage estimates, and evaluator caveats.

`run_all_constrained_evals.sh` runs **seven** suites: HallusionBench, MMBench,
ScienceQA test, AI2D test, MMMU validation, SEED-Bench-IMG, and MMStar. The
ShareGPT4V downloads in the master README are training data and do not supply
these benchmarks. The launcher and its `eval_*.py` loaders require local files.

The commands below place the files in this `evaluation/` directory with the
launcher's default layout. Source listings and HTTP download links were checked
on 2026-09-16; full datasets and inference were not downloaded/run for this guide.
The Hugging Face revisions are pinned to the file listings checked on that date.

## Sources and required formats

| Suite | Download source | Required local path relative to `evaluation/` |
| --- | --- | --- |
| HallusionBench | [LMMs-Lab Parquet distribution](https://huggingface.co/datasets/lmms-lab-encoder/HallusionBench) | `HallusionBench/data/{image-*,non_image-*}.parquet` |
| MMBench | [Original English dev TSV](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_en_20231003.tsv); [project](https://github.com/open-compass/MMBench) | `MMBench/mmbench_dev_en_20231003.tsv` |
| ScienceQA | [Official metadata](https://github.com/lupantech/ScienceQA/tree/main/data/scienceqa); [official image download script](https://github.com/lupantech/ScienceQA/blob/main/tools/download.sh) | `ScienceQA/data/{problems.json,pid_splits.json,images/test/<pid>/image.png}` |
| AI2D | [LMMs-Lab Parquet distribution](https://huggingface.co/datasets/lmms-lab-encoder/ai2d) | `AI2D/data/test-*.parquet` (both shards) |
| MMMU | [Official MMMU distribution](https://huggingface.co/datasets/MMMU/MMMU/tree/main) | `mmmu/data/<Subject>/validation-*.parquet` (30 subjects) |
| SEED-Bench-IMG | [Official original SEED-Bench files](https://huggingface.co/datasets/AILab-CVC/SEED-Bench/tree/main) | `SEEDBench/data/SEED-Bench.json` and `SEEDBench/data/SEED-Bench-image/<image>` |
| MMStar | [Official TSV](https://huggingface.co/datasets/Lin-Chen/MMStar/blob/main/MMStar.tsv) | `MMStar/MMStar.tsv` |

The old `lmms-lab/HallusionBench` and `lmms-lab/ai2d` URLs currently redirect to
`lmms-lab-encoder`. The selected Parquet and TSV files contain their images;
only ScienceQA and SEED-Bench need separate image archives.

## Download all seven

Run these blocks in the same Bash shell. Start from the workspace root:

```bash
cd /raid/lvchengtao/Minh/Distillation/reproduction/GRACE/evaluation
conda activate grace
```

You need `hf`, `curl`, and `unzip`. If `hf` is missing, install the CLI in a
separate small environment to avoid changing the training environment:

```bash
python -m venv /tmp/grace-dataset-download
/tmp/grace-dataset-download/bin/python -m pip install huggingface_hub
hf() { /tmp/grace-dataset-download/bin/hf "$@"; }
```

The [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
preserves repository-relative paths under `--local-dir`. This is why the
HallusionBench/AI2D destinations omit `data`, while the MMMU destination includes
it. These commands download only the files used by the seven-suite launcher.

```bash
set -euo pipefail

# 1. HallusionBench: image AND non-image partitions.
hf download lmms-lab-encoder/HallusionBench --repo-type dataset \
  --revision cd417161857aefb23d878d42cf1bb53aa9dd646f \
  --include 'data/image-*.parquet' \
  --include 'data/non_image-*.parquet' \
  --local-dir HallusionBench

# 2. MMBench: use the exact labeled English dev release expected by the launcher.
mkdir -p MMBench
curl -fL --retry 3 -C - \
  'https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_en_20231003.tsv' \
  -o MMBench/mmbench_dev_en_20231003.tsv

# 3. ScienceQA: metadata plus test images only.
mkdir -p ScienceQA/data/images
curl -fL --retry 3 -C - \
  'https://raw.githubusercontent.com/lupantech/ScienceQA/main/data/scienceqa/problems.json' \
  -o ScienceQA/data/problems.json
curl -fL --retry 3 -C - \
  'https://raw.githubusercontent.com/lupantech/ScienceQA/main/data/scienceqa/pid_splits.json' \
  -o ScienceQA/data/pid_splits.json
curl -fL --retry 3 -C - \
  'https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip' \
  -o ScienceQA/data/images/test.zip
unzip -n -q ScienceQA/data/images/test.zip -d ScienceQA/data/images

# 4. AI2D: test shards only; this loader reads EVERY Parquet in AI2D/data.
hf download lmms-lab-encoder/ai2d --repo-type dataset \
  --revision c83a9b9692933aff8349157c88a413df9d02c4e5 \
  --include 'data/test-*.parquet' --local-dir AI2D

# 5. MMMU: retain subject folders, download validation only.
hf download MMMU/MMMU --repo-type dataset \
  --revision 98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68 \
  --include '*/validation-*.parquet' --local-dir mmmu/data

# 6. Original SEED-Bench: annotations and image archive, no video chunks.
hf download AILab-CVC/SEED-Bench \
  SEED-Bench.json SEED-Bench-image.zip --repo-type dataset \
  --revision 81cb1571b8a3a30af2e62073b8d2054418350f0a \
  --local-dir SEEDBench/data
unzip -n -q SEEDBench/data/SEED-Bench-image.zip -d SEEDBench/data

# 7. MMStar: the TSV, not the alternative Parquet distribution.
hf download Lin-Chen/MMStar MMStar.tsv --repo-type dataset \
  --revision bc98d668301da7b14f648724866e57302778ab27 \
  --local-dir MMStar
```

With the installed `huggingface_hub==1.15.0`, repeat `--include` for each
pattern. Passing two patterns after one `--include` makes the second a literal
filename: the CLI warns that it is ignoring `--include`, then requests a path
containing `*` and fails. The command above was checked against the installed
CLI parser with downloads mocked. Use ordinary `--revision` / `--include`
options and `non_image`, without Markdown escape backslashes. Only the
end-of-line backslashes are needed for Bash continuation.

Hugging Face downloads can be rerun to reuse downloaded files; `curl -C -`
requests resumption and `unzip -n` preserves existing extracted files. If a
server refuses resumption, download to a fresh temporary filename and replace
the incomplete file only after the download succeeds. Keep the same revisions
and files for all three models. The ScienceQA metadata URLs follow `main`;
retain your downloaded copies for repeat comparisons.

The SEED image ZIP alone is about 1.68 GB; allow additional space for extracted
images and the other suites. Its video archives are unnecessary for this run.
ScienceQA's test ZIP is about 206 MB and the MMBench TSV about 136 MB. Download
and extraction use CPU only.

## Validate the layout

From `evaluation/`, after downloading:

```bash
# Clear a ShareGPT4V DATA_ROOT carried over from training, if present.
unset DATA_ROOT
bash run_all_constrained_evals.sh --check
```

This checks **both datasets and all three completed model exports** on CPU.
If the models are not ready, model errors are expected even with complete data.
`--dry-run` prints commands but does not check files. Individual path overrides
such as `MMMU_DIR` or `SEED_IMAGE_DIR` also take precedence over defaults; clear
any old overrides if they point elsewhere. `--check` checks file presence and
ScienceQA/SEED image references, not every Parquet/TSV row or file checksum.

Check these common layout mistakes:

- ScienceQA must have `ScienceQA/data/images/test/<pid>/image.png`, with exactly
  one `test` directory. Missing images otherwise become text-only questions in
  the evaluator; the launcher catches this before inference.
- SEED images must resolve as `SEEDBench/data/SEED-Bench-image/<data_id>`.
  Do not substitute SEED-Bench-2, SEED-Bench-2-Plus, or a TSV conversion.
- MMMU must have `mmmu/data/Accounting/validation-*.parquet`, etc., without an
  extra nested `data` directory. `MMMU_DEV_VAL.tsv` belongs to `run_mmmu.py`, a
  separate workflow that this launcher does not call.
- MMBench's newer VLMEvalKit TSV can encode repeated images as row references.
  This local loader directly decodes base64, so use the dated TSV above rather
  than substituting a different release. See the [MMBench format notes](https://github.com/open-compass/MMBench#data).
- AI2D must contain only test shards. Downloading a whole repository into an
  already populated data folder does not remove unwanted shards.

Once model and data checks pass, use the launch instructions in
[README_constrained.md](../qwen-vl-finetune/scripts/README_constrained.md#evaluate-the-actual-exported-models).
These downloads satisfy the local loaders; the local scoring procedures still
determine whether a result is comparable to an official benchmark leaderboard.

## Which evaluator to use

Use the local launcher for an initial SFT/QAT/GRACE comparison using these
downloaded files. Keep its results labeled as the local evaluation protocol.
For the main benchmark report, prefer a pinned lmms-eval revision and task
configuration as described in [README_lmms_constrained.md](README_lmms_constrained.md).
Its default tasks do not automatically consume this directory layout; reusing
these assets requires explicit local dataset configuration and schema checks.
Having downloaded the data does not determine which scoring protocol to use.

The GRACE upstream README claims lmms-eval, but the supplied ScienceQA Slurm
example runs a custom `eval_scienceqa.py`. Neither route alone establishes
exact reproduction of its reported scores. Match subsets, prompts, image
resolution, decoding, answer extraction and metrics before comparing results.

## Compare completed results

From the workspace root, combine all 21 merged result files on CPU:

```bash
python3 reproduction/GRACE/evaluation/summarize_constrained_evals.py --write
```

This prints all seven suite scores, HallusionBench group metrics, ScienceQA
image/text scores, and error/parse-failure counts for SFT, QAT, and GRACE.
It checks saved scores against individual predictions and verifies matching
question identities and gold labels across methods. SEED identity includes
question type because its source lists one question under two types.
The command writes `comparison.md` and `comparison.csv`, including every saved
category breakdown, to `experiments/constrained_32k_seed42/evaluation/`.
Omit `--write` to only display results. For another run, use
`--results-dir /path/to/evaluation`; `EXPERIMENT_DIR` and `OUT_DIR` also work.

These benchmark scores are in `evaluation/{method}/{suite}.json`, under
`summary`; `profiles/{method}/run_summary.json` describes training profiling,
not benchmark evaluation. The seven-suite mean is an unweighted descriptive
average using HallusionBench aAcc and full-test ScienceQA. Treat SFT as the
BF16 quality reference and QAT versus GRACE as the matched quantization
comparison; retain the local-protocol qualifications above.

## Other evaluation folders (optional)

These suites are present in this repository but are **not called** by
`run_all_constrained_evals.sh`.

| Folder | Source and preparation for its local loader |
| --- | --- |
| `OCRBench` | Download `data/test-*.parquet` from [echo840/OCRBench](https://huggingface.co/datasets/echo840/OCRBench/tree/main/data) into `OCRBench`; pass `--data_dir OCRBench/data`. |
| `MathVista` | Download `data/testmini-*.parquet` from [AI4Math/MathVista](https://huggingface.co/datasets/AI4Math/MathVista/tree/main/data) into `MathVista`; pass `--data_dir MathVista/data --split testmini`. |
| `RealWorldQA` | Its [dataset_utils.py](RealWorldQA/dataset_utils.py) downloads [RealWorldQA.tsv](https://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv) automatically into `--data-dir` / `LMUData`. Use `evaluation/RealWorldQA/data` as that directory. |
| `MathVision` | Its [dataset_utils.py](MathVision/dataset_utils.py) downloads [MathVision.tsv](https://opencompass.openxlab.space/utils/VLMEval/MathVision.tsv) or [MathVision_MINI.tsv](https://opencompass.openxlab.space/utils/VLMEval/MathVision_MINI.tsv) into `--data-dir` / `LMUData`. Use `evaluation/MathVision/data`. |
| `VideoMME` | See the [video benchmark guide](#3-video-mme-broad-video-reasoning-across-durations) below for pinned annotation/video downloads and the `data/` versus `videos/` layout required by the local loader. |
| `ODinW-13` | [Microsoft GLIP](https://github.com/microsoft/GLIP#download-odinw) provides `python odinw/download_datasets.py`; use its `configs/odinw_13` to identify the 13 tasks. Put extracted COCO datasets under `ODinW-13/data` following its [local README](ODinW-13/README.md#data-preparation). The local evaluator also requires `odinw13_config.py` containing `datasets` and `dataset_prefixes`, which is not supplied here. This requires configuration work beyond downloading; the same missing-config issue is [reported upstream](https://github.com/QwenLM/Qwen3-VL/issues/1904). |

For the two additional Parquet suites, from `evaluation/`:

```bash
hf download echo840/OCRBench --repo-type dataset \
  --include 'data/test-*.parquet' --local-dir OCRBench
hf download AI4Math/MathVista --repo-type dataset \
  --include 'data/testmini-*.parquet' --local-dir MathVista
```

Use each optional suite's own README and requirements for execution. Downloading
these files does not add the suites to the constrained launcher.

## Video-reasoning benchmarks for LLMC+ and token reduction

These are additional evaluations for the LLMC+ / token-reduction study, separate
from the seven image suites used for the GRACE reproduction. Project pages,
Hugging Face file inventories/revisions, and access gates were checked on
**2026-10-15**. Small directory ranges of public ZIPs were inspected to verify
extraction paths; full video downloads, decoding, and model inference were not run.

### What to download first

Our recommended starting set is **TempCompass multi-choice + Video-MME without
subtitles**, then **LongVideoBench validation** for long-context reasoning.
TempCompass isolates temporal sensitivity; the longer suites test whether token
reduction retains evidence spread across a video. MVBench adds task-level
diagnostics, MLVU adds long-video counting/order/retrieval, and Video-MMMU adds
knowledge acquisition. This prioritization is our experimental recommendation.

| Benchmark / evaluation scope | Why it is useful here | Download and access |
| --- | --- | --- |
| [TempCompass](https://llyx97.github.io/tempcompass/), multi-choice `test` | Action, direction, speed, event order, attribute change; conflicting clips share static content but differ temporally. A small first check for temporal token loss. | **0.42 GB** video ZIP; public, no gate. |
| [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), 20 tasks × 200 questions | Action sequence/prediction, counting, motion and state changes; report individual tasks as well as the mean. | **17.3 GB** public ZIPs, **incomplete for all 20 tasks**: NTU fine-grained-pose videos require separate access. |
| [Video-MME](https://github.com/MME-Benchmarks/Video-MME), 2,700 questions / 900 videos | Broad video understanding/reasoning; short <2 min, medium 4–15 min, long 30–60 min. Useful duration breakdowns. | **101.0 GB** video ZIPs; public. Optional subtitles add 7.5 MB. |
 
| [LongVideoBench](https://longvideobench.github.io/), **1,337 validation** questions | Referring reasoning across a long video, up to one hour; useful for evidence retention across distant frames. | **161.6 GB** multipart video TAR plus 0.12 GB subtitles; HF access acceptance required. Test labels are withheld. |
| [MLVU](https://github.com/JUNJIE99/MLVU), **Dev multiple-choice**, seven tasks | Plot/topic reasoning, needle retrieval, egocentric reasoning, action counting/order, anomaly recognition; videos span 3 min–2 h. | **291.2 GB** for the selected seven video folders; HF access acceptance required. Download individual MP4s, no extraction. |
| [Video-MMMU](https://videommmu.github.io/), 900 questions / 300 source lectures | Perception, comprehension and adaptation of information from educational videos. Complementary to motion-focused tasks. | **13.5 GB** including six discipline ZIPs, annotations and question-only videos; HF access acceptance required. |

Sizes above are decimal GB of downloads, calculated from the pinned repository
inventories. ZIP/TAR suites also need space for extraction: budget roughly twice
the archive size plus a margin when retaining archives. MLVU's figure excludes
its two generation-task folders. Download one suite at a time if space is tight.
The gates and filenames are visible in the corresponding HF listings linked in
the instructions below. Follow each source's dataset-use terms.

Long videos alone do not guarantee temporal reasoning. For the study, pair a
temporal diagnostic with a long-video suite, and keep temporal-category and
duration results visible instead of reporting only one overall accuracy.

### Shared download setup

Run from `evaluation/`; use the `hf` CLI setup near the top of this README if
needed. These commands download only the selected suite when its block is run.
All new assets live under ignored `data/` directories, separate from source code:

```bash
cd /raid/lvchengtao/Minh/Distillation/reproduction/GRACE/evaluation
set -euo pipefail
export VIDEO_BENCH_ROOT="$PWD/video_benchmarks"
mkdir -p "$VIDEO_BENCH_ROOT"
df -h "$VIDEO_BENCH_ROOT"
```

For **LongVideoBench, MLVU, or Video-MMMU**, first open the respective HF dataset
page, sign in, and accept its access conditions with your own account. Then run
`hf auth login` interactively with a token that can read the accepted repository.
A public file listing does not mean its contents are ungated. A 401/403 usually
means authentication/access is missing; downloading annotations alone does not
supply the videos. The public TempCompass and Video-MME downloads need no login.

Keep `--revision` values fixed across methods. `hf download` can be rerun to reuse
completed files. `unzip -n` preserves extracted files; if extraction was
interrupted, verify the archives and re-extract into a fresh destination rather
than trusting partial files. Do not concatenate independently readable ZIP chunks.

### 1. TempCompass: first temporal diagnostic

The [official preparation instructions](https://github.com/llyx97/TempCompass#data-preparation)
link to the processed clips, including reversed/conflicting variants. Use these
instead of rebuilding clips from upstream web videos. The current
[HF repository](https://huggingface.co/datasets/lmms-eval/TempCompass/tree/main)
is `lmms-eval/TempCompass`; `lmms-lab/TempCompass` redirects there.

```bash
hf download lmms-eval/TempCompass --repo-type dataset \
  --revision c8a67d88a3bc6fd4b2f9ae2f9e112668fbe05722 \
  --include README.md --include 'multi-choice/test-*.parquet' \
  --include tempcompass_videos.zip \
  --local-dir "$VIDEO_BENCH_ROOT/TempCompass/data"
unzip -n -q "$VIDEO_BENCH_ROOT/TempCompass/data/tempcompass_videos.zip" \
  -d "$VIDEO_BENCH_ROOT/TempCompass/data"
```

Expected paths: `TempCompass/data/multi-choice/test-*.parquet` and
`TempCompass/data/videos/<video_id>.mp4`. The annotation columns include
`video_id`, `question`, `answer`, and `dim`. This block selects **multi-choice
only**. Other formats have separate annotation directories in the same HF repo.
The [official evaluator](https://github.com/llyx97/TempCompass#-run-evaluation-)
supports `--disable_llm` for multi-choice/yes-no/caption matching. The pinned
lmms-eval implementation can still call a judge on parsing failure; choose and
record a scoring policy before running, rather than assuming MC implies no API.

### 2. MVBench: temporal task breakdowns, with a missing-data caveat

Download the original annotations and all currently available
[public video ZIPs](https://huggingface.co/datasets/OpenGVLab/MVBench/tree/main/video):

```bash
hf download OpenGVLab/MVBench --repo-type dataset \
  --revision 230a2d4fac8900333c61754641c7a13e069ac9c6 \
  --include README.md --include 'json/*.json' --include 'video/*.zip' \
  --include video/MVBench_videos_ntu.txt \
  --local-dir "$VIDEO_BENCH_ROOT/MVBench/data"
for archive in "$VIDEO_BENCH_ROOT"/MVBench/data/video/*.zip; do
  unzip -n -q "$archive" -d "$VIDEO_BENCH_ROOT/MVBench/data/video"
done
# The supplement contains additional Charades/CLEVRER files; merge it too.
cp -an "$VIDEO_BENCH_ROOT/MVBench/data/video/data0613/." \
  "$VIDEO_BENCH_ROOT/MVBench/data/video/"
```

Keep the extracted subdirectories: for example,
`video/star/Charades_v1_480/`, `video/clevrer/video_validation/` and
`video/ssv2_video/`. Keep `json/<task>.json` separate. Respect annotation `start`
and `end` bounds when sampling clips; TVQA episodic reasoning uses frame folders.
Use the [official evaluation notebook](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/mvbench.ipynb)
for the task-to-folder and video/frame mapping.

**The download is not a complete 20-task evaluation.** `nturgbd.zip` is absent
from the pinned release, which instead supplies
[`MVBench_videos_ntu.txt`](https://huggingface.co/datasets/OpenGVLab/MVBench/blob/main/video/MVBench_videos_ntu.txt).
Obtain the listed NTU RGB+D clips through the
[NTU access process](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
for fine-grained pose. Until then, explicitly exclude/report that task and name
the evaluated subset; never score missing videos as predictions or label a
partial mean as full MVBench. Validate every other selected annotation-to-asset
reference too. The pinned harness's `Charades_segment` / `ssv2_video_mp4` paths
differ from these original archives and need an explicit adapter.

### 3. Video-MME: broad video reasoning across durations

The [official project](https://github.com/MME-Benchmarks/Video-MME#-dataset)
points to [these annotations and archives](https://huggingface.co/datasets/lmms-eval/Video-MME/tree/main).
All 20 ZIPs are needed for the full benchmark; their indices are not duration
splits. Downloading the tiny Parquet alone does not download videos.

```bash
hf download lmms-eval/Video-MME --repo-type dataset \
  --revision ead1408f75b618502df9a1d8e0950166bf0a2a0b \
  --include README.md --include 'videomme/test-*.parquet' \
  --include 'videos_chunked_*.zip' \
  --local-dir "$VIDEO_BENCH_ROOT/VideoMME/data"
for archive in "$VIDEO_BENCH_ROOT"/VideoMME/data/videos_chunked_*.zip; do
  unzip -n -q "$archive" -d "$VIDEO_BENCH_ROOT/VideoMME/data"
done
# ZIP members are data/<videoID>.mp4; the local GRACE loader expects videos/.
ln -s data "$VIDEO_BENCH_ROOT/VideoMME/data/videos"
```

The last command is a one-time symlink; if `videos` already exists, check that it
resolves to the extracted `data` directory instead of replacing it. Final paths:
`VideoMME/data/videomme/test-*.parquet`, `VideoMME/data/data/<videoID>.mp4`, and
the alias `VideoMME/data/videos/<videoID>.mp4`. Point the existing local
[`run_videomme.py`](VideoMME/run_videomme.py) `--data-dir` at
`"$VIDEO_BENCH_ROOT/VideoMME/data"`, not at its nested `data/` directory.

For a separate **with-subtitles** experiment, additionally run:

```bash
hf download lmms-eval/Video-MME subtitle.zip --repo-type dataset \
  --revision ead1408f75b618502df9a1d8e0950166bf0a2a0b \
  --local-dir "$VIDEO_BENCH_ROOT/VideoMME/data"
unzip -n -q "$VIDEO_BENCH_ROOT/VideoMME/data/subtitle.zip" \
  -d "$VIDEO_BENCH_ROOT/VideoMME/data"
```

This creates `subtitle/<videoID>.srt`; subtitles are not present for every clip.
Report short/medium/long separately and label subtitle use. Official MC scoring
is rule based and does not require a judge; the local `VideoMME/` README's
judge-based workflow is a different protocol. See the
[official scoring and subtitle rules](https://github.com/MME-Benchmarks/Video-MME#-evaluation-pipeline).

### Download LongVideoBench, MLVU, and Video-MMMU together, with ntfy alerts

[`download_video_benchmarks.sh`](download_video_benchmarks.sh) downloads these
three suites sequentially at the revisions below, trying **Video-MMMU first,
then LongVideoBench, then MLVU**. It extracts the TAR/ZIP archives and retains
them for reruns. It uses the adjacent Python standard-library helper,
plus `hf`, `tar`, and `unzip`; it does not run inference. First accept **all three**
HF access gates linked in sections 4–6, then run `hf auth login`.

Subscribe to **`download-videobenchmarks-status`** on `https://ntfy.sh` in your
phone's ntfy app. Override with `--topic`
or `NTFY_TOPIC`; `--ntfy-url` / `NTFY_URL` selects another server. An optional
`NTFY_TOKEN` environment variable supplies bearer authentication. The notification
API follows the [ntfy publishing documentation](https://docs.ntfy.sh/publish/).

From the workspace root:

```bash
# Check notification delivery without downloading anything.
bash reproduction/GRACE/evaluation/download_video_benchmarks.sh --test-notification

# Run inside tmux so an SSH disconnect does not stop the download.
tmux new -s video-downloads
bash reproduction/GRACE/evaluation/download_video_benchmarks.sh \
  --datasets VideoMMMU

# Or try all three in the default order, checking space before each one.
bash reproduction/GRACE/evaluation/download_video_benchmarks.sh
```

Without `--root`, the destination is `evaluation/video_benchmarks` (or
`VIDEO_BENCH_ROOT`). Use `--root /path/to/video_benchmarks` to select another disk.
Before **each dataset's download**, the script checks free space on its destination
filesystem using these conservative budgets, including retained archives,
extracted videos/subtitles, and working headroom:

| Dataset | Total disk budget |
| --- | --- |
| Video-MMMU (`VideoMMMU`) | 30 GiB |
| LongVideoBench | 320 GiB |
| MLVU | 300 GiB |

Only already allocated files **inside that dataset's `data/` directory** are
subtracted from its budget; other datasets do not count toward its requirement.
Free space is checked again when advancing to the next dataset. If it cannot fit,
the script raises an error and sends an ntfy alert **before starting its download**;
earlier completed datasets remain available. With the approximately 171 GiB free
observed on 2026-10-15, Video-MMMU should fit, while a fresh LongVideoBench or MLVU
will fail preflight. `--datasets` selects suites in the given order. `--min-free-gb`
overrides the total budget **per selected dataset**; `0` disables space checks.
These budgets are estimates; other jobs can still consume disk space during a run.

Failure alerts include hostname, dataset/stage, exit status or caught error,
recent command output, and the full log path. This preserves messages such as
`No space left on device`, network errors, and HF access denials. A child killed
with SIGKILL is reported as a possible OOM/external kill; the script cannot
determine the cause from that signal alone. It also sends a completion alert.
`SIGINT`, `SIGTERM`, and `SIGHUP` stop the active process group and trigger an
alert. A watchdog stops the job and alerts after **30 minutes without data-file
changes or child disk I/O**, even if a CLI spinner keeps printing. Change this
with `--stall-minutes`; the check interval defaults to 30 seconds.

Logs live in `<root>/logs/`, with a `/tmp` fallback if opening a log fails. If
ntfy cannot be reached, three bounded attempts are made and the alert is queued
under `/tmp/video-download-<uid>-<root-id>-pending_ntfy.jsonl`, then retried at the
next invocation with the same root (including `--test-notification`). This queue
does not contain auth tokens, and HF/ntfy tokens are redacted from logged output.
Queued delivery needs the network to recover and the script to run again;
`/tmp` may be cleared on reboot. **Power loss, server loss, SIGKILL of the
supervisor, or a machine-wide OOM cannot be reported by this local script.**
Those cases require monitoring from another machine.

Rerun the same command to reuse HF downloads. Successfully completed archive
extractions are skipped using completion markers. Interrupted extractions run
again and overwrite partial files. Delete a corresponding `.extracted-*` marker
to force re-extraction if extracted assets were manually removed. The script
checks basic required files; it does not validate every annotation reference or
decode videos. Keep the validation steps below before evaluation.

```bash
# Print the pinned download plan without writes, network calls, or alerts.
bash reproduction/GRACE/evaluation/download_video_benchmarks.sh --dry-run

# See all options (including --max-workers, default 4).
bash reproduction/GRACE/evaluation/download_video_benchmarks.sh --help
```

### 4. LongVideoBench: long-context referring reasoning

Accept access at [the official HF release](https://huggingface.co/datasets/longvideobench/LongVideoBench/tree/main)
before downloading. Use labeled **validation** for local comparisons. The
[official loader instructions](https://github.com/longvideobench/LongVideoBench#custom-use-load-the-longvideobench-dataset)
describe the JSON and video/subtitle layout.

```bash
hf download longvideobench/LongVideoBench --repo-type dataset \
  --revision 60d1c89c1919a198b73be39c2babb213b29d6a5c \
  --include README.md --include lvb_val.json \
  --include 'validation-*.parquet' --include subtitles.tar \
  --include 'videos.tar.part.*' \
  --local-dir "$VIDEO_BENCH_ROOT/LongVideoBench/data"
# These ARE split pieces of one TAR. Concatenate in lexical order and stream
# extraction, avoiding an additional ~161.6 GB joined TAR on disk.
LC_ALL=C cat "$VIDEO_BENCH_ROOT"/LongVideoBench/data/videos.tar.part.* \
  | tar -xf - -C "$VIDEO_BENCH_ROOT/LongVideoBench/data"
tar -xf "$VIDEO_BENCH_ROOT/LongVideoBench/data/subtitles.tar" \
  -C "$VIDEO_BENCH_ROOT/LongVideoBench/data"
```

Expect `lvb_val.json`, `validation-*.parquet`, `videos/`, and `subtitles/` under
`LongVideoBench/data`. All **31** TAR parts (`aa` through `be`) are required even
for validation, since videos are packaged jointly with test assets. No test
annotations are requested here. The official interleaved frame/subtitle protocol
and a video-only run are different settings; record which one is used.

### 5. MLVU: long-video counting, order and retrieval

Use the **Dev multiple-choice** tasks for a direct accuracy comparison. Accept
access at [MLVU/MVLU](https://huggingface.co/datasets/MLVU/MVLU/tree/main)
(the official repo ID is spelled `MVLU`). Its nine-task release includes two
generation tasks; the following selects only task folders 1–7:

```bash
hf download MLVU/MVLU --repo-type dataset \
  --revision 06ddc388aa34746b3abba77972b7a7dfd977f7a3 \
  --include README.md --include 'MLVU/json/[1-7]_*.json' \
  --include 'MLVU/video/[1-7]_*/*' \
  --local-dir "$VIDEO_BENCH_ROOT/MLVU/data"
```

Files remain at `MLVU/data/MLVU/json/<task>.json` and
`MLVU/data/MLVU/video/<task>/<clip>.mp4`, relative to `VIDEO_BENCH_ROOT`.
There is no ZIP to extract. The seven folders total about **291.2 GB**; inspect
available disk space before running. Preserve their names when resolving JSON
video references. Use the [official evaluation code](https://github.com/JUNJIE99/MLVU/tree/main/evaluation)
and report the prescribed task mean (**M-Avg**), not a mixture with generation
scores. The pinned harness's `mlvu_dev` instead uses `sy1998/MLVU_dev` and its
`video_name` schema; downloading the original release does not populate that
mirror's cache layout. Dev and Test are distinct releases/protocols.

### 6. Video-MMMU: knowledge acquisition from lectures

Accept access at [lmms-eval/VideoMMMU](https://huggingface.co/datasets/lmms-eval/VideoMMMU/tree/main).
This is separate from the image-only MMMU dataset downloaded earlier.

```bash
hf download lmms-eval/VideoMMMU --repo-type dataset \
  --revision d1c35ac933123d79e877b7f1b9506afb0309cf1b \
  --include README.md --include '*/test-*.parquet' --include '*.zip' \
  --local-dir "$VIDEO_BENCH_ROOT/VideoMMMU/data"
for archive in "$VIDEO_BENCH_ROOT"/VideoMMMU/data/*.zip; do
  unzip -n -q "$archive" -d "$VIDEO_BENCH_ROOT/VideoMMMU/data"
done
```

The six discipline ZIPs supply Art, Business, Engineering, Humanities, Medicine
and Science videos. Annotations are under `Perception/`, `Comprehension/` and
`Adaptation/`; `question_only_videos.zip` supports the adaptation baseline.
Validate extracted paths against annotation IDs before evaluation. Preserve the
provided video endings containing question images; blindly dropping the final
frames changes this task. Report the three stages separately. The
[knowledge-gain metric](https://videommmu.github.io/) also needs the question-only
baseline; it cannot be inferred from video-conditioned accuracy alone.

### Check data and connect it to evaluation

Before inference, check archive integrity (`unzip -tq <archive.zip>`), then verify
**every selected question's referenced asset**, not just the number of files.
For example, after downloading TempCompass and Video-MME, this CPU-only check
uses `pandas`/`pyarrow` from the evaluation environment:

```bash
python - <<'PY'
import os
from pathlib import Path
import pandas as pd

root = Path(os.environ["VIDEO_BENCH_ROOT"])
for suite, pattern, column, folder, expected in [
    ("TempCompass", "multi-choice/test-*.parquet", "video_id", "videos", None),
    ("VideoMME", "videomme/test-*.parquet", "videoID", "videos", 2700),
]:
    base = root / suite / "data"
    shards = sorted(base.glob(pattern))
    assert shards, f"Missing annotations: {base / pattern}"
    rows = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    if expected is not None:
        assert len(rows) == expected, (suite, len(rows), expected)
    assets = {base / folder / f"{video_id}.mp4" for video_id in rows[column]}
    missing = sorted(str(p) for p in assets if not p.is_file() or p.stat().st_size == 0)
    assert not missing, f"{suite}: {len(missing)} missing/empty videos; first: {missing[:5]}"
    print(f"{suite}: {len(rows)} questions, {len(assets)} referenced videos present")
PY
```

This checks references and nonempty files, not successful decoding. Before a
full run, decode sampled frames on CPU with the actual evaluation backend and
check timestamps and clip bounds. Missing/corrupt videos should fail preflight;
do not silently skip questions and change the evaluation denominator.

The local `run_all_constrained_evals.sh`, `run_lmms_matrix.py`, and
`run_upstream_matrix.py` currently select the seven **image** suites. None of
these downloads adds video tasks to those wrappers. The vendored
`third_party/lmms-eval-v0.7.1` does contain these task definitions:

| Data | Pinned harness task(s) | Integration detail to resolve before inference |
| --- | --- | --- |
| TempCompass MC | `tempcompass_multi_choice` | Visual helper expects `$HF_HOME/tempcompass/videos/`; judge fallback is possible. |
| MVBench | `mvbench` group, or `mvbench_action_sequence`, etc. | Expects `$HF_HOME/mvbench_video/`; fix original-archive path differences explicitly and obtain NTU assets for all 20 tasks. |
| Video-MME | `videomme`, `videomme_w_subtitle` | Expects `$HF_HOME/videomme/data/` videos and `subtitle/`; local `VideoMME/` instead uses `--data-dir` and `videos/`. |
| LongVideoBench | `longvideobench_val_v`, `longvideobench_val_i` | Base cache is `$HF_HOME/datasets/longvideobench`; verify `video_path`/`subtitle_path` resolution and subtitle behavior for the selected task. |
| MLVU Dev MC | `mlvu_dev` | Uses the `sy1998/MLVU_dev` mirror and `$HF_HOME/mlvu/<video_name>`; needs mapping from the original release. |
| Video-MMMU | `videommmu` group | Expects `$HF_HOME/video_mmmu/<discipline>/<id>.mp4`; question-only adaptation has a separate task. |

These names were checked against the local task YAMLs; they are **integration
targets**, not a claim that the current wrappers can run them. `hf --local-dir`
does not fill the harness's dataset/asset caches automatically. A video runner
must explicitly select the local annotation files/revisions and asset roots,
audit the schema/scorer, and exercise the Qwen3-VL video processor and token
reducer together. Keep that work separate from the existing paper-reproduction
protocol; see [README_lmms_constrained.md](README_lmms_constrained.md).

For the LLMC+ comparison, hold source frames/timestamps, frame resolution,
subtitle/audio policy, prompts, decoding and scoring fixed. Compare the BF16
reference, LLMC+ alone, token reduction alone, and their combination. Start with
Video-MME without subtitles and label any subtitle ablation separately. Record
sampled frames, visual tokens **before and after reduction**, reduction location,
peak GPU memory, prefill latency, decode throughput and end-to-end time (including
video decoding). Sweep input frame budgets, for example 32/64/128, with the same
inputs across methods at each budget. Report quality versus token count and
latency: increasing video duration while holding the sampled frame count fixed
does not itself increase the model's visual-token workload.
