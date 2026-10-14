# Evaluation dataset downloads

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
| `VideoMME` | The [official project](https://github.com/MME-Benchmarks/Video-MME) links to [lmms-eval/Video-MME](https://huggingface.co/datasets/lmms-eval/Video-MME/tree/main), formerly `lmms-lab/Video-MME`. Download annotations under `videomme/`, the video ZIP chunks, and optionally `subtitle.zip`. Extract/arrange under `VideoMME/data` so `load_dataset(data_dir)['test']` can load annotations, videos are `videos/<videoID>.mp4`, and subtitles are `subtitle/<videoID>.srt`. This is a much larger download (roughly 100 GB of video archives); annotations alone are insufficient. |
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
