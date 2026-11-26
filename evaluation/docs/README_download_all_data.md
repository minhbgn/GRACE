# Download all GRACE workflow data

The self-contained Bash script [`download_all_data.sh`](../data_tools/download_all_data.sh)
contains its Python implementation. It downloads the seven image evaluation
suites, TempCompass MC, Video-MME **short + medium only**, Video-MMMU, and the
ShareGPT4V annotations/images described in the
[original GRACE guide](https://github.com/ForeverBlue816/GRACE#data-preparation).
No inference is performed.

## Run

From the workspace root:

```bash
DL=reproduction/GRACE/evaluation/data_tools/download_all_data.sh

# Preview sources, revisions and destinations. No writes or network requests.
bash "$DL" --dry-run

# First run: prepare a small isolated downloader environment, then download.
# Supply the fresh official SAM URL list described below.
bash "$DL" --install-deps --sam-links /path/to/sa1b_links.txt

# Resume later. Completed outputs are checked/skipped; partial transfers resume.
bash "$DL" --sam-links /path/to/sa1b_links.txt
```

Use `tmux` for a long download session. The default environment is
`reproduction/GRACE/.data-download-venv`; installation uses system site packages
from the selected base Python and installs any missing downloader dependencies
there. It does not install into the training Conda environment. `DATA_PYTHON`
selects a Python 3.10+ executable; `GRACE_DATA_ENV` selects the downloader venv.
Dependencies are requests, huggingface_hub, pyarrow, Pillow, ijson, and gdown 5.2.0.
`--install-deps` installs them and continues downloading; omit it on subsequent
runs. `--help` and `--dry-run` do not install dependencies.

Video-MMMU requires access approval at
[its dataset page](https://huggingface.co/datasets/lmms-eval/VideoMMMU).
Authenticate with `hf auth login` or supply `HF_TOKEN` in your environment.
The script also reads the existing Hugging Face login. Tokens are not written
into progress files or command arguments. Google Drive quota/access errors are
reported as deferred sources, and later datasets are still attempted.

## Download order and paths

The default order is:

1. HallusionBench, MMBench EN dev, ScienceQA test, AI2D test, MMMU validation,
   SEED-Bench-IMG, MMStar.
2. TempCompass multi-choice, Video-MME short/medium, Video-MMMU's three stages
   and the supplied question-only baseline files.
3. Both ShareGPT4V annotation JSONs, LLaVA pretraining images, COCO train2017,
   GQA images, TextVQA train/val images, both Visual Genome image archives,
   WebData, OCR-VQA, and SAM shards 000000–000050.

Evaluation files use the existing layouts in
[`README_datasets.md`](README_datasets.md). Training files default to
`reproduction/GRACE/ShareGPT4V`, with annotations at the root and images under
`data/`. This matches `run_constrained.py` and the original GRACE data guide.
Set `--training-root` or `SHAREGPT4V_ROOT` for another training-data directory;
set `--root` to move the GRACE/evaluation root.

LongVideoBench, MLVU, MVBench, and the optional image benchmarks outside the
seven-suite workflow are not selected. No Video-MME subtitles are downloaded.

Useful selections:

```bash
bash "$DL" --only evaluation
bash "$DL" --only training --sam-links /path/to/sa1b_links.txt
bash "$DL" --datasets tempcompass videomme videommmu
bash "$DL" --datasets scienceqa seedbench training-annotations
```

`--datasets` preserves the given order. When selecting individual training
sources, download the annotations first if OCR-VQA or the image-reference audit
is needed.

## Video-MME long exclusion

The original Video-MME ZIP chunks mix duration categories; choosing a few chunk
numbers does not select short/medium clips. The script downloads the pinned
annotation Parquet, validates its 900/900/900 question split, and selects the
600 unique short/medium video IDs.

It reads each ZIP's central directory using HTTP ranges and downloads only those
600 members' compressed bytes. It verifies each decompressed video's size/CRC,
then deletes that member's temporary compressed data. It retains the full
annotation file so existing loaders can still validate their source population;
this does not download the long videos. Evaluation should continue using the
existing short/medium population filter.

If the server refuses byte ranges, the script defers Video-MME. It never falls
back to downloading the complete mixed-duration ZIPs. Index requests and the
actual archive range support were checked on 2026-10-27; large media were not
downloaded as part of script development.

## SAM and Google Drive training sources

The default follows the [original data preparation guide](https://github.com/ForeverBlue816/GRACE#data-preparation):
download both annotations and the 51 SAM shards numbered 000000 through 000050.
Obtain fresh URLs through the [official SAM download page](https://ai.meta.com/datasets/segment-anything-downloads/)
and pass a text file containing a filename and its URL on each line:

```text
sa_000000.tar<TAB>https://...
sa_000001.tar<TAB>https://...
...
sa_000050.tar<TAB>https://...
```

Whitespace-separated and literal `\t` separators are also accepted. Other shard
numbers are ignored. The workspace's `sa1b_links.txt` is automatically detected
when present. **The existing recovered list covers only 000020–000050**; it
cannot supply the missing first 20 shards. The script attempts listed shards
and reports absent or expired links as incomplete. It does not manufacture CDN
URLs or silently substitute a smaller training population.

For an explicitly SFT-only download, `--sam-mode sft9k` selects the small Google
Drive subset linked in the guide. It may leave SAM references missing from the
100K caption annotation, which is reported by the reference audit.

WebData and OCR-VQA use the Drive links from the GRACE guide. Folder discovery
is cached and uses gdown without executing the downloaded OCR-VQA script. The
implementation reads OCR-VQA's `dataset.json`, downloads the images referenced
by the two training annotations, and writes them as actual JPEGs at their
annotation paths. Unavailable original image URLs remain explicit failures.
The final audit streams the large annotations using ijson and reports every
missing image group with counts and examples. It does not drop training rows.

## Time and space policy

Each dataset now gets **up to one hour of network time** by default,
regardless of its remaining byte count. This replaces the previous default
`remaining bytes / 5,000,000 + 60 seconds`, which gave small resumed transfers
very little time. Metadata/media download phases share the dataset allowance.
Extraction and local integrity checking pause the download clock; metadata/index
discovery retains its separate 90-second limit. The one-hour limit applies to
each dataset, so a full run can take longer than one hour.

```bash
# Default: one hour per dataset, with partial transfers retained for resumption.
bash "$DL" --only training --download-timeout 3600

# Optional legacy size-derived policy, still capped at one hour.
bash "$DL" --budget-mbps 2 --grace-seconds 120 --download-timeout 3600
```

On timeout the script retains partial files and attempts the next dataset.
Socket timeouts and retries still apply; an unreachable host can fail before
the hour is used. The limit does not force a broken source to keep retrying.

WebData selects only `share_textvqa.zip`, `web-celebrity.zip`,
`web-landmark.zip`, and `wikiart.zip` from the shared Drive folder.
It skips `sam_images_share-sft.zip` (handled by SAM) and `playground.zip`
(evaluation resources). Previously `playground.zip` failed the training-path
filter and prevented the actual WebData ZIPs from being attempted.

The original Visual Genome host refused connections from this machine on
2026-10-29. A third-party Hugging Face ZIP mirror is available explicitly:

```bash
bash "$DL" --datasets training-vg --vg-source huggingface
```

This uses `pppop7/VisualGenome_VG_100K_1_and_2`, pins its resolved revision in
the download metadata, and checks each archive against its LFS SHA-256. The
default `--vg-source stanford` retains the original official URLs. Mirror
checks verify integrity against the mirror, not equivalence to the original.

Older LLaVA downloads placed numeric image shard directories directly under
`data/llava/llava_pretrain/`. Selecting `training-llava` repairs these into
`data/llava/llava_pretrain/images/` with directory renames and updates its
completion record, avoiding a second download. New extractions use that layout.

Here “memory” means **free disk capacity**. Downloading/extraction stream data
rather than holding archives in RAM. Before each dataset, the script estimates
peak disk use while accounting for completed outputs and archive cleanup.
Before each archive and during writes it checks space again. ZIP/TAR extraction
uses actual declared expanded sizes. The default keeps 10 GiB free on every
involved filesystem, including a separate state/archive filesystem when used.

```bash
bash "$DL" --reserve-gib 20
# Tune the conservative expanded-size estimate if required (minimum 1).
bash "$DL" --expansion-factor 2
```

Insufficient space defers that dataset and attempts the next one. Other jobs can
still consume disk space concurrently; checks cannot reserve capacity globally.

## Resume, verification, and cleanup

Progress lives under `reproduction/GRACE/.data-download/` by default. Use
`--state-dir` to change this directory; keep the same directory when resuming.
It contains partial archives, pinned remote inventories, per-source completion
records, and `latest_summary.json`. Do not delete it while a recovery is in
progress. Source completion is bound to its actual output destination.

Files download into partial paths and become final only after successful
transfer and available hash/length validation. Hugging Face LFS SHA-256 hashes
are checked. Resumed HTTP responses must honor the requested range and retain
the remote identity. A server refusing resume leaves the partial untouched.

Extracted ZIP entries receive size and CRC verification. TAR entries are checked
against declared sizes and recorded with SHA-256. Extraction rejects traversal,
symlinks and device entries. Output files are promoted atomically, and the
extraction completion manifest is committed **before** the compressed archive
is deleted. Failed/interrupted extraction retains its archive. Reruns verify
recorded output presence/size and changed timestamps; `--verify` forces full
recorded hash/CRC rechecking. Missing extracted outputs trigger re-download
rather than being treated as complete.

An exclusive lock prevents concurrent downloaders from sharing the same state
or output roots. Ctrl-C/SIGTERM/SIGHUP stop an active helper process group and
retain its partial data.

Exit codes: **0** selected sources completed; **2** one or more sources deferred,
or a full training download has missing image references; **130** interrupted.
For a subset of training sources, the global image audit is informational;
its missing unrelated sources do not change successful selected-source status.

```bash
cat reproduction/GRACE/.data-download/latest_summary.json
```

## Validation

Local HTTP tests exercise partial resumption, refusal of unsafe range fallbacks,
archive cleanup/recovery, ZIP/TAR path protection, disk-space deferral, aggregate
space accounting, hard timeouts with a slow byte stream, helper termination,
destination-bound completion records, SAM selection, and selective Video-MME
member transfer/CRC checking. No real datasets or GPUs are used by these tests.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/air/miniconda3/envs/grace/bin/python \
  -m unittest discover -s reproduction/GRACE/evaluation/tests \
  -p test_all_data_downloads.py -v
```

The live check verified every pinned evaluation file pattern and a four-byte
Video-MME archive range request. COCO uses the original HTTP URL: its HTTPS
endpoint failed certificate validation on this machine. Full source downloads
and gated Google Drive/HF access have not been end-to-end tested here.
