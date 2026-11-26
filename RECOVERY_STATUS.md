# GRACE recovery status — 2026-10-27

This is a partial reconstruction of the deleted working tree, not an exact filesystem rollback. The available upstream source, local code, reports, configuration, editor history, temporary artifacts, and cached benchmark data have been restored or made accessible. No training or benchmark inference was launched.

## Restored source and documents

- Upstream GRACE source and Git history, checked out at the parent repository's recorded commit `b73fa9b4d427aa64ed3ad274def641efb3145877`. This is newly downloaded baseline Git metadata, not the deleted local `.git` directory; the previous local HEAD `e0c06b915707be5a7f71e3625ffe25383378c31e` is unavailable upstream. Reconstructed research changes remain uncommitted.
- Local training utilities, evaluation modules and organized package layout, test files, experiment specifications, launchers and available reports reconstructed from historical file bodies, patches and restricted editing operations. Unsupported operations were retained as evidence, not executed as shell commands.
- All four open files in `experiments/qat_int4_token_reduction_576_seed42/`, its experiment specification, and its evaluation implementation.
- The current `released_bf16_init_32k_b128_p589824_t4096_seed42` directory, using the known rename from the historical `t2048` directory. Both historical and current paths are retained. The latest available editor version of its completed image experiment report was restored.
- Approximately 667 upstream/custom source and document files, plus 2,762 pinned dependency source files, were installed by the reconstruction. The available editor versions, temporary artifacts and evidence are additionally retained under `recovery/`.
- Pinned lmms-eval source under `third_party/lmms-eval-v0.7.1`: commit `88b23e2bfa16a1edbc16e9e238ed82130b3a4f56`. The complete restored source tree was verified against the original Git tree `bafe0aefd1099821063f40dbd7d72a67c8746fe0`. Its exact signed commit object was reconstructed from GitHub's original commit payload/signature; its Git metadata is shallow and its working tree is clean.

## Surviving data

`recovered_datasets/` links to the surviving shared-cache snapshots:

| Dataset | Available snapshot files | Apparent size |
| --- | ---: | ---: |
| ScienceQA | 3 | 0.245 GiB |
| AI2D | 3 | 0.130 GiB |
| MMStar | 2 | 0.039 GiB |
| SEED-Bench | 274 | 25.294 GiB |

These are live links into the existing shared cache, not independent backup copies. The additional roughly 28 GiB Arrow cache remains linked at `recovery/available_dataset_cache`; these caches can overlap, so do not add their sizes as unique recovered data. Cached subsets have not been silently substituted for the original evaluation populations.

Available saved reports, CSV/JSON analysis tables, validation logs, plans and temporary experiment output are preserved under `recovery/20261027/temporary_artifacts/` and `additional_temporary_artifacts/`. These do not imply that complete raw prediction sets survived.

## Verification

- All 220 non-archived, non-third-party Python source files passed syntax parsing.
- All 98 original/reconstructed shell source files passed `bash -n` before installation; subsequent editor overlays are separately checked.
- All 11 lmms regression tests passed on CPU.
- Seven of eight token-reduction tests passed, including packed-weight equivalence/cache reuse and tiny-model image/video generation/compaction checks. The processor-budget test errored because the original local Qwen processor/tokenizer directory is missing. No GPU execution was tested.
- The installed token-reduction launcher successfully generated its expected 360-job plan using the surviving Conda Python.
- The reconstructed historical `validation.json` is explicitly labeled: its historical claims are not evidence that missing checkpoints/data were restored, and its regenerated hashes are labeled as reconstructed hashes.

The deleted `.venv-lmms-eval` environment has not been recreated. The original Conda environment survives. For planning with the recovered launcher:

```bash
LMMS_PYTHON=/home/air/miniconda3/envs/grace/bin/python \
  bash reproduction/GRACE/experiments/qat_int4_token_reduction_576_seed42/run_experiment.sh plan
```

## Still missing or uncertain

- Trained QAT/GRACE/SFT weights, optimizer states and complete native checkpoints.
- Previously downloaded base/released model weights and processor/tokenizer assets; the checked model-cache entries contain references without snapshots.
- Original training subset manifests and training images not present in surviving artifacts; no usable ShareGPT4V Hub snapshot was found.
- Original video assets and the full deleted evaluation datasets/predictions beyond the surviving cached data and saved reports.
- The original isolated evaluation environment and original local Git history.
- Exact fidelity of every custom edit: 67 recorded operations were not fully replayed, including checks and report generators that required missing inputs. This is not a count of missing files. See the issue/provenance records before relying on a particular experiment as runnable.

## Recovery evidence and alternatives

- `recovery/20261027/reconstruction_provenance.json`: source provenance.
- `recovery/20261027/reconstruction_issues.json`: operations not fully replayed.
- `recovery/20261027/all_recovery_events.jsonl`: chronological source records.
- `recovery/20261027/editor_history/`: preserved editor versions.
- `recovery/editor_overlay.json`: newer editor versions installed after reconstruction.
- `recovery/pre_reconstruction_files/` and `before_editor_overlay/`: differing earlier restored copies, preserved before replacement.
- `recovery/dataset_inventory.json`: available snapshot file inventory.
- `/tmp/grace-recovery-20261027/`: independent recovery staging on the root filesystem, including the successive reconstruction passes.

Some scripts and historical reports correctly reference files that are still missing. Treat old completion markers and historical scores as recovered evidence, not as a claim that their dependent models or predictions are currently available.
