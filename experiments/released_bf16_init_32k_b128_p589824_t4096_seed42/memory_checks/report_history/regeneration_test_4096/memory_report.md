# Memory feasibility for the current recipe

**Status: passed.** Fingerprint: `4970820da848284fcdd4637cd44bf7fe714819f8240de11bf26df5814fc992b5`.

Requested envelope: **576 visual / 4096 total tokens**. Reserved-memory guard: **34.0 GiB/GPU**.

| Arm | Peak allocated GiB | Peak reserved GiB | Device sampled peak GiB | Evidence |
| --- | ---: | ---: | ---: | --- |
| GRACE W4 | 29.31 | 31.89 | 32.94 | [run](memory_checks/4970820da848284fcdd4637cd44bf7fe714819f8240de11bf26df5814fc992b5/grace_w4/attempt_20261005T081745225532Z/profiles/grace/run_summary.json) |
| GRACE W8 | 29.31 | 31.89 | 32.94 | [run](memory_checks/4970820da848284fcdd4637cd44bf7fe714819f8240de11bf26df5814fc992b5/grace_w8/attempt_20261005T082204868025Z/profiles/grace/run_summary.json) |
| QAT W4 | 11.81 | 13.08 | 14.13 | [run](memory_checks/4970820da848284fcdd4637cd44bf7fe714819f8240de11bf26df5814fc992b5/qat_w4/attempt_20261005T082638481657Z/profiles/qat/run_summary.json) |
| QAT W8 | 11.81 | 13.08 | 14.13 | [run](memory_checks/4970820da848284fcdd4637cd44bf7fe714819f8240de11bf26df5814fc992b5/qat_w8/attempt_20261005T083057877140Z/profiles/qat/run_summary.json) |

These are three-update synthetic long-answer checks, with accumulation **2** versus production **64**, RCKA warmup **1** versus **25**, and forced maximum image size. They test optimizer updates, full GDKD/RCKA and checkpoint/export memory. They are not production training or quality measurements and do not prove that every production batch fits.

Device memory is sampled every 0.5 seconds and may miss brief spikes. Training peaks are rank-max PyTorch counters. The guard checks at optimizer-update boundaries and cannot prevent every transient OOM.

Rerun `run_memory_checks.sh` to regenerate missing reports and reuse only matching, validated completed checks. Recipe/code/model/data/environment changes select a new fingerprint; failed attempts remain separate. Older report snapshots are retained under `memory_checks/report_history/`; the original 2,048-token screen is also preserved under `memory_checks/legacy_2048_reports/`.

Full dataset preparation, training and evaluation are separate commands in [README.md](README.md).
