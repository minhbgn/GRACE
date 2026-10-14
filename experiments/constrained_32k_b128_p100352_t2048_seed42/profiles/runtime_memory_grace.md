# Runtime and memory measurements

Experiment: `/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/experiments/constrained_32k_b128_p100352_t2048_seed42`

**Stress profile:** repeated longest retained examples; not representative training throughput.

Medians exclude the first 1 update(s) of each invocation. Memory peaks include all logged updates.

| Method | Complete | Updates / measured | Batch | Max pixels | Tokens | Median s/update | Median s/cycle | Examples/s | Peak allocated GiB | Peak reserved GiB | Projected update-only hours |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| grace | True | 10 / 9 | 128 | 100352 | 2048 | 70.23 | 88.89 | 1.82 | 24.70 | 26.91 | 4.99 |

Cycle timing includes intervening checkpoint/data work. The hour projection excludes that work, loading and export; it is not an end-to-end guarantee.
PyTorch memory peaks are maxima across ranks at logged training boundaries, not total device occupancy or model-loading/export peaks.

Baseline: `/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/experiments/constrained_32k_seed42` (profiles; same timing calculation).

| Method | Baseline s/update | New s/update | Baseline examples/s | New examples/s | Throughput change | Reserved-memory change GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| grace | 19.97 | 70.23 | 3.21 | 1.82 | -43.1% | +3.06 |

Batch sizes and retained inputs differ. Compare examples/s as well as seconds/update. Stress profiles are input-envelope comparisons, not identical-data ablations.
