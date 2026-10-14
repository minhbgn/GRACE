# Constrained-stage runtime and memory

Kind: runs. Costs cover only these 4 configured runs, including logged resumed/failed invocations.

| Result | Updates | Trainable parameters | Process hours | GPU-hours (2 GPUs) | Recorded training hours | Other hours | Peak allocated GiB | Peak reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qat_w4 | 256 | 40771584 | 1.924 | 3.848 | 1.876 | 0.048 | 7.833 | 8.475 |
| grace_w4 | 256 | 40771584 | 2.390 | 4.780 | 2.340 | 0.050 | 24.696 | 26.068 |
| qat_w8 | 256 | 40771584 | 1.941 | 3.883 | 1.889 | 0.052 | 7.833 | 8.475 |
| grace_w8 | 256 | 40771584 | 2.382 | 4.765 | 2.332 | 0.051 | 24.696 | 26.068 |

Process time is measured around torchrun and includes loading, training, checkpoints and export. Other hours are process time outside recorded Trainer intervals, including loading/export and failures without a Trainer summary; those components are not separately instrumented. CPU preparation/provenance and the author’s prior training are excluded. A hard kill of the launcher can prevent recording its invocation.

Memory is the maximum PyTorch allocated/reserved value across ranks at logged training boundaries; loading/export peaks and total device occupancy are not measured. Profiles repeat the longest examples and do not measure representative training throughput.
