# lmms video protocol validation — 2026-10-26

The central controller is `evaluation/runners/run_lmms_matrix.py --video`.
The pinned harness source remains unchanged. No GPU benchmark or model-weight
inference was performed in this validation.

- 11 lmms unit tests passed (including four video protocol checks).
- 50 existing video tests: 43 passed inside the sandbox; seven downloader
  cases could not bind local HTTP sockets. All 10 downloader tests subsequently
  passed outside the sandbox.
- Real pinned harness integration passed for all five video suites, using two
  real local annotation rows per suite and dummy gold answers. These synthetic
  100% scores validate plumbing, not model performance.
- The new lmms model adapter passed real TempCompass video decode, resize,
  timestamps, prompt tokenization, telemetry and EOS checks with a fake CPU
  decoder. Its input was 1330 visual + 100 text tokens. All CUDA operations and
  checkpoint weight loading were mocked for that check.
- Bash syntax checks passed for the three experiment launchers.
- `check --methods vanilla_2b --smoke 1` built all five local harness tasks.

## Actual 720p CPU preflight

Command: `run_video_evaluation.sh prepare --methods vanilla_2b --smoke 1`.
Policy: 2 fps, short-edge ceiling 720, long-edge ceiling 1280, patch-grid rounding,
4096 output reserve. One first question per suite, not a full population audit.
The copied JSON/JSONL files retain the measured token counts.

| Suite | Visual input tokens | Context overflow |
| --- | ---: | --- |
| TempCompass MC | 1330 | No |
| VideoMME short/medium | 66000 | No |
| Video-MMMU perception | 312576 | Yes |
| Video-MMMU comprehension | 312576 | Yes |
| Video-MMMU adaptation | 312576 | Yes |

The native context is 262144 tokens. Preflight correctly exited unsuccessfully
for the three Video-MMMU inputs. VideoMME long exclusion does not imply every
remaining video fits at 720p. The runner does not reduce fps, shrink individual
inputs, truncate, or silently skip oversized questions. A full 720p baseline
matrix therefore remains blocked on those inputs until a separately declared
sampling/population/token-reduction protocol resolves the context requirement.
No full-population 720p GPU-memory feasibility claim is made.
