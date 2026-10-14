# Actual training token lengths — audit 2026-10-05

**None of the 32,768 retained training examples reached the 2,048-token limit. The maximum was 1,858 tokens: 1,777 text/formatting tokens plus 81 visual tokens.**

| Tokens per retained example | Mean | Median | 95th percentile | 99th percentile | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Total input | 314.50 | 234 | 612 | 751 | 1858 |
| Text + formatting (nonvisual) | 224.21 | 144 | 521 | 660 | 1777 |
| Merged visual | 90.28 | 88 | 96 | 96 | 98 |

“Text + formatting” includes user questions, assistant answers, chat role/turn delimiters and vision boundary tokens. It excludes expanded image-pad tokens. It is the nonvisual contribution to the training input budget, not just the answer tokens supervised by CE. These are actual retained sequence lengths, without padding to 2,048. Percentiles are computed independently and need not add together.

## How often was the limit approached?

- Exactly 2,048 tokens: **0 / 32,768**.
- At least 2,000 tokens: **0**.
- At least 1,800 tokens: **1**.
- At least 1,536 tokens: **5**.
- At least 1,024 tokens: **91 (0.278%)**.
- **99%** of retained examples have at most **751 total tokens**.
- The longest retained example has annotation index `457689`, 75 question/answer pairs and 190 unused tokens below the cap.

## Was this just an artifact of trimming?

All retained conversations were compared with their original annotation records: **0 conversations trimmed, 0 turn pairs removed**. The saved preparation log records **one rejected candidate** whose first complete question/answer pair exceeded the token limit. Thus the cap affected eligibility for at least that candidate, but did not shorten any of the final 32,768 retained conversations. These rejection counts describe candidates examined during subset preparation, not a global audit of every original example.

## Implication for the 576-token experiment

If every retained conversation kept the same text and every image used exactly 576 visual tokens, only **4 / 32,768 (0.0122%)** would exceed 2,048 total tokens. The longest hypothetical sequence would be **2,353** tokens. This is an upper-bound scenario: dynamic image sizes can use fewer visual tokens, and the newly prepared manifest can reject or shorten examples. It does not establish the new full subset's actual maximum.

The existing data therefore provides substantial sequence headroom for the larger visual budget. Keep the 2,048 total-token limit for the planned comparison, then inspect the new subset comparison instead of increasing the total limit preemptively.

## Verification and reproducibility

- Verified the entire saved manifest checksum and the original tokenizer/processor file fingerprints.
- Rendered/tokenized every retained conversation with the original processor. A rendered image contributes one placeholder; removing it gives the nonvisual token count. Subtracting this from the stored complete sequence length yields the expanded visual count.
- Reprocessed actual images for the longest 16 records plus every 1,024th record (**48 distinct checks**); complete token/label/grid hashes match the saved manifest, and the text/visual decomposition agrees.
- Streamed the original annotation array to compare selected conversations, verifying its complete SHA-256 against the preparation recipe.
- CPU only; no training/evaluation jobs or changes to saved training artifacts.

Artifacts: [summary JSON](token_lengths.json), [all 32,768 per-example counts](token_lengths.csv), [audit script](audit_token_lengths.py).

Run from the workspace root:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/air/miniconda3/envs/grace/bin/python reproduction/GRACE/experiments/released_bf16_init_32k_b128_p100352_t2048_seed42/reports/audit_token_lengths.py
```
