from pathlib import Path
root=Path('/raid/lvchengtao/Minh/Distillation')
exp=root/'reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t2048_seed42'
p=exp/'README.md';s=p.read_text()
s=s.replace('## What changes', '**Current revision (2026-10-05):** the user raised `MODEL_MAX_LENGTH` to **4,096** before production preparation. The directory retains its original `t2048` name to preserve existing artifact paths; `recipe.env` and `experiment.json` describe the current 4,096-token run. Old 2,048-token memory evidence remains archived.\n\n## What changes')
s=s.replace('| 2,048 | 2,048 |','| 2,048 | **4,096** |')
s=s.replace('Increasing visual tokens while holding the total limit at 2,048 can remove trailing complete conversation turns or change which examples fit.', 'The maximum visual count and total sequence limit both increase relative to the old experiment. These changes can alter which examples or complete conversation turns are retained.')
s=s.replace('This bounds the concern about lost text for the existing subset; still inspect the newly prepared manifest rather than assuming exact identity.', 'That audit motivated the original 2,048-token setup. The user subsequently selected 4,096 for additional headroom; inspect the newly prepared manifest rather than assuming exact identity.')
start=s.index('See [memory_report.md](memory_report.md)')
end=s.index('Full training retains', start)
s=s[:start]+'''Run the entire memory preflight with one command:

```bash
EXP=reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t2048_seed42
CUDA_VISIBLE_DEVICES=0,4 bash "$EXP/run_memory_checks.sh"
```

Choose two currently free A100 PCI indices. Each missing GPU check surveys ownership before launching; no workloads are reclaimed. The command reads `recipe.env` and runs GRACE W4/W8 then QAT W4/W8, with **576 visual / 4,096 total tokens** for the current settings. It exits nonzero on failure and does not launch production training or lock its full dataset recipe.

Generated/rebuilt outputs:

- `profile_memory.json`: current resolved production settings, diagnostic overrides, fingerprint and status.
- `memory_results.json`: per-arm measurements and overall `checking` / `failed` / `passed` status.
- [memory_report.md](memory_report.md): human-readable memory evidence and limitations.
- `memory_checks/<fingerprint>/<method>_w<bits>/attempt_*/`: isolated checkpoints, input envelope, GPU survey, measurements and validated completion evidence; the attempt log sits beside the attempt directory.

The fingerprint covers resolved settings, code, source model bytes, historical input manifest and relevant environment versions. Existing reports alone never count as a pass: only matching completed checks with intact hashed evidence and export artifacts are reused. Missing reports are regenerated from those checks; changed settings select a fresh fingerprint. A failed or incomplete attempt is preserved and a subsequent invocation retries it in a new directory. Older summaries are archived before replacing stale current reports. A concurrent second invocation is rejected by the matrix lock.

Each check performs three optimizer updates with diagnostic accumulation **2**, exercising accumulated gradients, and RCKA warmup **1**, exercising full RCKA on updates 2–3. The actual production accumulation remains **64**, warmup **25**, updates **256**. A synthetic long answer forces the requested sequence ceiling; checkpoint saving and merged/baked export are included. These **do not change the production recipe**, and diagnostic models must never be evaluated as trained results. The standalone `profile_memory.sh --method grace --bits 4` remains available for a single fresh check, but use `run_memory_checks.sh` for validated reuse and aggregate reports.

The current maximum-grid generator uses a square image and requires an exact square merged-token grid (576 = 24²). If a future image budget cannot produce that grid, it fails explicitly instead of silently testing fewer visual tokens. Use the actual-data `run_training.sh --profile` for other image-grid budgets.

''' +s[end:]
p.write_text(s)
p=exp/'experiment_report.md';s=p.read_text().replace('a 2,048 total-token sequence limit','a **4,096** total-token sequence limit (raised by the user before production preparation)')
s=s.replace('At fixed total sequence length, increased visual tokens can change retained text and data eligibility, so the cross-resolution comparison is not automatically an isolated image-resolution effect.', 'Both the image budget and total sequence limit differ from the previous experiment, potentially changing retained text and data eligibility; this is not an isolated image-resolution intervention.')
p.write_text(s)
(exp/'training_dry_run.txt').write_text(Path('/tmp/4096_training_dry_run.txt').read_text())
p=root/'CODEX.md';s=p.read_text().replace('holding batch 128, 256 updates and the total 2,048-token limit fixed.', 'holding batch 128 and 256 updates fixed. The user subsequently raised the total-token limit to **4,096** before production preparation; the directory retains its original `t2048` name. `run_memory_checks.sh` fingerprints current settings and runs/reuses all four diagnostic checks, regenerating `profile_memory.json`, `memory_results.json` and the memory report.')
p.write_text(s)
p=root/'plans/3_llmc_grace_proposal.md';s=p.read_text().replace('keeping the 2,048 total-token limit and other training settings.', 'initially keeping the 2,048 total-token limit; the user then increased it to **4,096 before production preparation**. Batch, update count and loss settings remain unchanged.')
p.write_text(s)
p=root/'plans/todo_plan.md';s=p.read_text();marker='- [ ] Prepare the new shared 32,768-example subset'
s=s.replace(marker, '- [~] User increased the total sequence limit to **4,096** before production preparation. Add/run the recipe-aware `run_memory_checks.sh` and validate report regeneration/reuse; preserve the prior 2,048-token evidence.\n'+marker)
s=s.replace('Larger images compete with conversation tokens under the unchanged total limit.', 'Larger images compete with conversation tokens; the user has now raised the total limit to 4,096 as well. Report both changes.')
p.write_text(s)
print('Current configuration and one-command workflow documented; historical evidence preserved.')
