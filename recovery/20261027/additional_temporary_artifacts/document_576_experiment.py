from pathlib import Path
root=Path('/raid/lvchengtao/Minh/Distillation')
name='released_bf16_init_32k_b128_p589824_t2048_seed42'
exp=root/'reproduction/GRACE/experiments'/name
(exp/'README.md').write_text('''# Released-SFT QAT/GRACE with a 576-visual-token ceiling

Created 2026-10-05 at the user's request, before LLMC+ integration and the next supervisor update. This is an independent copy of the **recipe**, initialized afresh from the authors' released BF16 SFT checkpoint. Historical checkpoints, predictions, subsets and reports remain in the [98-token experiment](../released_bf16_init_32k_b128_p100352_t2048_seed42/README.md).

## What changes

| Setting | Previous experiment | This experiment |
| --- | --- | --- |
| Maximum processed image area | 100,352 pixels | **589,824 pixels** |
| Maximum merged visual tokens | 98 | **576** |
| Minimum image area | 4,096 pixels | 4,096 pixels |
| Total training sequence limit, including image/text/answer tokens | 2,048 | 2,048 |
| Subset target / training seed / subset seed | 32,768 / 42 / 2026 | Same |
| Global batch / per-device microbatch / accumulation / GPUs | 128 / 1 / 64 / 2 | Same |
| Optimizer updates / RCKA warmup | 256 / 25 | Same |
| Precision arms | W4G128 and W8G128 | Same |
| Initialization / teacher | Released 2B BF16 SFT / vanilla 8B | Same |
| LoRA / optimizer / CE, GDKD and RCKA settings | Existing matched recipe | Same |

For Qwen3-VL's 16-pixel patches and 2×2 spatial merging, `589824 / (16×2)² = 576`; a 768×768 processed square reaches the ceiling. Dynamic resizing preserves aspect ratio and does not force small images up to the ceiling. **576 is a maximum, not a fixed count per image.** Matching LLaVA's nominal token count does not equate its image representation with Qwen3-VL's, and neither 98 nor 576 alone establishes redundancy.

Both QAT and GRACE share a newly prepared manifest. Increasing visual tokens while holding the total limit at 2,048 can remove trailing complete conversation turns or change which examples fit. `subset_comparison.json` records added/removed indices, retained-conversation changes, token-hash changes and order equality versus the previous experiment. Report these before interpreting a cross-resolution difference as an image-budget-only effect. Within this new experiment, all four arms use the same data and processor.

The released SFT initialization has no GDKD/RCKA stage according to the authors' training README. QAT is CE-only adaptation; GRACE adds the unchanged GDKD/RCKA losses and controller. No token pruning is implemented in this experiment.

## Memory feasibility

See [memory_report.md](memory_report.md) for measured feasibility results and limitations. `profile_memory.sh` runs a separate three-update synthetic stress check at 576 visual / 2,048 total tokens, including long supervised answers, optimizer updates and checkpoint export. Diagnostic accumulation is 1 and RCKA warmup is 1 so that full RCKA is exercised; these **do not change the production recipe**. Its outputs live only under `memory_checks/` and must never be evaluated as trained models.

Full training retains the existing 34-GiB reserved-memory guard and five-hour training-time pause limit. The memory guard checks at optimizer-update boundaries; it cannot prevent a transient OOM. A memory-screen pass is evidence of feasibility, not a guarantee for every training batch or a runtime estimate. A time-limit pause preserves a native checkpoint; resume that individual arm with `--resume`. Do not silently lower the image cap within this directory after preparation. If a lower cap proves necessary, create another named experiment and match both methods at that cap.

## Prepare and train

Run from the workspace root with the existing GRACE environment (the wrapper selects it by default):

```bash
EXP=reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t2048_seed42

# No GPU use: inspect all four commands, then validate/build the full shared subset.
bash "$EXP/run_training.sh" all --train --dry-run
bash "$EXP/run_training.sh" all --prepare-only

# Survey immediately before GPU work. Use two currently free A100 PCI indices.
GPU_GUARD_MINE=/raid/lvchengtao/Minh/Distillation gpuwho

# Optional additional profile on actual retained data, with the production batch.
# Three steps do not finish the production 25-step RCKA warmup;
# the synthetic screen separately checks its full-weight memory path.
CUDA_VISIBLE_DEVICES=0,4 PROFILE_STEPS=3 bash "$EXP/run_training.sh" all --profile

# Train all four arms sequentially: GRACE W4, QAT W4, GRACE W8, QAT W8.
CUDA_VISIBLE_DEVICES=0,4 bash "$EXP/run_training.sh" all --train
```

Indices `0,4` are examples, not a reservation. The wrapper sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`. It also clears old experiment/student/teacher overrides to avoid accidentally targeting the previous run. Reference model bytes are shared and verified; production outputs are new `w4g128/runs/{qat,grace}` and `w8g128/runs/{qat,grace}` directories.

Alternatively launch individual arms; this is also how to continue after a paused or completed arm:

```bash
CUDA_VISIBLE_DEVICES=0,4 bash "$EXP/run_training.sh" qat --bits 4 --train
CUDA_VISIBLE_DEVICES=0,4 bash "$EXP/run_training.sh" grace --bits 4 --train
# Resume only an unfinished arm with an existing native checkpoint:
CUDA_VISIBLE_DEVICES=0,4 bash "$EXP/run_training.sh" grace --bits 4 --train --resume
```

Repeat with `--bits 8` as needed. Completed exports are intentionally protected from overwrite. If `all --train` stops, inspect `run_summary.json`; launch/resume the unfinished arm individually rather than rerunning completed arms. Save completed per-bit runtime reports alongside the evaluation report. Do not reuse the old experiment's completion markers or resumable checkpoints.

## Latest seven-suite evaluation

The local wrapper defaults to the four new LoRA exports and **`local_native_greedy256_v1`**: native saved processors, deterministic greedy decoding, 256 generated-token cap, corrected deterministic scoring. Suites: AI2D, HallusionBench, MMBench, MMStar, MMMU, ScienceQA and SEED. This is the current local development protocol; it is not a claim of paper-protocol reproduction.

```bash
# Preview before training; inspect/validate actual completed exports afterwards.
bash "$EXP/run_local_evaluation.sh" plan
bash "$EXP/run_local_evaluation.sh" check

GPU_GUARD_MINE=/raid/lvchengtao/Minh/Distillation gpuwho
# One free A100 per evaluation worker. Smoke and full outputs are separate.
bash "$EXP/run_local_evaluation.sh" run --gpu 0 --smoke 2
bash "$EXP/run_local_evaluation.sh" run --gpu 0
bash "$EXP/run_local_evaluation.sh" summarize
```

Use `--methods qat_w4,grace_w4` for a completed W4 pair, or `--methods all` to include all shared reference checkpoints. References keep their native processor budgets and therefore are not resolution-matched controls. Add the same method selection to `check`, `run` and `summarize`. Evaluation output is `local_development/local_native_greedy256_v1/{smoke_2,full}/`; `full/local_scores.md` and `local_scores.json` summarize the full round.

The MMBench validator now compares histogram keys in their JSON representation, fixing integer-versus-string key mismatches without changing scoring. The new run records the updated source hashes in fresh provenance. Historical evaluation artifacts have not been rewritten or marked complete by this fix.

## Supervisor report checklist

- Confirm all four exports completed 256 updates, and include the actual elapsed time, GPU-hours and peak memory.
- Report the resolved image/sequence limits and measured token distribution; keep the changed input budget visible next to old results.
- Summarize `subset_comparison.json`, including conversation trimming and replacements.
- Present all seven task scores, unresolved/truncated generation rates and the paired GRACE−QAT differences at each bit width.
- Compare against the previous experiment's **latest local greedy-256 evaluation**, not its older seven-suite or four-suite harness scores.
- Keep any cross-resolution improvement separate from a token-reduction claim: this experiment establishes the new unpruned controls.
- Preserve the known native-versus-merged/baked export caveat; packed low-bit deployment speed/memory is not established by these research exports.
''')
# Fix copied comment: there are two trainable methods in this matrix.
p=exp/'recipe.env'; p.write_text(p.read_text().replace('Keep the three methods matched', 'Keep QAT and GRACE matched'))
p=root/'plans/todo_plan.md'; s=p.read_text()
s=s.replace('This session remains planning-only. Future cloning, implementation and GPU runs are tasks below; none was performed by this update. Total experiment compute allocation and quality-loss tolerances remain to be agreed before training.', 'The initial approval covered planning only. In a subsequent request on the same date, the user authorized setup and memory validation of a separate higher-resolution QAT/GRACE experiment before LLMC+ work, and will launch its training/evaluation for a supervisor update. LLMC+ integration compute allocation and quality-loss tolerances remain to be agreed.')
marker='### 1. Close bounded baseline checks'
insert='''### Immediate priority — larger visual budget, requested 2026-10-05

- [x] Create an isolated [576-token-ceiling experiment](../reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t2048_seed42/README.md), preserving the released SFT initialization, W4/W8 QAT/GRACE arms, batch 128, 256 updates and 2,048 total tokens. Change maximum image area from 100,352 to 589,824 pixels.
- [~] Validate the maximum-token training/export memory path on two free A100s; document measured limits in the experiment's memory report. Bounded diagnostic runs are separate from production training.
- [x] Connect the new exports to the latest seven-suite `local_native_greedy256_v1` evaluation in a fresh output tree; fix MMBench JSON histogram-key validation with regression coverage. Recovery of historical completion artifacts remains separate.
- [ ] Prepare the new shared 32,768-example subset and inspect `subset_comparison.json` for dropped turns, changed examples and order; measure the actual visual-token distribution.
- [ ] User runs the new W4/W8 QAT and GRACE training and latest evaluation; report per-task results, matched differences, memory and GPU-hours to the supervisor before LLMC+ integration.

This raises the unpruned input ceiling; it does not implement token reduction or guarantee redundancy. Keep native processors fixed within each new retention sweep. Larger images compete with conversation tokens under the unchanged total limit.

'''
assert marker in s; s=s.replace(marker,insert+marker)
s=s.replace('The 100,352-pixel cap permits at most about 98 merged visual tokens; do not assume substantial redundancy remains.', 'The historical 100,352-pixel cap permits at most 98 merged visual tokens; use the separately requested 589,824-pixel controls (up to 576) once trained/evaluated. Measure redundancy rather than assuming it from count.')
s=s.replace('Discuss any separately named higher-resolution experiment before changing the agreed processor policy.', 'Use the separately named higher-resolution experiment approved above, retaining its saved native processor throughout each sweep.')
p.write_text(s)
p=root/'plans/3_llmc_grace_proposal.md';s=p.read_text()
s=s.replace('**Planning update approved by the user on 2026-10-05; execution has not started.**', '**Planning update approved on 2026-10-05. Subsequent request: prepare and memory-check larger-resolution QAT/GRACE controls before LLMC+ work.**')
s=s.replace('No implementation, model downloads, repository clones, training, or evaluation were performed. The approved sequence', 'At the time of the initial review, no implementation, model downloads, repository clones, training, or evaluation were performed. The approved sequence')
s=s.replace('QAT and GRACE local exports share the 100,352-pixel cap, allowing their matched comparison.', 'Historical QAT and GRACE local exports share the 100,352-pixel cap. The newly requested, separately named controls use 589,824 pixels once trained; compare QAT and GRACE at the same cap in each experiment.')
s=s.replace('The present local cap permits at most about 98 merged visual tokens, versus the first pilot\'s 49. A 50% rule may leave fewer than 49 actual tokens. Start conservatively and inspect the distribution. If little useful redundancy remains, propose a separately named higher-resolution diagnostic rather than interpreting a negative result as a universal rejection of token reduction. Any resolution change requires new matched controls and discussion under the existing native-processor preference.', '''The completed experiment permits at most 98 merged visual tokens, versus the first pilot's 49. The user subsequently requested new matched unpruned controls at **589,824 pixels / at most 576 merged tokens**, keeping the 2,048 total-token limit and other training settings. The [new experiment](../reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t2048_seed42/README.md) has isolated configuration, outputs, memory checks and the latest evaluation wrapper. Train/evaluate these controls and report to the supervisor before LLMC+ integration. Actual token counts vary with image dimensions; 576 tokens do not prove redundancy or architectural equivalence to LLaVA. Check subset/retained-turn changes caused by the shared image/text token limit. Preserve the old results and each checkpoint's native processor within later sweeps.''')
s=s.replace('Approval of this planning update is not a request to launch jobs in the present planning-only session.', 'The original planning approval did not authorize jobs. The later explicit request authorizes the new higher-resolution experiment setup and bounded memory validation; production training/evaluation will be launched by the user. LLMC+ implementation and runs remain subsequent work.')
p.write_text(s)
p=root/'CODEX.md';s=p.read_text()
s=s.replace('The LLMC+ sequence is approved as a plan; cloning, implementation and new GPU runs have not begun in this planning session.', 'The LLMC+ sequence is approved as a plan. The subsequent 2026-10-05 request prioritizes separately named higher-resolution unpruned controls before LLMC+: `released_bf16_init_32k_b128_p589824_t2048_seed42` raises the cap to 589,824 pixels (at most 576 merged visual tokens), holding batch 128, 256 updates and the total 2,048-token limit fixed. Setup and bounded memory checks are authorized; the user will launch full training/evaluation. See that experiment\'s README and memory report. Actual counts vary, and larger images can change retained conversation turns/subset eligibility. Never mix these outputs with the historical 98-token controls.')
s=s.replace('Follow the approved current-focus checklist in `plans/todo_plan.md`: resolve bounded baseline checks,', 'Follow the approved current-focus checklist in `plans/todo_plan.md`: first complete the requested higher-resolution controls and supervisor report, then resolve bounded baseline checks,')
p.write_text(s)
print('Wrote experiment guide and updated active plans/CODEX.')
