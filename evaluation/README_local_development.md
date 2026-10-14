# Corrected local development evaluation v1

Implemented and CPU-validated on 2026-09-26. **Fresh GPU baselines are pending.** This is a separate deterministic local protocol, not the authors' replay, paper reproduction, or GPT-judged lmms-eval. The existing `run_re_evaluation.sh` and its paper-score gate remain unchanged.

## Run the evaluation

From the repository root, first select an available GPU using your usual ownership survey (`gpuwho`). The launcher runs one model/suite at a time on the selected physical GPU in PCI order. It never starts training.

```bash
RUN=reproduction/GRACE/experiments/reproduction_eval_only/run_local_development.sh

# Inspect all 63 commands and perform CPU preflight (no model inference).
bash "$RUN" plan --methods all
bash "$RUN" check --methods all

# Replace 0 with your available physical GPU. Smoke outputs are isolated.
bash "$RUN" run --methods all --smoke 2 --gpu 0

# Five references: 35 full jobs.
bash "$RUN" run --methods references --gpu 0

# Four local LoRA-GRACE/QAT exports: 28 full jobs, same protocol.
bash "$RUN" run --methods lora --gpu 0

# Revalidate completed artifacts and write the combined development table.
bash "$RUN" summarize --methods all
```

Alternatively, `run --methods all --gpu 0` runs all 63 jobs sequentially. Rerun the same command to resume; completed jobs are rescored and checked before skipping. Failed attempts remain on disk, and the next invocation creates a new attempt. Each attempt has `run.log`, `command.json`, `results.json`, a prediction JSONL shard and, after validation, `validated.json`. `COMPLETE.json` records hashes. Inspect failures in `run.log`; inference errors cannot earn a completion marker.

`LMMS_PYTHON` overrides the Python executable; the default is the existing GRACE `.venv-lmms-eval/bin/python`. `--methods` also accepts comma-separated experiment IDs. `--suites` accepts `hallusionbench,mmbench,scienceqa,ai2d,mmmu,seedbench_img,mmstar`. `--attn sdpa` is available if needed, but changing attention requires a different output base. `--experiment` selects another compatible matrix JSON.

Results default to:

```text
reproduction/GRACE/experiments/reproduction_eval_only/local_development/
  local_native_greedy256_v1/
    smoke_2/...
    full/
      protocol.json
      local_scores.md
      local_scores.json
      <model>/processor.json
      <model>/model_provenance.json
      <model>/<suite>/COMPLETE.json
      <model>/<suite>/attempt_.../results.json
```

Use `local_scores.md` or `summary.primary_score` for the primary scores. All scores in JSON are fractions except explicitly named `*_percent` fields. In particular, MMBench's compatibility field `overall_acc` is **row accuracy**, not its primary circular score. Missing suites remain visible in the table. A partial smoke run never reports circular accuracy, since the first N TSV rows do not contain complete rotation groups.

## Frozen generation and processor policy

The recommended protocol is **`local_native_greedy256_v1`**: 256 new tokens for every question in every suite, including both MMMU MC and open questions, for every model. This finite cap is a deliberate change from the archived 8/32/64-token limits. It improves the opportunity to finish an answer; it does not guarantee that every answer finishes. Decode greedily with one beam, repetition penalty 1, tokenizer EOS/padding, BF16 model loading, and FlashAttention 2 by default. Inherited checkpoint generation penalties are replaced with this common policy. Original local prompt templates, population and image assembly are retained; no judge or network inference is used.

For a controlled short-cap comparison, choose **`--protocol local_native_greedy_short_v1`** consistently for all models. It uses 8 for HallusionBench/MMBench/ScienceQA/AI2D/SEED-image, 64 for MMStar, and MMMU MC 8/open 32. This still uses corrected scoring and common greedy generation, so it is not an exact author replay. Protocol names automatically select separate output folders. There is no unversioned token-cap override.

Every checkpoint loads its own saved processor. No shared W4/W8 resolution, common chat template, or training-budget override is introduced. The CPU preflight confirmed these native pixel bounds:

| Models | Minimum pixels | Maximum pixels |
| --- | ---: | ---: |
| Vanilla 2B, vanilla 8B, released BF16 | 65,536 | 16,777,216 |
| Released GRACE W4 | 784 | 200,704 |
| Released GRACE W8 | 784 | 50,176 |
| Local QAT/GRACE W4/W8 exports | 4,096 | 100,352 |

Thus the comparison uses the same **native-processor policy**, with different actual image budgets. The five references are `vanilla_2b,vanilla_8b_teacher,released_bf16,released_grace_w4,released_grace_w8`; the local models are `qat_w4,grace_w4,qat_w8,grace_w8`. `sft_2b` is not substituted for released BF16.

The run freezes code hashes (including pinned benchmark helpers), package versions, attention backend, experiment configuration, question-asset hashes and model/processor provenance. External ScienceQA/SEED images are inventoried by path, byte size and mtime; other benchmark images reside in hashed TSV/parquet assets. Changes reject resume and require a new `--output` base. Checkpoint validation hashes weight files, so preflight can take several minutes. One writer locks each output directory.

## Scoring rules

`local_scoring.py` defines **`deterministic_local_v1`**, enabled in the workers by `--local_scoring`. Use the new launcher; older launchers are not the frozen corrected protocol.

| Suite | Population | Primary metric |
| --- | --- | --- |
| HallusionBench | 1,129 rows: 951 image + 178 text | Rule-based per-answer accuracy, `rule_aAcc_all_image_and_text_rows` |
| MMBench EN dev 20231003 | 4,329 supplied rotations / 1,164 base questions | Circular accuracy; all supplied rotations correct |
| ScienceQA | All 4,241 test questions, image and text | Micro accuracy |
| AI2D | 3,088 test questions | Micro accuracy |
| MMMU | All 900 validation questions: 847 MC + 53 open | Micro accuracy across MC and open |
| SEEDBench | 14,233 image questions | Micro accuracy |
| MMStar | All 1,500 questions | Overall accuracy; category breakdowns are diagnostics |

- **MMBench:** calls the exact pinned `MMBench_Evaluator.prefetch_answer` used by the validated offline audit, and reuses its validated circular grouping (`index % 1_000_000`). Coverage must match the complete TSV, with no duplicate rotations or missing base questions. Unresolved rows count as wrong. Group evidence and conservative lower/possible-resolution upper bounds are saved. Judge/network/random fallback is disabled. A group containing a definite wrong answer stays wrong even if another rotation is unresolved.
- **Choice extraction:** MMMU/MMStar, ScienceQA, AI2D and SEED use the same deterministic parser. Explicit answer/choice markers, boxed labels or answer tags take priority; conflicting explicit labels are unresolved. Then prefer a leading letter with answer punctuation, then a unique case-sensitive standalone uppercase option. Prose is never globally uppercased and letters inside words never match. An initial article `A` is not used as an unmarked answer. MMMU additionally accepts exact unique option text. Completed `<think>...</think>` blocks are removed; an unfinished thinking block is unresolved. This is a declared local convention, not a semantic judge or the upstream MMMU random fallback. It fixes `The correct choice is A`, `The answer is B` and `C: A forest`, while retaining ambiguities for inspection.
- **MMMU open:** uses the pinned MMMU `parse_open_response`, `normalize_str`, `extract_numbers`, `check_is_number` and `eval_open` functions without importing judge initialization. Preserve the entire candidate list and numeric types; never stringify the first candidate. Rules include final/key-response extraction, lowercase string matching, comma/scientific-number extraction, two-decimal numeric normalization, string substring matching and gold answer lists. These heuristics are the documented benchmark matching convention, not arbitrary numeric tolerance or semantic equivalence. Source: [MMMU official evaluator](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/eval_utils.py); implementation: local pinned lmms-eval v0.7.1 commit `88b23e2bfa16a1edbc16e9e238ed82130b3a4f56`, `lmms_eval/tasks/mmmu/utils.py`. Candidate ordering is sorted only for stable saved evidence.
- **HallusionBench:** parses a leading Yes/No, optionally after an explicit answer prefix, without guessing Yes from words such as `is`, `are` or `correct`. Primary score includes image and text rows. Existing qAcc/fAcc and modality breakdowns remain diagnostics of these rule-based decisions, not GPT correctness/consistency judgments.
- **MMStar aggregation:** primary score is overall micro accuracy, not the mean of the 18 L2 categories. Under the full balanced dataset it also equals the mean of the six L1 category accuracies.

Every row retains raw output, prediction/candidates, correctness, extraction method, unresolved flag and inference error. Unresolved answers count as incorrect and stay in the denominator. They are not silently guessed, excluded or judged. The same convention is applied to references and local models; a fixed parser can still disagree with semantic adjudication on unusual prose, so review unresolved outputs and close score differences.

## Generation and integrity evidence

Each row's `generation` records exact generated token IDs, token count (including EOS if generated), cap, `finish_reason`, `hit_token_cap` and `truncated`. An EOS on the final allowed token is `eos`, with `hit_token_cap=true` and `truncated=false`; a cap stop without EOS is `length` and `truncated=true`. An unexpected early stop is `other` and prevents full validation. “Truncated” means stopped at the cap, not proof that the answer itself was incomplete. A complete answer at a length stop is still scored. Historical outputs without token IDs are explicitly `generation_unknown`, never inferred to have zero truncations.

The runner validates source IDs and gold labels, duplicate/missing coverage, recorded native processor size, generation policy and token evidence, zero inference errors, row scores recomputed from raw text, circular completeness, summary arithmetic and artifact hashes. Summaries expose unresolved counts, length stops and token totals. `artifact_ready` means selected full results pass these checks; it does not certify judge-equivalent answer interpretation or paper agreement. LoRA runs have no paper-score prerequisite. Do not use smoke scores as baselines.

## CPU validation performed

All 26 evaluation unit/integration tests passed, including 10 focused local-scoring tests (one exercises all seven workers).

- Regression tests cover the known parser failures, ambiguous and invalid choices, empty/truncated reasoning, Yes/No copula traps, MMMU strings/numbers/gold aliases, and EOS at the token boundary.
- Circular tests reject missing/duplicate rotations and verify unresolved-group bounds.
- Integrity tests reject altered predictions, gold labels, IDs, processor size, telemetry and summary scores.
- All seven actual worker entry points are exercised with fake generation on CPU, including shard and merged artifacts.
- All 35 saved reference pairs were rescored without inference. Circular scores exactly match the prior five-model audit: 76.804%, 84.708%, 91.151%, 88.660%, 88.402%. The saved-answer audit includes each changed decision and unresolved raw response. It explicitly labels historical generation telemetry unknown.
- CPU preflight passed for all nine checkpoints and seven source populations. No real GPU generation has been run for this protocol.

Run regression/integration tests:

```bash
reproduction/GRACE/.venv-lmms-eval/bin/python -m unittest discover \
  -s reproduction/GRACE/evaluation/tests -p 'test_local*.py' -v
```

Reproduce the archived-answer validation separately:

```bash
reproduction/GRACE/.venv-lmms-eval/bin/python reproduction/GRACE/evaluation/audit_local_scoring.py \
  --full-root reproduction/GRACE/experiments/released_bf16_init_32k_b128_p100352_t2048_seed42/re_evaluation/full \
  --circular-reference reproduction/GRACE/experiments/reproduction_eval_only/metric_protocol_audit.json \
  --output /tmp/local_scoring_validation.json
```

The checked-in evidence is [local_scoring_validation.json](../experiments/reproduction_eval_only/local_scoring_validation.json). These rescored old responses do not establish the new 256-token baselines. Run the reference and LoRA matrices above before making development comparisons.
