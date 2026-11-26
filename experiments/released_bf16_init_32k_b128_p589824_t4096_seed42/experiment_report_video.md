> **Protocol update (2026-10-26):** New video runs use the central
> `evaluation/runners/run_lmms_matrix.py --video` controller through this
> experiment’s bash scripts. Defaults are 720p (720 short edge / 1280 long edge,
> rounded to patch grid), 2 fps, and VideoMME short + medium only. TempCompass MC
> uses an explicitly labelled local scoring fallback; tasks requiring an API
> judge without a local scorer are skipped. The results below remain historical
> 256-pixel local evaluations and are not 720p lmms-eval results. See the
> [current protocol](../../evaluation/docs/README_video_evaluation.md).

# Video evaluation results and input-token audit

Prepared **2026-10-19** from the saved full-population artifacts in
[`video_unsaturated_v2_3a74c7231e27/full`](video_evaluation/video_unsaturated_v2_3a74c7231e27/full/).
Updated the same day with an inspection of **all 50 worker logs**, including the five completed runs.

**The artifacts currently support 5 completed model/task results out of 50 planned pairs, rather than a completed ten-model comparison.** TempCompass has three complete models; Video-MMMU Perception and Comprehension each have one. Video-MME and Video-MMMU Adaptation have no complete results. None of the four local QAT/GRACE checkpoints has a completed video result in this directory, so a video distillation gain cannot yet be measured. All scores below use validated complete populations; partial runs contribute only coverage and input-accounting evidence.

Among completed TempCompass runs, the **8B teacher scores 71.58%**, versus **65.06% for local SFT 2B** and **64.43% for vanilla 2B**. Vanilla 2B scores **52.67% on Video-MMMU Perception** and **36.00% on Comprehension**. The full input audit is available for all five tasks, including tasks whose inference failed or stopped early.

## Protocol and populations

The pinned [protocol](video_evaluation/video_unsaturated_v2_3a74c7231e27/full/protocol.json) uses **2 fps across the full clip**, a **256-pixel longest side** before rounding to a 32-pixel spatial grid, and two-frame temporal patches. There is no frame-count cap or fixed per-video visual-token cap. Native adaptive video resizing and frame sampling are bypassed. Tokenizer truncation is disabled. Generation is greedy with a **4,096-token output allowance**, FlashAttention 2, and BF16 loading of the saved exports. The native context limit is **262,144 tokens**. Audio and subtitles are excluded.

The experiment's **576-image-token ceiling and 4,096-token training sequence limit do not constrain these video inputs**. W4/W8 denote the checkpoints' training/export arms; this evaluator loads their unpacked weights in BF16. These results do not measure packed low-bit deployment speed or memory savings.

| Task | Questions | Videos within task | Complete models / 10 |
| --- | ---: | ---: | ---: |
| TempCompass, multiple choice | 1,580 | 410 | 3 |
| Video-MME, without subtitles | 2,700 | 900 | 0 |
| Video-MMMU Perception | 300 | 300 | 1 |
| Video-MMMU Comprehension | 300 | 300 | 1 |
| Video-MMMU Adaptation | 300 | 300 | 0 |
| **Total question evaluations per model** | **5,180** | — | **5 / 50 pairs** |

The saved dataset manifests report no missing videos. The three Video-MMMU tasks use the same ordered video IDs with different questions, so their video counts should not be added as distinct assets.

Scoring follows the local deterministic choice parser, with unresolved answers counted wrong; non-multiple-choice Video-MMMU responses use the pinned MMMU open-answer matcher. Perception/Comprehension prompts ask the model to ignore the final quiz question. Adaptation uses the full video, including the final question image. These are local-protocol scores, rather than an official leaderboard replay.

## Input tokens

Visual tokens are merged video placeholders: `prod(video_grid_thw) / merge_size²`, with `merge_size=2`. The grid's temporal dimension already includes temporal patch grouping. Text input is every remaining input token, including the question, choices, timestamps, vision boundary markers, and chat-template markers. Generated text is reported separately and includes EOS.

The following are **question-weighted full-population counts per model under the common input policy**. TempCompass and Video-MMMU Perception/Comprehension are also measured over complete inference populations. Video-MME and Adaptation are CPU reconstructions from the hashed historical source-geometry/frame-count audits, using the current 256-pixel geometry and saved text counts. They are not evidence that every model processed the full population. Reconstruction matches all **9,697 saved inference rows**, including partial attempts, with no token-count differences. Matching input identities were also verified wherever multiple models produced the same question.

| Task | Mean visual input | Mean text input | Mean total input | Total visual tokens / full task | Total text-input tokens / full task |
| --- | ---: | ---: | ---: | ---: | ---: |
| TempCompass | 489.12 | 153.13 | 642.25 | 772,808 | 241,947 |
| Video-MME | 32,732.90 | 10,671.22 | 43,404.12 | 88,378,824 | 28,812,299 |
| Video-MMMU Perception | 16,939.89 | 5,186.78 | 22,126.67 | 5,081,968 | 1,556,033 |
| Video-MMMU Comprehension | 16,939.89 | 5,190.00 | 22,129.90 | 5,081,968 | 1,557,001 |
| Video-MMMU Adaptation | 16,939.89 | 5,162.19 | 22,102.09 | 5,081,968 | 1,548,658 |
| **All 5,180 questions** | **20,153.96** | **6,508.87** | **26,662.83** | **104,397,536** | **33,715,938** |

Totals count repeated videos each time a question is evaluated. The aggregate weights questions, rather than averaging five task means equally. The full-matrix theoretical input total is ten times the per-model total; it is not the number of tokens actually processed by the incomplete matrix.

| Task | Visual min / median / p95 / max | Text-input p95 / max | Total-input p95 / max |
| --- | --- | --- | --- |
| TempCompass | 64 / 440 / 920 / 1,440 | 257 / 365 | 1,158 / 1,805 |
| Video-MME | 384 / 15,632 / 97,536 / 114,560 | 32,412 / 38,354 | 129,660 / 152,914 |
| Video-MMMU Perception | 1,696 / 15,456 / 35,040 / 67,776 | 10,036 / 21,468 | 45,244 / 86,428 |
| Video-MMMU Comprehension | 1,696 / 15,456 / 35,040 / 67,776 | 10,058 / 21,390 | 44,966 / 86,350 |
| Video-MMMU Adaptation | 1,696 / 15,456 / 35,040 / 67,776 | 9,928 / 21,517 | 44,970 / 86,477 |

Percentiles use the nearest-rank definition; medians average the two central values. Column maxima are independent and need not occur on the same question.

**Input-budget observations:**

- Text occupies **24.41% of all input tokens** across the full population. In this representation it includes per-temporal-group timestamps and markers, so it is much larger than the question alone. A visual-only reduction would leave this text cost unless the representation also changes.
- Long videos dominate the budget: Video-MME averages **66.92 times** TempCompass's visual count. Its mean visual count also exceeds its median substantially, so a single mean hides a long input tail.
- All projected questions fit native context with the output reserve: the largest input plus allowance is **152,914 + 4,096 = 157,010**, below 262,144. This establishes context feasibility, not GPU-memory feasibility; the teacher's actual OOMs are documented below.
- Identical video inputs imply identical visual counts across vanilla, SFT, QAT, and GRACE. These checkpoints implement no visual-token pruning. Missing model results cannot be interpreted as reduced token usage. The training image ceiling is not a video-token-reduction baseline.

## Model performance

Accuracy is percent correct over the full task population. **C** = certified complete; **P n** = partial, with n saved predictions and no explicit failure record; **OOM n** = explicit CUDA OOM after n predictions; **N** = an attempt exists but has zero predictions. An absent score is not zero accuracy.

| Checkpoint | TempCompass, 1,580 | Video-MME, 2,700 | Video-MMMU Perception, 300 | Video-MMMU Comprehension, 300 | Video-MMMU Adaptation, 300 |
| --- | --- | --- | --- | --- | --- |
| Vanilla 2B (`vanilla_2b`) | **64.43% C** | P 1,413 | **52.67% C** | **36.00% C** | P 57 |
| Vanilla 8B teacher (`vanilla_8b_teacher`) | **71.58% C** | OOM 1,800 | OOM 2 | OOM 2 | OOM 2 |
| Local SFT 2B (`sft_2b`) | **65.06% C** | P 1,079 | N | N | N |
| Released BF16 SFT (`released_bf16`) | N | N | N | N | N |
| Released GRACE W4 (`released_grace_w4`) | N | N | P 2 | N | N |
| Released GRACE W8 (`released_grace_w8`) | N | N | N | N | N |
| Local QAT W4 (`qat_w4`) | N | N | N | N | N |
| Local GRACE W4 (`grace_w4`) | N | N | N | N | N |
| Local QAT W8 (`qat_w8`) | N | N | N | N | N |
| Local GRACE W8 (`grace_w8`) | N | N | N | N | N |

Coverage is **5 complete, 4 explicit OOM failures, 4 partial attempts without an explicit failure record, and 37 zero-prediction attempts**. The saved comparison file labels the latter two groups “pending”; the distinction above comes from inspecting their prediction files. Merely reaching the final model in the matrix does not establish successful completion of the earlier jobs.

The five complete results have **zero unresolved answers and zero generation cutoffs**. Their raw responses, recomputed scores, source/input identities, token accounting, population ordering, and completion-file hashes pass the current CPU validator. The pinned evaluator source hashes also match the current implementation.

| Complete result | Correct / total | Accuracy % | Mean text output | Maximum text output | Mean input + output text |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vanilla 2B — TempCompass | 1,018 / 1,580 | 64.43 | 3.20 | 15 | 156.33 |
| 8B teacher — TempCompass | 1,131 / 1,580 | 71.58 | 4.02 | 25 | 157.15 |
| Local SFT 2B — TempCompass | 1,028 / 1,580 | 65.06 | 2.00 | 2 | 155.13 |
| Vanilla 2B — Video-MMMU Perception | 158 / 300 | 52.67 | 6.06 | 34 | 5,192.83 |
| Vanilla 2B — Video-MMMU Comprehension | 108 / 300 | 36.00 | 6.47 | 38 | 5,196.48 |

No three-benchmark mean or Video-MMMU overall score is reported: no checkpoint has all the required complete task results. Partial Video-MME populations are especially unsuitable for comparison because question order groups videos by duration, and the stopping points differ by model.

### TempCompass diagnostics

| Category | Questions | Vanilla 2B % | 8B teacher % | Local SFT 2B % |
| --- | ---: | ---: | ---: | ---: |
| Action | 338 | 94.97 | 97.04 | 95.56 |
| Attribute change | 288 | 67.71 | 79.86 | 68.06 |
| Direction | 335 | 48.36 | 51.64 | 50.45 |
| Order | 302 | 67.88 | 73.51 | 68.87 |
| Speed | 317 | 42.59 | 56.15 | 41.64 |

The teacher's overall gain over vanilla 2B is **+7.15 pp**, with its largest category gain on speed (**+13.56 pp**). Direction remains difficult even for the teacher. Local SFT improves the aggregate by **+0.63 pp**, or **10 net correct answers**, and regresses slightly on speed. Paired predictions show **74 SFT-only correct versus 64 vanilla-only correct**; this is a small observed gain, rather than evidence of a broad temporal-reasoning improvement.

Vanilla 2B's Video-MMMU Comprehension result separates into **46.78% concept comprehension** (80/171) and **21.71% problem-solving comprehension** (28/129). Its Perception score is **52.71% OCR** (146/277) and **52.17% ASR-labelled questions** (12/23). The ASR category is evaluated without audio in this protocol, which limits interpretation as an audio-understanding result.

## Failures and additional observations

### Context-overflow check and exclusion policy

**There are zero native-context overflows in this 256-pixel run**, zero saved `blocked.json` records, and no context-overflow exception in any worker log. The full-population reconstruction checks `input_tokens + 4096 > text_config.max_position_embeddings` for every question and all ten models. Each model's configuration has a **262,144-token context**.

| Task | Native-context overflows / questions | Maximum input + output reserve |
| --- | ---: | ---: |
| TempCompass | 0 / 1,580 | 5,901 |
| Video-MME | 0 / 2,700 | 157,010 |
| Video-MMMU Perception | 0 / 300 | 90,524 |
| Video-MMMU Comprehension | 0 / 300 | 90,446 |
| Video-MMMU Adaptation | 0 / 300 | 90,573 |

The requested policy of excluding context-overflow samples would therefore retain **all 5,180 questions per model**. It leaves the five completed accuracies unchanged and does not provide predictions for the incomplete runs. Their missing samples are not established context exclusions, so an observed-prefix score would not be accuracy over the non-overflow population. No such score is substituted for a full result here. The source-resolution audits from October 17 have genuine overflows, but describe a different input policy and cannot supply exclusions for this run.

**Tokenizer warnings are distinct from context failures.** Exactly **25 logs contain one tokenizer-length warning each**. The warning says the sequence is longer than the “specified maximum sequence length” and predicts indexing errors, but the installed Transformers method only logs a warning against `tokenizer.model_max_length`; it does not raise an exception. It suppresses subsequent copies, so 25 messages is not a count of affected samples.

| Checkpoint group | Saved tokenizer length setting | Logs with this warning |
| --- | ---: | ---: |
| Vanilla 2B and 8B teacher | 262,144 | 0 |
| Local SFT 2B | 2,048 | 4: Video-MME and all three Video-MMMU tasks |
| Released BF16 SFT | 16,384 | 3: all Video-MMMU tasks |
| Released GRACE W4 and W8 | 8,192 | 3 per model: all Video-MMMU tasks |
| Local QAT/GRACE W4/W8 | 4,096 | 3 per model: all Video-MMMU tasks |

The worker explicitly sets `truncation=False` and checks native context before loading weights. As direct evidence that the warnings are nonfatal, released GRACE W4 Perception logs its warning and then successfully generates two EOS-complete predictions with **25,771 and 44,298 input tokens**, both above its 8,192-token tokenizer setting. Its subsequent unexplained stop is not proof that the warning caused a failure.

### GPU memory, rather than context, blocked the teacher

The teacher has explicit `OutOfMemoryError` records on all four long-video tasks:

| Task | Saved predictions before failure | Failed question ID | Requested allocation |
| --- | ---: | --- | ---: |
| Video-MME | 1,800 / 2,700 | `601-1` | 1.74 GiB |
| Video-MMMU Perception | 2 / 300 | `test_Geography_8` | 1.58 GiB |
| Video-MMMU Comprehension | 2 / 300 | `test_Geography_8` | 1.58 GiB |
| Video-MMMU Adaptation | 2 / 300 | `test_Geography_8` | 1.59 GiB |

The logs report a **39.39 GiB GPU** and approximately **37.84–38.94 GiB process memory** at failure. By comparison, the complete vanilla-2B Perception/Comprehension runs record maximum allocated GPU memory of **20.76/20.75 GiB**. Complete TempCompass maxima are **4.33 GiB** for vanilla/local-SFT 2B and **16.90 GiB** for the teacher. These are recorded inference allocations, not packed quantization-memory measurements.

### Log evidence for all 45 incomplete runs

The following groups account for every incomplete pair. “Three Video-MMMU tasks” means Perception, Comprehension, and Adaptation. “Seven checkpoints” means released BF16, released GRACE W4/W8, and the four local QAT/GRACE checkpoints.

| Checkpoints and tasks | Pairs | Evidence in the logs | Established stopping reason |
| --- | ---: | --- | --- |
| 8B teacher — Video-MME | 1 | 1,800 predictions, decoder warnings, CUDA traceback | CUDA OOM |
| 8B teacher — three Video-MMMU tasks | 3 | Two predictions each, CUDA traceback | CUDA OOM |
| Vanilla 2B — Video-MME | 1 | 1,413 predictions, decoder warnings, no traceback | Unknown |
| Local SFT 2B — Video-MME | 1 | 1,079 predictions, tokenizer/decoder warnings, no traceback | Unknown |
| Vanilla 2B — Adaptation | 1 | 57 predictions, no warning or traceback | Unknown |
| Released GRACE W4 — Perception | 1 | Two predictions, tokenizer warning, no traceback | Unknown |
| Seven checkpoints — TempCompass and Video-MME | 14 | 100% weight-loader bar; no prediction, warning, or traceback | Unknown |
| Released BF16, released GRACE W8, four local QAT/GRACE — three Video-MMMU tasks | 18 | Tokenizer warning, 100% weight-loader bar; no prediction or traceback | Unknown |
| Released GRACE W4 — Comprehension and Adaptation | 2 | Tokenizer warning, 100% weight-loader bar; no prediction or traceback | Unknown |
| Local SFT 2B — three Video-MMMU tasks | 3 | Tokenizer warning, 100% weight-loader bar; no prediction or traceback | Unknown |
| **Total** | **45** | **Four explicit CUDA OOMs; 41 unexplained stops** | — |

For the **37 zero-prediction attempts**, the log always ends at the completed weight-loader progress bar. That bar does not certify a successful GPU transfer or first generation: the worker calls `.to('cuda:0')` after `from_pretrained`, then starts inference. A process kill, native crash, or resource limit is possible, but the logs do not identify one. These attempts cannot be labelled CUDA OOMs from their absence of predictions.

For the **four unexplained partial attempts**, the next unscored question also fits context:

| Model / task | Last saved question | Next unscored question | Next input + output reserve |
| --- | --- | --- | ---: |
| Vanilla 2B / Video-MME | `471-3` | `472-1` | 35,273 |
| Local SFT 2B / Video-MME | `360-2` | `360-3` | 18,682 |
| Vanilla 2B / Adaptation | `validation_Accounting_27` | `validation_Economics_1` | 20,402 |
| Released GRACE W4 / Perception | `dev_Geography_5` | `test_Geography_8` | 73,221 |

The matrix runner checks each subprocess's return code but saves neither that code nor a termination signal in the attempt directory. For nonzero exits it prints `FAILED` to its own stdout and continues. The worker saves `failure.json` only when its Python exception handler runs. Consequently, a worker ending outside that handler can leave exactly the observed incomplete artifacts. No saved parent stdout or matching host-kernel error record was found to resolve the 41 unknown cases. Recording worker exit codes/signals and retaining parent stdout would make future failures diagnosable; this audit did not change the evaluator or rerun inference.

### Decoder warnings in three incomplete Video-MME runs

Vanilla 2B, local SFT 2B, and the teacher log H.264 **`mmco: unref short failure`** messages. They continue to produce successful predictions after their last decoder warning:

| Video-MME model | Decoder warning messages | Predictions logged after the last warning |
| --- | ---: | ---: |
| Vanilla 2B | 60 | 463 |
| Local SFT 2B | 60 | 129 |
| 8B teacher | 111 | 337 |

These messages were not fatal where they occurred and do not explain the later abrupt stops; the teacher's later fatal error is explicitly CUDA OOM. They are nevertheless a video-decoding quality caveat for those partial runs. The logs alone do not establish whether decoded frames were affected, so no frame-corruption count or corrected accuracy is inferred.

### Inspection of the five completed logs

**All five completed logs have zero tokenizer-length warnings, decoder warnings, tracebacks, or other unexpected log lines.** Each contains only weight-loading output and the expected ordered prediction progress records:

| Completed model / task | Progress records / expected | Last progress record | Log finding |
| --- | ---: | --- | --- |
| Vanilla 2B / TempCompass | 1,580 / 1,580 | `1580/1580 1579` | Clean |
| 8B teacher / TempCompass | 1,580 / 1,580 | `1580/1580 1579` | Clean |
| Local SFT 2B / TempCompass | 1,580 / 1,580 | `1580/1580 1579` | Clean |
| Vanilla 2B / Perception | 300 / 300 | `300/300 new_Psychology_5` | Clean |
| Vanilla 2B / Comprehension | 300 / 300 | `300/300 new_Psychology_5` | Clean |

Every progress record was matched to its saved prediction's question ID, visual/text-input counts, output count, and finish reason. Completion hashes and full scoring validation also passed again. All responses finish at EOS, with no unresolved answer or output cutoff. **No log or saved-telemetry issue was found that requires excluding samples from these five results.** The declared 2 fps, 256-pixel, no-audio/no-subtitles protocol still defines their interpretation.

### One Adaptation annotation follows the wrong prompt/scoring path

`validation_Accounting_13` has **`question_type: "None"`**, despite ten choices and a letter gold answer, **`C`**. The current adapter treats every type other than `multiple-choice` as open-ended: this row's prompt omits the options, asks “Answer briefly,” and its output is scored by the open-answer matcher against `C`. The saved vanilla-2B response is `$12,000.`, scored incorrect. That numerical response is also different from the C option (`$72,000`), so this observation alone does not recover a correct answer.

The row needs an explicit annotation normalization and a new, provenance-separated evaluation before claiming clean Adaptation results. Its maximum direct influence is **1/300 = 0.33 pp** on that task. Existing predictions and completion artifacts have been left unchanged.

### What the available results establish

The completed runs establish the teacher's TempCompass advantage, a small local-SFT change there, and vanilla 2B's weakness on Video-MMMU problem solving. They establish neither a GRACE-versus-QAT advantage nor a W4-versus-W8 effect. A complete comparison requires the missing/failed populations, consistent handling of the malformed Adaptation row, and successful long-video teacher inference under a declared memory-feasible setup. Reducing resolution or frame rate would change this protocol and require a separate comparison.

## Evidence and verification

- [Machine-readable analysis](reports/video_results_analysis_20261019.json): all 50 attempt states, observed/expected counts, validated completed scores, complete-population token distributions, prediction/source hashes, paired TempCompass counts, and the annotation anomaly.
- [All-50-run log audit](reports/video_run_log_audit_20261019.json): each log's path/hash, matched progress records, warning counts, native/tokenizer limits, context-overflow counts, last/next question, and explicit failure evidence. All 45 incomplete runs and all five completed runs are covered.
- [Saved comparison](video_evaluation/video_unsaturated_v2_3a74c7231e27/full/comparison.md) and [JSON](video_evaluation/video_unsaturated_v2_3a74c7231e27/full/comparison.json): the original summary, consistent with five certified completions.
- [Resolution-feasibility audit](reports/video_resolution_feasibility_20261017.json): hashed historical source-geometry audits used to reconstruct full-population 256-pixel inputs. Earlier source-resolution overflow counts describe another policy and are not failures of this run.
- [Evaluator validation](../../evaluation/runners/run_video_matrix.py), [token construction](../../evaluation/protocols/video_protocol.py), and [video worker](../../evaluation/workers/video_worker.py): scoring, completion certification, exact input accounting, and recorded generation/memory telemetry.

This analysis used CPU checks on saved artifacts; no model inference or training was launched. The report reflects the files present in this experiment on 2026-10-19. The older implementation-validation JSON and README statements that full scoring had not run predate these five results.
