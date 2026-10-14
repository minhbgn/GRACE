# Checkpoint, dataset and evaluation integrity audit

Audit date: 2026-09-25. Scope: the completed **40 lmms-eval jobs** on
ScienceQA-image, AI2D, SEED-image and MMStar. This was a CPU/read-only audit of
model and dataset artifacts and saved predictions. No model inference, training,
judge calls or changes to the pinned evaluator were performed. Derived audit
scripts, JSON evidence and this report were added.

**Conclusion: no wrong checkpoint, swapped model path, corrupted benchmark
source, denominator/averaging error, or parser advantage explaining the local
low-bit models' results was found.** The internal four-suite comparison is
supported by the checks below. This does not establish exact reproduction of
the paper, superiority across all seven benchmarks, or identical packed-model
behavior.

## 1. Model identity and download verification

All **114 recorded files across ten checkpoints** were rehashed from the local
filesystem and matched the manifests saved before evaluation. For the five
Hugging Face reference models, all **59 downloaded files** additionally matched
the official Hub manifest at the recorded commit: SHA-256 for LFS files and Git
blob SHA-1 for ordinary repository files. This includes weights, model configs,
tokenizers, chat templates and processor assets where present.

| Evaluated reference | Hugging Face repository | Verified commit |
| --- | --- | --- |
| Vanilla 2B | `Qwen/Qwen3-VL-2B-Instruct` | `89644892e4d85e24eaac8bacfd4f463576704203` |
| Vanilla 8B teacher | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| Released BF16 initialization | `ForeverBlue/Qwen3-VL-2B-GRACE-BF16` | `82f9a267340f65a3619be0e19daf6582d3998a5f` |
| Released W4 | `ForeverBlue/Qwen3-VL-2B-GRACE-W4G128` | `57dcaeed3fd7739a0566bfc4b8651f451068825c` |
| Released W8 | `ForeverBlue/Qwen3-VL-2B-GRACE-W8G128` | `32c351aed1388ae602d59f7c26101dd16aca8c80` |

All five commits were also the respective repositories' current heads at audit
time, with no differences in the compared files. We therefore found neither a
stale release nor a locally modified reference checkpoint. W4 and W8 have
different weight hashes and the expected bit-width metadata; their names have
not been interchanged.

The experiment mapping, each saved job's `model_path`, model provenance and
`results.json`'s actual `pretrained=` argument agree for all 40 jobs. The local
SFT reference points to the previous SFT export, and the four local low-bit rows
point to their separate final merged exports. Their bytes still match what was
recorded at evaluation. This rules out accidentally evaluating the BF16
initialization under a QAT/GRACE label in the recorded workflow.

The selected released repositories are the **QAT research exports**, which are
intended to load with standard Transformers. Their floating-point tensors store
quantized values. They are different from the separately released packed AWQ
deployment model. This is the appropriate artifact class for comparison with
our floating-point research exports; changing only the released row to AWQ
would introduce another deployment/quantization difference.
[Official W4 card](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-W4G128),
[official W8 card](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-W8G128).

Evidence: [checkpoint checksums and Hub comparisons](audit/checkpoints.json),
[reproducible audit script](audit/audit_checkpoints.py).

## 2. Loading, quantization and preprocessing checks

- The low-bit configs consistently specify **196 LLM linear modules**, G128,
  and the expected W4/W8 precision. Vision, merger, embeddings and output head
  are excluded from that weight-quantization scope. The compared files contain
  ordinary floating-point tensors and no unmerged LoRA or packed `qweight`/
  `qzeros` keys.
- For released W4, LoRA-QAT W4 and LoRA-GRACE W4, **9,408 sampled G128 groups per
  model across all 196 target matrices** each had at most 16 distinct values.
  This directly checks that our evaluated W4 exports are grid-projected rather
  than accidentally retaining unrestricted BF16 weights. It is a sampled
  W4 check, not a proof for every group or an independent W8 lattice audit.
- For every 2B checkpoint that stores both embeddings and `lm_head.weight`,
  the tied tensors are elementwise equal. Vanilla 2B omits the duplicate head
  as expected. An inconsistent duplicated head therefore does not explain the
  released models' scores. The 8B model is not included in this 2B tie check.
- The recorded runtime source matches the pinned commit and the runner, worker
  and checkpoint-helper hashes in the run lock. All 40 completed logs were
  scanned for missing, unused, unexpectedly initialized or shape-mismatched
  weights and tracebacks; none of those messages was found.
- A real AI2D example was run through **all ten processors on CPU**, using the
  saved evaluation prompt and the backend's actual image preparation path.
  The 600×449 source image became 608×448. Token IDs, attention mask, image grid,
  pixel tensors and rendered prompt hashes were identical across models.
  Effective pixel bounds were 200,704/1,605,632 for all ten. The old per-release
  native processor bounds did not silently override this test.
- Loaded RoPE settings agree, including theta 5,000,000 and MRoPE sections
  `[24,20,20]`, despite different serialization layouts and recorded Transformers
  versions. All checkpoints are evaluated by the same Qwen3-VL backend in BF16
  with SDPA. The backend explicitly supplies greedy decoding, one beam, common
  tokenizer EOS/pad handling and task-specific token limits. The saved sampling
  defaults therefore do not secretly make the released rows stochastic.

The processor test establishes parity on one real example; it is not an
independent replay of every historical tensor or a cross-version forward-pass
comparison. Historical runs did not log all input tensors. No new GPU forward
pass was needed for these checks.

Evidence: [processor, tied-weight and W4-grid results](audit/loading.json),
[CPU probe script](audit/audit_loading.py).

## 3. Benchmark source and split verification

The four selected task definitions point to the expected dataset repositories.
Hub metadata confirms the `lmms-lab/*` names resolve to the pinned
`lmms-lab-encoder/*` repositories; the redirect is not an unrelated dataset.
All **278 cached Parquet files, totaling 27,603,981,783 bytes**, matched the
Hub's LFS SHA-256 manifests at the exact pinned dataset commits.

| Suite | Resolved repository | Source verification | Evaluated split/score |
| --- | --- | --- | --- |
| ScienceQA | `lmms-lab-encoder/ScienceQA` | 2 cached Parquets: image validation and test | `ScienceQA-IMG` / test; 2,017 rows |
| AI2D | `lmms-lab-encoder/ai2d` | 2 test Parquets | test; 3,088 rows |
| MMStar | `Lin-Chen/MMStar` | 1 Parquet | val; 1,500 rows |
| SEED-Bench | `lmms-lab-encoder/SEED-Bench` | 273 test Parquets | test; 17,990 rows, including 14,233 image and 3,757 video rows |

Every selected source row's saved question/option/metadata fields and target
were compared with the evaluated sample records in the correct order. Source
images are present for all rows; selected images were decoded successfully.
The preceding compilation audit also established identical saved documents and
targets across all ten model rows, so comparison is on a shared example set.
The source-file check includes images, while the source-to-prediction alignment
check verifies saved non-image fields; historical pixel tensors were not logged.

SEED's reported score is computed only over the **14,233 image questions**,
not all 17,990 generated answers. ScienceQA reports only its **2,017 image
questions**. Neither training/validation data nor the full text-plus-image
ScienceQA score was substituted into these test results. MMStar's macro score
uses 18 L2 categories, as the pinned task implements.

These checks establish faithful download of the selected public dataset
snapshots. They do not establish that the authors used those same snapshots,
that repackaged datasets are identical to every original benchmark release,
or that no training/evaluation overlap exists. Dataset contamination was not
audited in this task.

Evidence: [dataset revisions, per-file checksums and row checks](audit/datasets.json),
[source-data audit script](audit/audit_datasets.py).

## 4. Scoring, prompts and answer-format sensitivity

The previous report independently recomputed all 40 primary scores from sample
records. This audit additionally verifies identical effective task configs and
saved prompt arguments across models, correct actual checkpoint paths, and
unchanged evaluator source. Prompt builders use question/context/options; the
selected zero-shot workflow does not append the ground-truth answer to the
model prompt.

A potentially concerning format difference was investigated explicitly:
GRACE-W8 often responds with `B.` or `B. New Hampshire`, while both released
low-bit models consistently return a single letter. Specifically, GRACE-W8
has non-single-letter responses on 1,093/2,017 ScienceQA, 1,670/3,088 AI2D and
9,831/14,233 SEED-image examples, but none on MMStar. A whole-string scorer
requiring exactly `B` would turn many clearly correct answers into failures;
its much lower score would measure format compliance as well as answer quality.

An independent anchored choice parser was therefore used: it accepts an
explicit option such as `B`, `B. explanation`, `(B)` or an answer prefix, and
rejects an arbitrary prose word merely beginning with a valid letter. **Every
sample's correctness agrees with the harness for both released low-bit models,
all four local QAT/GRACE models, released BF16 and local SFT.** Their reported
scores and all four local-versus-release contrasts remain unchanged. This is
strong evidence that the observed low-bit-model advantages are not produced by
a lenient first-character parser.

Three discrepancies occurred only in the vanilla references:

- Vanilla 2B/MMStar, document 1277: `The correct answer is (B) 1.2.` is a correct
  explicit B that the harness missed. Accepting it raises its Mean-4 by about
  **0.0095 pp**.
- Teacher/SEED-image, documents 7917 and 8120: `A, C` and `C, D` are ambiguous
  multi-option answers. The harness credited their first letters. Rejecting
  them lowers the teacher's Mean-4 by about **0.0035 pp**.

These are genuine small parser limitations, but neither explains the local
low-bit models' advantage over the releases. Original published harness scores
remain unchanged; the independent parser is an audit, not a replacement metric.

Evidence: [paths, prompt/config hashes and rescoring results](audit/evaluation.json),
[independent audit script](audit/audit_evaluation.py).

## 5. What the result does and does not mean

The premise needs a qualification: **some** of our models outperform the
corresponding releases on Mean-4; not every local model does.

| Local model | Local Mean-4 | Corresponding release Mean-4 | Local − release |
| --- | ---: | ---: | ---: |
| LoRA-QAT W4 | 77.86 | 77.68 | +0.18 pp |
| LoRA-GRACE W4 | 76.84 | 77.68 | −0.84 pp |
| LoRA-QAT W8 | 79.35 | 76.72 | +2.63 pp |
| LoRA-GRACE W8 | 77.78 | 76.72 | +1.06 pp |

The small W4 QAT difference should be described as close performance, not a
clear repeatable training improvement. More plausible explanations for the
larger differences, distinguished from proven causes, are:

1. **We inherit the released BF16 student's strength.** Its Mean-4 is 78.94,
   already 1.26 pp above released W4 and 2.22 pp above released W8. QAT W8 is
   just 0.41 pp above that initialization. Its +2.63 pp over the W8 release
   mostly reflects avoiding the release's drop relative to this strong starting
   point, rather than demonstrating that a short local stage reproduces the
   authors' entire training pipeline.
2. **Different adaptation can preserve different capabilities.** LoRA, our
   32k examples, one-pass schedule and optimization settings differ from full
   training. Restricted updates can plausibly preserve useful pretrained
   behavior, whereas longer/full training can trade some benchmarks for others.
   This is a hypothesis; we do not have a matched full-parameter retraining
   ablation that isolates the cause.
3. **This is a four-suite protocol, not the paper's full evaluation.** MMMU,
   HallusionBench and MMBench are missing. Image resolution, prompt formatting,
   dataset versions and scoring may differ from the authors' setup. Equal
   settings give a useful internal comparison, but they do not establish the
   same ranking under all protocols or on omitted tasks. The new large image
   budget especially needs to match the intended deployment budget.
4. **The W8 release really has the unexpected ordering in these saved runs.**
   It is authentic and current, yet falls below released W4 on all four tested
   suites. A mislabeled local path or damaged download is no longer a plausible
   explanation supported by this audit. The authors' artifact/export history
   or a remaining runtime/protocol interaction is still possible. Which one is
   responsible cannot be inferred from these accuracy tables alone.

The checks support the wording: “Our constrained QAT/GRACE research exports
achieve competitive or higher scores on this verified four-suite protocol,
conditional on initialization from the released BF16 student.” They do not
support “our method universally beats full GRACE,” “the paper is wrong,” or a
packed INT4 memory/speed claim.

## 6. Remaining checks with the highest value

First fix the known MMMU task-reimport/unused-judge initialization bug in a
separately versioned protocol and test the full loading path. Its selected
standard metric does not intrinsically require paid judging. The original
seven-suite gap cannot be resolved while the previously most discrepant task
remains untested.

For an independent runtime test, select a fixed small set of the released/local
W8 disagreements and compare saved harness predictions with direct Transformers
generation under the same explicit inputs and decoding settings. Only then
change one factor at a time: authors' Transformers version, native versus shared
processor settings, image budget, prompt or output cap. Keep an ordinary
representative sample as well; a disagreement-only set is diagnostic and not an
unbiased benchmark. No such GPU run was launched during this audit.

Request the authors' exact evaluation commands, task versions and raw
predictions for the same checkpoint commits. Use packed export/re-evaluation
and multiple training seeds before making a deployment or statistically stable
method-superiority claim. Retain the completed results and the present audit
as immutable evidence rather than overwriting them during follow-up work.
