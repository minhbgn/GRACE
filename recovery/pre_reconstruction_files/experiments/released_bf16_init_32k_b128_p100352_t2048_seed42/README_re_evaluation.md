# Two-stage local re-evaluation

Use [run_re_evaluation.sh](run_re_evaluation.sh). It runs the recovered upstream
**local** protocol on all seven suites, with checkpoint-native image settings,
the authors' prompts/scorers and their per-suite token limits. There are no API
judge calls. See the [protocol audit](../../evaluation/README_paper_protocol.md)
for the differences from the standardized harness and the remaining uncertainty
about the paper's exact recipe.

Stage one evaluates these five checkpoints (35 model/suite jobs):

- `vanilla_2b`
- `vanilla_8b_teacher`
- `released_bf16` — the authors' released BF16/SFT initialization, as confirmed;
  this is not our earlier local `sft_2b` export.
- `released_grace_w4`
- `released_grace_w8`

Stage two evaluates `qat_w4`, `grace_w4`, `qat_w8`, and `grace_w8` (28 jobs),
using the same protocol and output root. The stages run separately; stage one
never launches stage two automatically.

## Commands

From this experiment directory:

```bash
# Optional: inspect the 35-job plan and run CPU-only preflight.
bash run_re_evaluation.sh stage1 --plan
bash run_re_evaluation.sh stage1 --check

# Choose a currently free physical GPU. Replace 0 below as needed.
gpuwho

# Stage 1: five evaluation-only references × seven suites.
bash run_re_evaluation.sh stage1 --gpu 0

# Read the per-suite paper comparison and gate decision.
cat re_evaluation/stage1_gate.md

# Stage 2: four LoRA models × seven suites, only if the reference gate passes.
bash run_re_evaluation.sh stage2 --gpu 0
```

By default **every one of the 35 reference scores must be within ±3 percentage
points** of its published counterpart. If all scores are within ±5 points and
you explicitly accept the barely acceptable range, use:

```bash
bash run_re_evaluation.sh stage2 --gpu 0 --accept-barely
```

The ±8-point judge allowance does not apply: this local protocol uses unjudged
scoring for HallusionBench and MMBench too. Numeric agreement does not establish
that their local proxy metrics match the paper's intended metrics.

Stage two verifies saved artifact hashes, full sample coverage, model paths,
shared question IDs/labels and recomputed scores before starting model inference.
Missing, failed, changed or out-of-range results block it. A summary file alone
cannot unlock stage two. There is no force-through option.

To recheck the gate without evaluation:

```bash
bash run_re_evaluation.sh status
# Or assess the wider, explicitly accepted tolerance:
bash run_re_evaluation.sh status --accept-barely
```

`status` exits 0 on a passing gate and 2 on a blocked gate; integrity errors fail
separately. Stage one exits successfully when all evaluations completed, even if
the score gate is blocked, and prints that distinction. `stage2 --plan` and
`stage2 --check` are available before stage one finishes; neither runs inference.

## Outputs and resuming

All evaluation outputs stay in this experiment's `re_evaluation/`:

```text
re_evaluation/
  logs/stage1_<timestamp>_<pid>.log
  logs/stage2_<timestamp>_<pid>.log
  stage1_gate.json
  stage1_gate.md
  full/
    protocol.json
    paper_comparison.json
    paper_comparison.md
    <model>/
      model_provenance.json
      processor.json
      <suite>/
        COMPLETE.json
        attempt_<timestamp>/
          command.json
          run.log
          results.json
          results.json.rank0.jsonl
          validated.json
```

After stage two, `full/paper_comparison.md` includes all nine evaluated models.
LoRA rows show their local scores with `no_published_target`; published scores
are compared only against the matching five reference models. Stage-one gate
reports remain separate from this combined table.

Rerun the same stage command after an interruption. Validated completed pairs
are skipped; incomplete pairs get a new attempt directory. Changing the pinned
source/environment/protocol is rejected rather than mixing results. The script
uses `.venv-lmms-eval/bin/python` by default; `LMMS_PYTHON` can select another
compatible interpreter. GPU indices use PCI order, and jobs run sequentially on
one selected GPU. No new training or model downloads are initiated.

Validation of this launcher: Bash syntax, 35/28-job plans and 16 evaluation
regression tests passed, including missing/tampered/smoke-result rejection and
the ±3 versus explicit ±5 gate. No GPU evaluation was launched while creating it.
