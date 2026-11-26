# Vanilla Qwen3-VL-2B baseline

The authors' [Qwen3-VL release results](https://github.com/ForeverBlue816/GRACE#-results)
explicitly identify **vanilla Qwen3-VL-2B-Instruct** as their BF16 baseline.
It has the upstream instruction tuning, but no additional SFT, QAT, or GRACE
training from this experiment. The paper itself evaluates LLaVA and Qwen2-VL;
the Qwen3-VL comparison belongs to the released repository.

`run_all_constrained_evals.sh` now runs `baseline sft qat grace` by default
(28 evaluations). To add the baseline to the already completed experiment:

```bash
conda activate grace
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4  # choose a GPU with enough available memory
export NPROC=1
export METHODS=baseline
bash evaluation/run_all_constrained_evals.sh --dry-run
bash evaluation/run_all_constrained_evals.sh --check
bash evaluation/run_all_constrained_evals.sh
python evaluation/summarize_constrained_evals.py --write
```

These commands run all seven configured suites: HallusionBench, MMBench English
dev, ScienceQA full test, AI2D test, MMMU validation, SEED-Bench-IMG, and MMStar.
Predictions are saved under `<EXPERIMENT_DIR>/evaluation/baseline/`.
The summarizer adds Vanilla 2B scores and GRACE−Vanilla 2B deltas to
`comparison.md` and `comparison.csv`, including category breakdowns.
It validates identical question identities and gold labels across all methods,
and refuses an incomplete baseline directory. Older result directories without
a baseline still support the original three-method comparison.

## Matched preprocessing

`BASELINE_MODEL` defaults to `<GRACE>/models/Qwen3-VL-2B-Instruct`.
**Model weights always come from this original checkpoint**, in BF16.
`BASELINE_PROCESSOR` defaults to `<EXPERIMENT_DIR>/runs/sft` and loads only
its tokenizer, chat template, and image preprocessing configuration.
For `constrained_32k_seed42`, this matches the existing evaluations' minimum
4,096 and maximum 50,176 image pixels. The prompts, splits, decoding limits,
and scorers are the same as the SFT/QAT/GRACE runs. Processor source and image
budget are recorded in every new merged result file.

To measure the vanilla checkpoint with its original, larger visual budget,
set `BASELINE_PROCESSOR` to the same path as `BASELINE_MODEL` and set `OUT_DIR`
to a fresh directory. That is a different protocol and should be reported
separately from the matched training comparison. Local scores should not be
assumed equivalent to the authors' published benchmark scores.

`EXPERIMENT_DIR`, `OUT_DIR`, `BASELINE_MODEL`, `BASELINE_PROCESSOR`, `PYTHON`,
`NPROC`, and dataset paths remain configurable through environment variables.
The launcher refuses existing merged results or rank shards; a fresh `OUT_DIR`
is required to repeat a run.
