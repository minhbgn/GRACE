# Released-SFT QAT/GRACE controls, 576-cap / 4,096 tokens

All four production arms completed 256 updates. The historical local evaluation reports QAT W4/W8 means **75.77/77.37**, GRACE **74.56/74.86**, and GRACE−QAT **−1.22/−2.51 pp**. Read [the control report](experiment_report.md) and [the combined experiment report](../REPORT.md). The original 28 evaluation predictions are absent in this checkout; their scores are labelled report-only in the merged results. The checkpoint exports are present and pass the current CPU preflight.

The shared initialization is the authors' **BF16 SFT** checkpoint, without GDKD/RCKA. QAT uses CE-only LoRA-QAT; GRACE adds GDKD/RCKA. The recipe uses 32,768 shared examples, batch 128, two GPUs, microbatch one, accumulation 64, seed 42, LoRA rank/alpha 32/64, W4/W8 group size 128, minimum/maximum image pixels 4,096/589,824 and a 4,096 total-token ceiling. Mean visual count is 254.64, maximum 576; maximum total length is 2,209. The `t2048` directory is retained only as historical preparation evidence.

From `reproduction/GRACE`, reproduce the seven native-processor image suites:

```bash
bash evaluate.sh plan --workflow controls-images
bash evaluate.sh check --workflow controls-images
bash evaluate.sh run --workflow controls-images --gpus 0 --smoke 2
bash evaluate.sh run --workflow controls-images --gpus 0
bash evaluate.sh summarize --workflow controls-images
```

New outputs go under `results/reruns/`. `evaluation.json` selects the four local QAT/GRACE arms and available frozen references, excluding the old local `sft_2b`. `--models all` adds the available references; they retain different native processor budgets. These research exports load in BF16. Packed INT4 evaluation is the separate [token-reduction study](../qat_int4_token_reduction_576_seed42/README.md).

Completed training need not be repeated for evaluation. The original training and memory helpers remain available for reproduction, with completed outputs protected from overwrite:

```bash
EXP=experiments/released_bf16_init_32k_b128_p589824_t4096_seed42
bash "$EXP/run_training.sh" all --train --dry-run
bash "$EXP/run_training.sh" all --prepare-only
# Select free cards before GPU work. Do not overwrite completed production arms.
CUDA_VISIBLE_DEVICES=0,2 bash "$EXP/run_memory_checks.sh"
# Individual unfinished training arm, if reproducing in a fresh experiment:
# CUDA_VISIBLE_DEVICES=0,2 bash "$EXP/run_training.sh" qat --bits 4 --train
```

See the [original recipe/readme](../../reports/sources/released_bf16_init_32k_b128_p589824_t4096_seed42/README.md) for preparation, export, memory-screen and resume details. Synthetic memory checks use a different input envelope from production; their peaks are not production memory measurements. `recipe.env`, `experiment.json`, checkpoint provenance and subsets remain unchanged.

Older video launchers preserve distinct 256-pixel local and 720p lmms protocols. The old video report is [historical](experiment_report_video.md); use [the unified experiment command reference](../README.md) for the completed packed-INT4 suite. Dataset/code/protocol mismatches must select a fresh output rather than editing saved hashes.
