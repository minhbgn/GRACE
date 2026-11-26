#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
NPROC=${NPROC:-2}
OUT_DIR="$EXPERIMENT_DIR/evaluation"

run_eval() {
    "$PYTHON" -m torch.distributed.run \
        --standalone \
        --nproc_per_node="$NPROC" \
        "$@"
}

for method in sft qat grace; do
    model="$EXPERIMENT_DIR/runs/$method"
    out="$OUT_DIR/$method"

    test -f "$model/EXPORT_COMPLETE" || {
        echo "Incomplete or missing export: $model"
        exit 1
    }

    mkdir -p "$out"

    echo "===== $method: HallusionBench ====="
    run_eval evaluation/HallusionBench/eval_hallusion.py \
        --model_path "$model" \
        --data_dir "$DATA_ROOT/HallusionBench/data" \
        --output "$out/hallusionbench.json" \
        --max_new_tokens 8

    echo "===== $method: MMBench ====="
    run_eval evaluation/MMBench/eval_mmbench.py \
        --model_path "$model" \
        --tsv "$DATA_ROOT/mmbench/mmbench_dev_en_20231003.tsv" \
        --output "$out/mmbench.json" \
        --max_new_tokens 8

    echo "===== $method: ScienceQA ====="
    run_eval evaluation/ScienceQA/eval_scienceqa.py \
        --model_path "$model" \
        --problems "$SCIENCEQA_META/problems.json" \
        --splits "$SCIENCEQA_META/pid_splits.json" \
        --image_root "$DATA_ROOT/ScienceQA" \
        --split test \
        --output "$out/scienceqa.json" \
        --max_new_tokens 8

    echo "===== $method: AI2D ====="
    run_eval evaluation/AI2D/eval_ai2d.py \
        --model_path "$model" \
        --data_dir "$DATA_ROOT/AI2D/data" \
        --output "$out/ai2d.json" \
        --max_new_tokens 8

    echo "===== $method: MMMU ====="
    run_eval evaluation/mmmu/eval_mmmu.py \
        --model_path "$model" \
        --data_dir "$DATA_ROOT/MMMU" \
        --split validation \
        --output "$out/mmmu.json" \
        --max_new_tokens 32

    echo "===== $method: SEED-Bench-IMG ====="
    run_eval evaluation/SEEDBench/eval_seedbench.py \
        --model_path "$model" \
        --json_path "$DATA_ROOT/SEED-Bench/SEED-Bench.json" \
        --image_dir "$DATA_ROOT/SEED-Bench/SEED-Bench-image" \
        --output "$out/seedbench_img.json" \
        --max_new_tokens 8

    echo "===== $method: MMStar ====="
    run_eval evaluation/MMStar/eval_mmstar.py \
        --model_path "$model" \
        --tsv "$DATA_ROOT/MMStar/MMStar.tsv" \
        --output "$out/mmstar.json" \
        --max_new_tokens 8
done