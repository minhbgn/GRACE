#!/usr/bin/env bash
# Compare vanilla Qwen3-VL-2B-Instruct and the final SFT/QAT/GRACE exports on the seven GRACE table benchmarks.
set -euo pipefail

EVAL_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GRACE_ROOT=$(cd -- "$EVAL_DIR/.." && pwd)
PYTHON=${PYTHON:-python}
NPROC=${NPROC:-2}
ATTN=${ATTN:-flash_attention_2}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-"$GRACE_ROOT/experiments/constrained_32k_seed42"}
OUT_DIR=${OUT_DIR:-"$EXPERIMENT_DIR/evaluation"}
BASELINE_MODEL=${BASELINE_MODEL:-"$GRACE_ROOT/models/Qwen3-VL-2B-Instruct"}
# Reuse preprocessing only; baseline weights are always the original checkpoint.
BASELINE_PROCESSOR=${BASELINE_PROCESSOR:-"$EXPERIMENT_DIR/runs/sft"}
read -r -a METHODS <<< "${METHODS:-baseline sft qat grace}"
CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_DEVICE_ORDER
# Benchmark assets live beside their evaluation code, independently of training.
DATA_ROOT=${DATA_ROOT:-"$EVAL_DIR"}

HALLUSION_DIR=${HALLUSION_DIR:-"$DATA_ROOT/HallusionBench/data"}
MMBENCH_TSV=${MMBENCH_TSV:-"$DATA_ROOT/MMBench/mmbench_dev_en_20231003.tsv"}
SCIENCEQA_META=${SCIENCEQA_META:-"$DATA_ROOT/ScienceQA/data"}
SCIENCEQA_PROBLEMS=${SCIENCEQA_PROBLEMS:-"$SCIENCEQA_META/problems.json"}
SCIENCEQA_SPLITS=${SCIENCEQA_SPLITS:-"$SCIENCEQA_META/pid_splits.json"}
SCIENCEQA_IMAGE_ROOT=${SCIENCEQA_IMAGE_ROOT:-"$SCIENCEQA_META/images"}
AI2D_DIR=${AI2D_DIR:-"$DATA_ROOT/AI2D/data"}
MMMU_DIR=${MMMU_DIR:-"$DATA_ROOT/mmmu/data"}
SEED_ROOT=${SEED_ROOT:-"$DATA_ROOT/SEEDBench/data"}
SEED_JSON=${SEED_JSON:-"$SEED_ROOT/SEED-Bench.json"}
SEED_IMAGE_DIR=${SEED_IMAGE_DIR:-"$SEED_ROOT/SEED-Bench-image"}
MMSTAR_TSV=${MMSTAR_TSV:-"$DATA_ROOT/MMStar/MMStar.tsv"}

usage() {
    cat <<'USAGE'
Usage: bash evaluation/run_all_constrained_evals.sh [OPTIONS]

Run from reproduction/GRACE (or any directory after activating the grace env).
Default: evaluate baseline, sft, qat, grace sequentially on HallusionBench, MMBench,
ScienceQA test, AI2D, MMMU validation, SEED-Bench-IMG, and MMStar.

  --dry-run  Print all 28 commands without checking inputs or using GPUs.
  --check    Validate selected models, processors and dataset paths on CPU, then exit.
  --resume   Skip suites with a merged result and retry suites that have only rank shards.
  --keep-going
             Attempt every selected method/suite after an evaluation command fails, then
             exit nonzero after reporting all failures.

Use METHODS=baseline to add only the seven vanilla-model results to an existing
comparison. METHODS can select any of baseline, sft, qat, grace (space-separated).
BASELINE_MODEL defaults to <GRACE>/models/Qwen3-VL-2B-Instruct (local weights).
BASELINE_PROCESSOR defaults to <EXPERIMENT_DIR>/runs/sft to match the saved
image budget and prompts. This does not load any SFT model weights. Override
BASELINE_PROCESSOR with BASELINE_MODEL to measure the original visual budget
in a separate OUT_DIR; those scores are not a matched training comparison.

Select available GPUs with CUDA_VISIBLE_DEVICES; NPROC defaults to 2.
PYTHON defaults to python in the active environment. ATTN defaults to
flash_attention_2 (can also be sdpa or eager).

Default roots (absolute, resolved relative to this script):
  EXPERIMENT_DIR   <GRACE>/experiments/constrained_32k_seed42
  OUT_DIR          <EXPERIMENT_DIR>/evaluation
  DATA_ROOT        <GRACE>/evaluation
  SCIENCEQA_META   <DATA_ROOT>/ScienceQA/data

Expected dataset layout under DATA_ROOT:
  HallusionBench/data/{image-*,non_image-*}.parquet
  MMBench/mmbench_dev_en_20231003.tsv
  ScienceQA/data/{problems.json,pid_splits.json,images/test/<pid>/<image>}
  AI2D/data/*.parquet                          (test shards only)
  mmmu/data/<Subject>/validation-*.parquet
  SEEDBench/data/{SEED-Bench.json,SEED-Bench-image/<image>}
  MMStar/MMStar.tsv

All roots and individual dataset variables near the top of this script can be
overridden through environment variables. AI2D_DIR must contain ONLY the test
parquet shards; the evaluator reads every parquet in that directory.
SCIENCEQA_IMAGE_ROOT must directly contain test/<pid>/<image>.
No datasets are downloaded. Without --resume, any existing result or rank shard
stops the launch. With --resume, merged results are preserved and incomplete rank
shards are replaced while retrying that suite. Use a fresh OUT_DIR to repeat scores.
These calls use the local Transformers eval_*.py loaders. The separate
mmmu/run_mmmu.py vLLM/TSV downloader is a different workflow.
USAGE
}

MODE=run
RESUME=0
KEEP_GOING=0
while (( $# > 0 )); do
    case "$1" in
        --dry-run)
            [[ "$MODE" == run ]] || { echo 'Choose only one of --dry-run and --check.' >&2; exit 2; }
            MODE=dry-run
            ;;
        --check)
            [[ "$MODE" == run ]] || { echo 'Choose only one of --dry-run and --check.' >&2; exit 2; }
            MODE=check
            ;;
        --resume) RESUME=1 ;;
        --keep-going) KEEP_GOING=1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
    shift
done
[[ "$NPROC" =~ ^[1-9][0-9]*$ ]] || { echo 'NPROC must be a positive integer.' >&2; exit 2; }
case "$ATTN" in
    flash_attention_2|sdpa|eager) ;;
    *) echo 'ATTN must be flash_attention_2, sdpa, or eager.' >&2; exit 2 ;;
esac

(( ${#METHODS[@]} > 0 )) || { echo 'METHODS must not be empty.' >&2; exit 2; }
declare -A seen_methods=()
for method in "${METHODS[@]}"; do
    case "$method" in baseline|sft|qat|grace) ;; *) echo "Unknown method: $method" >&2; exit 2 ;; esac
    [[ -z "${seen_methods[$method]:-}" ]] || { echo "Duplicate method: $method" >&2; exit 2; }
    seen_methods[$method]=1
done

if [[ "$MODE" != dry-run ]]; then
    command -v "$PYTHON" >/dev/null || {
        echo "Python executable not found: $PYTHON. Activate grace or set PYTHON to its executable." >&2
        exit 1
    }
    # Use only the standard library: no torch import and no GPU allocation.
    "$PYTHON" - "$EXPERIMENT_DIR" "$HALLUSION_DIR" "$MMBENCH_TSV" \
        "$SCIENCEQA_PROBLEMS" "$SCIENCEQA_SPLITS" "$SCIENCEQA_IMAGE_ROOT" \
        "$AI2D_DIR" "$MMMU_DIR" "$SEED_JSON" "$SEED_IMAGE_DIR" "$MMSTAR_TSV" \
        "$BASELINE_MODEL" "$BASELINE_PROCESSOR" "${METHODS[@]}" <<'PY'
import json
import sys
from pathlib import Path

(experiment, hallusion, mmbench, problems, splits, science_images,
 ai2d, mmmu, seed_json, seed_images, mmstar, baseline, baseline_processor) = map(Path, sys.argv[1:14])
methods = sys.argv[14:]
errors = []

def require_file(path, label):
    if not path.is_file():
        errors.append(f"{label}: missing {path}")

def require_glob(root, pattern, label):
    if not any(p.is_file() for p in root.glob(pattern)):
        errors.append(f"{label}: no files matching {root / pattern}")

for method in methods:
    model = baseline if method == 'baseline' else experiment / 'runs' / method
    processor = baseline_processor if method == 'baseline' else model
    require_file(model / 'config.json', method)
    require_file(processor / 'tokenizer_config.json', f'{method} processor')
    if not any((processor / name).is_file() for name in ('processor_config.json', 'preprocessor_config.json')):
        errors.append(f"{method}: missing saved processor configuration in {processor}")
    if method != 'baseline':
        require_file(model / 'EXPORT_COMPLETE', method)
        try:
            summary = json.loads((model / 'run_summary.json').read_text())
            if summary.get('complete') is not True or summary.get('profile') is not False:
                errors.append(f"{method}: run must be complete and not a profile")
        except (OSError, ValueError) as exc:
            errors.append(f"{method}: cannot read run_summary.json: {exc}")
    if not any(model.glob('model*.safetensors')) and not any(model.glob('pytorch_model*.bin')):
        errors.append(f"{method}: missing HF model weights in {model}")

require_glob(hallusion, 'image-*.parquet', 'HALLUSION_DIR')
require_glob(hallusion, 'non_image-*.parquet', 'HALLUSION_DIR')
require_file(mmbench, 'MMBENCH_TSV')
require_file(mmstar, 'MMSTAR_TSV')
require_glob(ai2d, '*.parquet', 'AI2D_DIR (test shards only)')
require_glob(mmmu, '*/validation-*.parquet', 'MMMU_DIR')

# The ScienceQA evaluator silently treats absent images as text-only questions.
# Fail here instead, so its image-modality score is meaningful.
try:
    questions = json.loads(problems.read_text())
    test_ids = json.loads(splits.read_text())['test']
    missing = []
    if not test_ids:
        errors.append('ScienceQA: empty test split')
    for pid in test_ids:
        item = questions[str(pid)]
        if item.get('image'):
            path = science_images / 'test' / str(pid) / item['image']
            if not path.is_file():
                missing.append(path)
    if missing:
        errors.append(f"SCIENCEQA_IMAGE_ROOT: {len(missing)} missing images; first: {missing[0]}")
except (OSError, ValueError, KeyError, TypeError) as exc:
    errors.append(f"ScienceQA metadata ({problems}, {splits}): {exc}")

try:
    data = json.loads(seed_json.read_text())
    items = data['questions'] if isinstance(data, dict) else data
    items = [item for item in items if item.get('data_type') == 'image']
    if not items:
        errors.append(f"SEED_JSON: no image questions in {seed_json}")
    missing = []
    for item in items:
        data_id = item.get('data_id') or ''
        names = [data_id]
        if '.' not in Path(data_id).name:
            names += [data_id + suffix for suffix in ('.jpg', '.png', '.jpeg')]
        if not any((seed_images / name).is_file() for name in names):
            missing.append(data_id)
    if missing:
        errors.append(f"SEED_IMAGE_DIR: {len(missing)} missing images under {seed_images}; first: {missing[0]}")
except (OSError, ValueError, KeyError, TypeError) as exc:
    errors.append(f"SEED_JSON ({seed_json}): {exc}")

if errors:
    print('Preflight failed; no evaluation was launched:', file=sys.stderr)
    for error in errors:
        print(f'  - {error}', file=sys.stderr)
    sys.exit(1)
print(f'Preflight passed: {len(methods)} selected models/processors and all seven dataset paths.')
PY
    [[ "$MODE" != check ]] || exit 0

    if (( ! RESUME )); then
        # Check every destination before starting any suite.
        for method in "${METHODS[@]}"; do
            for suite in hallusionbench mmbench scienceqa ai2d mmmu seedbench_img mmstar; do
                output="$OUT_DIR/$method/$suite.json"
                if [[ -e "$output" ]] || compgen -G "$output.rank*.jsonl" >/dev/null; then
                    echo "Existing results: $output (or rank shards). Set OUT_DIR to a fresh directory or use --resume." >&2
                    exit 1
                fi
            done
        done
    fi
fi

run_eval() {
    local suite=$1 script=$2 tokens=$3
    shift 3
    local output="$OUT_DIR/$method/$suite.json"
    local command=("$PYTHON" -m torch.distributed.run --standalone
        --nproc_per_node="$NPROC" "$EVAL_DIR/$script"
        --model_path "$model_path"
        --output "$output"
        --attn "$ATTN" --max_new_tokens "$tokens" "$@")
    if [[ "$method" == baseline ]]; then
        command+=(--processor_path "$BASELINE_PROCESSOR")
    fi
    printf '\n[%s / %s]\n' "$method" "$suite"
    if [[ "$MODE" != check ]] && (( RESUME )) && [[ -e "$output" ]]; then
        printf 'Skipping completed result: %s\n' "$output"
        return 0
    fi
    if [[ "$MODE" != check ]] && (( RESUME )) && compgen -G "$output.rank*.jsonl" >/dev/null; then
        printf 'Retrying incomplete result; existing rank shards will be replaced: %s.rank*.jsonl\n' "$output"
    fi
    if [[ "$MODE" == dry-run ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
}

failures=()
run_or_record_failure() {
    local suite=$1
    shift
    if run_eval "$suite" "$@"; then
        return 0
    else
        local status=$?
        failures+=("$method/$suite (exit $status)")
        echo "Evaluation failed: $method/$suite (exit $status)" >&2
        (( KEEP_GOING )) || exit "$status"
    fi
}

for method in "${METHODS[@]}"; do
    model_path="$EXPERIMENT_DIR/runs/$method"
    [[ "$method" != baseline ]] || model_path="$BASELINE_MODEL"
    run_or_record_failure hallusionbench HallusionBench/eval_hallusion.py 8 --data_dir "$HALLUSION_DIR"
    run_or_record_failure mmbench MMBench/eval_mmbench.py 8 --tsv "$MMBENCH_TSV"
    run_or_record_failure scienceqa ScienceQA/eval_scienceqa.py 8 \
        --problems "$SCIENCEQA_PROBLEMS" --splits "$SCIENCEQA_SPLITS" \
        --image_root "$SCIENCEQA_IMAGE_ROOT" --split test
    run_or_record_failure ai2d AI2D/eval_ai2d.py 8 --data_dir "$AI2D_DIR"
    run_or_record_failure mmmu mmmu/eval_mmmu.py 32 --data_dir "$MMMU_DIR" --split validation
    run_or_record_failure seedbench_img SEEDBench/eval_seedbench.py 8 \
        --json_path "$SEED_JSON" --image_dir "$SEED_IMAGE_DIR"
    run_or_record_failure mmstar MMStar/eval_mmstar.py 8 --tsv "$MMSTAR_TSV"
done

if (( ${#failures[@]} > 0 )); then
    printf '\n%d evaluation(s) failed; all selected evaluations were attempted:\n' "${#failures[@]}" >&2
    printf '  - %s\n' "${failures[@]}" >&2
    exit 1
fi
