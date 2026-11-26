# Evaluate the constrained exports with lmms-eval

**Executable follow-up (2026-09-23):** See the [released-BF16 experiment workflow](../experiments/released_bf16_init_32k_b128_p100352_t2048_seed42/lmms_evaluation.md) for the pinned runner, environment setup, integrity checks and 70-job matrix. The findings and manual commands below are the earlier experiment notes.

Yes: use the `qwen3_vl` backend with the final directories
`experiments/constrained_32k_seed42/runs/{sft,qat,grace}`. Do not point it at
`profiles/` or native `checkpoint-*` directories. The exports have merged
adapters and ordinary Hugging Face model weights. QAT/GRACE weights have already
been projected onto the quantization grid; do not quantize them again when loading.
Their BF16 storage is suitable for accuracy evaluation, not packed-INT4 speed or
memory measurements.

The existing `run_all_constrained_evals.sh` runs custom Transformers evaluators,
not lmms-eval. The original ScienceQA Slurm example also invokes a custom
evaluator, expects `MODEL_PATH` rather than `MODEL`, and has cluster-specific
paths. Using lmms-eval provides a common evaluation protocol for our models;
it does not by itself reproduce the paper's exact prompts, splits and scores.

Use the local launcher for an initial comparison when the seven local datasets
are already available; prefer this pinned lmms-eval workflow for the main
benchmark report. Keep the two sets of results separate. The local ScienceQA
overall score includes text-only questions, whereas `scienceqa_img` selects
image questions. Local HallusionBench includes image and non-image partitions
and uses rule-based scoring; `hallusion_bench_image` selects the image partition
and its default scorer uses a judge. Even identically named metrics need not
measure the same protocol.

Inspect raw answers before trusting local scores: the local multiple-choice
parsers scan characters for the first valid letter (for example, "The answer
is B" can be scored as A). The local MMMU open-answer scorer is also a simplified
numeric/text matcher. Shared evaluation code alone does not eliminate scoring
bias when models produce different response formats.

## Findings from this experiment (2026-09-16)

CPU inspection of the saved summaries, metrics, configurations and safetensors
headers found:

| Run | Completed updates | Training-loop time | Peak reserved GPU memory |
| --- | ---: | ---: | ---: |
| SFT | 512 | 1.45 h | 6.59 GiB |
| QAT | 512 | 1.90 h | 6.64 GiB |
| GRACE | 512 | 2.37 h | 24.13 GiB |

All three have `EXPORT_COMPLETE`, `complete=true`, `profile=false`, consecutive
steps 1–512, and no nonfinite numeric values in their step metrics. Each export
has 626 BF16 tensors and no remaining `lora_` tensor keys. This is a structural
check, not a check of every weight value or a successful inference test.

GRACE's logged IB beta reaches its upper bound of 1.0 at step 7 and stays there
for 506/512 steps. This is consistent with the controller's clipping rule, but
means this run exercises little adaptation after the beginning. Keep that in
mind when interpreting the strategy. GRACE's total training loss includes
distillation and RCKA, so comparing it numerically with SFT/QAT CE loss is invalid.

The existing `grace` environment contains lmms-eval **0.7.1**, Transformers
**5.9.0**, Torch **2.5.1+cu121**, and qwen-vl-utils **0.0.14**. Two concrete
installation problems were found:

- Importing its Qwen3-VL backend fails with `ModuleNotFoundError: decord`.
- MMStar, MMMU and MMBench YAMLs reference extensionless `_default_template*`
  files absent from the installed package.

The commands below use an editable source installation to retain those files.
No installation, dataset download or GPU evaluation was performed for this guide.

## Prepare an evaluation environment

Run from GRACE. This small virtual environment inherits the existing training
dependencies while installing evaluation additions locally. It does not replace
packages in the `grace` conda environment.

```bash
cd /raid/lvchengtao/Minh/Distillation/reproduction/GRACE
conda activate grace
python -m venv --system-site-packages .venv-lmms-eval
source .venv-lmms-eval/bin/activate

mkdir -p third_party
git clone --branch v0.7.1 --depth 1 \
  https://github.com/EvolvingLMMs-Lab/lmms-eval.git third_party/lmms-eval
python -m pip install --no-deps decord==0.6.0
python -m pip install --no-deps -e third_party/lmms-eval

# CPU import and source-template check; no model loading.
CUDA_VISIBLE_DEVICES='' python - <<'PY'
from pathlib import Path
import lmms_eval
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL

root = Path(lmms_eval.__file__).resolve().parent
print('Using:', root)
for relative in (
    'tasks/mmstar/_default_template_yaml',
    'tasks/mmmu/_default_template_yaml',
    'tasks/mmbench/_default_template_mmbench_en_yaml',
):
    assert (root / relative).is_file(), relative
print('Backend import and template checks passed')
PY
```

If the clone already exists, inspect its revision instead of cloning over it.
Keep one source revision and environment for every comparison. `--no-deps`
preserves the inherited dependency versions; the CPU check and smoke test are
still required to establish compatibility.

lmms-eval loads the datasets declared in its task YAMLs through Hugging Face.
It does not use the local TSV/JSON paths in `README_datasets.md`. Allow network
access on the first run, then reuse the same cached datasets for every model.
Some v0.7.1 tasks specify `token: True`; use `hf auth login` if no saved Hugging
Face token is available. Old `lmms-lab/*` dataset names may need a source-path
update if redirects no longer resolve; record any YAML changes and use them
for all models.

## Smoke test, then full comparison

First select an available GPU. These commands use one GPU and batch size one.
SFT, QAT and GRACE are evaluated sequentially. No teacher is needed.

```bash
export CUDA_VISIBLE_DEVICES=0  # replace with an available physical GPU
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
export EXPERIMENT_DIR="$PWD/experiments/constrained_32k_seed42"

set -euo pipefail
# mkdir intentionally fails if this result directory already exists.
OUT="$EXPERIMENT_DIR/lmms_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir "$OUT"
git -C third_party/lmms-eval rev-parse HEAD > "$OUT/lmms_revision.txt"
python -m pip freeze > "$OUT/environment.txt"

for method in sft qat grace; do
  python -m lmms_eval \
    --model qwen3_vl \
    --model_args "pretrained=$EXPERIMENT_DIR/runs/$method,device_map=cuda:0,attn_implementation=sdpa,min_pixels=4096,max_pixels=50176" \
    --tasks scienceqa_img,ai2d,mmstar \
    --batch_size 1 --num_fewshot 0 --seed 42 \
    --gen_kwargs "max_new_tokens=32,temperature=0,do_sample=False,num_beams=1" \
    --limit 8 --log_samples \
    --output_path "$OUT/$method"
done
```

Inspect the logged responses and confirm images load, answers address the
question, and generations are not empty or repeatedly cut off. Eight examples
per task check the pipeline; their accuracy is not a performance estimate.

For the full comparison, use a fresh output root and remove `--limit`. Include
the original student as a fourth row to distinguish fine-tuning gains from
quantization/distillation effects:

```bash
OUT="$EXPERIMENT_DIR/lmms_full_$(date +%Y%m%d_%H%M%S)"
mkdir "$OUT"
git -C third_party/lmms-eval rev-parse HEAD > "$OUT/lmms_revision.txt"
python -m pip freeze > "$OUT/environment.txt"

for method in base sft qat grace; do
  MODEL_PATH="$EXPERIMENT_DIR/runs/$method"
  if [[ "$method" == base ]]; then
    MODEL_PATH="$PWD/models/Qwen3-VL-2B-Instruct"
  fi
  python -m lmms_eval \
    --model qwen3_vl \
    --model_args "pretrained=$MODEL_PATH,device_map=cuda:0,attn_implementation=sdpa,min_pixels=4096,max_pixels=50176" \
    --tasks scienceqa_img,ai2d,mmstar \
    --batch_size 1 --num_fewshot 0 --seed 42 \
    --gen_kwargs "max_new_tokens=32,temperature=0,do_sample=False,num_beams=1" \
    --log_samples --output_path "$OUT/$method"
done
```

The explicit pixel bounds matter: lmms-eval's backend overrides the saved
processor defaults. The chosen bounds match the constrained training exports,
including for the untuned base model. SDPA avoids a FlashAttention dependency;
using `flash_attention_2` consistently for every row is also possible.
Generation is greedy with a common 32-token cap. These are our declared
comparison settings, not a claim about the paper's settings.

To use two available GPUs, set `CUDA_VISIBLE_DEVICES` accordingly and replace
`python -m lmms_eval` with:

```bash
python -m accelerate.commands.launch --multi_gpu --num_processes 2 \
  --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  --main_process_port 29531 -m lmms_eval
```

Append the same arguments from above. The Qwen3-VL backend assigns each process
to its local GPU; choose a free rendezvous port if another job uses 29531.

## Extend to the seven benchmark families

Check the selected revision's YAML and scorer before including additional
tasks. Do not combine local-evaluator and lmms-eval scores in one comparison.

| Benchmark | lmms-eval task | Interpretation / extra requirements |
| --- | --- | --- |
| ScienceQA image subset | `scienceqa_img` | Use `exact_match`; `scienceqa` is a different scope. |
| AI2D | `ai2d` | Use `exact_match`; keep diagram variant fixed. |
| MMStar | `mmstar` | Use overall `average`, and retain category scores. |
| MMMU validation | `mmmu_val` | Use `mmmu_acc`; smoke-test multi-image cases separately. Its Qwen3-VL prompt override also formats open questions with options; inspect the saved prompts before accepting the protocol. |
| SEED-Bench | `seedbench` | Select `seed_image` for the image comparison. The task also processes video-category questions represented as images; `seed_all` is not SEED-Bench-IMG. |
| MMBench English dev | `mmbench_en_dev` | The installed scorer uses an external judge for answer matching and includes submission output. Configure and record the judge protocol. |
| HallusionBench image | `hallusion_bench_image` | The installed scorer uses an external judge and reports `aAcc`, `qAcc`, `fAcc`; it is not the local script's combined image/non-image yes/no accuracy. |

The first three tasks above offer a useful initial comparison without external
judge calls. Add each further task to a small smoke run before a full run.
External judge tasks need credentials, can incur cost, and must use the same
judge configuration for every model.

## Decide whether the strategy helps

Use **GRACE minus QAT** per benchmark as the main distillation comparison; use
**QAT minus SFT** for the combined quantization-training/export penalty, and
compare SFT with the original base model for the fine-tuning effect. SFT is a
BF16 reference, not an INT4 baseline. Report accuracy in percentage points,
normalizing metric scales before computing any explicitly defined macro mean.

Retain sample IDs, raw predictions, task configurations, dataset fingerprints,
sample counts and the environment/revision records. Compare paired correctness
on identical examples and estimate uncertainty (for example, a paired bootstrap;
resample image/question groups where examples are dependent). Do not discard
inference failures from the denominator or report a partial run as complete.
Check overlap with the actual training subset before making generalization claims.

If GRACE loses, examine its beta saturation, CE/distillation balance, response
format errors, and quantization/export loss before changing the training recipe.
Native LoRA computes `Q(W)x + LoRA(x)`, whereas export computes
`Q(W + delta_W)x`; the gap can change quality. Comparing native and exported
models requires the native training loader, not pointing lmms-eval at a native
checkpoint. The 50,176-pixel cap can also limit text/diagram tasks: a second,
separately labeled evaluation at higher resolution should apply to every model.
Use validation data for tuning and later repeat with matched training seeds;
one seed's benchmark result alone cannot establish robustness or absence of bugs.

Sources: [lmms-eval v0.7.1](https://github.com/EvolvingLMMs-Lab/lmms-eval/releases/tag/v0.7.1),
[Qwen3-VL backend](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/v0.7.1/lmms_eval/models/simple/qwen3_vl.py),
[task definitions](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/v0.7.1/lmms_eval/tasks).
