from pathlib import Path
import json
root=Path('/raid/lvchengtao/Minh/Distillation/reproduction/GRACE')
old=root/'experiments/released_bf16_init_32k_b128_p100352_t2048_seed42'
new=root/'experiments/released_bf16_init_32k_b128_p589824_t2048_seed42'
new.mkdir(exist_ok=False)
spec=json.loads((old/'experiment.json').read_text())
spec['previous_experiment']=str(old.relative_to(root))
spec['report_title']='Released-BF16 SFT initialization: 576-visual-token ceiling'
spec['report_note']='New independent QAT/GRACE W4/W8 runs from the same released SFT initialization; maximum image area 589824 pixels (at most 576 merged Qwen3-VL visual tokens), total training input limit 2048. Compare with the previous experiment using its saved predictions only under explicitly labeled protocols; resolution and retained conversations may differ.'
spec['visual_budget']={'max_pixels':589824,'min_pixels':4096,'merged_token_ceiling':576,'patch_size':16,'merge_size':2,'model_max_length':2048}
(new/'experiment.json').write_text(json.dumps(spec,indent=2)+'\n')
recipe=(root/'qwen-vl-finetune/scripts/recipes/constrained_32k_b128_p100352_t2048_seed42.env').read_text()
recipe=recipe.replace('# Source through run_constrained_b128_p100352_t2048.sh.', '# Experiment-local recipe: source through run_training.sh.').replace('export MAX_PIXELS=100352','export MAX_PIXELS=589824')
(new/'recipe.env').write_text(recipe)
(new/'run_training.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
source "$HERE/recipe.env"
# Never redirect this experiment into a previous run through a stale shell export.
unset EXPERIMENT_DIR BASE_MODEL_DIR TEACHER_MODEL_PATH SFT_MODEL_DIR
PYTHON_PATH="${PYTHON:-/home/air/miniconda3/envs/grace/bin/python}"
exec "$PYTHON_PATH" "$ROOT/qwen-vl-finetune/scripts/run_constrained_matrix.py" --config "$HERE/experiment.json" "$@"
''')
(new/'run_local_evaluation.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
unset EXPERIMENT_DIR BASE_MODEL_DIR TEACHER_MODEL_PATH SFT_MODEL_DIR
PYTHON_PATH="${LMMS_PYTHON:-$ROOT/.venv-lmms-eval/bin/python}"
# This output root is separate from all historical evaluations.
exec "$PYTHON_PATH" "$ROOT/evaluation/run_local_development.py" \\
  --experiment "$HERE/experiment.json" --output "$HERE/local_development" \\
  --protocol local_native_greedy256_v1 --methods lora "$@"
''')
(new/'profile_memory.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/recipe.env"
unset EXPERIMENT_DIR BASE_MODEL_DIR TEACHER_MODEL_PATH SFT_MODEL_DIR
exec "${PYTHON:-/home/air/miniconda3/envs/grace/bin/python}" "$HERE/profile_memory.py" "$@"
''')
for p in new.glob('*.sh'): p.chmod(0o755)
print(new)
