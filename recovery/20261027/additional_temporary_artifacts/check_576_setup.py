import ast, json, os, shlex, subprocess
from pathlib import Path
root=Path('/raid/lvchengtao/Minh/Distillation')
exp=root/'reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t2048_seed42'
for p in exp.glob('*.sh'):
    subprocess.run(['bash','-n',str(p)],check=True)
ast.parse((exp/'profile_memory.py').read_text())
source=exp.parent/'released_bf16_init_32k_b128_p100352_t2048_seed42'
new=json.loads((exp/'experiment.json').read_text()); old=json.loads((source/'experiment.json').read_text())
assert new['models']==old['models'] and new['bit_widths']==old['bit_widths']
cmds=subprocess.check_output(['bash',str(exp/'run_training.sh'),'all','--train','--dry-run'],text=True,
    env=dict(os.environ, EXPERIMENT_DIR='/tmp/wrong',BASE_MODEL_DIR='/tmp/wrong',TEACHER_MODEL_PATH='/tmp/wrong',SFT_MODEL_DIR='/tmp/wrong'))
lines=cmds.strip().splitlines(); assert len(lines)==4
for line in lines:
    c=shlex.split(line.split('] ',1)[1])
    for flag,value in {'--max_pixels':'589824','--min_pixels':'4096','--model_max_length':'2048',
                       '--max_steps':'256','--gradient_accumulation_steps':'64','--rcka_warmup_steps':'25',
                       '--profile_run':'False','--lora_r':'32'}.items():
        assert c[c.index(flag)+1]==value,(flag,line)
    assert c[c.index('--output_dir')+1].startswith(str(exp))
    assert c[c.index('--model_name_or_path')+1].endswith('models/Qwen3-VL-2B-GRACE-BF16')
    assert '/tmp/wrong' not in line
(exp/'training_dry_run.txt').write_text(cmds)
plan=subprocess.check_output(['bash',str(exp/'run_local_evaluation.sh'),'plan'],text=True)
(exp/'evaluation_plan.json').write_text(plan)
obj=json.loads(plan)
print('Training: all four commands isolated; cap 589824, length2048, batch128, updates256, warmup25.')
print('Evaluation plan keys:',list(obj) if isinstance(obj,dict) else len(obj))
# Verify plan without making assumptions about its outer JSON structure.
assert str(exp/'w4g128/runs/qat') in plan and str(exp/'w8g128/runs/grace') in plan
assert 'local_native_greedy256_v1' in plan
assert '--max_pixels' not in plan and '--min_pixels' not in plan
print('Shell/Python syntax and native-processor evaluation planning: passed.')
