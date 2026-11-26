import json,sys
from pathlib import Path
E=Path('/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t2048_seed42');R=E.parents[1]
sys.path.insert(0,str(R/'qwen-vl-finetune/scripts'))
from constrained_experiment import Experiment,sha
new=Experiment(E/'experiment.json');old=Experiment(E.parent/'released_bf16_init_32k_b128_p100352_t2048_seed42/experiment.json')
a=json.loads((E/'reports/comparison_audit.json').read_text())
for key,v in a['evaluations'].items():
 label,method=key.split('/');spec=new if label=='new' else old
 for suite,entry in v['suites'].items():
  payload=json.loads(Path(entry['results_path']).read_text())
  assert Path(payload['summary']['model_path']).resolve()==spec.paths[method]
for method in ['qat_w4','grace_w4','qat_w8','grace_w8']:
 new.validate_export(method)
 run=new.paths[method];launch=json.loads((run/'launch_recipe.json').read_text());comp=launch['comparison'];bits=new.models[method]['bits']
 assert comp==json.loads((E/f'w{bits}g128/comparison_recipe.json').read_text())
 assert comp['experiment_config_sha256']==sha(E/'experiment.json')
 assert comp['subset_sha256']==a['subset']['records_sha256']
 assert json.loads((run/'trainer_state.json').read_text())['global_step']==256
 for name in ['student','teacher']:
  model=getattr(new,name);assert comp[name+'_provenance_sha256']==sha(E/'provenance'/f'{model}.json')
a['supplemental_verification']=dict(all_91_result_model_paths_match_config=True,all_four_new_exports_pass=True,
 current_experiment_hash_matches_training=True,all_four_share_subset_and_pinned_recipe=True,
 current_checkpoint_weight_bytes_rehashed=False)
(E/'reports/comparison_audit.json').write_text(json.dumps(a,indent=2)+'\n')
print(a['supplemental_verification'])
