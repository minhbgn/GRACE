"""CPU audit and aggregation of new and evaluation-only saved predictions."""
from collections import defaultdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import sys

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0,str(ROOT/'evaluation'))
from run_local_development import validate, metadata, sha, verified_completion
from summarize_constrained_evals import ID_FIELDS
from rescore_saved_protocols import hallusion

SUITES = ['hallusionbench','mmbench','scienceqa','ai2d','mmmu','seedbench_img','mmstar']
LOCAL = ['qat_w4','grace_w4','qat_w8','grace_w8']
REFS = ['vanilla_2b','vanilla_8b_teacher','released_bf16','released_grace_w4','released_grace_w8']
BASE = ROOT/'experiments/reproduction_eval_only'
OLD = ROOT/'experiments/released_bf16_init_32k_b128_p100352_t2048_seed42'
SUFFIX = Path('local_development/local_native_greedy256_v1/full')


def read(p): return json.loads(p.read_text())
def differences(a,b): return {k:dict(old=a.get(k),new=b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
def stats(a):
 import numpy as np
 x=np.array(a);return dict(count=len(a),min=float(x.min()),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.percentile(x,95)),p99=float(np.percentile(x,99)),max=float(x.max()))
def rows_by_id(rows,suite):return {tuple(r[k] for k in ID_FIELDS[suite]):r for r in rows}


def main():
 import numpy as np
 result=dict(date='2026-10-12',protocol='local_native_greedy256_v1',evaluations={},training={},comparisons={},sources={})
 data={}; identities={}
 for label,exp,methods in [('baseline',BASE,REFS+LOCAL),('new',EXP,LOCAL)]:
  full=exp/SUFFIX; policy=read(full/'protocol.json');combined=read(full/'local_scores.json')
  summary_map={(r['model'],r['suite']):r for r in combined['rows']}
  result['sources'][label]=dict(root=str(full),policy_sha256=sha(full/'protocol.json'),artifact_ready=combined['artifact_ready'],policy=policy)
  for method in methods:
   proc=read(full/method/'processor.json'); entries={}; data[label,method]={}
   for suite in SUITES:
    job=full/method/suite;done=verified_completion(job)
    if done:
     path=job/done['attempt']/'results.json'
     for k,f in [('protocol_sha256',full/'protocol.json'),('processor_sha256',full/method/'processor.json'),('model_provenance_sha256',full/method/'model_provenance.json')]:
      assert done[k]==sha(f),(label,method,suite,k)
    else:
     assert label=='baseline' and suite=='mmbench'
     files=list(job.glob('*/results.json'));assert len(files)==1,(job,len(files));path=files[0]
    payload=read(path)
    values=validate(payload,suite,Path(payload['summary']['model_path']),policy['name'],policy['attention'],None,proc)
    assert values['sample_identity']==identities.setdefault(suite,values['sample_identity'])
    if done:
     for k,v in values.items():assert done[k]==v,(label,method,suite,k)
     for k,v in values.items():assert summary_map[method,suite][k]==v,(label,method,suite,k)
    entry=dict(**values,completion_marker=bool(done),results_path=str(path),results_sha256=sha(path),
               errors=sum(bool(r.get('error')) for r in payload['results']),image_size=proc['image_size'])
    if suite=='mmbench':entry['groups']=payload['summary']['circular']['groups']
    entries[suite]=entry;data[label,method][suite]=payload['results']
   result['evaluations'][label+'/'+method]=dict(suites=entries,mean=sum(e['score_percent'] for e in entries.values())/7,
      six_suite_mean=sum(e['score_percent'] for s,e in entries.items() if s!='mmstar')/6,
      unresolved=sum(e['unresolved'] for e in entries.values()),truncated=sum(e['truncated'] for e in entries.values()))
   print('Validated',label,method,flush=True)
 result['policy_differences']=differences(result['sources']['baseline']['policy'],result['sources']['new']['policy'])
 pairs=[('new',m,'baseline',m) for m in LOCAL]+[('new',f'grace_w{b}','new',f'qat_w{b}') for b in (4,8)]+[('baseline',f'grace_w{b}','baseline',f'qat_w{b}') for b in (4,8)]
 for ll,l,rr,r in pairs:
  left,right=result['evaluations'][ll+'/'+l],result['evaluations'][rr+'/'+r]
  value=dict(mean_delta=left['mean']-right['mean'],six_suite_delta=left['six_suite_mean']-right['six_suite_mean'],
             deltas={s:left['suites'][s]['score_percent']-right['suites'][s]['score_percent'] for s in SUITES})
  a,b=rows_by_id(data[ll,l]['mmstar'],'mmstar'),rows_by_id(data[rr,r]['mmstar'],'mmstar');assert a.keys()==b.keys()
  value['mmstar_paired']=dict(only_left_correct=sum(a[k]['correct'] and not b[k]['correct'] for k in a),
                             only_right_correct=sum(b[k]['correct'] and not a[k]['correct'] for k in a))
  result['comparisons'][ll+'/'+l+' minus '+rr+'/'+r]=value
 result['mmstar_categories']={};result['hallusion_grouped']={}
 for label,method in data:
  key=label+'/'+method;groups=defaultdict(list)
  for row in data[label,method]['mmstar']:groups[row['category']].append(row['correct'])
  result['mmstar_categories'][key]={cat:dict(count=len(v),accuracy=100*sum(v)/len(v)) for cat,v in groups.items()}
  result['hallusion_grouped'][key]=hallusion(data[label,method]['hallusionbench'])
  result['evaluations'][key]['mmstar_bare_letters']=sum(r['raw'].strip() in 'ABCD' and len(r['raw'].strip())==1 for r in data[label,method]['mmstar'])
 for label,exp in [('new',EXP),('baseline',OLD)]:
  for bits in (4,8):
   for method in ('qat','grace'):
    run=exp/f'w{bits}g128/runs/{method}';summary=read(run/'run_summary.json');recipe=read(run/'launch_recipe.json');c=recipe['comparison']['config']
    assert summary['complete'] and not summary['profile'] and summary['global_step']==256 and (run/'EXPORT_COMPLETE').exists()
    inv=[json.loads(x) for x in (run/'invocations.jsonl').read_text().splitlines()];assert all(x['exit_code']==0 for x in inv)
    steps=[json.loads(x) for x in (run/'step_metrics.jsonl').read_text().splitlines()];assert len(steps)==256
    entry=dict(summary=summary,config=c,implementation=recipe['comparison']['implementation'],environment=recipe['comparison']['environment'],
      wall_hours=sum(x['wall_seconds'] for x in inv)/3600,gpu_hours=sum(x['wall_seconds']*x['gpu_count'] for x in inv)/3600,
      start_utc=datetime.fromtimestamp(inv[0]['started_unix'],timezone.utc).isoformat(),
      end_utc=datetime.fromtimestamp(inv[-1]['started_unix']+inv[-1]['wall_seconds'],timezone.utc).isoformat(),
      mean_ce=sum(x['ce'] for x in steps)/len(steps),final_step=steps[-1])
    if method=='grace':
     entry['beta_cap_updates']=sum(x['beta']>=c['ib_beta_max']-1e-8 for x in steps)
     entry['first_cap_step']=next(x['step'] for x in steps if x['beta']>=c['ib_beta_max']-1e-8)
    result['training'][label+'/'+method+f'_w{bits}']=entry
 result['training_differences']={m:dict(config=differences(result['training']['baseline/'+m]['config'],result['training']['new/'+m]['config']),
     implementation=differences(result['training']['baseline/'+m]['implementation'],result['training']['new/'+m]['implementation'])) for m in LOCAL}
 old,new=read(OLD/'subset.json'),read(EXP/'subset.json');a={x['index']:x for x in old['records']};b={x['index']:x for x in new['records']}
 from hashlib import sha256
 for manifest in (old,new):assert sha256(json.dumps(manifest['records'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()==manifest['records_sha256']
 diff=read(EXP/'subset_comparison.json');assert diff['records_sha256']==new['records_sha256']
 result['subset']=dict(comparison={k:len(v) if isinstance(v,list) else v for k,v in diff.items()},
   added=[dict(index=i,length=b[i]['length'],turns=len(b[i]['sample']['conversations'])//2,source=b[i]['source']) for i in b.keys()-a.keys()],
   removed=[dict(index=i,length=a[i]['length'],source=a[i]['source']) for i in a.keys()-b.keys()],
   same_position_count=sum(x['index']==y['index'] for x,y in zip(old['records'],new['records'])),
   old_lengths=stats([x['length'] for x in old['records']]),new_lengths=stats([x['length'] for x in new['records']]),
   new_above_2048=sum(x['length']>2048 for x in new['records']),new_at_4096=sum(x['length']==4096 for x in new['records']),
   records_sha256=new['records_sha256'],rejections=new['rejected'])
 for key in list(result['sources']):result['sources'][key]['current_source_hash_differences']={p:dict(saved=h,current=sha(ROOT/p)) for p,h in result['sources'][key]['policy']['source_sha256'].items() if sha(ROOT/p)!=h}
 # Validate result model identities against the declared matrices and new exports.
 sys.path.insert(0,str(ROOT/'qwen-vl-finetune/scripts'))
 from constrained_experiment import Experiment
 specs={'new':Experiment(EXP/'experiment.json'),'baseline':Experiment(OLD/'experiment.json')}
 for key,values in result['evaluations'].items():
  label,method=key.split('/')
  for entry in values['suites'].values():
   payload=read(Path(entry['results_path']))
   assert Path(payload['summary']['model_path']).resolve()==specs[label].paths[method]
 for method in LOCAL:
  specs['new'].validate_export(method)
  run=specs['new'].paths[method];recipe=read(run/'launch_recipe.json')['comparison'];bits=specs['new'].models[method]['bits']
  assert recipe==read(EXP/f'w{bits}g128/comparison_recipe.json')
  assert recipe['experiment_config_sha256']==sha(EXP/'experiment.json')
  assert recipe['subset_sha256']==result['subset']['records_sha256']
  assert read(run/'trainer_state.json')['global_step']==256
  for name in ('student','teacher'):
   model=getattr(specs['new'],name)
   assert recipe[name+'_provenance_sha256']==sha(EXP/'provenance'/f'{model}.json')
 result['supplemental_verification']=dict(all_91_result_model_paths_match_config=True,all_four_new_exports_pass=True,
  current_experiment_hash_matches_training=True,all_four_share_subset_and_pinned_recipe=True,current_checkpoint_weight_bytes_rehashed=False)
 (EXP/'reports/comparison_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 with (EXP/'reports/scores_comparison.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['experiment','model',*SUITES,'mean','unresolved','length_stops'])
  for key,v in result['evaluations'].items():w.writerow([*key.split('/'),*[v['suites'][s]['score_percent'] for s in SUITES],v['mean'],v['unresolved'],v['truncated']])
 print('Saved comparison_audit.json and scores_comparison.csv',flush=True)

if __name__=='__main__':main()
