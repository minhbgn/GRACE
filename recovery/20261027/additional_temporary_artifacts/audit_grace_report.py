import json,sys,copy,hashlib
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path('reproduction/GRACE/evaluation').resolve()))
import run_local_development as r
hub=Path('reproduction/GRACE/experiments/reproduction_eval_only').resolve()
p=hub/'local_development/local_native_greedy256_v1/full'
protocol=r.read(p/'protocol.json')
assert all(r.sha(r.ROOT/k)==v for k,v in protocol['source_sha256'].items())
exp=r.Experiment(r.DEFAULT/'experiment.json')
assert r.sha(exp.config_path)==protocol['experiment_sha256']
out={'protocol':protocol['name'],'audit_date':'2026-09-29','original_artifact_ready':r.read(p/'local_scores.json')['artifact_ready'],'source_hashes_match':True,'rows':[]}
data={}; identities={}
for m in r.REFERENCES+r.LORA:
 data[m]={}
 for s in r.SUITES:
  job=p/m/s; fs=list(job.glob('attempt_*/results.json'));assert len(fs)==1
  f=fs[0];j=r.read(f);data[m][s]=j;done=r.verified_completion(job)
  adjustment=None
  if s=='mmbench':
   assert done is None
   try:r.validate(j,s,exp.paths[m],protocol['name'],protocol['attention'],None,r.read(p/m/'processor.json'))
   except ValueError as e:assert str(e)=='Summary disagrees with corrected scoring/diagnostics'
   else:raise AssertionError('Expected original histogram validation failure')
   recomputed=r.corrected_summary(s,r.score_rows(s,copy.deepcopy(j['results']),r.metadata(s)),r.metadata(s),partial=False)
   assert [k for k,v in recomputed.items() if j['summary'].get(k)!=v]==['circular']
   assert json.loads(json.dumps(recomputed))=={k:j['summary'][k] for k in recomputed}
   checked=copy.deepcopy(j)
   hist=checked['summary']['circular']['rotation_count_histogram']
   checked['summary']['circular']['rotation_count_histogram']={int(k):v for k,v in hist.items()}
   adjustment='In-memory conversion of circular.rotation_count_histogram keys from JSON strings to integers only; original files untouched.'
  else:
   assert done is not None
   for key,artifact in [('protocol_sha256',p/'protocol.json'),('processor_sha256',p/m/'processor.json'),('model_provenance_sha256',p/m/'model_provenance.json')]:assert done[key]==r.sha(artifact)
   assert done['attempt']==f.parent.name
   checked=j
  v=r.validate(checked,s,exp.paths[m],protocol['name'],protocol['attention'],None,r.read(p/m/'processor.json'))
  if done:assert all(done[k]==value for k,value in v.items())
  assert identities.setdefault(s,v['sample_identity'])==v['sample_identity']
  row=dict(model=m,suite=s,results=str(f.relative_to(hub)),results_sha256=r.sha(f),completion_marker=bool(done),cpu_audit='pass',normalization=adjustment,**v)
  row['errors']=j['summary']['errors'];row['generation_unknown']=j['summary']['generation_unknown']
  if s=='mmbench':row['circular_bounds_percent']=j['summary']['circular']['judge_completion_bounds_percent']
  out['rows'].append(row)
 print('AUDITED',m,flush=True)
out['total_rows']=sum(x['count'] for x in out['rows'])
out['completion_markers']=sum(x['completion_marker'] for x in out['rows'])
out['mmstar_paired']=[]
for a,b in [('grace_w4','released_grace_w4'),('grace_w8','released_grace_w8'),('grace_w4','qat_w4'),('grace_w8','qat_w8')]:
 ra={v['index']:v for v in data[a]['mmstar']['results']};rb={v['index']:v for v in data[b]['mmstar']['results']};assert ra.keys()==rb.keys()
 wins=sum(ra[k]['correct'] and not rb[k]['correct'] for k in ra);losses=sum(rb[k]['correct'] and not ra[k]['correct'] for k in ra)
 out['mmstar_paired'].append(dict(left=a,right=b,left_only_correct=wins,right_only_correct=losses,net_correct=wins-losses))
out['mmstar_bare_labels']={m:sum(v['raw'].strip() in ['A','B','C','D'] for v in data[m]['mmstar']['results']) for m in data}
(hub/'experiment_report_audit.json').write_text(json.dumps(out,indent=2)+'\n')
Path('/tmp/grace_report_data.json').write_text(json.dumps({m:{s:j['summary'] for s,j in suites.items()} for m,suites in data.items()}))
print('TOTAL',out['total_rows'],'MARKERS',out['completion_markers']);print('PAIRED',out['mmstar_paired']);print('BARE',out['mmstar_bare_labels'])
