import json, math, hashlib, sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path('reproduction/GRACE').resolve()))
from evaluation.runners.run_video_matrix import completed, read, summary_from_rows
from evaluation.protocols.video_protocol import geometry, digest
from evaluation.paths import source_file

exp=Path('reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42')
base=exp/'video_evaluation/video_unsaturated_v2_3a74c7231e27/full'
protocol=read(base/'protocol.json')['policy']
models=list(read(exp/'experiment.json')['models'])
suites=['tempcompass','videomme','videommmu_perception','videommmu_comprehension','videommmu_adaptation']
feasibility=read(exp/'reports/video_resolution_feasibility_20261017.json')
def checksum(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(vals):
 vals=sorted(vals)
 return dict(count=len(vals),total=sum(vals),mean=sum(vals)/len(vals),min=vals[0],median=(vals[(len(vals)-1)//2]+vals[len(vals)//2])/2,p95=vals[math.ceil(.95*len(vals))-1],max=vals[-1])
report=dict(date='2026-10-19',base=str(base.relative_to(exp)),policy=protocol,source_hashes={},jobs=[],input_populations={},paired_tempcompass={})
reference_rows={}
for name,expected in read(base/'protocol.json')['code'].items():
 path=source_file(name); assert checksum(path)==expected
 report['source_hashes'][str(path)]=expected
for suite in suites:
 p=next((base/models[0]/suite).glob('attempt_*/job.json'))
 rows=read(p)['rows']; reference_rows[suite]=rows
 assert len(rows)==read(base/'datasets'/f'{suite}.json')['questions']
 assert digest(rows)==read(base/'datasets'/f'{suite}.json')['sample_identity']
 for method in models:
  job_dir=base/method/suite
  attempts=sorted(job_dir.glob('attempt_*'))
  attempt=attempts[-1]; job=read(attempt/'job.json')
  assert job['rows']==rows and job['protocol']==protocol
  pred=attempt/'predictions.jsonl'
  results=[json.loads(line) for line in pred.open()] if pred.exists() else []
  assert [r['id'] for r in results]==[r['id'] for r in rows[:len(results)]]
  for r,src in zip(results,rows):
   assert r['source_identity']==digest(src)
   counts={k:r['tokens'][k] for k in ['input_visual_tokens','input_text_tokens','input_tokens']}
   assert r['input_identity']==digest(dict(source=src,sampling=r['sampling'],tokens=counts,input_ids_sha256=r['input_ids_sha256']))
   assert counts['input_tokens']+protocol['max_new_tokens']<=r['context_limit']
  if results: summary_from_rows(results) # Recompute scores and validate telemetry on partial rows too.
  done=completed(job_dir, rows, protocol)
  failure=read(attempt/'failure.json') if (attempt/'failure.json').exists() else None
  state='complete' if done else 'failed_oom' if failure and 'OutOfMemoryError' in failure['error'] else 'partial' if results else 'no_predictions'
  entry=dict(method=method,suite=suite,state=state,expected=len(rows),observed=len(results),attempt=str(attempt.relative_to(exp)),prediction_sha256=checksum(pred) if pred.exists() else None)
  if failure:entry['failure']=failure
  if done:
   entry['summary']=done['summary']
   entry['correct']=sum(r['correct'] for r in results)
   entry['runtime_seconds_sum']=sum(r['elapsed_seconds'] for r in results)
   entry['max_peak_gpu_allocated_gib']=max(r['peak_gpu_allocated_bytes'] for r in results)/2**30
  report['jobs'].append(entry)
 src=feasibility['inputs'][suite]; p=Path(src['path']); assert checksum(p)==src['sha256']
 report['source_hashes'][str(p)]=checksum(p)
 preflight=[json.loads(l) for l in p.open()]
 assert [r['id'] for r in preflight]==[r['id'] for r in rows]
 projected={}
 for r in preflight:
  assert 'error' not in r
  g=geometry(r['sampled_frames'],r['source_height'],r['source_width'],16,2,2,256)
  projected[r['id']]=dict(input_visual_tokens=g['visual_tokens'], input_text_tokens=r['input_text_tokens'],input_tokens=g['visual_tokens']+r['input_text_tokens'])
 checked=0
 seen_identities={}
 for method in models:
  for pred in (base/method/suite).glob('attempt_*/predictions.jsonl'):
   for line in pred.open():
    row=json.loads(line)
    if row['id'] in seen_identities: assert row['input_identity']==seen_identities[row['id']]
    seen_identities[row['id']]=row['input_identity']
    for key,value in projected[row['id']].items():assert row['tokens'][key]==value,(method,suite,row['id'],key)
    checked+=1
 report['input_populations'][suite]=dict(questions=len(rows),videos=len({r['video_id'] for r in rows}),evidence='CPU reconstruction from hashed historical preflight; cross-checked against every saved inference row',inference_rows_crosschecked=checked,context_overflows=sum(r['input_tokens']+4096>262144 for r in projected.values()),tokens={key:stats([r[key] for r in projected.values()]) for key in ['input_visual_tokens','input_text_tokens','input_tokens']},question_types=dict(Counter(r['question_type'] for r in rows)))
all_inputs={k:[] for k in ['input_visual_tokens','input_text_tokens','input_tokens']}
for k in all_inputs:
 for suite in suites:
  stat=report['input_populations'][suite]['tokens'][k]
  all_inputs[k].append(stat)
report['combined_input']=dict(questions=sum(r['questions'] for r in report['input_populations'].values()),tokens={k:dict(total=sum(s['total'] for s in ss),mean=sum(s['total'] for s in ss)/5180) for k,ss in all_inputs.items()})
report['states']=dict(Counter(r['state'] for r in report['jobs']))
for lhs,rhs in [('sft_2b','vanilla_2b'),('vanilla_8b_teacher','vanilla_2b'),('vanilla_8b_teacher','sft_2b')]:
 def get(method):
  done=read(base/method/'tempcompass/COMPLETE.json')
  return [json.loads(l) for l in (base/method/'tempcompass'/done['attempt']/'predictions.jsonl').open()]
 a,b=get(lhs),get(rhs)
 assert [r['input_identity'] for r in a]==[r['input_identity'] for r in b]
 only_a=sum(x['correct'] and not y['correct'] for x,y in zip(a,b));only_b=sum(y['correct'] and not x['correct'] for x,y in zip(a,b))
 report['paired_tempcompass'][f'{lhs}_minus_{rhs}']=dict(only_left_correct=only_a,only_right_correct=only_b,net_correct=only_a-only_b,delta_pp=100*(only_a-only_b)/1580)
report['source_hashes'][str(base/'protocol.json')]=checksum(base/'protocol.json')
report['source_hashes'][str(exp/'reports/video_resolution_feasibility_20261017.json')]=checksum(exp/'reports/video_resolution_feasibility_20261017.json')
report['annotation_anomalies']=[dict(suite=suite,id=row['id'],question_type=row['question_type'],gold=row['gold'],options=row['options'],impact='Any non-multiple-choice type follows the open-answer path; option list is omitted from the prompt.') for suite,rows in reference_rows.items() for row in rows if row['question_type'] not in ('multiple-choice','open')]
output=exp/'reports/video_results_analysis_20261019.json'
output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print('STATES',report['states'])
for suite,d in report['input_populations'].items():
 print('\nINPUT',suite,'observed',d['inference_rows_crosschecked'],'types',d['question_types'])
 for k,s in d['tokens'].items():print(k,s)
print('COMBINED',report['combined_input'])
print('PAIRED',report['paired_tempcompass'])
for job in report['jobs']:
 if job['state']=='complete':print('COMPLETE',job['method'],job['suite'],job['correct'],job['summary']['accuracy'],job['summary']['token_usage']['output_text_tokens'],job['max_peak_gpu_allocated_gib'])
print('WROTE',output)
