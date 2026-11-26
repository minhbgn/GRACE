from pathlib import Path
import collections,datetime,hashlib,json,re,struct
exp=Path('/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42')
root=exp.parents[1];retry=exp/'video_evaluation/parallel_retry_20261019T034141Z_kbsu66';reports=exp/'reports'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
 with p.open() as f:return [json.loads(s) for s in f]
def witness(p):return dict(path=str(p),sha256=sha(p))
report=dict(date='2026-10-23',timezone='Asia/Singapore',experiment=str(exp))
parent=retry/'gpu4';jobs=re.findall(r'^Running ([^;]+); log: (.+)$',(parent/'matrix.log').read_text(),re.M);exits=[]
for line in (parent/'worker_exits.log').read_text().splitlines():
 m=re.match(r'([\d.]+) --- SIGCHLD .*si_pid=(\d+),.*si_status=(\w+)',line)
 if m and int(m[2])>=2047355:exits.append(m.groups())
assert len(jobs)==len(exits)==20
historical=[]
for (label,log),(ts,pid,status) in zip(jobs,exits):
 attempt=Path(log).parent;job=read(attempt/'job.json');preds=rows(attempt/'predictions.jsonl')
 historical.append(dict(pair=label,worker_pid=int(pid),exit_status=status,
  exited_at_singapore=datetime.datetime.fromtimestamp(float(ts),datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
  attempt=str(attempt),predictions=len(preds),expected=len(job['rows']),
  complete=(attempt.parent/'COMPLETE.json').exists(),last_question=preds[-1]['id'] if preds else None,
  failure=read(attempt/'failure.json') if (attempt/'failure.json').exists() else None,
  log=witness(Path(log))))
report['historical_local_jobs']=historical
report['exit_evidence']=[witness(parent/'matrix.log'),witness(parent/'worker_exits.log')]
report['startup_termination']=dict(zero_prediction_sigkill_jobs=sum(x['predictions']==0 and x['exit_status']=='SIGKILL' for x in historical),
 total_sigkill_jobs=sum(x['exit_status']=='SIGKILL' for x in historical),sender=None,
 limitation='wait4/SIGCHLD records child status, not SIGKILL sender; no matching kernel OOM record found in accessible host logs')
config=read(exp/'experiment.json');checkpoints=[]
for method in ['vanilla_2b','qat_w4','grace_w4','qat_w8','grace_w8']:
 spec=config['models'][method];p=(exp if spec.get('relative_to')=='experiment' else root)/spec['path'];cfg=read(p/'config.json');headers={}
 for f in p.glob('*.safetensors'):
  with f.open('rb') as stream:n=struct.unpack('<Q',stream.read(8))[0];h=json.loads(stream.read(n))
  headers.update({k:v for k,v in h.items() if k!='__metadata__'})
 checkpoints.append(dict(method=method,path=str(p),tensors=len(headers),
  saved_bytes=sum(v['data_offsets'][1]-v['data_offsets'][0] for v in headers.values()),
  saved_dtypes=sorted({v['dtype'] for v in headers.values()}),qat=cfg.get('qat'),qat_lora=cfg.get('qat_lora'),
  quantization_config=cfg.get('quantization_config'),context=cfg['text_config']['max_position_embeddings']))
report['checkpoint_headers']=checkpoints
protocol=read(parent/'full/protocol.json');report['pinned_environment']=protocol['environment']
source_map={'paths.py':'paths.py','run_video_matrix.py':'runners/run_video_matrix.py','video_worker.py':'workers/video_worker.py','video_protocol.py':'protocols/video_protocol.py','local_scoring.py':'protocols/local_scoring.py'}
report['source_verification']={n:dict(expected=protocol['code'][n],current=sha(root/'evaluation'/rel),matches=protocol['code'][n]==sha(root/'evaluation'/rel)) for n,rel in source_map.items()}
diags=[]
for folder in ['startup_diagnostic_20261023','allocator_diagnostic_20261023']:
 base=reports/folder
 for result in read(base/'results.json'):
  out=base/result['name'];ev=rows(out/'events.jsonl')
  diags.append(dict(group=folder,exit=result,events=ev,evidence=[witness(out/'events.jsonl'),witness(out/'exit.json'),witness(out/'run.log')]))
report['diagnostics']=diags
qat=rows(retry/'gpu4/full/qat_w4/videomme/attempt_20261019T035547.718615Z/predictions.jsonl')
van=rows(retry/'gpu2/full/vanilla_2b/videomme/attempt_20261019T034148.338111Z/predictions.jsonl');by_id={r['id']:r for r in van}
report['historical_input_comparison']=dict(prefix=len(qat),identical_input_id_hashes=sum(r['input_ids_sha256']==by_id[r['id']]['input_ids_sha256'] for r in qat),
 qat_prefix_peak=max(r['peak_gpu_allocated_bytes'] for r in qat),vanilla_prefix_peak=max(by_id[r['id']]['peak_gpu_allocated_bytes'] for r in qat),
 vanilla_target={k:by_id['618-1'][k] for k in ['id','tokens','input_ids_sha256','peak_gpu_allocated_bytes','elapsed_seconds']},
 vanilla_videomme_predictions=len(van),vanilla_complete=witness(retry/'gpu2/full/vanilla_2b/videomme/COMPLETE.json'))
report['diagnostic_script']=witness(reports/'investigate_video_startup_20261023.py')
(reports/'video_startup_investigation_20261023.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(historical_local_jobs=len(historical),sigkill=report['startup_termination'],diagnostics=[dict(group=d['group'],name=d['exit']['name'],returncode=d['exit']['returncode']) for d in diags],pinned_sources_match=all(v['matches'] for v in report['source_verification'].values()))))
