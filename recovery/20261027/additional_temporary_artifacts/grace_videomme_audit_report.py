import hashlib,json,math,sys
from pathlib import Path
sys.path.insert(0,'reproduction/GRACE')
from evaluation.protocols.video_protocol import load_suite
exp=Path('reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42')
output=exp/'video_evaluation/videomme_complete_20261017/full'
source,info=load_suite('videomme')
records=[json.loads(line) for line in (output/'preflight/videomme.jsonl').read_text().splitlines()]
assert len(records)==2700 and {r['id'] for r in records}=={r['id'] for r in source}
summary=json.loads((output/'preflight/summary.json').read_text())['videomme']
protocol=json.loads((output/'protocol.json').read_text())
by_id={r['id']:r for r in source}
assets={r['video_id']:r['asset'] for r in source}
assert len(assets)==900 and all('path' in a for a in assets.values())
def stats(values):
 values=sorted(values)
 return dict(count=len(values),mean=sum(values)/len(values) if values else None,
 min=values[0] if values else None,max=values[-1] if values else None,
 p95=values[math.ceil(.95*len(values))-1] if values else None)
def summarize(rows):
 good=[r for r in rows if 'error' not in r]
 return dict(questions=len(rows),videos=len({by_id[r['id']]['video_id'] for r in rows}),
 errors=sum('error' in r for r in rows),fits_context=sum(r['fits_context'] for r in good),
 context_overflows=sum(not r['fits_context'] for r in good),
 input_visual_tokens=stats([r['input_visual_tokens'] for r in good]),
 input_text_tokens=stats([r['input_text_tokens'] for r in good]),
 input_tokens=stats([r['input_tokens'] for r in good]))
check=json.loads(Path('/tmp/grace_videomme_complete_check.json').read_text())
report=dict(date='2026-10-17',experiment=exp.name,inventory=info,all_videos_extracted=True,
 extracted_video_root=str(Path(next(iter(assets.values()))['path']).parent),
 policy=protocol['policy'],environment=protocol['environment'],code_sha256=protocol['code'],
 models_checked=list(check['model_paths']),
 model_context_limits={m:json.loads((Path(p)/'config.json').read_text())['text_config']['max_position_embeddings'] for m,p in check['model_paths'].items()},
 summary=summary,overall=summarize(records),
 by_duration={c:summarize([r for r in records if by_id[r['id']]['category']==c]) for c in ('short','medium','long')},
 measurement='CPU first-frame/metadata probes plus exact prompt-token reconstruction; proposed input tokens include context-blocked questions, not observed model inference usage',
 decode_scope='900 first frames and video metadata; not an exhaustive decode of all sampled frames',
 historical_hashes_modified=False,gpu_inference_started=False,training_started=False,
 output=str(output),sha256={str(p.relative_to(exp)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
 (output/'protocol.json',output/'datasets/videomme.json',output/'preflight/videomme.jsonl',output/'preflight/summary.json')})
path=exp/'reports/videomme_input_audit_20261017.json'
path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(overall=report['overall'],by_duration=report['by_duration']),indent=2))
print('Saved',path)
