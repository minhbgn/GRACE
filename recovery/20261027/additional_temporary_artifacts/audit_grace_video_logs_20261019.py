from pathlib import Path
from collections import Counter
import json,re,sys,hashlib,math
sys.path.insert(0,str(Path('reproduction/GRACE').resolve()))
from evaluation.runners.run_video_matrix import completed,read,summary_from_rows
from evaluation.protocols.video_protocol import geometry,digest
exp=Path('reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42')
base=exp/'video_evaluation/video_unsaturated_v2_3a74c7231e27/full'
policy=read(base/'protocol.json')['policy']
analysis=read(exp/'reports/video_results_analysis_20261019.json')
feasibility=read(exp/'reports/video_resolution_feasibility_20261017.json')
projected={}
for suite,src in feasibility['inputs'].items():
 records=[json.loads(l) for l in Path(src['path']).open()]
 assert hashlib.sha256(Path(src['path']).read_bytes()).hexdigest()==src['sha256']
 projected[suite]={r['id']:geometry(r['sampled_frames'],r['source_height'],r['source_width'],16,2,2,256)['visual_tokens']+r['input_text_tokens'] for r in records}
progress_re=re.compile(r'^(\d+)/(\d+) (\S+): visual=(\d+) text_in=(\d+) text_out=(\d+) finish=(\S+)$')
report=dict(date='2026-10-19',policy=policy,coverage={'logs':0},jobs=[],model_limits={},source='All full-matrix attempt run.log, job.json, predictions.jsonl, COMPLETE.json, failure.json, blocked.json; saved model/tokenizer configs; hashed historical preflight projections.')
for method in read(exp/'experiment.json')['models']:
 for suite in ['tempcompass','videomme','videommmu_perception','videommmu_comprehension','videommmu_adaptation']:
  attempts=sorted((base/method/suite).glob('attempt_*'))
  assert len(attempts)==1
  attempt=attempts[-1];job=read(attempt/'job.json');lp=attempt/'run.log';lines=lp.read_text(errors='replace').splitlines()
  predictions=[json.loads(l) for l in (attempt/'predictions.jsonl').open()]
  done=completed(attempt.parent,job['rows'],policy)
  if predictions:summary_from_rows(predictions)
  config=read(Path(job['model_path'])/'config.json');tok=read(Path(job['model_path'])/'tokenizer_config.json')
  context=config['text_config']['max_position_embeddings']; tokenizer_limit=tok['model_max_length']
  report['model_limits'][method]=dict(native_context=context,tokenizer_model_max_length=tokenizer_limit)
  progress=[];warnings=[];decoder=[];unexpected=[]
  for i,line in enumerate(lines,1):
   match=progress_re.fullmatch(line)
   if match:progress.append((i,match));continue
   if not line.strip() or 'Loading weights:' in line:continue
   if 'Token indices sequence length is longer' in line:
    limits=re.search(r'\((\d+) > (\d+)\)',line)
    warnings.append(dict(line=i,message=line,logged_length=int(limits.group(1)),warning_threshold=int(limits.group(2))));continue
   if '[h264' in line:decoder.append(dict(line=i,message=line));continue
   unexpected.append(dict(line=i,message=line))
  assert len(progress)==len(predictions)
  for index,((line,match),row) in enumerate(zip(progress,predictions),1):
   num,den,uid,visual,text,out,finish=match.groups()
   assert (int(num),int(den),uid)==(index,len(job['rows']),row['id'])
   assert (int(visual),int(text),int(out),finish)==(row['tokens']['input_visual_tokens'],row['tokens']['input_text_tokens'],row['tokens']['output_text_tokens'],row['generation']['finish_reason'])
  projected_tokens=list(projected[suite].values());assert len(projected_tokens)==len(job['rows'])
  overflow_ids=[r['id'] for r in job['rows'] if projected[suite][r['id']]+policy['max_new_tokens']>context]
  failure=read(attempt/'failure.json') if (attempt/'failure.json').exists() else None
  block=read(attempt/'blocked.json') if (attempt/'blocked.json').exists() else None
  if done:state='complete';reason='Complete log and saved predictions agree; no logged warnings or errors' if not(warnings or decoder or unexpected) else 'Complete with log findings'
  elif failure:
   assert 'OutOfMemoryError' in failure['error'];state='cuda_oom';reason='Explicit CUDA OOM in failure.json and traceback'
  else:
   state='unexplained_partial' if predictions else 'unexplained_zero_predictions'
   reason='Log ends after successful predictions without traceback or saved failure record' if predictions else 'Log ends after 100% weight loading without a first prediction or saved failure record'
  item=dict(method=method,suite=suite,state=state,reason=reason,log_path=str(lp.relative_to(exp)),log_sha256=hashlib.sha256(lp.read_bytes()).hexdigest(),log_lines=len(lines),expected_questions=len(job['rows']),saved_predictions=len(predictions),progress_records_matched=len(progress),context_overflows=len(overflow_ids),context_overflow_ids=overflow_ids,inputs_above_tokenizer_setting=sum(t>tokenizer_limit for t in projected_tokens),tokenizer_warnings=warnings,decoder_warning_count=len(decoder),decoder_warning_examples=decoder[:2],unexpected_line_count=len(unexpected),traceback_count=sum('Traceback (most recent call last):' in r['message'] for r in unexpected),max_projected_input=max(projected_tokens),native_context=context,tokenizer_limit=tokenizer_limit)
  if predictions:
   last=predictions[-1]
   item['last_prediction']=dict(id=last['id'],log_line=progress[-1][0],tokens=last['tokens'])
   item['saved_predictions_above_tokenizer_setting']=sum(r['tokens']['input_tokens']>tokenizer_limit for r in predictions)
   item['all_predictions_finish_eos']=all(r['generation']['finish_reason']=='eos' for r in predictions)
  if not done:
   nextrow=job['rows'][len(predictions)]
   item['next_unscored_question']=dict(id=nextrow['id'],projected_input=projected[suite][nextrow['id']],with_output_reserve=projected[suite][nextrow['id']]+policy['max_new_tokens'])
  if failure:
   item['failure']=failure
   assert nextrow['id']==failure['id']
  if block:item['block']=block
  if done:
   item['validated_accuracy']=done['summary']['accuracy'];item['generation_cutoffs']=done['summary']['generation_cutoffs'];item['unresolved']=done['summary']['unresolved']
   assert not (warnings or decoder or unexpected),(method,suite,unexpected)
  if decoder:
   last_decoder=max(r['line'] for r in decoder)
   item['predictions_logged_after_last_decoder_warning']=sum(i>last_decoder for i,_ in progress)
   item['last_decoder_warning_line']=last_decoder
  report['jobs'].append(item)
report['coverage']=dict(logs=len(report['jobs']),states=dict(Counter(j['state'] for j in report['jobs'])),logs_with_tokenizer_warning=sum(bool(j['tokenizer_warnings']) for j in report['jobs']),tokenizer_warning_messages=sum(len(j['tokenizer_warnings']) for j in report['jobs']),logs_with_decoder_warnings=sum(bool(j['decoder_warning_count']) for j in report['jobs']),decoder_warnings=sum(j['decoder_warning_count'] for j in report['jobs']),explicit_context_blockers=sum('block' in j for j in report['jobs']),native_context_overflows_model_questions=sum(j['context_overflows'] for j in report['jobs']),unique_population_context_overflows=sum(t+4096>262144 for ss in projected.values() for t in ss.values()),saved_predictions=sum(j['saved_predictions'] for j in report['jobs']))
report['interpretation']={'unknown_failures':'41 incomplete attempts have no exit status/signal, exception, or context blocker saved. A 100% weight-loader bar does not certify GPU transfer or generation. A kill/crash/resource limit is possible, but cannot be assigned as the cause from these artifacts.','context_policy':'There are zero native-context overflows. The user-authorized accuracy excluding context overflows therefore cannot recover any missing predictions; no incomplete score is promoted to a complete result.','tokenizer_warning':'Installed Transformers warning method logs once per tokenizer when model_max_length is exceeded; it does not raise. Worker explicitly uses truncation=False and checks the real model context.','decoder_warning':'H.264 mmco warnings are nonfatal at their logged positions; successful predictions continue afterwards. Decoded-frame fidelity was not verified by a log-only audit.'}
output=exp/'reports/video_run_log_audit_20261019.json';output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report['coverage'],indent=2))
for j in report['jobs']:
 if j['state']=='complete': print('COMPLETE LOG',j['method'],j['suite'],j['log_lines'],'lines',j['saved_predictions'],'progress records; zero warning/error lines')
 if j['decoder_warning_count']:print('DECODER',j['method'],j['suite'],j['decoder_warning_count'],'later successful predictions',j['predictions_logged_after_last_decoder_warning'])
 if j['state'].startswith('unexplained') and j['saved_predictions']:print('UNKNOWN PARTIAL',j['method'],j['suite'],'last',j['last_prediction']['id'],'next',j['next_unscored_question'])
print('WARNING MODELS',[(m,sum(bool(j['tokenizer_warnings']) for j in report['jobs'] if j['method']==m)) for m in report['model_limits']])
print('WROTE',output)
