"""Audit evaluated paths/configs/prompt parity and raw-answer parsing sensitivity."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,re,sys,subprocess
EXP=Path(__file__).resolve().parents[2];ROOT=EXP.parents[1]
sys.path.insert(0,str(ROOT/'evaluation'))
from run_lmms_matrix import TASKS,verified_completion,sha
FULL=EXP/'lmms_evaluation/full';protocol=json.loads((FULL/'protocol.json').read_text())
for key,path in [('runner_sha256',ROOT/'evaluation/run_lmms_matrix.py'),('worker_sha256',ROOT/'evaluation/lmms_worker.py'),('checkpoint_helper_sha256',ROOT/'qwen-vl-finetune/scripts/constrained_experiment.py')]:
    assert sha(path)==protocol[key],key
source=ROOT/'third_party/lmms-eval-v0.7.1'
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==protocol['protocol']['harness_commit']
assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=all'],text=True).strip()
base={};prompts={};out={'source_matches_run_lock':True,'models':{}}
models=list(protocol['experiment']['models'])
for model in models:
    out['models'][model]={}
    for suite in ('scienceqa','ai2d','seedbench','mmstar'):
        directory=FULL/model/suite;done=verified_completion(directory);attempt=directory/done['attempt']
        job=json.loads((attempt/'job.json').read_text());results=json.loads((attempt/'results.json').read_text());provenance=json.loads((FULL/model/'model_provenance.json').read_text())
        assert job['model_path']==provenance['path']
        assert f'pretrained={job["model_path"]},' in results['config']['model_args']
        assert job['protocol']==protocol['protocol']
        # Mask memory addresses from serialized callables before comparing effective task configs.
        task_config=results['configs'][TASKS[suite][0]]
        config_json=re.sub(r'0x[0-9a-f]+','0xADDR',json.dumps(task_config,sort_keys=True))
        assert config_json==base.setdefault(suite,config_json),(model,suite,'task drift')
        samples=[json.loads(l) for l in (attempt/'samples.jsonl').open()]
        p=hashlib.sha256(json.dumps([s['arguments'] for s in samples],sort_keys=True).encode()).hexdigest()
        assert p==prompts.setdefault(suite,p),(model,suite,'prompt drift')
        rawgroups=defaultdict(list);explicitgroups=defaultdict(list);disagreements=[];scored=0;nonletter=[];empty=0
        for s in samples:
            if suite=='seedbench' and 'seed_image' not in s:continue
            raw=s['resps'][0][0].strip();target=s['target'].strip()
            scored+=1;empty+=int(not raw)
            if not re.fullmatch('[A-Z]',raw,re.I):nonletter.append({'doc_id':s['doc_id'],'response':raw,'target':target})
            group=s['average']['l2_category'] if suite=='mmstar' else 'all'
            rawgroups[group].append(float(raw.lower()==target.lower()))
            match=re.match(r'^\s*(?:\(([A-Z])\)|([A-Z])(?=$|[.):]|\n)|(?:the (?:correct )?answer is|answer:|option)\s*\(?([A-Z])(?=$|[.):\s]))',raw,re.I)
            explicit=next((g.upper() for g in match.groups() if g is not None),None) if match else None
            explicit_correct=float(explicit==target.upper())
            explicitgroups[group].append(explicit_correct)
            if suite=='mmstar':original=s['average']['score']
            elif suite=='seedbench':original=float(s['seed_image']['pred'].lower().strip()==target.lower())
            else:original=s['exact_match']
            if explicit_correct!=original:disagreements.append({'doc_id':s['doc_id'],'response':raw,'target':target,'harness_correct':original,'explicit_correct':explicit_correct})
        strict=100*sum(sum(v)/len(v) for v in rawgroups.values())/len(rawgroups)
        explicit_score=100*sum(sum(v)/len(v) for v in explicitgroups.values())/len(explicitgroups)
        out['models'][model][suite]={'harness_score':done['score_percent'],'strict_letter_score':strict,
            'explicit_choice_score':explicit_score,'explicit_choice_disagreements':len(disagreements),'disagreement_examples':disagreements[:5],
            'harness_minus_strict_pp':done['score_percent']-strict,'scored':scored,'non_single_letter':len(nonletter),
            'empty':empty,'non_letter_examples':nonletter[:3], 'model_path':job['model_path'],
            'raw_prompt_arguments_sha256':p,'effective_config_sha256':hashlib.sha256(config_json.encode()).hexdigest()}
    out['models'][model]['explicit_mean4']=sum(out['models'][model][s]['explicit_choice_score'] for s in ('scienceqa','ai2d','seedbench','mmstar'))/4
    out['models'][model]['strict_mean4']=sum(out['models'][model][s]['strict_letter_score'] for s in ('scienceqa','ai2d','seedbench','mmstar'))/4
(EXP/'reports/audit/evaluation.json').write_text(json.dumps(out,indent=2)+'\n')
for m,v in out['models'].items():
 print(m,'explicit mean4',round(v['explicit_mean4'],4),'disagreements',{s:d['explicit_choice_disagreements'] for s,d in v.items() if isinstance(d,dict)},flush=True)
