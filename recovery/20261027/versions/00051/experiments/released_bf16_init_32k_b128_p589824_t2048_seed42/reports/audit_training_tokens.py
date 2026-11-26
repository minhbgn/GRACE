"""CPU full-manifest token decomposition, cross-checked against saved hashes."""
import json
from pathlib import Path
import sys
EXP=Path(__file__).resolve().parents[1]; ROOT=EXP.parents[1]
sys.path.insert(0,str(ROOT/'qwen-vl-finetune'))
from qwenvl.data.constrained_data import json_hash,processor_fingerprint,preprocess,token_hash
from qwenvl.data.data_processor import _build_messages,IMAGE_TOKEN_INDEX,update_processor_pixels
from qwenvl.train.argument import DataArguments
from transformers import AutoProcessor
import numpy as np
import torch

def stats(v):
 a=np.array(v);return dict(min=int(a.min()),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)),max=int(a.max()))

def main():
 torch.set_num_threads(2)
 m=json.loads((EXP/'subset.json').read_text());records=m['records'];assert json_hash(records)==m['records_sha256']
 model=ROOT/'models/Qwen3-VL-2B-GRACE-BF16';assert processor_fingerprint(model)==m['recipe']['processor']
 p=AutoProcessor.from_pretrained(model,local_files_only=True);p.tokenizer.model_max_length=4096
 update_processor_pixels(p,DataArguments(max_pixels=589824,min_pixels=4096))
 rows=[]
 for i in range(0,len(records),512):
  batch=records[i:i+512]
  rendered=[p.apply_chat_template(_build_messages(r['sample'],Path(r['sample']['data_path'])),tokenize=False) for r in batch]
  tokens=p.tokenizer(rendered,add_special_tokens=False,truncation=False)['input_ids']
  for r,ids in zip(batch,tokens):
   assert ids.count(IMAGE_TOKEN_INDEX)==1
   text=len(ids)-1;visual=r['length']-text;assert 4<=visual<=576
   rows.append(dict(index=r['index'],total=r['length'],nonvisual=text,visual=visual))
  if len(rows)%8192==0:print('Tokenized',len(rows),flush=True)
 indices=set(range(0,len(records),1024))|set(sorted(range(len(records)),key=lambda i:records[i]['length'],reverse=True)[:16])
 for i in sorted(indices):
  item=preprocess(records[i]['sample'],p,4096,589824)
  assert token_hash(item)==records[i]['token_sha256']
  assert int((item['input_ids']==IMAGE_TOKEN_INDEX).sum())==rows[i]['visual']
 out=dict(count=len(rows),subset_sha256=m['records_sha256'],full_image_hash_checks=len(indices),
  statistics={k:stats([r[k] for r in rows]) for k in ['total','nonvisual','visual']},
  visual_at_least={str(k):sum(r['visual']>=k for r in rows) for k in [98,192,256,400,512,576]},
  above_2048=[r for r in rows if r['total']>2048],at_4096=sum(r['total']==4096 for r in rows))
 (EXP/'reports/training_token_audit.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
