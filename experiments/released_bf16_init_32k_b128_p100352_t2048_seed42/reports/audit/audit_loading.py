"""CPU-only processor parity, tied weights and W4 grid checks; no model instantiated."""
from pathlib import Path
from collections import Counter
from io import BytesIO
import hashlib,json,sys
import torch
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig,AutoProcessor
from qwen_vl_utils import process_vision_info
EXP=Path(__file__).resolve().parents[2];ROOT=EXP.parents[1]
sys.path.insert(0,str(ROOT/'qwen-vl-finetune/scripts'))
from constrained_experiment import Experiment
exp=Experiment(EXP/'experiment.json');torch.set_num_threads(2)
# Use the same real AI2D image and saved evaluation prompt for every checkpoint.
f=Path('/raid/lvchengtao/cache/huggingface/hub/datasets--lmms-lab-encoder--ai2d/snapshots/c83a9b9692933aff8349157c88a413df9d02c4e5/data/test-00000-of-00002.parquet')
rec=next(pq.ParquetFile(f).iter_batches(batch_size=1)).to_pylist()[0]
im=Image.open(BytesIO(rec['image']['bytes'])).convert('RGB')
d=EXP/'lmms_evaluation/full/released_grace_w4/ai2d';c=json.loads((d/'COMPLETE.json').read_text());sample=json.loads(next((d/c['attempt']/'samples.jsonl').open()))
context=sample['arguments'][0]
messages=[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':[{'type':'image','image':im,'max_pixels':1605632,'min_pixels':200704},{'type':'text','text':context}]}]
image_inputs,video_inputs,video_kwargs=process_vision_info([messages],return_video_kwargs=True,image_patch_size=16,return_video_metadata=True)
out={'probe':{'suite':'ai2d','doc_id':0,'source_size':im.size,'resized_size':image_inputs[0].size},'models':{}}
for name,path in exp.paths.items():
    processor=AutoProcessor.from_pretrained(path,local_files_only=True,max_pixels=1605632,min_pixels=200704)
    texts=processor.apply_chat_template([messages],tokenize=False,add_generation_prompt=True)
    inputs=processor(text=texts,images=image_inputs,videos=video_inputs,**video_kwargs,do_resize=False,return_tensors='pt')
    signatures={k:{'shape':list(v.shape),'sha256':hashlib.sha256(v.numpy().tobytes()).hexdigest()} for k,v in inputs.items()}
    config=AutoConfig.from_pretrained(path,local_files_only=True)
    row={'input_signatures':signatures,'rendered_prompt_sha256':hashlib.sha256(texts[0].encode()).hexdigest(),
         'processor_class':type(processor).__name__,'image_processor_class':type(processor.image_processor).__name__,
         'tokenizer_eos':processor.tokenizer.eos_token_id,'tokenizer_pad':processor.tokenizer.pad_token_id,
         'effective_image_size':processor.image_processor.size,
         'rope_parameters':getattr(config.text_config,'rope_parameters',None),
         'tie_word_embeddings':config.tie_word_embeddings}
    if name!='vanilla_8b_teacher':
        with safe_open(str(path/'model.safetensors'),framework='pt',device='cpu') as sf:
            keys=list(sf.keys()); embed=next(k for k in keys if k.endswith('embed_tokens.weight'))
            if 'lm_head.weight' in keys:
                a,b=sf.get_slice(embed),sf.get_slice('lm_head.weight')
                n=a.get_shape()[0]; unequal=0
                for start in range(0,n,2048):
                    unequal+=int((a[start:start+2048]!=b[start:start+2048]).sum())
                row['tied_embedding_unequal_elements']=unequal
            else:row['tied_embedding_unequal_elements']='head omitted (tied)'
            if name.endswith('w4'):
                linears=[k for k in keys if '.layers.' in k and any(k.endswith('.'+suffix+'.weight') for suffix in ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')) and 'visual' not in k]
                distinct=Counter()
                for key in linears:
                    view=sf.get_slice(key);shape=view.get_shape()
                    for r in (0,shape[0]//2,shape[0]-1):
                        vals=view[r:r+1].float().numpy().reshape(-1,128)
                        for group in vals[:16]:distinct[len(np.unique(group))]+=1
                row['w4_group_probe']={'linears':len(linears),'groups':sum(distinct.values()),'unique_counts':dict(sorted(distinct.items())), 'all_at_most_16_levels':max(distinct)<=16}
    out['models'][name]=row
    (EXP/'reports/audit/loading.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
    print('CHECKED',name,'inputs',signatures['input_ids']['sha256'][:12], 'tie mismatch',row.get('tied_embedding_unequal_elements'), 'grid',row.get('w4_group_probe',{}).get('all_at_most_16_levels'),flush=True)
