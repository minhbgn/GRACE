import json, sys
from pathlib import Path
root=Path('/raid/lvchengtao/Minh/Distillation/reproduction/GRACE')
exp=root/'experiments/released_bf16_init_32k_b128_p589824_t2048_seed42'
sys.path.insert(0,str(root/'qwen-vl-finetune'))
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
from qwenvl.data.data_processor import update_processor_pixels
from qwenvl.train.argument import DataArguments
from qwenvl.data.constrained_data import image_path

torch.set_num_threads(2)
old_path=root/'experiments/released_bf16_init_32k_b128_p100352_t2048_seed42/subset.json'
old=json.loads(old_path.read_text())
proc=AutoProcessor.from_pretrained(root/'models/Qwen3-VL-2B-GRACE-BF16',local_files_only=True)
records=old['records'][:256]
rows=[]
for r in records:
    path=Path(r['sample']['data_path'])/image_path(r['sample'])
    with Image.open(path) as im:
        im=im.convert('RGB')
        row=dict(index=r['index'],source=r['source'],width=im.width,height=im.height)
        for cap in [100352,589824]:
            update_processor_pixels(proc,DataArguments(max_pixels=cap,min_pixels=4096))
            encoded=proc.image_processor(images=im,return_tensors='pt')
            row[str(cap)]=int(encoded['image_grid_thw'].prod())//proc.image_processor.merge_size**2
    rows.append(row)
summary={}
for cap in [100352,589824]:
    x=np.array([r[str(cap)] for r in rows])
    summary[str(cap)]=dict(count=len(x),min=int(x.min()),mean=float(x.mean()),
                          median=float(np.median(x)),p90=float(np.percentile(x,90)),max=int(x.max()),
                          at_least_400=int((x>=400).sum()),at_ceiling=int((x==cap//1024).sum()))
result=dict(purpose='CPU preview only: first 256 rows of the shuffled historical subset, not the new training manifest',
            source_manifest=str(old_path),source_records_sha256=old['records_sha256'],min_pixels=4096,
            patch_size=proc.image_processor.patch_size,merge_size=proc.image_processor.merge_size,
            summary=summary,records=rows)
(exp/'token_budget_preview.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))
