"""Verify cached source parquet bytes against Hub and compare text/labels with evaluated rows."""
from pathlib import Path
import hashlib,json,os
from huggingface_hub import HfApi
import pyarrow.parquet as pq
EXP=Path(__file__).resolve().parents[2]
FULL=EXP/'lmms_evaluation/full'
CACHE=Path(os.environ.get('HF_HOME','~/.cache/huggingface')).expanduser()/'hub'
locks=json.loads((FULL/'datasets.lock.json').read_text())
api=HfApi();out={}
for suite in ('scienceqa','ai2d','mmstar','seedbench'):
    lock=locks[suite]
    info=api.dataset_info(lock['requested_repo'],revision=lock['revision'],files_metadata=True)
    assert info.id==lock['repo'] and info.sha==lock['revision']
    remote={x.rfilename:x for x in info.siblings}
    snapshot=CACHE/('datasets--'+info.id.replace('/','--'))/'snapshots'/info.sha
    files=sorted(snapshot.rglob('*.parquet'))
    select=[f for f in files if ('test-' in f.name if suite!='mmstar' else f.name=='mmstar.parquet')]
    assert select
    row={'repository':info.id,'revision':info.sha,'files':{},'evaluated_source_files':[str(f.relative_to(snapshot)) for f in select]}
    for f in files:
        h=hashlib.sha256()
        with f.open('rb') as handle:
            for chunk in iter(lambda:handle.read(16*1024*1024),b''):h.update(chunk)
        rel=str(f.relative_to(snapshot));r=remote[rel]
        assert r.lfs and h.hexdigest()==r.lfs.sha256,(suite,rel)
        row['files'][rel]={'sha256':h.hexdigest(),'bytes':f.stat().st_size,'matches_hub':True}
    directory=FULL/'released_grace_w4'/suite
    complete=json.loads((directory/'COMPLETE.json').read_text())
    saved=[json.loads(line) for line in (directory/complete['attempt']/'samples.jsonl').open()]
    offset=0;image_counts={};source_counts={};image_samples=[]
    for f in select:
        parquet=pq.ParquetFile(f)
        # Text/label comparison avoids decoding every image; checks exact source row order.
        columns=[name for name in parquet.schema_arrow.names if name!='image']
        for batch in parquet.iter_batches(batch_size=256,columns=columns):
            for original in batch.to_pylist():
                s=saved[offset]
                for k,v in original.items():
                    if k in s['doc']:
                        assert v==s['doc'][k],(suite,offset,k)
                if suite=='scienceqa':target=chr(65+original['answer'])
                elif suite=='ai2d':target=chr(65+int(original['answer']))
                else:target=original['answer']
                assert target==s['target'],(suite,offset,'target')
                kind=original.get('data_type','image'); source_counts[kind]=source_counts.get(kind,0)+1
                offset+=1
        # Presence checks across all rows; decode first image of first/middle/last selected shard.
        for batch in parquet.iter_batches(batch_size=32,columns=['image']):
            for rec in batch.to_pylist():
                vals=rec['image'] if isinstance(rec['image'],list) else [rec['image']]
                assert vals and all(v is not None and (bool(v) if isinstance(v, bytes) else (v.get('bytes') or v.get('path'))) for v in vals),(suite,f,'missing image')
                image_counts[len(vals)]=image_counts.get(len(vals),0)+1
        if f in (select[0],select[len(select)//2],select[-1]):
            from PIL import Image
            from io import BytesIO
            rec=next(parquet.iter_batches(batch_size=1,columns=['image'])).to_pylist()[0]['image']
            val=rec[0] if isinstance(rec,list) else rec
            im=Image.open(BytesIO(val if isinstance(val,bytes) else val['bytes'])).convert('RGB')
            image_samples.append({'file':str(f.relative_to(snapshot)),'size':im.size,'rgb_sha256':hashlib.sha256(im.tobytes()).hexdigest()})
    assert offset==len(saved)==complete['count'],(suite,offset,len(saved))
    row.update(verified_rows=offset,source_counts=source_counts,image_counts=image_counts,decoded_image_samples=image_samples)
    out[suite]=row
    (EXP/'reports/audit/datasets.json').write_text(json.dumps(out,indent=2)+'\n')
    print('VERIFIED',suite,len(files),'parquet files',offset,'rows',source_counts,flush=True)
