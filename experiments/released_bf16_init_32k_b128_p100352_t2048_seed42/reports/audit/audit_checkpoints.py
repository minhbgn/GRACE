"""Read-only local byte/hash and Hub-manifest verification; no model inference."""
from pathlib import Path
import hashlib,json,sys,time
from huggingface_hub import HfApi
ROOT=Path(__file__).resolve().parents[4]
EXP=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'qwen-vl-finetune/scripts'))
from constrained_experiment import Experiment
exp=Experiment(EXP/'experiment.json')
api=HfApi()
out={'checked_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'models':{}}
for name,info in exp.models.items():
    recorded=json.loads((EXP/'lmms_evaluation/full'/name/'model_provenance.json').read_text())
    path=exp.paths[name]
    assert path==Path(recorded['path']).resolve()
    row={'path':str(path),'repository':recorded.get('repository'),'revision':recorded.get('revision'),'files':{}}
    remote={}; latest={}
    if row['repository']:
        server=api.model_info(row['repository'],revision=row['revision'],files_metadata=True)
        head=api.model_info(row['repository'],files_metadata=True)
        row.update(resolved_repository=server.id,verified_revision=server.sha,latest_revision=head.sha)
        remote={f.rfilename:f for f in server.siblings}; latest={f.rfilename:f for f in head.siblings}
        assert server.sha==row['revision']
    for filename,expected in recorded['files_sha256'].items():
        f=path/filename
        h=hashlib.sha256(); g=hashlib.sha1(f'blob {f.stat().st_size}\0'.encode())
        with f.open('rb') as handle:
            for b in iter(lambda:handle.read(16*1024*1024),b''):
                h.update(b)
                if filename in remote and remote[filename].lfs is None:g.update(b)
        actual=h.hexdigest()
        item={'sha256':actual,'bytes':f.stat().st_size,'matches_evaluation_manifest':actual==expected}
        assert actual==expected,(name,filename,'changed since evaluation')
        if remote:
            assert filename in remote,(name,filename,'not in Hub commit')
            r=remote[filename]
            expected_remote=r.lfs.sha256 if r.lfs else r.blob_id
            observed=actual if r.lfs else g.hexdigest()
            item.update(hub_hash=expected_remote,hub_hash_type='sha256' if r.lfs else 'git_blob_sha1',matches_hub=observed==expected_remote)
            assert observed==expected_remote,(name,filename,'Hub mismatch')
            l=latest.get(filename)
            item['same_as_latest_hub']=bool(l and (l.lfs.sha256 if l.lfs else l.blob_id)==expected_remote)
        row['files'][filename]=item
    # Safetensors headers provide structural checks without materializing weights.
    tensor_meta={}
    for f in path.glob('*.safetensors'):
        with f.open('rb') as h:
            length=int.from_bytes(h.read(8),'little'); header=json.loads(h.read(length))
        tensor_meta.update({k:v for k,v in header.items() if k!='__metadata__'})
    row['tensor_count']=len(tensor_meta)
    row['tensor_dtypes']=sorted(set(v['dtype'] for v in tensor_meta.values()))
    row['adapter_or_packed_keys']=[k for k in tensor_meta if any(t in k for t in ('lora_','qweight','qzeros'))]
    row['qat_config']=json.loads((path/'config.json').read_text()).get('qat')
    row['stored_tensor_elements']=sum(__import__('math').prod(v['shape']) for v in tensor_meta.values())
    row['weights_sha256']={n:d['sha256'] for n,d in row['files'].items() if n.endswith('.safetensors')}
    out['models'][name]=row
    (EXP/'reports/audit/checkpoints.json').write_text(json.dumps(out,indent=2)+'\n')
    print('VERIFIED',name,row['revision'],'files',len(row['files']),'tensors',row['tensor_count'],flush=True)
print('All checkpoint bytes match recorded evaluation manifests; all downloaded reference files match their exact Hub commits.',flush=True)
