from pathlib import Path
import json,hashlib,zlib,subprocess,tarfile
root=Path('/raid/lvchengtao/Minh/Distillation/reproduction/GRACE');src=root/'third_party/lmms-eval-v0.7.1'
d=json.loads((root/'recovery/20261027/lmms_original_commit.json').read_text())
payload=d['verification']['payload'];headers,message=payload.split('\n\n',1)
signature=d['verification']['signature'];raw=headers+'\ngpgsig '+signature.replace('\n','\n ').rstrip(' ')+'\n'+message
# A signature ends with a newline. Preserve Git's header/body separator exactly.
variants=[]
for sig in [signature,signature.rstrip('\n')]:
 variants.append(headers+'\ngpgsig '+sig.replace('\n','\n ')+'\n\n'+message)
 variants.append(headers+'\ngpgsig '+sig.replace('\n','\n ')+'\n'+message)
raw=next((x.encode() for x in variants if hashlib.sha1(b'commit '+str(len(x.encode())).encode()+b'\0'+x.encode()).hexdigest()==d['sha']),None)
if raw is None:raise SystemExit('Exact commit hash could not be reconstructed; no Git metadata installed')
with tarfile.open('/tmp/grace-recovery-20261027/lmms-eval-88b23e2.tar.gz') as tar:
 for m in tar.getmembers():
  if m.isfile():(src/Path(m.name).relative_to(m.name.split('/')[0])).chmod(m.mode&0o777)
meta=Path('/tmp/grace-recovery-20261027/lmms_git_metadata')
subprocess.run(['git','init','--quiet',str(meta)],check=True)
cmd=['git','--git-dir',str(meta/'.git'),'--work-tree',str(src)]
subprocess.run(cmd+['-c','core.autocrlf=false','add','-f','-A'],check=True)
tree=subprocess.check_output(cmd+['write-tree'],text=True).strip()
if tree!=d['tree']['sha']:raise SystemExit('Archive tree hash mismatch: '+tree)
sha=d['sha'];obj=meta/'.git/objects'/sha[:2]/sha[2:];obj.parent.mkdir(exist_ok=True);obj.write_bytes(zlib.compress(b'commit '+str(len(raw)).encode()+b'\0'+raw))
(meta/'.git/HEAD').write_text(sha+'\n');(meta/'.git/shallow').write_text(sha+'\n')
import shutil
shutil.copytree(meta/'.git',src/'.git',dirs_exist_ok=True)
print('Restored exact pinned Git commit and verified full source tree:',sha,tree)
print(subprocess.check_output(['git','-C',str(src),'status','--porcelain'],text=True))
