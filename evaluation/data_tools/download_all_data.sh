#!/usr/bin/env bash
# One entry point for GRACE evaluation and ShareGPT4V data. See --help.
set -euo pipefail
GRACE_DOWNLOAD_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export GRACE_DOWNLOAD_SCRIPT_DIR
GRACE_DOWNLOAD_ROOT="$(cd -- "$GRACE_DOWNLOAD_SCRIPT_DIR/../.." && pwd)"
GRACE_DOWNLOAD_ENV="${GRACE_DATA_ENV:-$GRACE_DOWNLOAD_ROOT/.data-download-venv}"
if [[ -z "${DATA_PYTHON:-}" ]]; then
  if [[ -x "$GRACE_DOWNLOAD_ENV/bin/python" ]]; then
    DATA_PYTHON="$GRACE_DOWNLOAD_ENV/bin/python"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    DATA_PYTHON="$CONDA_PREFIX/bin/python"
  elif [[ -x "${HOME}/miniconda3/envs/grace/bin/python" ]]; then
    DATA_PYTHON="${HOME}/miniconda3/envs/grace/bin/python"
  else
    DATA_PYTHON="$(command -v python3)"
  fi
fi
if ! "$DATA_PYTHON" -c 'import sys; assert sys.version_info >= (3, 10)' 2>/dev/null; then
  echo 'Python 3.10+ is required. Set DATA_PYTHON=/path/to/python.' >&2
  exit 1
fi
GRACE_DOWNLOAD_INSTALL=0
GRACE_DOWNLOAD_PREVIEW=0
GRACE_DOWNLOAD_ARGS=()
for GRACE_DOWNLOAD_ARG in "$@"; do
  case "$GRACE_DOWNLOAD_ARG" in
    --install-deps) GRACE_DOWNLOAD_INSTALL=1 ;;
    --dry-run|--help|-h) GRACE_DOWNLOAD_PREVIEW=1; GRACE_DOWNLOAD_ARGS+=("$GRACE_DOWNLOAD_ARG") ;;
    *) GRACE_DOWNLOAD_ARGS+=("$GRACE_DOWNLOAD_ARG") ;;
  esac
done
if [[ "$GRACE_DOWNLOAD_INSTALL" == 1 && "$GRACE_DOWNLOAD_PREVIEW" == 0 ]]; then
  if [[ ! -x "$GRACE_DOWNLOAD_ENV/bin/python" ]]; then
    "$DATA_PYTHON" -m venv --system-site-packages "$GRACE_DOWNLOAD_ENV"
  fi
  "$GRACE_DOWNLOAD_ENV/bin/python" -m pip install \
    'requests>=2.31,<3' 'huggingface_hub>=0.30,<2' 'pyarrow>=14' \
    'Pillow>=10' 'ijson>=3.2,<4' 'gdown==5.2.0'
  DATA_PYTHON="$GRACE_DOWNLOAD_ENV/bin/python"
fi
exec "$DATA_PYTHON" -u - "${GRACE_DOWNLOAD_ARGS[@]}" <<'GRACE_DATA_PY'
# BEGIN EMBEDDED PYTHON
"""Sequential GRACE data recovery. Embedded in download_all_data.sh; no GPU use."""
from __future__ import annotations
import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import fnmatch
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tarfile
import time
import zipfile
import zlib

GIB = 1024**3
MIB = 1024**2
CHUNK = MIB
SOURCE_GUIDE = 'https://github.com/ForeverBlue816/GRACE#data-preparation'
SFT = 'sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json'
CAPTION = 'sharegpt4v_instruct_gpt4-vision_cap100k.json'
ORDER = ['hallusionbench','mmbench','scienceqa','ai2d','mmmu','seedbench','mmstar',
         'tempcompass','videomme','videommmu','training-annotations','training-llava',
         'training-coco','training-gqa','training-textvqa','training-vg',
         'training-web','training-ocrvqa','training-sam']
HF = {
 'hallusionbench': ('lmms-lab-encoder/HallusionBench','cd417161857aefb23d878d42cf1bb53aa9dd646f', ['data/image-*.parquet','data/non_image-*.parquet'],'HallusionBench'),
 'ai2d': ('lmms-lab-encoder/ai2d','c83a9b9692933aff8349157c88a413df9d02c4e5',['data/test-*.parquet'],'AI2D'),
 'mmmu': ('MMMU/MMMU','98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68',['*/validation-*.parquet'],'mmmu/data'),
 'seedbench': ('AILab-CVC/SEED-Bench','81cb1571b8a3a30af2e62073b8d2054418350f0a',['SEED-Bench.json','SEED-Bench-image.zip'],'SEEDBench/data'),
 'mmstar': ('Lin-Chen/MMStar','bc98d668301da7b14f648724866e57302778ab27',['MMStar.tsv'],'MMStar'),
 'tempcompass': ('lmms-eval/TempCompass','c8a67d88a3bc6fd4b2f9ae2f9e112668fbe05722',['multi-choice/test-*.parquet','tempcompass_videos.zip'],'video_benchmarks/TempCompass/data'),
 'videomme': ('lmms-eval/Video-MME','ead1408f75b618502df9a1d8e0950166bf0a2a0b',['videomme/test-*.parquet','videos_chunked_*.zip'],'video_benchmarks/VideoMME/data'),
 'videommmu': ('lmms-eval/VideoMMMU','d1c35ac933123d79e877b7f1b9506afb0309cf1b',['Perception/test-*.parquet','Comprehension/test-*.parquet','Adaptation/test-*.parquet','*.zip'],'video_benchmarks/VideoMMMU/data'),
}

class Deferred(RuntimeError): pass
class Deadline(Deferred): pass
class Space(Deferred): pass
class Integrity(Deferred): pass


def log(message):
    # Never print auth tokens or signed request query strings.
    print(time.strftime('[%Y-%m-%d %H:%M:%S] ')+str(message), flush=True)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def read_json(path, default=None):
    try: return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError): return default


def digest_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(CHUNK),b''):h.update(block)
    return h.hexdigest()


def key(text): return hashlib.sha256(text.encode()).hexdigest()[:24]


def clean_relative(name):
    p=PurePosixPath(name.replace('\\','/'))
    if p.is_absolute() or '..' in p.parts or '\x00' in name or (p.parts and ':' in p.parts[0]):
        raise Integrity('Unsafe archive/output path: '+repr(name))
    return Path(*p.parts)


def target_path(base,name):
    path=base/clean_relative(name)
    # A preexisting directory symlink must not redirect extraction outside base.
    if not path.resolve().is_relative_to(base.resolve()):
        raise Integrity('Extraction target escapes destination')
    return path


@dataclass
class Item:
    source: str
    url: str
    path: Path
    size: int=0
    sha256: str|None=None
    archive: bool=False
    extract_to: Path|None=None
    layout: str='keep'
    drive_id: str|None=None
    estimate: int=0


class Budget:
    def __init__(self,remaining,mbps=5.0,grace=60.0,max_seconds=None):
        self.seconds=grace+max(0,remaining)/(mbps*1_000_000)
        if max_seconds is not None:self.seconds=min(self.seconds,max_seconds)
        self.end=time.monotonic()+self.seconds
    def left(self):
        remaining=self.end-time.monotonic()
        if remaining<=0:raise Deadline('Dataset download time budget exhausted; partial files retained')
        return remaining
    @contextmanager
    def guard(self):
        # Socket read timeouts alone do not stop a peer that dribbles bytes.
        remaining=self.left()
        old_handler=signal.getsignal(signal.SIGALRM)
        old_timer=signal.getitimer(signal.ITIMER_REAL)
        start=time.monotonic()
        def expired(signum,frame):raise Deadline('Dataset download time budget exhausted; partial files retained')
        signal.signal(signal.SIGALRM,expired)
        signal.setitimer(signal.ITIMER_REAL,remaining)
        try:yield
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGALRM,old_handler)
            if old_timer[0]:signal.setitimer(signal.ITIMER_REAL,max(.001,old_timer[0]-(time.monotonic()-start)),old_timer[1])
    @contextmanager
    def paused(self):
        start=time.monotonic()
        try:yield
        finally:self.end+=time.monotonic()-start


class HTTP:
    def __init__(self,timeout=20):
        import requests
        self.requests=requests;self.session=requests.Session();self.timeout=timeout
        self.token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        if not self.token:
            try:
                from huggingface_hub import get_token
                self.token=get_token()
            except ImportError:pass
    def open(self,url,*,start=None,end=None,budget=None,method='GET'):
        from urllib.parse import urlparse
        headers={'Accept-Encoding':'identity','User-Agent':'GRACE-sequential-data-recovery/1'}
        if self.token and urlparse(url).hostname=='huggingface.co':headers['Authorization']='Bearer '+self.token
        if start is not None:headers['Range']=f'bytes={start}-'+('' if end is None else str(end))
        timeout=min(self.timeout,max(.1,budget.left())) if budget else self.timeout
        with (budget.guard() if budget else nullcontext()):
            r=self.session.request(method,url,headers=headers,stream=True,timeout=(timeout,timeout),allow_redirects=True)
        if r.status_code>=400:
            code=r.status_code;r.close()
            raise Deferred(f'HTTP {code} from {urlparse(url).hostname}; check access/login or expired link')
        if start is not None:
            cr=r.headers.get('Content-Range','')
            match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+|\*)',cr)
            if r.status_code!=206 or not match or int(match[1])!=start or (end is not None and int(match[2])!=end):
                r.close();raise Deferred('Server did not honor exact HTTP byte ranges; refusing a full-file fallback')
        if r.headers.get('Content-Encoding','identity') not in ('identity',''):
            r.close();raise Deferred('Unexpected content encoding for resumable transfer')
        return r
    def json(self,url,budget=None):
        with (budget.guard() if budget else nullcontext()), self.open(url,budget=budget) as r:return r.json()
    def range(self,url,start,end,budget=None):
        with (budget.guard() if budget else nullcontext()), self.open(url,start=start,end=end,budget=budget) as r:
            data=r.raw.read(end-start+2)
        if len(data)!=end-start+1:raise Integrity('Truncated or oversized range response')
        return data


class RemoteZIP(io.RawIOBase):
    """Seekable HTTP range reader: ZIP indexes only; never fetch whole archives."""
    def __init__(self,http,url,size,budget=None):
        self.http,self.url,self.size,self.budget=http,url,size,budget
        self.pos=0;self.cache={}
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if self.pos<0:raise ValueError('Negative seek')
        return self.pos
    def read(self,n=-1):
        n=min(self.size-self.pos,n if n>=0 else self.size-self.pos)
        if n<=0:return b''
        # Tail caching avoids a request for each central-directory field.
        start=self.pos;end=start+n-1
        for (a,b),data in self.cache.items():
            if a<=start and end<=b:self.pos+=n;return data[start-a:end-a+1]
        fetch_end=min(self.size-1,max(end,start+65535))
        data=self.http.range(self.url,start,fetch_end,self.budget)
        self.cache[(start,fetch_end)]=data
        if len(self.cache)>8:self.cache.pop(next(iter(self.cache)))
        self.pos+=n;return data[:n]


class Runner:
    def __init__(self,args):
        self.args=args;self.root=args.root.resolve();self.ev=self.root/'evaluation'
        self.train=(args.training_root or self.root/'ShareGPT4V').resolve()
        self.state=(args.state_dir or self.root/'.data-download').resolve()
        self.http=HTTP(args.socket_timeout);self.budget=None;self.dataset=''
        self.records=self.state/'records';self.parts=self.state/'archives'
        self.metadata=self.state/'metadata';self.current_process=None
        self.selected_video_ids=set();self.references=None;self.discovery=None
    def marker(self,item):
        destination=item.extract_to if item.archive else item.path
        return self.records/(key(item.source+'|destination='+str(destination.resolve()))+'.json')
    def file_record(self,path,sha=None):
        st=path.stat()
        return {'path':str(path),'size':st.st_size,'mtime_ns':st.st_mtime_ns,**({'sha256':sha} if sha else {})}
    def valid_outputs(self,record):
        if not record or not record.get('complete') or not record.get('outputs'):return False
        for row in record['outputs']:
            p=Path(row['path'])
            try:st=p.stat()
            except OSError:return False
            if st.st_size!=row['size']:return False
            if self.args.verify or st.st_mtime_ns!=row.get('mtime_ns'):
                if row.get('sha256'):
                    if digest_file(p)!=row['sha256']:return False
                elif row.get('crc') is not None:
                    if self.crc(p)!=row['crc']:return False
                else:return False
        return True
    @staticmethod
    def crc(path):
        crc=0
        with path.open('rb') as f:
            for block in iter(lambda:f.read(CHUNK),b''):crc=zlib.crc32(block,crc)
        return crc&0xffffffff
    def complete(self,item):
        r=read_json(self.marker(item))
        return self.valid_outputs(r)
    def done(self,item,outputs,**extra):
        atomic_json(self.marker(item),{'source':item.source,'complete':True,'outputs':outputs,**extra})
    def space(self,path,needed):
        path=Path(path)
        while not path.exists():path=path.parent
        free=shutil.disk_usage(path).free;required=max(0,needed)+int(self.args.reserve_gib*GIB)
        if free<required:raise Space(f'Insufficient disk space: need {required/GIB:.2f} GiB including reserve; free {free/GIB:.2f} GiB')
    def pending_bytes(self,item):
        if self.complete(item):return 0
        if item.archive and item.path.exists():return 0
        partial=item.path.with_name(item.path.name+'.part')
        sizes=[partial.stat().st_size] if partial.exists() else []
        if item.drive_id:sizes += [p.stat().st_size for p in item.path.parent.glob(item.path.name+'*.part')]
        return max(0,(item.size or item.estimate)-max(sizes,default=0))
    def check_dataset_space(self,items):
        # Sequential archives are removed after extraction. Reserve the peak, not
        # the sum of all compressed archives, while accounting for retained outputs.
        allocated={};peaks={};paths={}
        def device(path):
            p=Path(path)
            while not p.exists():p=p.parent
            dev=p.stat().st_dev;paths[dev]=p
            return dev
        for item in items:
            if self.complete(item):continue
            src=device(item.path.parent);remaining=self.pending_bytes(item)
            if item.archive:
                dst=device(item.extract_to)
                expanded=int((item.size or item.estimate)*self.args.expansion_factor)
                for dev in {src,dst}:
                    demand=allocated.get(dev,0)+(remaining if dev==src else 0)+(expanded if dev==dst else 0)
                    peaks[dev]=max(peaks.get(dev,0),demand)
                allocated[dst]=allocated.get(dst,0)+expanded
            else:
                allocated[src]=allocated.get(src,0)+remaining
                peaks[src]=max(peaks.get(src,0),allocated[src])
        for dev,needed in peaks.items():self.space(paths[dev],needed)
    def set_budget(self,items):
        self.check_dataset_space(items)
        remaining=sum(self.pending_bytes(i) for i in items)
        self.budget=self.download_budget(remaining)
        log(f'{self.dataset}: {remaining/1e9:.3f} GB remaining; download budget {self.budget.seconds/60:.1f} min (local extraction/checks excluded)')
    def download_budget(self,remaining):
        # Metadata/media phases of one dataset share the same fixed allowance.
        if self.args.budget_mbps is None:return self.budget or Budget(0,grace=self.args.download_timeout)
        return Budget(remaining,self.args.budget_mbps,self.args.grace_seconds,max_seconds=self.args.download_timeout)
    @contextmanager
    def pause(self):
        if self.budget:
            with self.budget.paused():yield
        else:yield
    def run_child(self,command,timeout):
        self.current_process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,text=True)
        try:
            out,err=self.current_process.communicate(timeout=timeout)
            if self.current_process.returncode:raise Deferred('Download/discovery helper failed; check Google Drive permissions/quota or install dependencies')
            return out
        except subprocess.TimeoutExpired:
            self.stop_child()
            self.current_process.communicate()
            raise Deadline('Helper timed out; partial files retained')
        finally:self.current_process=None
    def stop_child(self):
        p=self.current_process
        if p is None:return
        try:os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            p.wait()
    def hf_items(self,repo,rev,patterns,dest,archive_dest=None):
        # Resolve mutable training releases once and pin them in state for all reruns.
        cache=self.metadata/(key(repo+'@'+rev)+'.json')
        info=read_json(cache)
        if info is None:
            info=self.http.json('https://huggingface.co/api/datasets/'+repo+'/revision/'+rev+'?blobs=true',self.discovery)
            atomic_json(cache,info)
        sha=info['sha'];items=[];matched={p:0 for p in patterns}
        for f in info.get('siblings',[]):
            name=f['rfilename'];pats=[p for p in patterns if fnmatch.fnmatchcase(name,p)]
            if not pats:continue
            for p in pats:matched[p]+=1
            size=f.get('size') or f.get('lfs',{}).get('size',0)
            if not size:raise Deferred('Missing remote file size: '+repo+'/'+name)
            archive=name.lower().endswith(('.zip','.tar','.tar.gz','.tgz'))
            source=f'hf:{repo}@{sha}/{name}'
            path=(archive_dest or self.parts/key(repo+'@'+sha))/Path(name).name if archive else dest/clean_relative(name)
            items.append(Item(source,'https://huggingface.co/datasets/'+repo+'/resolve/'+sha+'/'+name,path,size,f.get('lfs',{}).get('sha256'),archive,dest))
        if any(not n for n in matched.values()):raise Deferred('Source listing lacks required patterns: '+','.join(p for p,n in matched.items() if not n))
        return items
    def direct(self,name,url,path,*,archive=False,extract_to=None,layout='keep',estimate=0):
        # HEAD is metadata only; do not download an archive just to determine its size.
        try:
            if self.complete(Item(name,url,path,archive=archive,extract_to=extract_to,layout=layout)):size=0
            else:
                with self.http.open(url,method='HEAD',budget=self.discovery) as r:size=int(r.headers.get('Content-Length',0))
        except Deadline:raise
        except Exception:size=0
        return Item(name,url,path,size,archive=archive,extract_to=extract_to,layout=layout,estimate=estimate)
    def drive(self,source,file_id,path,estimate=0,archive=False,extract_to=None,layout='keep'):
        return Item(source,'',path,drive_id=file_id,estimate=estimate,archive=archive,extract_to=extract_to,layout=layout)
    def drive_listing(self,folder):
        p=self.metadata/('drive-'+folder+'.json');rows=read_json(p)
        if rows is None:
            code=('import gdown,json,sys; fs=gdown.download_folder(id=sys.argv[1],skip_download=True,quiet=True); '
                  'assert fs; print(json.dumps([dict(id=f.id,path=f.path) for f in fs]))')
            rows=json.loads(self.run_child([sys.executable,'-c',code,folder],self.discovery.left() if self.discovery else self.args.discovery_timeout))
            atomic_json(p,rows)
        return rows
    def download(self,item):
        item.path.parent.mkdir(parents=True,exist_ok=True)
        existing=item.path.stat().st_size if item.path.exists() else 0
        proof=read_json(item.path.with_name(item.path.name+'.download.json'),{})
        proof_ok=proof.get('source')==item.source and proof.get('size')==existing and proof.get('mtime_ns')==item.path.stat().st_mtime_ns if existing else False
        if existing and ((item.size and existing==item.size) or proof_ok or item.sha256):
            with self.pause():
                good=not item.sha256 or digest_file(item.path)==item.sha256
            if good:return
        if existing:
            os.replace(item.path,item.path.with_name(item.path.name+'.unverified-'+str(time.time_ns())))
        if item.drive_id:
            self.space(item.path,self.pending_bytes(item))
            log(f'{self.dataset}: Google Drive {item.path.name}')
            code=('import gdown,sys; p=gdown.download(id=sys.argv[1],output=sys.argv[2],resume=True,quiet=True); '
                  'assert p is not None')
            self.run_child([sys.executable,'-c',code,item.drive_id,str(item.path)],self.budget.left())
            if not item.path.is_file() or not item.path.stat().st_size:raise Integrity('Empty Google Drive result')
            atomic_json(item.path.with_name(item.path.name+'.download.json'),dict(source=item.source,**self.file_record(item.path)))
            return
        part=item.path.with_name(item.path.name+'.part');side=part.with_name(part.name+'.json')
        identity=read_json(side,{})
        if part.exists() and identity.get('source')!=item.source:
            os.replace(part,part.with_name(part.name+'.unverified-'+str(time.time_ns())))
        for attempt in range(self.args.retries+1):
            offset=part.stat().st_size if part.exists() else 0
            if item.size and offset>item.size:raise Integrity('Partial download exceeds expected source size')
            self.space(item.path,max(0,(item.size or item.estimate)-offset))
            if item.size and offset==item.size:break
            try:
                with self.budget.guard(), self.http.open(item.url,start=offset if offset else None,budget=self.budget) as r:
                    etag=r.headers.get('ETag') or r.headers.get('Last-Modified')
                    if offset and identity.get('etag') and etag!=identity['etag']:
                        raise Integrity('Remote identity changed; partial retained for inspection')
                    if 'text/html' in r.headers.get('Content-Type','') and item.path.suffix not in ('.html','.htm'):
                        raise Integrity('Source returned HTML instead of dataset data')
                    atomic_json(side,{'source':item.source,'etag':etag,'size':item.size})
                    identity=read_json(side)
                    with part.open('ab' if offset else 'wb') as f:
                        reported=time.monotonic()
                        for block in r.iter_content(CHUNK):
                            self.budget.left()
                            if not block:continue
                            self.space(part,len(block));f.write(block)
                            if time.monotonic()-reported>=30:
                                log(f'{self.dataset}: {item.path.name}: {f.tell()/1e9:.3f} GB saved; {self.budget.left()/60:.1f} min budget left');reported=time.monotonic()
                    if item.size and part.stat().st_size!=item.size:raise Integrity('Download length mismatch')
                break
            except (Deadline,Space,Integrity,Deferred):raise
            except self.http.requests.RequestException as e:
                if attempt==self.args.retries:raise Deferred(f'Network transfer failed ({type(e).__name__}); partial retained') from e
                self.budget.left()
        if not part.exists():raise Integrity('No downloaded file')
        with self.pause():
            if item.sha256 and digest_file(part)!=item.sha256:
                os.replace(part,part.with_name(part.name+'.corrupt-'+str(time.time_ns())))
                raise Integrity('SHA-256 mismatch; corrupt bytes quarantined for inspection')
        os.replace(part,item.path)
        atomic_json(item.path.with_name(item.path.name+'.download.json'),dict(source=item.source,**self.file_record(item.path)))
        side.unlink(missing_ok=True)
    @staticmethod
    def layout_name(name,layout):
        p=clean_relative(name);parts=list(p.parts)
        if layout=='llava':
            return str(p if parts and parts[0]=='images' else Path('images')/p)
        if layout=='sam':
            return str(Path('images')/p.name) if p.suffix.lower() in ('.jpg','.jpeg','.png') else None
        if layout=='textvqa':
            if parts and parts[0] in ('train_images','train_val_images','trainval_images'):parts=parts[1:]
            return str(Path('train_images',*parts))
        if layout.startswith('web'):
            anchors={'share_textvqa','web-celebrity','web-landmark','wikiart'}
            pos=next((i for i,x in enumerate(parts) if x in anchors),None)
            if pos is None:
                anchor=layout.partition(':')[2]
                if not anchor:return None
                if parts and parts[0]=='images':parts=parts[1:]
                return str(Path(anchor,'images',*parts))
            return str(Path(*parts[pos:]))
        return str(p)
    def extract(self,item):
        base=item.extract_to
        if base is None:raise Integrity('Archive destination is unspecified')
        base.mkdir(parents=True,exist_ok=True)
        log(f'{self.dataset}: extracting {item.path.name}')
        with self.pause():
            if zipfile.is_zipfile(item.path):
                with zipfile.ZipFile(item.path) as z:
                    entries=[]
                    for info in z.infolist():
                        clean_relative(info.filename)
                        if info.is_dir():continue
                        if stat.S_ISLNK(info.external_attr>>16):raise Integrity('Archive contains a symlink')
                        name=self.layout_name(info.filename,item.layout)
                        if name is not None:entries.append((name,info.file_size,info.CRC,lambda info=info:z.open(info)))
                    outputs=self.extract_entries(base,entries)
            elif tarfile.is_tarfile(item.path):
                with tarfile.open(item.path,'r:*') as tar:
                    entries=[]
                    for info in tar:
                        clean_relative(info.name)
                        if info.isdir():continue
                        if not info.isfile():raise Integrity('TAR links/devices are not supported')
                        name=self.layout_name(info.name,item.layout)
                        if name is not None:entries.append((name,info.size,None,lambda info=info:tar.extractfile(info)))
                    outputs=self.extract_entries(base,entries)
            else:raise Integrity('Downloaded archive is not a supported ZIP/TAR')
            if not outputs:raise Integrity('Archive contained no selected data files')
            self.done(item,outputs)
            # Marker is committed before deletion. If interrupted, rerun validates outputs first.
            item.path.unlink()
        log(f'{self.dataset}: extracted and removed {item.path.name}')
    def extract_entries(self,base,entries):
        # ZIP entries describe actual uncompressed sizes, so space is checked precisely here.
        missing=sum(size for name,size,_,_ in entries if not (base/clean_relative(name)).is_file())
        self.space(base,missing)
        outputs=[];seen=set()
        for name,size,crc,opener in entries:
            target=target_path(base,name)
            if str(target) in seen:raise Integrity('Archive has duplicate output paths')
            seen.add(str(target))
            valid=target.is_file() and target.stat().st_size==size and crc is not None and self.crc(target)==crc
            if not valid:
                target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.name+'.extracting')
                with opener() as src,tmp.open('wb') as dst:
                    total=0;actual=0
                    for block in iter(lambda:src.read(CHUNK),b''):
                        self.space(tmp,len(block));total+=len(block)
                        if total>size:raise Integrity('Expanded archive entry exceeds declared size')
                        actual=zlib.crc32(block,actual);dst.write(block)
                if total!=size or (crc is not None and actual&0xffffffff!=crc):raise Integrity('Archive member length/CRC mismatch')
                os.replace(tmp,target)
            row=self.file_record(target)
            if crc is not None:row['crc']=crc
            else:row['sha256']=digest_file(target)
            outputs.append(row)
        return outputs
    def process(self,item):
        if self.complete(item):
            # Complete extraction plus a leftover archive means a crash happened before cleanup.
            if item.archive and item.path.exists():item.path.unlink()
            return
        reserve=int((item.size or item.estimate)*self.args.expansion_factor) if item.archive else 0
        # State/archive and final destinations may be on different filesystems.
        same_fs=False
        if item.archive:
            a=item.path.parent;b=item.extract_to
            while not a.exists():a=a.parent
            while not b.exists():b=b.parent
            same_fs=a.stat().st_dev==b.stat().st_dev
        self.space(item.path,self.pending_bytes(item)+(reserve if same_fs else 0))
        if item.archive and not same_fs:self.space(item.extract_to,reserve)
        log(f'{self.dataset}: downloading {item.path.name}')
        self.download(item)
        if item.archive:self.extract(item)
        else:
            with self.pause():self.done(item,[self.file_record(item.path,digest_file(item.path))])
    def configure_regular(self,name):
        if name in HF:
            repo,rev,pats,relative=HF[name]
            return self.hf_items(repo,rev,pats,self.ev/relative)
        if name=='mmbench':
            return [self.direct('mmbench-en-dev-20231003','https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_en_20231003.tsv',self.ev/'MMBench/mmbench_dev_en_20231003.tsv',estimate=140_000_000)]
        if name=='scienceqa':
            base=self.ev/'ScienceQA/data'
            items=[self.direct('scienceqa-'+f,'https://raw.githubusercontent.com/lupantech/ScienceQA/main/data/scienceqa/'+f,base/f,estimate=10_000_000) for f in ['problems.json','pid_splits.json']]
            items.append(self.direct('scienceqa-test-images','https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip',self.parts/'scienceqa-test.zip',archive=True,extract_to=base/'images',estimate=220_000_000))
            return items
        if name=='training-annotations':return self.hf_items('Lin-Chen/ShareGPT4V','main',[CAPTION,SFT],self.train)
        if name=='training-vg' and self.args.vg_source=='huggingface':
            return self.hf_items('pppop7/VisualGenome_VG_100K_1_and_2','main',['images.zip','images2.zip'],self.train/'data/vg')
        if name=='training-llava':
            items=self.hf_items('liuhaotian/LLaVA-Pretrain','main',['images.zip'],self.train/'data/llava/llava_pretrain')
            for item in items:
                item.layout='llava'
                self.repair_llava_layout(item)
            return items
        sources={
          'training-coco':[('coco-train2017','http://images.cocodataset.org/zips/train2017.zip','coco','keep',19_000_000_000)],
          'training-gqa':[('gqa-images','https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip','gqa','keep',25_000_000_000)],
          'training-textvqa':[('textvqa-trainval','https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip','textvqa','textvqa',8_000_000_000)],
          'training-vg':[
            ('vg-part1','https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip','vg','keep',10_000_000_000),
            ('vg-part2','https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip','vg','keep',6_000_000_000)]}
        if name in sources:return [self.direct(k,url,self.parts/(k+'.zip'),archive=True,extract_to=self.train/'data'/dest,layout=layout,estimate=estimate) for k,url,dest,layout,estimate in sources[name]]
        if name=='training-web':
            rows=self.drive_listing('1tCUQ-sq6vdshZVkF0ZeF3K4eztkXJgax')
            # The same folder also contains evaluation resources (playground.zip)
            # and a separate SAM subset. Neither belongs to WebData training.
            required={'share_textvqa.zip','web-celebrity.zip','web-landmark.zip','wikiart.zip'}
            zips=[r for r in rows if Path(r['path']).name in required]
            missing=required-{Path(r['path']).name for r in zips}
            if missing:raise Deferred('Official Drive folder lacks WebData ZIPs: '+','.join(sorted(missing)))
            items=[]
            for r in zips:
                anchor=next((a for a in ['share_textvqa','web-celebrity','web-landmark','wikiart'] if a in Path(r['path']).name),None)
                items.append(self.drive('web-'+r['id'],r['id'],self.parts/('web-'+Path(r['path']).name),estimate=2*GIB,archive=True,extract_to=self.train/'data',layout='web'+(':'+anchor if anchor else '')))
            return items
        if name=='training-sam':return self.sam_items()
        raise Deferred('Unknown dataset: '+name)
    def repair_llava_layout(self,item):
        # Older runs extracted numeric shards beside images/ instead of inside it.
        # Directory renames preserve every image and avoid downloading 13 GB again.
        base=item.extract_to
        if not base.exists():return
        shards=[p for p in base.iterdir() if re.fullmatch(r'\d{5}',p.name) and p.is_dir() and not p.is_symlink()]
        if shards:(base/'images').mkdir(exist_ok=True)
        if any((base/'images'/shard.name).exists() for shard in shards):raise Integrity('Conflicting legacy LLaVA shard directories')
        for shard in shards:
            destination=base/'images'/shard.name
            shard.rename(destination)
        record=read_json(self.marker(item))
        changed=False
        if record:
            for row in record.get('outputs',[]):
                relative=Path(row['path']).relative_to(base)
                if re.fullmatch(r'\d{5}',relative.parts[0]) and (base/'images'/relative).is_file():
                    row['path']=str(base/'images'/relative);changed=True
            if changed:atomic_json(self.marker(item),record)
        if shards:log(f'training-llava: repaired {len(shards)} legacy shard directories under images/')
    def sam_items(self):
        if self.args.sam_mode=='sft9k':
            return [self.drive('sam-sft9k','1dKumdOKSXtV7lIXdrG7jsIK_z2vZv2gs',self.parts/'sam-sft9k.zip',estimate=4*GIB,archive=True,extract_to=self.train/'data/sam',layout='sam')]
        links=self.args.sam_links
        if links is None:raise Deferred('Full SAM release needs --sam-links with fresh official sa_000000.tar through sa_000050.tar URLs')
        rows={}
        for line in links.read_text().splitlines():
            match=re.search(r'(?:sa_)?(\d{6})\.tar',line)
            url=re.search(r'https?://\S+',line)
            if match and url and 0<=int(match[1])<=50:rows[int(match[1])]=url[0]
        self.missing_sam=[i for i in range(51) if i not in rows]
        if not rows:raise Deferred('SAM URL list contains no selected shard links')
        return [self.direct(f'sam-shard-{i:06}',url,self.parts/f'sa_{i:06}.tar',archive=True,extract_to=self.train/'data/sam',layout='sam',estimate=12*GIB) for i,url in sorted(rows.items())]
    def video_mme(self):
        items=self.configure_regular('videomme');annotations=[i for i in items if not i.archive];archives=[i for i in items if i.archive]
        # Annotation download has its own small budget; media has a budget from selected compressed bytes.
        self.set_budget(annotations)
        for item in annotations:self.process(item)
        import pyarrow.parquet as pq
        rows=[]
        for item in annotations:rows.extend(pq.read_table(item.path).to_pylist())
        from collections import Counter
        counts=Counter(r['duration'] for r in rows)
        if counts!={'short':900,'medium':900,'long':900}:raise Integrity('Unexpected Video-MME annotation population')
        ids={str(r['videoID']) for r in rows if r['duration'] in ('short','medium')}
        if len(ids)!=600:raise Integrity('Expected 600 distinct short/medium Video-MME clips')
        selection=[];found=set()
        index_budget=Budget(0,grace=self.args.discovery_timeout)
        for item in archives:
            cache=self.metadata/('zip-index-'+key(item.source)+'.json');entries=read_json(cache)
            if entries is None:
                log('videomme: reading range index '+item.path.name)
                remote=RemoteZIP(self.http,item.url,item.size,index_budget)
                with zipfile.ZipFile(remote) as z:
                    entries=[dict(filename=i.filename,header_offset=i.header_offset,compress_size=i.compress_size,file_size=i.file_size,CRC=i.CRC,compress_type=i.compress_type,flag_bits=i.flag_bits) for i in z.infolist() if not i.is_dir()]
                atomic_json(cache,entries)
            for info in entries:
                clean_relative(info['filename'])
                if Path(info['filename']).suffix.lower()=='.mp4' and Path(info['filename']).stem in ids:
                    vid=Path(info['filename']).stem
                    if vid in found:raise Integrity('Duplicate Video-MME member across chunks')
                    found.add(vid);selection.append((item,info))
        if found!=ids:raise Integrity('ZIP indexes do not cover all 600 short/medium videos')
        media=[]
        for archive,info in selection:
            vid=Path(info['filename']).stem
            item=Item(archive.source+'#'+info['filename'],archive.url,self.ev/'video_benchmarks/VideoMME/data/data'/f'{vid}.mp4',info['compress_size'])
            media.append((item,archive,info))
        remaining=0
        for item,archive,info in media:
            part=self.parts/(key(item.source)+'.compressed.part')
            if not self.complete(item):remaining+=max(0,info['compress_size']-(part.stat().st_size if part.exists() else 0))
        self.budget=self.download_budget(remaining)
        log(f'videomme: selected 600 short/medium clips; zero long members selected; remaining {remaining/1e9:.3f} GB')
        pending=[(item,info) for item,_,info in media if not self.complete(item)]
        if pending:
            needed=sum(info['file_size'] for _,info in pending);largest=max(info['compress_size'] for _,info in pending)
            self.parts.mkdir(parents=True,exist_ok=True)
            base=self.ev/'video_benchmarks/VideoMME/data/data';base.mkdir(parents=True,exist_ok=True)
            same=base.stat().st_dev==self.parts.stat().st_dev
            self.space(base,needed+(largest if same else 0))
            if not same:self.space(self.parts,largest)
        for index,(item,archive,info) in enumerate(media,1):
            self.zip_member(item,archive,info)
            if index%25==0:log(f'videomme: verified {index}/{len(media)} selected clips')
        alias=self.ev/'video_benchmarks/VideoMME/data/videos'
        if not alias.exists() and not alias.is_symlink():alias.symlink_to('data',target_is_directory=True)
        return [self.file_record(item.path) for item,_,_ in media]
    def zip_member(self,item,archive,info):
        part=self.parts/(key(item.source)+'.compressed.part')
        if self.complete(item):
            part.unlink(missing_ok=True)
            return
        if info['flag_bits']&1 or info['compress_type'] not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED):raise Integrity('Encrypted/unsupported remote ZIP entry')
        a=item.path.parent;b=part.parent
        while not a.exists():a=a.parent
        while not b.exists():b=b.parent
        partial=part.stat().st_size if part.exists() else 0
        extra=max(0,info['compress_size']-partial) if a.stat().st_dev==b.stat().st_dev else 0
        self.space(item.path,info['file_size']+extra)
        start=info['header_offset'];header=self.http.range(item.url,start,start+29,self.budget)
        fields=struct.unpack('<4s5H3L2H',header)
        if fields[0]!=b'PK\x03\x04':raise Integrity('Invalid ZIP local header')
        data_start=start+30+fields[-2]+fields[-1]
        part=self.parts/(key(item.source)+'.compressed.part');part.parent.mkdir(parents=True,exist_ok=True)
        size=info['compress_size'];offset=part.stat().st_size if part.exists() else 0
        if offset>size:raise Integrity('Oversized compressed partial')
        self.space(part,size-offset)
        if offset<size:
            with self.budget.guard(), self.http.open(item.url,start=data_start+offset,end=data_start+size-1,budget=self.budget) as r,part.open('ab') as dst:
                for block in r.iter_content(CHUNK):
                    self.budget.left()
                    if block:self.space(part,len(block));dst.write(block)
        if part.stat().st_size!=size:raise Integrity('Compressed ZIP member incomplete')
        with self.pause():
            item.path.parent.mkdir(parents=True,exist_ok=True);tmp=item.path.with_name(item.path.name+'.extracting')
            dec=zlib.decompressobj(-15) if info['compress_type']==zipfile.ZIP_DEFLATED else None
            crc=0;total=0
            with part.open('rb') as src,tmp.open('wb') as dst:
                for block in iter(lambda:src.read(CHUNK),b''):
                    pending=block
                    while pending:
                        out=dec.decompress(pending,CHUNK) if dec else pending
                        pending=dec.unconsumed_tail if dec else b''
                        total+=len(out)
                        if total>info['file_size']:raise Integrity('ZIP member expands beyond expected size')
                        self.space(tmp,len(out));crc=zlib.crc32(out,crc);dst.write(out)
                if dec and not dec.eof:raise Integrity('Incomplete deflate stream')
            if total!=info['file_size'] or crc&0xffffffff!=info['CRC']:
                part.rename(part.with_name(part.name+'.corrupt-'+str(time.time_ns())))
                raise Integrity('ZIP member CRC/length mismatch')
            os.replace(tmp,item.path)
            row=self.file_record(item.path);row['crc']=info['CRC'];self.done(item,[row],selection='short_medium')
            part.unlink()
    def ocr_vqa(self):
        rows=self.drive_listing('1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_')
        selected=[r for r in rows if Path(r['path']).name=='dataset.json']
        if len(selected)!=1:raise Deferred('Official OCR-VQA folder lacks a unique dataset.json')
        base=self.train/'data/ocr_vqa';meta=self.drive('ocrvqa-metadata',selected[0]['id'],base/'dataset.json',estimate=20*MIB)
        self.set_budget([meta]);self.process(meta)
        docs=json.loads(meta.path.read_text());required=self.training_references().get('ocr_vqa',set())
        if not required:raise Deferred('No OCR-VQA references found; training annotations must be available first')
        items=[]
        for image_path in sorted(required):
            image_id=Path(image_path).stem
            if image_id not in docs:raise Integrity('OCR-VQA reference absent from official metadata')
            item=Item('ocrvqa-image-'+image_id,docs[image_id]['imageURL'],self.parts/('ocr-'+image_id),estimate=300_000)
            items.append((item,self.train/'data'/clean_relative(image_path)))
        self.set_budget([i for i,_ in items]);outputs=[];failures=0
        from PIL import Image
        for item,target in items:
            if self.complete(item):outputs+=read_json(self.marker(item))['outputs'];continue
            try:
                self.download(item)
                with self.pause():
                    target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.name+'.extracting')
                    with Image.open(item.path) as im:im.convert('RGB').save(tmp,format='JPEG',quality=95)
                    os.replace(tmp,target);row=self.file_record(target,digest_file(target));self.done(item,[row]);outputs.append(row);item.path.unlink()
            except (Deadline,Space):raise
            except Exception:failures+=1;log('training-ocrvqa: failed image '+target.name+'; will retry on next run')
        if failures:raise Deferred(f'{failures} OCR-VQA images failed; source URLs may be stale')
        return outputs
    def training_references(self):
        if self.references is not None:return self.references
        try:import ijson
        except ImportError:raise Deferred('ijson is required to inspect the large training annotations without loading them into RAM')
        groups={}
        for file in [self.train/CAPTION,self.train/SFT]:
            if not file.exists():raise Deferred('Both training annotation JSONs are needed for image-reference validation')
            with file.open('rb') as f:
                for row in ijson.items(f,'item'):
                    name=row.get('image')
                    if not name:continue
                    path=clean_relative(name)
                    if path.parts and path.parts[0]=='data':path=Path(*path.parts[1:])
                    if not path.parts:continue
                    groups.setdefault(path.parts[0],set()).add(str(path))
        self.references=groups;return groups
    def validate_training(self):
        try:groups=self.training_references()
        except Deferred as e:return {'status':'incomplete','reason':str(e)}
        missing={};counts={}
        for group,names in groups.items():
            counts[group]=len(names);bad=[n for n in sorted(names) if not (self.train/'data'/n).is_file() or not (self.train/'data'/n).stat().st_size]
            if bad:missing[group]={'count':len(bad),'examples':bad[:10]}
        return {'status':'complete' if not missing else 'incomplete','referenced_images':counts,'missing':missing}
    def datasets(self,names):
        summary={};self.state.mkdir(parents=True,exist_ok=True);self.records.mkdir(exist_ok=True);self.parts.mkdir(exist_ok=True);self.metadata.mkdir(exist_ok=True)
        for name in names:
            self.dataset=name;self.budget=None;self.discovery=Budget(0,grace=self.args.discovery_timeout);start=time.time();log('Starting '+name)
            try:
                if name=='videomme':outputs=self.video_mme()
                elif name=='training-ocrvqa':outputs=self.ocr_vqa()
                else:
                    items=self.configure_regular(name);self.set_budget(items)
                    for item in items:self.process(item)
                    if name=='training-sam' and self.args.sam_mode=='full51' and self.missing_sam:
                        raise Deferred('SAM URL list is missing shard numbers: '+','.join(f'{i:06}' for i in self.missing_sam))
                    outputs=[o for item in items for o in read_json(self.marker(item),{}).get('outputs',[])]
                if not outputs:raise Integrity('Dataset produced no completed output files')
                summary[name]={'status':'complete','outputs':len(outputs)};log(name+': complete')
            except (KeyboardInterrupt,SystemExit):raise
            except Exception as e:
                # Do not print request exception URLs, which can include signed credentials.
                message=str(e) if isinstance(e,Deferred) else type(e).__name__+' (see source access/dependencies; rerun to resume)'
                summary[name]={'status':'deferred','reason':message};log(name+': '+message+'; continuing with next dataset')
            summary[name]['elapsed_seconds']=round(time.time()-start,2)
            atomic_json(self.state/'latest_summary.json',summary)
        if any(n.startswith('training-') for n in names):
            with self.pause():summary['training-reference-check']=self.validate_training()
            atomic_json(self.state/'latest_summary.json',summary)
        log('Summary: '+str(self.state/'latest_summary.json'))
        required=[summary[n] for n in names]
        if all(n in names for n in ORDER if n.startswith('training-')):required.append(summary['training-reference-check'])
        return 0 if all(v['status']=='complete' for v in required) else 2


def parse_args(argv=None):
    default_root=Path(os.environ.get('GRACE_DOWNLOAD_SCRIPT_DIR','.')).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--root',type=Path,default=default_root,help='GRACE root; evaluation/ is placed here')
    p.add_argument('--training-root',type=Path,default=Path(os.environ['SHAREGPT4V_ROOT']) if os.environ.get('SHAREGPT4V_ROOT') else None)
    p.add_argument('--state-dir',type=Path,help='Partial archives, resumable records, metadata and summary')
    p.add_argument('--datasets',nargs='+',choices=ORDER,default=None,help='Dataset IDs, in the requested order')
    p.add_argument('--only',choices=['all','evaluation','training'],default='all')
    p.add_argument('--download-timeout',type=float,default=3600.,help='Per-dataset network time limit in seconds; extraction/checks pause the clock')
    p.add_argument('--budget-mbps',type=float,default=None,help='Optional legacy byte-derived budget, capped by --download-timeout; NOT a bandwidth cap')
    p.add_argument('--grace-seconds',type=float,default=60.,help='Additional per-dataset setup/network allowance')
    p.add_argument('--reserve-gib',type=float,default=10.,help='Disk headroom retained on every destination filesystem')
    p.add_argument('--expansion-factor',type=float,default=1.5,help='Conservative pre-download extraction space estimate; actual sizes checked before extraction')
    p.add_argument('--socket-timeout',type=float,default=15.)
    p.add_argument('--discovery-timeout',type=float,default=90.)
    p.add_argument('--retries',type=int,default=2)
    p.add_argument('--sam-mode',choices=['full51','sft9k'],default='full51')
    p.add_argument('--vg-source',choices=['stanford','huggingface'],default='stanford',help='Original VG host or third-party ZIP mirror with pinned revision and SHA-256 verification')
    p.add_argument('--sam-links',type=Path,help='Fresh official SAM URLs, one filename and URL per line')
    p.add_argument('--verify',action='store_true',help='Rehash/recheck existing completed outputs before skipping')
    p.add_argument('--dry-run',action='store_true',help='Print selected sources/layout; no writes or network')
    p.add_argument('--install-deps',action='store_true',help='Bash launcher installs dependencies in an isolated environment before running')
    args=p.parse_args(argv)
    values=[args.download_timeout,args.grace_seconds,args.reserve_gib,args.expansion_factor,args.socket_timeout,args.discovery_timeout]
    if args.budget_mbps is not None:values.append(args.budget_mbps)
    if any(not math.isfinite(v) for v in values) or args.download_timeout<=0 or (args.budget_mbps is not None and args.budget_mbps<=0) or args.grace_seconds<0 or args.reserve_gib<0 or args.expansion_factor<1 or args.socket_timeout<=0 or args.discovery_timeout<=0 or args.retries<0:p.error('Invalid budget/space/timeout/retry setting')
    if args.sam_links is None:
        local=args.root.resolve().parents[1]/'sa1b_links.txt'
        if local.exists():args.sam_links=local
    return args


def main(argv=None):
    args=parse_args(argv)
    names=args.datasets or [n for n in ORDER if args.only=='all' or (n.startswith('training-'))==(args.only=='training')]
    if len(set(names))!=len(names):raise SystemExit('Duplicate dataset IDs are not allowed')
    if args.dry_run:
        print(json.dumps({'datasets':names,'root':str(args.root),'training_root':str(args.training_root or args.root/'ShareGPT4V'),
            'budget':{'download_timeout_seconds':args.download_timeout,'budget_mbps':args.budget_mbps,'grace_seconds':args.grace_seconds,'local_checks_pause_clock':True},
            'sam_mode':args.sam_mode,'sam_links':str(args.sam_links) if args.sam_links else None,
            'vg_source':args.vg_source,
            'videomme':'600 short/medium clips only, selected by annotation IDs and HTTP ZIP ranges; no long video payloads',
            'sources':{n:{'repository':HF[n][0],'revision':HF[n][1],'files':HF[n][2]} for n in names if n in HF},
            'training_guide':SOURCE_GUIDE,'writes':False,'network':False},indent=2));return 0
    import fcntl
    runner=Runner(args);runner.state.mkdir(parents=True,exist_ok=True)
    # One manifest owner per state AND per training/evaluation root, even with custom state dirs.
    locks=[]
    for path in sorted({runner.state/'.lock',runner.root/'.data-download.lock',runner.train/'.data-download.lock'}):
        path.parent.mkdir(parents=True,exist_ok=True);f=path.open('a+')
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another downloader owns '+str(path))
        locks.append(f)
    def stop(signum,frame):
        runner.stop_child();raise KeyboardInterrupt
    for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,stop)
    try:return runner.datasets(names)
    except KeyboardInterrupt:
        log('Interrupted; partial downloads retained. Rerun the same command to resume.');return 130
    finally:
        runner.stop_child()
        for f in locks:f.close()

if __name__=='__main__':raise SystemExit(main())

# END EMBEDDED PYTHON
GRACE_DATA_PY
