from pathlib import Path as RealPath
import ast,json,re,shutil,types,hashlib,copy,math,textwrap,collections,posixpath,builtins,shlex,csv,pprint
OUT=RealPath('/tmp/grace-recovery-20261027')
STAGE=OUT/'reconstructed';STAGE.mkdir(exist_ok=True)
WORK=STAGE/'workspace';ROOT=WORK/'reproduction/GRACE';ROOT.mkdir(parents=True,exist_ok=True)
TMP=STAGE/'tmp';TMP.mkdir(exist_ok=True)
PREFIX='/raid/lvchengtao/Minh/Distillation'
LOG=[];PROV={};context={}
def physical(p):
 s=str(p)
 if s.startswith(PREFIX+'/'):s=s[len(PREFIX)+1:]
 elif s==PREFIX:return WORK
 elif s.startswith('/tmp/'):
  return TMP/s[len('/tmp/'):]
 elif s.startswith('/'):raise ValueError('outside reconstruction: '+s)
 if not (s=='reproduction/GRACE' or s.startswith('reproduction/GRACE/')):raise ValueError('outside GRACE: '+s)
 q=WORK/s
 if '.git' in q.parts or '..' in q.parts:raise ValueError('protected path')
 return q
class P:
 def __init__(self,*args):self.s=posixpath.join(*(str(x) for x in args)) if args else '.'
 def __str__(self):return self.s
 def __fspath__(self):return self.s
 def __repr__(self):return 'Path('+repr(self.s)+')'
 def __hash__(self):return hash(self.s)
 def __eq__(self,o):return str(self)==str(o)
 def __lt__(self,o):return str(self)<str(o)
 def __truediv__(self,o):return P(posixpath.join(self.s,str(o)))
 @property
 def parent(self):return P(posixpath.dirname(self.s))
 @property
 def parents(self):return [P(str(x)) for x in RealPath(self.s).parents]
 @property
 def name(self):return posixpath.basename(self.s)
 @property
 def stem(self):return RealPath(self.s).stem
 @property
 def suffix(self):return RealPath(self.s).suffix
 @property
 def parts(self):return RealPath(self.s).parts
 def resolve(self):return P(self.s if self.s.startswith('/') else PREFIX+'/'+self.s)
 absolute=resolve
 def relative_to(self,o):return P(str(RealPath(self.s).relative_to(str(o))))
 def with_name(self,s):return self.parent/s
 def with_suffix(self,s):return P(str(RealPath(self.s).with_suffix(s)))
 def read_text(self,*args,**kw):return physical(self).read_text(*args,**kw)
 def read_bytes(self):return physical(self).read_bytes()
 def write_text(self,data,*args,**kw):
  q=physical(self);q.parent.mkdir(parents=True,exist_ok=True);r=q.write_text(data,*args,**kw);PROV[str(q.relative_to(STAGE))]=dict(context);return r
 def write_bytes(self,data):
  q=physical(self);q.parent.mkdir(parents=True,exist_ok=True);r=q.write_bytes(data);PROV[str(q.relative_to(STAGE))]=dict(context);return r
 def exists(self):return physical(self).exists()
 def is_file(self):return physical(self).is_file()
 def is_dir(self):return physical(self).is_dir()
 def stat(self):return physical(self).stat()
 def mkdir(self,*args,**kw):return physical(self).mkdir(*args,**kw)
 def glob(self,pattern):return (P(str(self/x.relative_to(physical(self)))) for x in physical(self).glob(pattern))
 def rglob(self,pattern):return self.glob('**/'+pattern)
 def iterdir(self):return (self/x.name for x in physical(self).iterdir())
 def rename(self,target):
  q=physical(target);q.parent.mkdir(parents=True,exist_ok=True);physical(self).rename(q);return P(str(target))
 replace=rename
 def unlink(self,missing_ok=False):
  q=physical(self)
  if q.exists():q.unlink()
  elif not missing_ok:raise FileNotFoundError(str(self))
 def open(self,mode='r',*args,**kw):
  q=physical(self)
  if any(x in mode for x in 'wax+'):q.parent.mkdir(parents=True,exist_ok=True);PROV[str(q.relative_to(STAGE))]=dict(context)
  return q.open(mode,*args,**kw)
 def chmod(self,mode):physical(self).chmod(mode)
 @classmethod
 def cwd(cls):return P(PREFIX)
 @classmethod
 def home(cls):raise ValueError('home not available')
def safe_copy(src,dst,*a,**kw):
 srcq=physical(src);dstq=physical(dst);dstq.parent.mkdir(parents=True,exist_ok=True)
 return shutil.copy2(srcq,dstq)
def safe_move(src,dst):return P(str(src)).rename(dst)
modules={'pathlib':types.SimpleNamespace(Path=P),'json':json,'re':re,'hashlib':hashlib,'copy':copy,'math':math,'textwrap':textwrap,'collections':collections,'ast':ast,'shlex':shlex,'csv':csv,'pprint':pprint,
 'shutil':types.SimpleNamespace(copy2=safe_copy,copy=safe_copy,copyfile=safe_copy,move=safe_move),
 'os':types.SimpleNamespace(path=posixpath,environ={},replace=safe_move)}
def safe_git_list(args,**kw):
 if args[0]=='git' and 'ls-files' in args:
  base=physical(args[args.index('-C')+1]);return '\n'.join(str(p.relative_to(base)) for p in base.rglob('*.md'))
 raise ValueError('subprocess blocked')
modules['subprocess']=types.SimpleNamespace(check_output=safe_git_list)
modules['sys']=types.SimpleNamespace(path=[],argv=[])
def imp(name,globals=None,locals=None,fromlist=(),level=0):
 if name not in modules:raise ValueError('import blocked: '+name)
 return modules[name]
B={k:getattr(builtins,k) for k in ['str','int','float','bool','bytes','len','range','enumerate','zip','sorted','list','tuple','set','dict','min','max','sum','any','all','isinstance','repr','abs','round','Exception','ValueError','FileNotFoundError','RuntimeError','AssertionError','KeyError','type','reversed','next','map','filter']}
B.update(__import__=imp,open=lambda path,*a,**k:P(str(path)).open(*a,**k),print=lambda *a,**k:None)
def pyedit(code,meta):
 code=code.replace('values=eval(s[','values=ast.literal_eval(s[')
 try:tree=ast.parse(code)
 except Exception as e:return 'parse: '+str(e)
 env={'__builtins__':B,'__name__':'recovery_record','__file__':'/tmp/recovery_edit.py','ast':ast}
 failures=[]
 for statement in tree.body:
  try:
   for n in ast.walk(statement):
    if isinstance(n,(ast.Import,ast.ImportFrom)):
     names=[x.name for x in n.names] if isinstance(n,ast.Import) else [n.module]
     if any(x not in modules for x in names):raise ValueError('unsupported imports: '+str(names))
    if isinstance(n,ast.Name) and n.id in {'eval','exec','compile','getattr','setattr','globals','locals','input','breakpoint','exit','quit'}:raise ValueError('unsupported dynamic code')
    if isinstance(n,ast.Attribute) and n.attr.startswith('_'):raise ValueError('unsupported private attribute')
   one=ast.Module(body=[statement],type_ignores=[])
   exec(compile(one,'<historical-edit>','exec'),env)
  except Exception as e:
   for n in ast.walk(statement):
    if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store):env.pop(n.id,None)
   failures.append(str(statement.lineno)+': '+type(e).__name__+': '+str(e)[:200])
 return 'ok' if not failures else '; '.join(failures[:4])

def patch(body):
 lines=body.splitlines();i=1;ok=0
 while i<len(lines):
  head=lines[i];i+=1
  if head=='*** End Patch':break
  if not head.startswith(('*** Update File: ','*** Add File: ','*** Delete File: ')):continue
  kind,path=head[4:].split(' File: ',1)
  try:q=physical(path)
  except Exception: 
   while i<len(lines) and not lines[i].startswith('*** '):i+=1
   continue
  chunk=[]
  while i<len(lines) and not lines[i].startswith('*** '):chunk.append(lines[i]);i+=1
  if kind=='Add':
   if q.exists():continue
   data='\n'.join(x[1:] for x in chunk if x.startswith('+'))+'\n';q.parent.mkdir(parents=True,exist_ok=True);q.write_text(data);ok+=1
  elif kind=='Update':
   if not q.is_file():LOG.append(dict(**context,issue='patch target absent',path=path));continue
   s=q.read_text();oldlines=s.splitlines(keepends=True);cursor=0;newlines=list(oldlines);delta=0
   hunks=[];h=[]
   for x in chunk:
    if x.startswith('@@'):
     if h:hunks.append(h);h=[]
    elif x!='*** End of File':h.append(x)
   if h:hunks.append(h)
   failed=False
   for h in hunks:
    before=[x[1:] for x in h if x.startswith((' ','-'))];after=[x[1:] for x in h if x.startswith((' ','+'))]
    lines2=s.splitlines();matches=[j for j in range(cursor,len(lines2)-len(before)+1) if lines2[j:j+len(before)]==before]
    if not matches:
     # tolerate whitespace differences only when the entire context matches uniquely
     matches=[j for j in range(cursor,len(lines2)-len(before)+1) if [t.rstrip() for t in lines2[j:j+len(before)]]==[t.rstrip() for t in before]]
    if not matches:failed=True;break
    j=matches[0];lines2[j:j+len(before)]=after;s='\n'.join(lines2)+'\n';cursor=j+len(after)
   if failed:LOG.append(dict(**context,issue='patch context absent',path=path));continue
   q.write_text(s);ok+=1
  elif kind=='Delete':continue
  PROV[str(q.relative_to(STAGE))]=dict(context)
 return ok

def events():
 result=[];seen=set()
 for p in sorted(RealPath('/home/air/.codex/sessions').rglob('*.jsonl')):
  if '/2026/10/27/' in str(p):continue
  try:
   first=json.loads(p.open().readline());project_session='Minh/Distillation' in first.get('payload',{}).get('cwd','')
  except Exception:project_session=False
  for ln,line in enumerate(p.open(),1):
   if not project_session and 'GRACE' not in line and 'apply_patch' not in line:continue
   try:d=json.loads(line);a=d.get('payload',{})
   except Exception:continue
   if a.get('type') not in ('function_call','custom_tool_call'):continue
   key=a.get('call_id',a.get('id'))
   if key in seen:continue
   seen.add(key)
   raw=a.get('input',a.get('arguments',''))
   if not isinstance(raw,str):continue
   items=[]
   for m in re.finditer(r'"(?:[^"\\]|\\.)*"',raw):
    try:val=json.loads(m[0])
    except Exception:continue
    if val.startswith('*** Begin Patch'):items.append((m.start(),'patch',val))
   for m in re.finditer(r'\b(?:cmd|command)"?\s*:\s*("(?:[^"\\]|\\.)*")',raw):
    try:items.append((m.start(),'command',json.loads(m[1])))
    except Exception:pass
   if raw.startswith('*** Begin Patch'):items.append((0,'patch',raw))
   if not items:
    try:
     obj=json.loads(raw)
     for k in ['cmd','command','patch']:
      if isinstance(obj.get(k),str):items.append((0,'patch' if k=='patch' else 'command',obj[k]))
    except Exception:pass
   for _,kind,s in sorted(items):
    if project_session or 'GRACE' in s or kind=='patch':result.append((d.get('timestamp',''),str(p),ln,kind,s))
 return sorted(result,key=lambda x:(x[0],x[1],x[2]))

def replay():
 shutil.copytree('/tmp/grace-recovery-upstream-20261027',ROOT,dirs_exist_ok=True,ignore=shutil.ignore_patterns('.git'))
 for tempbase in [OUT/'temporary_artifacts',OUT/'additional_temporary_artifacts']:
  for f in tempbase.iterdir():
   if f.is_file():shutil.copy2(f,TMP/f.name)
 ev=events();(OUT/'all_recovery_events.jsonl').write_text(''.join(json.dumps(dict(timestamp=t,session=p,line=l,kind=k,text=s))+'\n' for t,p,l,k,s in ev))
 stats=collections.Counter()
 copied_current=False
 layout_done=False
 for t,p,l,kind,s in ev:
  context.clear();context.update(timestamp=t,session=p,line=l)
  if not copied_current and t>='2026-10-12':
   old=ROOT/'experiments/released_bf16_init_32k_b128_p589824_t2048_seed42';new=ROOT/'experiments/released_bf16_init_32k_b128_p589824_t4096_seed42'
   if old.exists():shutil.copytree(old,new,dirs_exist_ok=True)
   copied_current=True
  if kind=='patch':stats['patch_files']+=patch(s);continue
  # Parse heredocs without executing shell, downloads, workloads, or arbitrary scripts.
  pattern=r"(?m)^\s*(?:(cat)\s+(>>?)\s*([^\s]+)\s*|((?:[^\s]*python[^\s]*|python3?)\s+-\s*))<<\s*(['\"]?)([A-Za-z_][A-Za-z_0-9]*)\5[^\n]*\n"
  spans=[]
  for m in re.finditer(pattern,s):
   end=re.search(r'(?m)^'+re.escape(m[6])+r'\s*$',s[m.end():])
   if not end:continue
   body=s[m.end():m.end()+end.start()];spans.append((m.start(),m.end()+end.end()))
   if m[1]:
    try:
     q=physical(m[3].strip('\"\''));q.parent.mkdir(parents=True,exist_ok=True)
     if m[2]=='>>' and not q.exists():LOG.append(dict(**context,issue='append without base',path=m[3]));continue
     q.write_text((q.read_text() if m[2]=='>>' else '')+body);PROV[str(q.relative_to(STAGE))]=dict(context);stats['heredocs']+=1
    except Exception:pass
   elif any(x in body for x in ['write_text','write_bytes','.rename(', 'shutil.copy','shutil.move','.write(']):
    status=pyedit(body,context);stats['python_'+('ok' if status=='ok' else 'incomplete')]+=1
    if status!='ok':LOG.append(dict(**context,issue='python edit not fully replayed',detail=status,code=body))
  # Retain direct copy/rename operations, using paths only within the isolated reconstruction.
  cleaned=s
  for a,b in reversed(spans):cleaned=cleaned[:a]+'\n'+cleaned[b:]
  shellvars={}
  for line in cleaned.splitlines():
   try:args=shlex.split(line)
   except Exception:continue
   if not args:continue
   if len(args)==1 and '=' in args[0]:
    k,v=args[0].split('=',1)
    if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$',k):shellvars[k]=v
   args=[re.sub(r'\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?',lambda m:shellvars.get(m[1],m[0]),a) for a in args]
   if args[0] in ('cp','mv'):
    args=[x for x in args if not x.startswith('-')]
    if len(args)!=3:continue
    try:
     src,dst=physical(args[1]),physical(args[2])
     if not src.exists():continue
     if dst.is_dir():dst=dst/src.name
     dst.parent.mkdir(parents=True,exist_ok=True)
     if src.is_dir():shutil.copytree(src,dst,dirs_exist_ok=True)
     else:shutil.copy2(src,dst)
     # Preserve old files as recovery versions even for historical moves.
     stats['copies']+=1
    except Exception:pass
  for tm in re.finditer(r'(?m)^python3? (/tmp/[A-Za-z_0-9-]+\.py)(?:\s|$)',cleaned):
   q=physical(tm[1])
   if not q.exists():continue
   code=q.read_text()
   if q.name=='reorganize_grace_evaluation.py':
    if layout_done:continue
    for backupdir in (OUT/'temporary_backups').glob('grace_layout_intermediate_*'):
     for source in backupdir.glob('*'):
      if source.is_file():shutil.copy2(source,ROOT/'evaluation'/source.name)
    layout_done=True
    code=RealPath('/tmp/reorganize_grace_evaluation.py').read_text()
    code=code.replace('for old in mapping: capture(root/old)',"mapping={k:v for k,v in mapping.items() if (root/k).is_file()}\nfor old in mapping: capture(root/old)")
    code=code.replace('def capture(path):\n', 'def capture(path):\n if not path.is_file():return\n')
   if any(x in code for x in ['write_text','write_bytes','.rename(', 'shutil.copy','shutil.move']):
    status=pyedit(code,context)
    if status!='ok':LOG.append(dict(**context,issue='temporary editing script incomplete',script=tm[1],detail=status))
 (OUT/'reconstruction_issues.json').write_text(json.dumps(LOG,indent=2)+'\n')
 (OUT/'reconstruction_provenance.json').write_text(json.dumps(PROV,indent=2)+'\n')
 print(dict(stats));print('issues',len(LOG),'files',sum(1 for x in ROOT.rglob('*') if x.is_file()))
if __name__=='__main__':replay()
