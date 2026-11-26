from pathlib import Path
import json,re,hashlib,shutil
from urllib.parse import unquote
out=Path('/tmp/grace-recovery-20261027');out.mkdir(exist_ok=True)
records=[]; versions=[]; count=0
prefix='/raid/lvchengtao/Minh/Distillation/'
def relative(path):
 path=path.strip().strip('\"\'')
 if path.startswith(prefix):path=path[len(prefix):]
 if not path.startswith('reproduction/GRACE/'): return None
 p=Path(path)
 if '..' in p.parts:return None
 return str(p.relative_to('reproduction/GRACE'))
def save(path,body,meta):
 global count
 rel=relative(path)
 if rel is None:return
 count+=1
 target=out/'literal_snapshots'/rel;target.parent.mkdir(parents=True,exist_ok=True)
 target.write_text(body)
 archive=out/'versions'/f'{count:05d}'/rel;archive.parent.mkdir(parents=True,exist_ok=True);archive.write_text(body)
 versions.append(dict(path=rel,sha256=hashlib.sha256(body.encode()).hexdigest(),**meta))
for p in sorted(Path('/home/air/.codex/sessions').rglob('*.jsonl')):
 if '/2026/10/27/' in str(p):continue
 for ln,line in enumerate(p.open(),1):
  try:d=json.loads(line);a=d.get('payload',{})
  except Exception:continue
  if a.get('type') not in ('function_call','custom_tool_call'):continue
  raw=a.get('input',a.get('arguments',''))
  if not isinstance(raw,str):continue
  if 'GRACE' not in raw:continue
  meta=dict(timestamp=d.get('timestamp'),session=str(p),line=ln,tool=a.get('name'))
  cmds=[]
  try:
   obj=json.loads(raw)
   for key in ['cmd','command','patch']: 
    if isinstance(obj.get(key),str):cmds.append(obj[key])
  except Exception:pass
  for m in re.finditer(r'\b(?:cmd|command)\s*:\s*("(?:[^"\\]|\\.)*")',raw):
   try:cmds.append(json.loads(m.group(1)))
   except Exception:pass
  if raw.startswith('*** Begin Patch'):cmds.append(raw)
  if not cmds:cmds=[raw]
  for cmd in cmds:
   records.append(dict(**meta,text=cmd))
   # Extract quoted heredoc bodies as data; never run historical commands.
   pattern=r"(?m)^\s*cat\s+(>>?)\s*([^\s]+)\s*<<\s*(['\"])([A-Za-z_][A-Za-z_0-9]*)\3[^\n]*\n"
   for m in re.finditer(pattern,cmd):
    end=re.search(r'(?m)^'+re.escape(m[4])+r'\s*$',cmd[m.end():])
    if not end:continue
    body=cmd[m.end():m.end()+end.start()]
    rel=relative(m[2])
    if not rel:continue
    if m[1]=='>>':
     prev=out/'literal_snapshots'/rel
     if prev.exists():body=prev.read_text()+body
     else:continue
    save(m[2],body,dict(**meta,source='quoted heredoc'))
   for m in re.finditer(r'(?m)^\*\*\* Add File: ([^\n]+)\n',cmd):
    end=re.search(r'(?m)^\*\*\* ',cmd[m.end():]);body=cmd[m.end():m.end()+end.start()] if end else cmd[m.end():]
    lines=body.splitlines()
    if all(x.startswith('+') for x in lines):save(m[1],'\n'.join(x[1:] for x in lines)+'\n',dict(**meta,source='add-file patch'))
(out/'historical_commands.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
(out/'snapshot_provenance.json').write_text(json.dumps(versions,indent=2)+'\n')
editor=[]
for p in Path('/home/air/.vscode-server/data/User/History').rglob('entries.json'):
 try:d=json.loads(p.read_text())
 except Exception:continue
 resource=unquote(d.get('resource',''));idx=resource.find(prefix)
 if idx<0:continue
 rel=relative(resource[idx:])
 if not rel:continue
 for e in d.get('entries',[]):
  source=p.parent/e['id']
  if not source.is_file():continue
  target=out/'editor_history'/str(e['timestamp'])/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
  editor.append(dict(path=rel,source=str(source),timestamp=e['timestamp']))
(out/'editor_provenance.json').write_text(json.dumps(editor,indent=2)+'\n')
for p in Path('/tmp').iterdir():
 if (p.name.startswith(('grace','token_','retention_','int4_','create_token','write_grace','update_grace','reorganize_grace','video_token','question_tokens','master_suite_tokens'))) and p.is_file():
  target=out/'temporary_artifacts'/p.name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
for p in [Path('/home/air/.vscode-server/data/logs/20261027T115405/exthost1/vscode.git/Git.log'),Path('/home/air/.vscode-server/data/logs/20261027T115405/exthost1/openai.chatgpt/Codex.log')]:
 target=out/'evidence'/p.name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
print(json.dumps(dict(recovery_directory=str(out),historical_commands=len(records),literal_snapshots=count,unique_snapshot_paths=len(set(x['path'] for x in versions)),editor_snapshots=len(editor)),indent=2))
