from pathlib import Path
import ast,hashlib,json,os,re,shutil,subprocess

workspace=Path('/raid/lvchengtao/Minh/Distillation')
root=workspace/'reproduction/GRACE'
evalroot=root/'evaluation'
archive=evalroot/'provenance/layout_20261017'
resuming=archive.exists()
groups={
 'runners': ['run_constrained_matrix_evals.py','run_lmms_matrix.py','run_local_development.py','run_upstream_matrix.py','run_video_matrix.py','run_all_constrained_evals.sh'],
 'workers': ['lmms_worker.py','video_worker.py'],
 'protocols': ['local_scoring.py','matched_protocol.py','paper_acceptance.py','video_protocol.py'],
 'audits': ['audit_image_visual_tokens.py','audit_local_scoring.py','check_re_evaluation_gate.py','rescore_saved_protocols.py','summarize_constrained_evals.py'],
 'data_tools': ['download_video_benchmarks.py','download_video_benchmarks.sh','prepare_lmms_eval.sh']}
mapping={f'evaluation/{name}':f'evaluation/{group}/{name}' for group,names in groups.items() for name in names}
for path in sorted((archive/'original/evaluation' if resuming else evalroot).glob('*.md')):
 mapping['evaluation/'+path.name]='evaluation/docs/'+path.name
for dirname in ['VideoMME','RealWorldQA','mmmu','ODinW-13','MathVision']:
 mapping[f'evaluation/{dirname}/README.md']=f'evaluation/docs/benchmarks/{dirname}.md'
if resuming:
 for source in sorted((archive/'original').rglob('*')):
  if source.is_file():
   rel=str(source.relative_to(archive/'original'))
   target=root/mapping.get(rel,rel)
   target.parent.mkdir(parents=True,exist_ok=True)
   shutil.copy2(source,target)
files={}
def capture(path):
 rel=str(path.relative_to(root))
 if rel not in files:
  dest=archive/'original'/rel
  data=(dest if dest.exists() else path).read_bytes()
  dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists(): shutil.copy2(path,dest)
  files[rel]=dict(sha256=hashlib.sha256(data).hexdigest(),size=len(data),current_path=mapping.get(rel,rel))
for old in mapping: capture(root/old)
bench=[]
for folder in evalroot.iterdir():
 if folder.is_dir() and folder.name in ('AI2D','HallusionBench','MMBench','MMStar','MathVision','MathVista','OCRBench','ODinW-13','RealWorldQA','SEEDBench','ScienceQA','VideoMME','mmmu'):
  bench += [p for p in folder.iterdir() if p.is_file() and p.suffix in ('.py','.sh','.slurm')]
for path in bench+list((evalroot/'tests').glob('*.py')): capture(path)
launchers=[p for p in (root/'experiments').glob('*/*.sh') if 'evaluation/' in p.read_text()]
for path in launchers: capture(path)
for old,new in mapping.items():
 dest=root/new
 dest.parent.mkdir(parents=True,exist_ok=True)
 if (root/old).exists(): (root/old).rename(dest)
for group in [*groups,'tests']:
 (evalroot/group/'__init__.py').write_text('"""GRACE evaluation '+group.replace('_',' ')+' package."""\n')
(evalroot/'__init__.py').write_text('"""Offline evaluation tools. See evaluation/docs/README.md for the entry points."""\n')
lookup={Path(old).name:str(Path(new).relative_to('evaluation')) for old,new in mapping.items() if old.endswith(('.py','.sh'))}
lookup['paths.py']='paths.py'
(evalroot/'paths.py').write_text('''"""Stable asset roots and source lookup for the organized evaluation package."""
from pathlib import Path

EVALUATION_ROOT = Path(__file__).resolve().parent
GRACE_ROOT = EVALUATION_ROOT.parent
SOURCE_PATHS = '''+repr(lookup)+'''


def source_file(name):
    """Resolve an owned source filename for subprocess launch and provenance."""
    return EVALUATION_ROOT / SOURCE_PATHS[name]
''')
modules={Path(name).stem:'evaluation.'+str(Path(new).relative_to('evaluation').with_suffix('')).replace('/','.') for name,new in mapping.items() if name.endswith('.py')}
bootstrap='''# Support both direct script execution and `python -m evaluation.<group>.<module>`.
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.paths import EVALUATION_ROOT, source_file

'''
py_paths=[root/new for old,new in mapping.items() if old.endswith('.py')]+bench+list((evalroot/'tests').glob('*.py'))
for path in py_paths:
 if path.suffix!='.py':continue
 if path in bench and not any(x in path.read_text() for x in ('from local_scoring import','from matched_protocol import')):continue
 text=path.read_text()
 for old,new in modules.items():
  text=re.sub(r'\bfrom '+re.escape(old)+r' import\b','from '+new+' import',text)
  text=re.sub(r'(patch\([\'\"])'+re.escape(old)+r'\.',r'\1'+new+'.',text)
 text=text.replace('import download_video_benchmarks as downloader','from evaluation.data_tools import download_video_benchmarks as downloader')
 text=text.replace('ROOT = Path(__file__).resolve().parents[1]','from evaluation.paths import GRACE_ROOT as ROOT')
 text=text.replace('Path(__file__).resolve().parents[1] / "experiments/constrained_32k_seed42"','EVALUATION_ROOT.parent / "experiments/constrained_32k_seed42"')
 text=re.sub(r'Path\(__file__\)(?:\.resolve\(\))?\.with_name\(([\'\"])([^\'\"]+)\1\)',lambda m:'source_file('+repr(m[2])+')' if m[2] in lookup else m[0],text)
 text=text.replace('sha(Path(__file__).with_name(name))','sha(source_file(name))')
 for old,new in mapping.items():
  text=text.replace(old,new)
 text=text.replace("Path(__file__).parent / 'provenance/upstream_20260925'","EVALUATION_ROOT / 'provenance/upstream_20260925'")
 if path.name=='download_video_benchmarks.py':
  text=text.replace('Path(__file__).resolve().parent / "video_benchmarks"','EVALUATION_ROOT / "video_benchmarks"')
 if path.name=='test_video_downloads.py':
  text=text.replace('Path(__file__).resolve().parents[1] / "download_video_benchmarks.sh"','source_file("download_video_benchmarks.sh")')
 if path.name=='run_video_matrix.py':
  text=text.replace("version='video_unsaturated_v1'","version='video_unsaturated_v2'")
  text=text.replace('sha(Path(__file__).parent / name)','sha(source_file(name))')
  text=text.replace("'run_video_matrix.py', 'video_worker.py', 'video_protocol.py', 'local_scoring.py')","'run_video_matrix.py', 'video_worker.py', 'video_protocol.py', 'local_scoring.py', 'paths.py')")
 if path.name=='run_local_development.py':
  text=text.replace("paths = [Path(__file__),","paths = [source_file('paths.py'), Path(__file__),")
 if path.name=='run_constrained_matrix_evals.py':
  text=text.replace("scripts += [ROOT /", "scripts += [source_file('paths.py'), ROOT /",1)
 if path.name=='run_lmms_matrix.py':
  text=text.replace("'runner_sha256': sha(__file__),","'runner_sha256': sha(__file__), 'layout_helper_sha256': sha(source_file('paths.py')),")
 if path.name=='run_upstream_matrix.py':
  text=text.replace("'runner_sha256': sha(__file__),","'runner_sha256': sha(__file__), 'layout_helper_sha256': sha(source_file('paths.py')),")
 # All owned executable modules are exactly one directory below evaluation/.
 tree=ast.parse(text)
 nodes=[n for n in tree.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)) and not (isinstance(n,ast.ImportFrom) and n.module=='__future__')]
 line=nodes[0].lineno-1 if nodes else len(text.splitlines())
 lines=text.splitlines(keepends=True)
 lines.insert(line,bootstrap)
 text=''.join(lines)
 # Remove the previous flat-directory sys.path injection (not the package bootstrap).
 text=text.replace('sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n','')
 text=text.replace('sys.path.insert(0, str(SCRIPT.parent))\n','')
 path.write_text(text)
# Shell tools keep asset roots at evaluation/, independently of launcher location.
for path in [root/new for old,new in mapping.items() if old.endswith('.sh')]+launchers:
 text=path.read_text()
 for old,new in mapping.items():text=text.replace(old,new)
 if path.name=='run_all_constrained_evals.sh':
  text=text.replace('"$(dirname -- "${BASH_SOURCE[0]}")" && pwd)','"$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)',1)
 if path.name=='prepare_lmms_eval.sh':
  text=text.replace('"${BASH_SOURCE[0]}")/.."','"${BASH_SOURCE[0]}")/../.."',1)
 path.write_text(text)
# Rewrite live documentation links relative to each document's new directory.
# Preserve upstream/dataset documents and historical result/provenance artifacts.
old_origins={root/new:root/old for old,new in mapping.items() if old.endswith('.md')}
md_paths=set(old_origins)
for repo in [workspace,root]:
 names=subprocess.check_output(['git','-C',str(repo),'ls-files','--','*.md'],text=True).splitlines()
 for name in names:
  path=repo/name
  if not path.is_file():continue
  if any(part in {'provenance','reports','local_development','re_evaluation','video_evaluation','third_party','models','data'} for part in path.relative_to(repo).parts):continue
  md_paths.add(path)
# Current untracked video guide has already been included above.
for path in sorted(md_paths):
 if root in path.parents and 'video_benchmarks' in path.relative_to(root).parts:continue
 old_origin=old_origins.get(path,path)
 text=path.read_text()
 def link(match):
  target=match[1]
  if '://' in target or target.startswith(('#','/','mailto:')):return match[0]
  plain,sep,anchor=target.partition('#')
  if not plain:return match[0]
  absolute=Path(os.path.normpath(str(old_origin.parent/plain)))
  try:rel=str(absolute.relative_to(root))
  except ValueError:rel=None
  if rel in mapping:absolute=root/mapping[rel]
  if path!=old_origin or rel in mapping:
   return ']('+os.path.relpath(absolute,path.parent)+(sep+anchor if sep else '')+')'
  return match[0]
 text=re.sub(r'\]\(([^)\s]+)\)',link,text)
 for old,new in mapping.items():text=text.replace(old,new)
 # Explicit commands in the main guides often use only a filename.
 if path in old_origins and path.parent==evalroot/'docs':
  for old,new in mapping.items():
   if old.endswith(('.py','.sh')):
    name=Path(old).name
    # Keep dataset-working-directory examples runnable from evaluation/.
    text=re.sub(r'(?m)^(\s*(?:bash|python3?|\.venv-lmms-eval/bin/python)\s+)'+re.escape(name)+r'(?=\s|$)',lambda m:m[1]+str(Path(new).relative_to('evaluation')),text)
 if text!=path.read_text():
  if root in path.parents and path not in old_origins:capture(path)
  path.write_text(text)
manifest=dict(schema=1,date='2026-10-17',purpose='Pre-reorganization source snapshot; historical experiment manifests/results are not rewritten.',
 mapping=mapping,files=files,root='GRACE',snapshot='original/',
 note='Verify old source bytes using this manifest. New code uses package imports and new source hashes; reruns need fresh outputs, not edits to old hash manifests.')
(archive/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print('Moved',len(mapping),'files; snapshotted',len(files),'original files.')
