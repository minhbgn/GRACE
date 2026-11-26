import tempfile,subprocess,os,json,shutil
from pathlib import Path
source=Path('reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/run_video_evaluation_parallel.sh').resolve()
models=['vanilla_2b','vanilla_8b_teacher','sft_2b','released_bf16','released_grace_w4','released_grace_w8','qat_w4','grace_w4','qat_w8','grace_w8']
suites=['tempcompass','videomme','videommmu_perception','videommmu_comprehension','videommmu_adaptation']
complete={('vanilla_2b','tempcompass'),('vanilla_2b','videommmu_perception'),('vanilla_2b','videommmu_comprehension'),('vanilla_8b_teacher','tempcompass'),('sft_2b','tempcompass')}
stub='''#!/usr/bin/env python3
import sys,fcntl,json,time,os
from pathlib import Path
root=Path(__file__).resolve().parent
base=root/'video_evaluation/video_unsaturated_v2_3a74c7231e27/full'
args=sys.argv[1:]
if args[0]=='summarize':
 assert len(list(base.glob('*/*/attempt_cpu_stub')))==45
 print('Combined all attempted pairs');sys.exit(0)
methods=args[args.index('--methods')+1].split(',')
output=Path(args[args.index('--output')+1])
gpu=args[args.index('--gpu')+1]
lock=(output/'full/.matrix.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
with (base/'.matrix.lock').open('r') as shared:
 try:fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise AssertionError('Coordinator did not hold shared lock')
(output.parent/('ready'+gpu)).write_text('ready')
for _ in range(100):
 if len(list(output.parent.glob('ready*')))==2:break
 time.sleep(.05)
else:raise AssertionError('Groups did not run concurrently')
attempted=0
for method in methods:
 for suite in ['tempcompass','videomme','videommmu_perception','videommmu_comprehension','videommmu_adaptation']:
  job=output/'full'/method/suite
  if (job/'COMPLETE.json').exists():continue
  (job/'attempt_cpu_stub').mkdir();attempted+=1
expected=25 if gpu=='2' else 20
assert attempted==expected,(gpu,attempted)
print('GPU',gpu,'attempted',attempted)
sys.exit(8 if os.environ.get('SIMULATE_LOCAL_FAILURE')=='1' and gpu=='4' else 0)
'''
trace='''#!/usr/bin/env bash
while [[ "$1" != bash ]]; do
 if [[ "$1" == -o ]]; then : > "$2"; shift 2; else shift; fi
done
exec "$@"
'''
for failure in [False,True]:
 with tempfile.TemporaryDirectory(prefix='grace-parallel-test-') as t:
  root=Path(t);shutil.copy2(source,root/source.name)
  base=root/'video_evaluation/video_unsaturated_v2_3a74c7231e27/full';base.mkdir(parents=True)
  (base/'protocol.json').write_text('{}');(base/'datasets').mkdir();(base/'.matrix.lock').touch()
  for method in models:
   for suite in suites:
    d=base/method/suite;d.mkdir(parents=True)
    if (method,suite) in complete:(d/'COMPLETE.json').write_text('{"original":true}')
  (root/'stub_runner.py').write_text(stub)
  runner=root/'run_video_evaluation.sh';runner.write_text('#!/usr/bin/env bash\nexec python3 "$(dirname -- "$0")/stub_runner.py" "$@"\n');runner.chmod(0o755)
  binpath=root/'bin';binpath.mkdir();tracer=binpath/'strace';tracer.write_text(trace);tracer.chmod(0o755)
  env=dict(os.environ,PATH=str(binpath)+':'+os.environ['PATH'],SIMULATE_LOCAL_FAILURE='1' if failure else '0')
  result=subprocess.run(['bash',str(root/source.name),'run','2','4'],env=env,text=True,capture_output=True,timeout=15)
  assert result.returncode==(1 if failure else 0),(result.stdout,result.stderr,result.returncode)
  assert len(list(base.glob('*/*/attempt_cpu_stub')))==45
  assert len(list(base.glob('*/*/COMPLETE.json')))==5
  for method,suite in complete:assert (base/method/suite/'COMPLETE.json').read_text()=='{"original":true}'
  print('PASS:', 'one group fails' if failure else 'both groups succeed', '; concurrent groups; 25 + 20 retries; five completions preserved; shared lock held; combined summary runs')
