from pathlib import Path
import json, shutil
exp=Path('/raid/lvchengtao/Minh/Distillation/reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t2048_seed42')
# Preserve the previous diagnostic implementation alongside its unchanged artifacts.
archive=exp/'memory_checks/legacy_2048_reports'
archive.mkdir(exist_ok=True)
for name in ['memory_results.json','memory_report.md','profile_memory.py','training_dry_run.txt']:
 if (exp/name).exists() and not (archive/name).exists():shutil.copy2(exp/name,archive/name)
p=exp/'profile_memory.py';s=p.read_text()
s=s.replace('Three optimizer updates with accumulation=1', 'Three optimizer updates with accumulation=2 by default')
s=s.replace("    args = p.parse_args()", "    p.add_argument('--output', type=Path, help='New diagnostic directory selected by the matrix runner')\n    p.add_argument('--accumulation', type=int, default=2)\n    args = p.parse_args()\n    if args.accumulation < 1:\n        p.error('--accumulation must be positive')")
s=s.replace("qat_bits=args.bits, subset_size=8, global_batch=2, grad_accum=1,", "qat_bits=args.bits, subset_size=8 * args.accumulation, global_batch=2 * args.accumulation, grad_accum=args.accumulation,")
s=s.replace("    work = HERE / 'memory_checks' / f'{args.method}_w{args.bits}_{stamp}'", "    work = args.output.resolve() if args.output else HERE / 'memory_checks' / f'{args.method}_w{args.bits}_{stamp}'")
s=s.replace("for i in range(8)]", "for i in range(c['subset_size'])]")
s=s.replace("accumulation=1, updates=3,", "accumulation=args.accumulation, updates=3,")
p.write_text(s)
p=exp/'experiment.json';c=json.loads(p.read_text());c['visual_budget']['model_max_length']=4096;c['report_note']=c['report_note'].replace('limit 2048','limit 4096');p.write_text(json.dumps(c,indent=2)+'\n')
(exp/'run_memory_checks.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/recipe.env"
unset EXPERIMENT_DIR BASE_MODEL_DIR TEACHER_MODEL_PATH SFT_MODEL_DIR
exec "${PYTHON:-/home/air/miniconda3/envs/grace/bin/python}" "$HERE/run_memory_checks.py" "$@"
''')
(exp/'run_memory_checks.sh').chmod(0o755)
