import csv, json, shutil
from pathlib import Path
root=Path('/raid/lvchengtao/Minh/Distillation')
exp=root/'reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t2048_seed42'
rows=[]
for work in sorted((exp/'memory_checks').iterdir()):
    for f in work.glob('profiles/*/run_summary.json'):
        s=json.loads(f.read_text())
        if not s['complete'] or not (f.parent/'EXPORT_COMPLETE').exists():
            raise RuntimeError(f'Incomplete check: {f}')
        c=json.loads((work/'comparison_recipe.json').read_text())['config']
        steps=[json.loads(line) for line in (f.parent/'step_metrics.jsonl').read_text().splitlines()]
        peaks={}
        with (work/'device_memory.csv').open() as source:
            for row in csv.DictReader(source,skipinitialspace=True):
                gpu=row['index'].strip()
                used=float(row['memory.used [MiB]'].split()[0])/1024
                peaks[gpu]=max(peaks.get(gpu,0),used)
        invocations=[json.loads(x) for x in (f.parent/'invocations.jsonl').read_text().splitlines()]
        assert len(invocations)==1 and invocations[0]['exit_code']==0
        envelope=json.loads((work/'input_envelope.json').read_text())
        assert envelope['visual_tokens']==576 and envelope['sequence_tokens']==2048
        if s['method']=='grace':
            assert steps[-1]['rcka_scale']==1.0 and steps[-1]['dkd_scale']==1.0
        row=dict(method=s['method'],bits=c['qat_bits'],complete=True,updates=s['global_step'],
                 peak_allocated_gib=max(x['peak_allocated_gib'] for x in steps),
                 peak_reserved_gib=s['peak_reserved_gib'],device_peak_used_gib=peaks,
                 training_seconds=s['metrics']['train_runtime'],
                 launch_wall_seconds=invocations[0]['wall_seconds'],
                 artifact=str(work.relative_to(exp)),envelope=envelope)
        rows.append(row)
        logfile=Path(f"/tmp/{s['method']}_576_profile_w{c['qat_bits']}.log")
        if logfile.exists(): shutil.copy2(logfile,work/'run.log')
assert {(x['method'],x['bits']) for x in rows}=={('qat',4),('qat',8),('grace',4),('grace',8)},rows
rows.sort(key=lambda x:(x['bits'],x['method']))
gpu_hours=sum(x['launch_wall_seconds'] for x in rows)*2/3600
result=dict(date='2026-10-05',status='all_four_synthetic_memory_checks_passed',
            physical_gpu_ids=[0,4],gpu_type='NVIDIA A100-SXM4-40GB',
            production_training_started=False,production_evaluation_started=False,
            profile_launch_gpu_hours=gpu_hours,rows=rows)
(exp/'memory_results.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# 576-token memory feasibility — 2026-10-05','',
'**All four bounded checks passed, including native checkpoint saving and merged/baked export. Keep the requested 589,824-pixel / 576-token ceiling; no lower-cap fallback was needed. Full training and evaluation have not started.**','',
'Two free NVIDIA A100-SXM4-40GB GPUs (PCI indices 0 and 4) were surveyed before each run. No other workloads were stopped. Source weights, processor compatibility, implementation hashes, launch recipes and GPU surveys are recorded per check. Machine-readable results: [memory_results.json](memory_results.json).','',
'| Arm | Peak allocated GiB/GPU | Peak reserved GiB/GPU | Highest sampled device use GiB | Updates / export | Evidence |',
'| --- | ---: | ---: | ---: | --- | --- |']
for r in rows:
 lines.append(f"| {r['method'].upper()} W{r['bits']} | {r['peak_allocated_gib']:.2f} | {r['peak_reserved_gib']:.2f} | {max(r['device_peak_used_gib'].values()):.2f} | 3 / complete | [run]({r['artifact']}/profiles/{r['method']}/run_summary.json) |")
lines += ['',
'Training allocated/reserved values are the maximum across ranks at optimizer-step boundaries (using PyTorch peak counters). Device use is sampled every 0.5 seconds across loading, training, checkpointing and export and includes runtime overhead; a sampled trace can miss brief spikes. The production reserved-memory guard remains **34 GiB**. It checks at optimizer-update boundaries and cannot prevent every transient OOM.','',
'## What was exercised','',
'- A real square image forced to a 768×768 processed grid: `[1,48,48]` patches, exactly **576 merged visual tokens**.',
'- Exactly **2,048 total tokens**, including **1,458 supervised answer tokens** from a synthetic repeated-word answer. This stresses the long-answer distillation path; losses and accuracy on this artificial example have no research interpretation.',
'- The actual trainer, released SFT initialization, QAT modules, LoRA rank/targets, merger, optimizer, ZeRO-2, gradient checkpointing, BF16 storage and W4/W8 settings.',
'- For GRACE: the actual 8B teacher, GDKD and **full-weight RCKA at updates 2–3**.',
'- Three optimizer updates, native checkpoints, LoRA merge, quantization bake, model/processor save and quantized research export.','',
'## Limits and production handoff','',
'To bound diagnostic cost, this screen used accumulation **1** (global batch 2), RCKA warmup **1** and a forced minimum image area equal to the ceiling. Production retains accumulation **64** (global batch 128), warmup **25**, minimum **4,096**, and **256** updates. The eight repeated synthetic records are isolated under `memory_checks/`; they are never the production subset.','',
'A fixed-envelope screen is evidence of memory feasibility, not an exhaustive test of every aspect ratio, retained conversation, accumulation cycle or later optimization state. For additional assurance, the documented `run_training.sh all --profile` uses the actual freshly prepared data and production accumulation. Training is protected by the existing pause/resume guard; do not change image budgets midway through a pinned experiment.','',
'No production throughput or full-run duration is inferred from these short, checkpoint-heavy runs. The recorded two-GPU launch wall time across all four checks is ' + f'**{gpu_hours:.3f} GPU-hours**' + ' (including model loading/checkpoint/export within each launcher invocation; excluding CPU provenance/input preparation and idle time between invocations). This is diagnostic cost, not the cost of training the four production models.','',
'## Input distribution and non-GPU validation','',
'The [256-image CPU preview](token_budget_preview.json) increased mean/median visual counts from **89.76/88** to **250.84/260**, with maximum **576**. Small images remain below the ceiling. Measure the new full subset separately once prepared.','',
'All four production dry-run commands retain batch 128, 256 updates, 2,048 total tokens and 25-step RCKA warmup, and point to fresh production outputs. The latest evaluation plan covers four models × seven suites. Shell/Python syntax checks passed. The 11-test local scoring/worker suite passed; after strengthening histogram-tampering coverage, all 10 scoring tests passed again. The MMBench fix changes JSON-key comparison only, not scoring semantics or old completion artifacts.','',
'See [README.md](README.md) for preparation, training, resume and evaluation commands.']
(exp/'memory_report.md').write_text('\n'.join(lines)+'\n')
# Mark only finished setup and feasibility work, not full production jobs.
p=root/'plans/todo_plan.md';s=p.read_text().replace('- [~] Validate the maximum-token training/export memory path on two free A100s;', '- [x] Validate the maximum-token training/export memory path on two free A100s;')
s=s.replace('Bounded diagnostic runs are separate from production training.', 'All four three-update checks passed at 576 visual / 2,048 total tokens; see the memory report for measured peaks and diagnostic-versus-production settings. Bounded diagnostic runs are separate from production training.')
p.write_text(s)
p=root/'plans/have_done.md';s=p.read_text();s+='\n2026-10-05: Prepared isolated released-SFT QAT/GRACE W4/W8 controls with a 589,824-pixel (up to 576 merged visual tokens) ceiling and the latest seven-suite evaluation wrapper. All four bounded maximum-token memory/export checks passed on two A100 40GB GPUs. CPU token-count preview confirms increased headroom; full preparation/training/evaluation remain for the user to launch. Fixed MMBench JSON histogram-key validation with regression coverage. [Run guide](../reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t2048_seed42/README.md) and [memory evidence](../reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t2048_seed42/memory_report.md).\n';p.write_text(s)
p=root/'experiments/log.md';s=p.read_text();s=s.replace('(No experiments yet — first entry will be the GRACE smoke test in Week 2-3.)', 'Historical full-run records are under `reproduction/GRACE/experiments/`; the entries below record subsequent work.')
s+='\n## EXP-001: 576-token QAT/GRACE memory feasibility\n- **Date:** 2026-10-05\n- **Hypothesis:** The unchanged two-A100 adaptation recipe can accommodate a higher visual-token ceiling, supplying headroom for later token reduction.\n- **Config:** [Experiment recipe](../reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t2048_seed42/recipe.env). Production targets 589,824 pixels, 2,048 total tokens, batch 128 and 256 updates; not launched.\n- **Setup/run:** Four separate three-update diagnostic runs (QAT/GRACE × W4/W8), two A100 40GB GPUs, BF16, ZeRO-2, microbatch 1, checkpointing. Diagnostic accumulation=1 and RCKA warmup=1, with a synthetic long-answer maximum-grid input; not training-quality evidence.\n- **Result:** All four training/checkpoint/export checks passed. [Measured memory, costs, source artifacts and limitations](../reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t2048_seed42/memory_report.md).\n- **Next:** User prepares the full shared subset, trains/evaluates the new controls and reports to the supervisor before LLMC+ integration.\n';p.write_text(s)
print(json.dumps(result,indent=2))
