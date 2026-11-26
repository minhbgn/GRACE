#!/usr/bin/env python3
"""Compile the CPU-only QAT W4 report from the runner's verified summary.

First run `run_experiment.sh summarize` with the original evaluation Python.
This compiler independently verifies artifact hashes and raw QAT telemetry.
The interpretation is specific to the November 5 snapshot; after new jobs
complete, update its narrative before rebuilding. It never allocates a GPU.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
IMAGES = ['hallusionbench', 'mmbench', 'scienceqa', 'ai2d', 'mmmu', 'seedbench_img', 'mmstar']
VIDEOS = ['tempcompass', 'videomme', 'videommmu_perception', 'videommmu_comprehension', 'videommmu_adaptation']
CONFIGS = ['baseline_r100'] + [f'{m}_r{r:03}' for m in ('resolution', 'uniform', 'merge') for r in (50, 25, 10)]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', default='2026-11-05', help='Report snapshot date (Singapore)')
    args = parser.parse_args()
    output = EXP / 'results/full'
    protocol = read(output / 'protocol.json')
    changed = [name for name, value in protocol['code'].items() if sha(ROOT / name) != value]
    require(not changed, f'Pinned code changed: {changed}')
    require(read(EXP / 'experiment.json') == protocol['experiment'], 'Experiment differs from protocol')
    rows = read(output / 'comparison.json')['rows']
    expected = {(m, s, c) for m in protocol['experiment']['models'] for s in IMAGES + VIDEOS for c in CONFIGS}
    actual = {(r['model'], r['suite'], f"{r['method']}_r{round(100*r['retention']):03}") for r in rows}
    require(actual == expected and len(rows) == len(expected), 'Summary does not cover the full matrix')
    snapshot = collections.Counter((r['model'], r['status']) for r in rows)
    require(snapshot == collections.Counter({('qat_w4', 'complete'): 75, ('qat_w4', 'failed'): 45,
                ('vanilla_2b', 'complete'): 8, ('vanilla_2b', 'pending'): 112, ('released_bf16', 'pending'): 120}),
            'Results have advanced beyond this snapshot. The runner summarize action remains usable; '
            'update this compiler\'s narrative and snapshot check before regenerating the report.')
    audit = []
    flat = []
    for row in rows:
        config = f"{row['method']}_r{round(100*row['retention']):03}"
        directory = output / row['model'] / row['suite'] / config
        attempts = sorted(directory.glob('attempt_*'))
        marker = directory / 'COMPLETE.json'
        done = read(marker) if marker.exists() else None
        item = dict(model=row['model'], suite=row['suite'], config=config,
                    status='complete' if done else ('incomplete' if attempts else 'pending'),
                    attempts=len(attempts), marker=str(marker.relative_to(EXP)) if done else None)
        if done:
            require(row['result'] == done and row['status'] == 'complete', f'Stale summary: {directory}')
            attempt = directory / done['attempt']
            for name, checksum in done['sha256'].items():
                require(sha(attempt / name) == checksum, f'Artifact changed: {attempt / name}')
            require(read(attempt / 'exit.json')['returncode'] == 0, f'Nonzero exit: {attempt}')
            validated = read(attempt / 'validated.json')
            require(all(done[k] == v for k, v in validated.items()), f'Validation differs: {attempt}')
            job = read(attempt / 'job.json')
            require(job['policy'] == protocol['policy'] and job['limit'] is None, f'Job protocol differs: {attempt}')
            item.update(count=done['count'], gpu=job['gpu'], attempt=done['attempt'],
                        sample_identity=done['sample_identity'], output_cutoffs=done['output_cutoffs'])
            if row['model'] == 'qat_w4':
                sums = collections.defaultdict(float)
                maxima = collections.defaultdict(float)
                profile_means = []
                count = cutoffs = 0
                with (attempt / 'telemetry.jsonl').open() as stream:
                    for line in stream:
                        t = json.loads(line)
                        require(t['index'] == count and t['fused_int4_calls'] > 0, f'Invalid INT4 telemetry: {attempt}')
                        require(t['decoder_layer_visual_tokens'] == [t['input_visual_tokens']] * 28,
                                f'Decoder token counts differ: {attempt}')
                        require(len(t['generated_token_ids']) == t['output_text_tokens'], f'Token count differs: {attempt}')
                        cutoffs += t['generated_token_ids'][-1] != 151645
                        for key in done['telemetry']:
                            if isinstance(done['telemetry'][key], dict) and key != 'fixed_output_profile':
                                sums[key] += t[key]
                                maxima[key] = max(maxima[key], t[key])
                        profile = t['fixed_output_profile_seconds']
                        if profile:
                            require(len(profile) == 3 and t['fixed_output_profile_tokens'] == 32, f'Profile differs: {attempt}')
                            profile_means.append(sum(profile) / len(profile))
                        count += 1
                require(count == done['count'] and cutoffs == done['output_cutoffs'], f'Coverage differs: {attempt}')
                for key in sums:
                    for stat, value in [('total', sums[key]), ('mean', sums[key] / count), ('max', maxima[key])]:
                        require(math.isclose(value, done['telemetry'][key][stat], rel_tol=1e-9, abs_tol=1e-9),
                                f'Telemetry aggregate differs: {attempt}/{key}/{stat}')
                require(len(profile_means) == 3, f'Profile sample coverage differs: {attempt}')
                require(math.isclose(sum(profile_means) / 3, done['telemetry']['fixed_output_profile']['mean_seconds']),
                        f'Profile mean differs: {attempt}')
                runtime = read(attempt / 'runtime.json')
                require(runtime['integer_codes_verified'] and runtime['converted_modules'] == 196,
                        f'Packing evidence missing: {attempt}')
                item['raw_telemetry_verified'] = True
            t = done['telemetry']
            flat.append(dict(model=row['model'], suite=row['suite'], config=config, count=done['count'],
                             score_percent=done['score_percent'], actual_retention=t['actual_retention'],
                             mean_visual_tokens=t['input_visual_tokens']['mean'],
                             mean_text_input_tokens=t['input_text_tokens']['mean'],
                             mean_output_tokens=t['output_text_tokens']['mean'],
                             mean_model_seconds=t['model_seconds']['mean'],
                             fixed_32_token_seconds=t['fixed_output_profile']['mean_seconds'],
                             fixed_output_speedup=row['fixed_output_speedup'],
                             max_peak_allocated_gib=t['peak_gpu_allocated_bytes']['max'] / 2**30,
                             output_cutoffs=done['output_cutoffs'], gpu=job['gpu']))
        elif attempts:
            attempt = attempts[-1]
            log = (attempt / 'run.log').read_text() if (attempt / 'run.log').exists() else ''
            exit_info = read(attempt / 'exit.json') if (attempt / 'exit.json').exists() else None
            reason = ('CUDA OOM' if 'CUDA out of memory' in log else
                      'missing TempCompass YAML' if 'FileNotFoundError' in log and 'multi-choice.yaml' in log else
                      'interrupted/uncommitted' if exit_info is None else 'worker failure')
            item.update(reason=reason, last_attempt=str(attempt.relative_to(EXP)), exit=exit_info,
                        validated_present=(attempt / 'validated.json').exists())
        audit.append(item)

    report_dir = EXP / 'reports'
    report_dir.mkdir(exist_ok=True)
    (report_dir / 'completion_audit.json').write_text(json.dumps(dict(date=args.date,
        protocol_sha256=sha(output / 'protocol.json'), pinned_code_verified=True, jobs=audit), indent=2) + '\n')
    with (report_dir / 'completed_metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    q = [r for r in rows if r['model'] == 'qat_w4']
    complete = [r for r in q if r['result']]
    by = {(r['suite'], f"{r['method']}_r{round(100*r['retention']):03}"): r for r in complete}
    stats = collections.Counter((a['model'], a['status']) for a in audit)
    failures = collections.Counter(a.get('reason') for a in audit if a['model'] == 'qat_w4' and a['status'] != 'complete')
    lines = ['# QAT W4: packed INT4 and visual-token reduction', '',
        f'Prepared {args.date} (Asia/Singapore). Snapshot of `results/full`; all scores below are percentages.', '',
        f'**QAT W4 has {len(complete)}/120 verified completions: 70/70 image jobs and 5/50 video jobs.** '
        'Every configuration was attempted, but an attempted job is not necessarily a successful evaluation. '
        f"The remaining video jobs failed: {failures['CUDA OOM']} CUDA OOMs and {failures['missing TempCompass YAML']} missing-YAML failures. "
        'Vanilla 2B and released BF16 comparisons remain pending; no claim about INT4 versus BF16 accuracy, memory saving or speedup is supported yet.', '',
        '## Completion and evidence', '',
        table(['Model', 'Verified complete / planned', 'Incomplete attempts', 'Unstarted'],
              [[m, f"{stats[m, 'complete']}/120", stats[m, 'incomplete'], stats[m, 'pending']]
               for m in ('qat_w4', 'vanilla_2b', 'released_bf16')]), '',
        'Vanilla 2B already has eight completed HallusionBench configurations. Its `merge_r025` attempt has '
        '`validated.json` but no `exit.json` or `COMPLETE.json`, so it is uncommitted and will restart; `merge_r010` is unstarted. '
        'Those partial baseline results are retained in the machine summary, but are not used for cross-model conclusions here.', '',
        'The compiler independently verified every completed artifact SHA-256, successful exit and saved validation payload. '
        'For all completed QAT jobs it also checked raw telemetry coverage/order, positive fused INT4 calls on every question, '
        'all 28 decoder layers seeing the retained token count, timing/token/memory aggregates, three fixed-output profile samples, '
        '196 packed modules and integer-code verification. All pinned evaluation source hashes match the saved protocol. '
        'This is an artifact/telemetry audit, not an independent rerun of generation or benchmark scoring.', '',
        'Evidence: [full runner summary](results/full/comparison.md), [summary JSON](results/full/comparison.json), '
        '[completion audit](reports/completion_audit.json), [completed metrics CSV](reports/completed_metrics.csv), '
        '[frozen protocol](results/full/protocol.json), and per-job `COMPLETE.json` / attempt artifacts. '
        'Smoke results are excluded.', '',
        '## Checkpoint and evaluation protocol', '',
        '- Checkpoint: `../released_bf16_init_32k_b128_p589824_t4096_seed42/w4g128/runs/qat`; '
        'training completed step 256 (CE loss 0.596977; training time 6,779 s). W4, group size 128, '
        'merged LoRA rank 32/alpha 64 on 112 attention linears; 196 language linears are quantized.',
        '- Runtime: strict packed Triton W4A16, FP16 activation tiles/scales with FP32 accumulation. '
        'Vision, embeddings and head remain floating point. This is weight-only INT4 with floating-point arithmetic inside the kernel. '
        'Both saved GPU numerical gates passed with maximum absolute error 0.0078125.',
        '- QAT sidecar SHA-256: `3383915972b9685a2cc51d900684de09144475f6b13a2c34244ffb8bfc1e0379`. '
        'The provenance file describes the source safetensors as floating point; `runtime.json` and telemetry establish packed execution.',
        '- Maximum image/frame area 589,824 pixels: at most 576 visual tokens per image or video two-frame temporal group, '
        'not per whole video. Video sampling is 2 fps without duration/frame truncation. Smaller images retain their actual baseline count.',
        '- One shared 100% baseline, then resolution/uniform/merge at 50%, 25%, 10%. '
        'Resolution reduces input pixels; uniform/merge first encode the full visual input and then compact tokens before decoder layer 0. '
        'Merge is deterministic contiguous-bin averaging, not a learned token selector.',
        '- FlashAttention 2; seed 42; greedy single-beam decoding; 256-token answer cap; batch size one. '
        'Fixed-output timing uses the first three questions, one warm-up and three timed repeats of exactly 32 output tokens.',
        '- Saved environment: Python 3.11.0, torch 2.5.1+cu121, transformers 5.9.0, Triton 3.1.0, lmms-eval 0.7.1. '
        'Results were dispatched to physical GPUs 0 and 2; failure logs identify 39.39 GiB devices. '
        'Performance across different cards or concurrent workloads is not controlled by this report.', '',
        '## Image accuracy', '',
        'All seven image populations completed in all ten configurations: 29,420 rows per configuration, '
        '294,200 scored question inferences in total. No completed image answer hit the output cap. '
        'HallusionBench includes text-only rows and uses row accuracy; ScienceQA also retains text-only rows. '
        'MMBench uses circular accuracy on its supplied rotations. The other suites use their saved deterministic local primary metric. '
        'No aggregate accuracy across unlike benchmarks is calculated.', '',
        table(['Suite', 'Rows', '100%', 'Res 50%', 'Res 25%', 'Res 10%', 'Uniform 50%', 'Uniform 25%', 'Uniform 10%', 'Merge 50%', 'Merge 25%', 'Merge 10%'],
              [[s, by[s, 'baseline_r100']['result']['count']] +
               [f"{by[s, c]['result']['score_percent']:.2f}" for c in CONFIGS] for s in IMAGES]), '',
        '### Accuracy cost and resource change', '',
        'The ranges below span the seven image suites, without averaging their scores. '
        'Loss is each suite’s 100% score minus its reduced score (percentage points). '
        'Memory saving compares the maximum per-question allocated GPU memory within each suite. '
        'Actual retention is the ratio of summed retained to summed original visual tokens, excluding text.', '']
    resource_rows = []
    for config in CONFIGS[1:]:
        rr = [by[s, config] for s in IMAGES]
        losses = [by[s, 'baseline_r100']['result']['score_percent'] - by[s, config]['result']['score_percent'] for s in IMAGES]
        speeds = [r['fixed_output_speedup'] for r in rr]
        ret = [r['result']['telemetry']['actual_retention'] * 100 for r in rr]
        mem = [100 * (1 - by[s, config]['result']['telemetry']['peak_gpu_allocated_bytes']['max'] /
                       by[s, 'baseline_r100']['result']['telemetry']['peak_gpu_allocated_bytes']['max']) for s in IMAGES]
        resource_rows.append([config, f'{min(losses):.2f}–{max(losses):.2f}', f'{min(ret):.1f}–{max(ret):.1f}%',
                              f'{min(speeds):.3f}–{max(speeds):.3f}×', f'{min(mem):.2f}–{max(mem):.2f}%'])
    lines += [table(['Configuration', 'Accuracy loss pp', 'Actual retained', '32-token speedup', 'Peak allocation saving'], resource_rows), '',
        '**Observed tradeoff:** at 50% retention, merge preserves accuracy best on five of seven image suites '
        '(loss 0.19–1.78 pp), while resolution is best on HallusionBench and MMMU. '
        'At 25% and 10%, resolution has the highest score among the three reduction methods on every image suite. '
        'At 10%, resolution loses 2.66–10.65 pp; merge loses as much as 19.42 pp on MMBench. '
        'These are descriptive outcomes from one checkpoint/seed; small differences are not significance tests.', '',
        'For follow-up evaluation, 50% merge is a useful accuracy-preserving candidate, and 25% resolution is a useful '
        'stronger-reduction candidate. Neither is a universal optimum; select using per-suite accuracy requirements. '
        'Resolution retains about 45–48% for a 50% target and 8–9% for a 10% target because of patch alignment; '
        'method comparisons at the same nominal target therefore do not have exactly equal token budgets.', '',
        '## Timing and GPU memory', '',
        table(['Suite', 'Mean visual tokens at 100%', 'Mean answer model time s', '32-output-token time s', 'Max allocated GiB'],
              [[s, f"{by[s, 'baseline_r100']['result']['telemetry']['input_visual_tokens']['mean']:.1f}",
                f"{by[s, 'baseline_r100']['result']['telemetry']['model_seconds']['mean']:.4f}",
                f"{by[s, 'baseline_r100']['result']['telemetry']['fixed_output_profile']['mean_seconds']:.4f}",
                f"{by[s, 'baseline_r100']['result']['telemetry']['peak_gpu_allocated_bytes']['max']/2**30:.3f}"] for s in IMAGES]), '',
        'The observed image fixed-output gains are modest: about 1.004–1.161× over the same packed QAT model at full input. '
        'These do not establish a speedup over BF16. Removing 90% of visual tokens does not remove 90% of total work: '
        'text and autoregressive decoding remain, and uniform/merge still pay the full vision-encoder cost. '
        'Image peak allocation is about 2.12–2.29 GiB at full input; savings from token reduction are small compared '
        'with the token-count reductions. The report does not infer quantization memory savings without matched BF16 runs.', '',
        'Model timing includes synchronized GPU vision/embedding, reduction, decoder prefill and decode. It excludes '
        'checkpoint load/packing, CPU media decoding/preprocessing, transfer and file I/O. Mean normal-answer time varies '
        'with output length; use fixed-output timing for the within-model comparisons. Only three fixed questions per suite '
        'were profiled, so these estimates are exploratory and have no representative latency distribution or uncertainty estimate. '
        'Peak allocated memory is PyTorch allocation, not total/reserved VRAM or whole-process memory.', '',
        '## Video results and failures', '',
        'Only complete full-population jobs are scored below. Partial predictions from failed jobs are excluded. '
        'All full-token video baselines failed, so neither video accuracy deltas versus full input nor baseline-relative '
        'speedups are available.', '',
        table(['Suite / population', 'Configuration', 'Questions', 'Score %', 'Actual retained', 'Mean model s', 'Max allocated GiB', 'Capped answers'],
              [[r['suite'], f"{r['method']}_r{round(100*r['retention']):03}", r['result']['count'],
                f"{r['result']['score_percent']:.2f}", f"{r['result']['telemetry']['actual_retention']*100:.2f}%",
                f"{r['result']['telemetry']['model_seconds']['mean']:.3f}",
                f"{r['result']['telemetry']['peak_gpu_allocated_bytes']['max']/2**30:.3f}", r['result']['output_cutoffs']]
               for r in complete if r['suite'] in VIDEOS]), '',
        'VideoMME is the short+medium 1,800-question population, excluding long videos; Video-MMMU uses all 300 questions '
        'in each split. Video-MMMU adaptation has 40/300 capped answers (13.3%), comprehension 4/300 and perception 1/300; '
        'these outputs remain in the score and limit interpretation. There is no audio/subtitle input or API judge.', '',
        table(['Suite', 'Complete configurations', 'Failed configurations / reason'],
              [[s, ', '.join(c for c in CONFIGS if (s, c) in by) or 'None',
                f"{10-sum((s,c) in by for c in CONFIGS)} / " + ('missing YAML' if s == 'tempcompass' else 'CUDA OOM')]
               for s in VIDEOS]), '',
        'The TempCompass failure is deterministic: the pinned runner requests '
        '`third_party/lmms-eval-v0.7.1/lmms_eval/tasks/tempcompass/multi-choice.yaml`, while the clean pinned harness '
        'contains `tempcompass_mc.yaml`. These jobs failed before answering any questions. A future repaired evaluation '
        'must also verify the MC task adapter/metric and local deterministic scorer. Editing the runner changes a pinned '
        'source hash, so a repaired protocol needs a separate output base; do not rewrite the current `protocol.json`.', '',
        'OOMs include vision-encoder allocations before token selection. Uniform/merge therefore cannot prevent those '
        'full-input vision peaks. Most lower-resolution video configurations also exceeded the roughly 40 GiB card capacity. '
        'Freeing a busy GPU is necessary for launch but does not by itself resolve these recorded per-process OOMs. '
        'Retry unchanged on a sufficiently larger idle card, or implement a separately validated memory optimization in '
        'a new protocol/output. Do not silently reduce fps, cap duration, drop questions or change the population.', '',
        '## Resume from the last committed job', '',
        'From the workspace root, after the selected GPUs are available and ownership has been checked:', '',
        '```bash', 'export LMMS_PYTHON=/home/air/miniconda3/envs/grace/bin/python',
        'EXP=reproduction/GRACE/experiments/qat_int4_token_reduction_576_seed42', '',
        '# CPU-only plan: continue the two baseline models on all image suites.',
        'bash "$EXP/run_experiment.sh" plan --models vanilla_2b,released_bf16 \\',
        '  --suites hallusionbench,mmbench,scienceqa,ai2d,mmmu,seedbench_img,mmstar \\',
        '  --output "$EXP/results"', '',
        '# Launch only once physical GPUs 0 and 2 are idle; use --gpu 2 if only one is available.',
        'bash "$EXP/run_experiment.sh" run --gpus 0,2 --reserve-gpus \\',
        '  --models vanilla_2b,released_bf16 \\',
        '  --suites hallusionbench,mmbench,scienceqa,ai2d,mmmu,seedbench_img,mmstar \\',
        '  --output "$EXP/results"', '',
        '# Refresh the entire matrix summary without using a GPU.',
        'bash "$EXP/run_experiment.sh" summarize --output "$EXP/results"',
        '```', '',
        'This baseline-image continuation plans 140 jobs, skips eight verified completions and has 132 remaining '
        'at this snapshot. Completed jobs are verified by hashes and skipped. A job without `COMPLETE.json` restarts '
        'in a new attempt from its first question; there is no per-question continuation, even if predictions or '
        '`validated.json` survived. Old attempts remain intact. `--output` takes the parent results directory: '
        'the runner appends `/full`; do not pass `results/full`. Do not use `--smoke` to resume full results.', '',
        'To resume the original entire matrix instead (including all failed QAT video jobs):', '',
        '```bash', 'bash "$EXP/run_experiment.sh" run --gpus 0,2 --reserve-gpus \\',
        '  --models qat_w4,vanilla_2b,released_bf16 --output "$EXP/results"', '```', '',
        'At this snapshot the full continuation has 277 uncommitted jobs (45 QAT, 112 vanilla, 120 released BF16). '
        'It retries failed video jobs; it does not repair them. Selecting only `--models vanilla_2b,released_bf16` '
        'with no suite filter continues both baseline image and video matrices (232 uncommitted jobs). '
        'The controller reserves idle GPUs and runs one independent job per card; multiple GPUs do not pool their VRAM.', '',
        'Keep the original evaluation Python/packages, source files, dataset assets, policy, checkpoints and output base. '
        'Subset selection and available GPU indices may change; source/policy/environment changes cause the existing '
        'protocol pin to reject reuse. Do not remove completion markers, edit pinned hashes, or launch two controllers '
        'against this output. No GPU jobs were launched while preparing this report.', '',
        '## Reproduce this synthesis', '',
        'The report compiler contains the interpretation for this November 5 snapshot and rejects a changed completion '
        'inventory. After more jobs complete, the runner’s `summarize` action remains usable; update the compiler’s '
        'narrative and snapshot check before rebuilding the experiment report.', '',
        '```bash', 'export LMMS_PYTHON=/home/air/miniconda3/envs/grace/bin/python',
        'EXP=reproduction/GRACE/experiments/qat_int4_token_reduction_576_seed42',
        'bash "$EXP/run_experiment.sh" summarize', 'python3 "$EXP/compile_report.py" --date ' + args.date, '```', '']
    (EXP / 'experiment_report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(EXP / 'experiment_report.md'), verified_completed=len(flat),
                          qat_completed=len(complete), qat_questions=sum(r['result']['count'] for r in complete)), indent=2))


if __name__ == '__main__':
    main()
