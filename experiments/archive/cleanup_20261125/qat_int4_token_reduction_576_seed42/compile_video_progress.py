#!/usr/bin/env python3
"""Refresh video progress/analysis from verified completions, without GPU work.

Writes reports/video_progress.{md,json,csv} and updates the main experiment report.
The active controller's locked output files and pinned code are left untouched.
"""
import collections
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
ORIGINAL = EXP / 'results/full'
RECOVERY = EXP / 'results_video_retry_gpu1/full'
SUITES = ['tempcompass', 'videomme', 'videommmu_perception', 'videommmu_comprehension', 'videommmu_adaptation']
CONFIGS = ['baseline_r100'] + [f'{method}_r{ratio:03}'
    for method in ('resolution', 'uniform', 'merge') for ratio in (50, 25, 10)]
EXPECTED = dict(tempcompass=1580, videomme=1800, videommmu_perception=300,
                videommmu_comprehension=300, videommmu_adaptation=300)
START, END = '<!-- CURRENT_VIDEO_PROGRESS_START -->', '<!-- CURRENT_VIDEO_PROGRESS_END -->'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def jsonlines(path, live=False):
    if not path.exists():
        return []
    result = []
    with path.open() as stream:
        for line in stream:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                if not live:
                    raise
    return result


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def complete(directory):
    marker = directory / 'COMPLETE.json'
    if not marker.exists():
        return None
    done = read(marker)
    attempt = directory / done['attempt']
    for name, checksum in done['sha256'].items():
        require(sha(attempt / name) == checksum, f'Completed artifact changed: {attempt / name}')
    require(read(attempt / 'exit.json')['returncode'] == 0, f'Nonzero completed exit: {attempt}')
    require(all(done[k] == v for k, v in read(attempt / 'validated.json').items()),
            f'Validation differs: {attempt}')
    rows = jsonlines(attempt / 'telemetry.jsonl')
    require(len(rows) == done['count'], f'Telemetry population mismatch: {attempt}')
    for i, row in enumerate(rows):
        require(row['index'] == i and row['fused_int4_calls'] > 0, f'INT4 evidence mismatch: {attempt}')
    for key, value in done['telemetry'].items():
        if isinstance(value, dict) and 'total' in value:
            require(math.isclose(sum(r[key] for r in rows), value['total'], rel_tol=1e-9, abs_tol=1e-8),
                    f'Telemetry sum mismatch: {attempt}/{key}')
    started = dt.datetime.strptime(done['attempt'], 'attempt_%Y%m%dT%H%M%S_%fZ').replace(tzinfo=dt.timezone.utc)
    ended = dt.datetime.fromtimestamp((attempt / 'exit.json').stat().st_mtime, tz=dt.timezone.utc)
    seconds = sorted(r['model_seconds'] for r in rows)
    return dict(result=done, directory=str(directory.relative_to(EXP)),
        marker_sha256=sha(marker), attempt_directory=str(attempt.relative_to(EXP)),
        wall_hours=(ended - started).total_seconds() / 3600,
        p50_model_seconds=seconds[len(seconds) // 2], p95_model_seconds=seconds[math.ceil(.95 * len(seconds)) - 1])


def correctness(suite, sample):
    if suite == 'tempcompass':
        return int(sample['avg_accuracy']['rating'])
    return int(sample['videomme_perception_score']['score'])


def paired(suite, baseline, reduced):
    require(baseline['result']['sample_identity'] == reduced['result']['sample_identity'], 'Population differs')
    left = jsonlines(EXP / baseline['attempt_directory'] / 'samples.jsonl')
    right = jsonlines(EXP / reduced['attempt_directory'] / 'samples.jsonl')
    indexed = {(r['doc_id'], r['doc_hash'], str(r['target'])): r for r in right}
    require(len(left) == len(right) == len(indexed) == baseline['result']['count'], 'Paired sample count differs')
    groups = collections.defaultdict(lambda: dict(count=0, baseline_correct=0, reduced_correct=0))
    gained = lost = 0
    for l in left:
        r = indexed[(l['doc_id'], l['doc_hash'], str(l['target']))]
        a, b = correctness(suite, l), correctness(suite, r)
        gained += b and not a
        lost += a and not b
        group = l['avg_accuracy']['dim'] if suite == 'tempcompass' else l['videomme_perception_score']['duration']
        groups[group]['count'] += 1
        groups[group]['baseline_correct'] += a
        groups[group]['reduced_correct'] += b
    require(math.isclose(100 * sum(correctness(suite, s) for s in left) / len(left),
                         baseline['result']['score_percent']), 'Baseline score differs from samples')
    require(math.isclose(100 * sum(correctness(suite, s) for s in right) / len(right),
                         reduced['result']['score_percent']), 'Reduced score differs from samples')
    return dict(gained=int(gained), lost=int(lost), net=int(gained - lost), groups=dict(groups))


def update_block(path, text, historical_heading=False):
    content = path.read_text()
    block = START + '\n' + text + '\n' + END
    if START in content:
        first, rest = content.split(START, 1)
        _, last = rest.split(END, 1)
        content = first + block + last
    else:
        title, rest = content.split('\n', 1)
        heading = '\n\n## Historical report — 2026-11-05\n' if historical_heading else ''
        content = title + '\n\n' + block + heading + '\n' + rest.lstrip('\n')
    path.write_text(content)


def main():
    now = dt.datetime.now(dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=8), 'Asia/Singapore'))
    original_protocol, recovery_protocol = read(ORIGINAL / 'protocol.json'), read(RECOVERY / 'protocol.json')
    require(sha(ORIGINAL / 'protocol.json') == recovery_protocol['source_protocol_sha256'], 'Original protocol changed')
    for name, checksum in {**recovery_protocol['source_code'], **recovery_protocol['repair_code']}.items():
        require(sha(ROOT / name) == checksum, f'Pinned evaluation source changed: {name}')
    entries, metrics = [], []
    for suite in SUITES:
        for config in CONFIGS:
            entry = dict(suite=suite, config=config, status='pending', origin=None)
            for base, origin in ((RECOVERY, 'retry'), (ORIGINAL, 'original')):
                info = complete(base / 'qat_w4' / suite / config)
                if info is not None:
                    require(info['result']['count'] == EXPECTED[suite], f'Full population differs: {suite}')
                    entry.update(info, status='complete', origin=origin)
                    break
            if entry['status'] != 'complete':
                attempts = sorted((RECOVERY / 'qat_w4' / suite / config).glob('attempt_*'))
                if attempts:
                    attempt = attempts[-1]
                    rows = jsonlines(attempt / 'telemetry.jsonl', live=True)
                    entry.update(status='failed' if (attempt / 'exit.json').exists() else 'in progress / uncommitted',
                        partial_questions=len(rows), expected_questions=EXPECTED[suite],
                        attempt_directory=str(attempt.relative_to(EXP)))
            entries.append(entry)
    by = {(r['suite'], r['config']): r for r in entries}
    for entry in entries:
        if entry['status'] != 'complete':
            continue
        d, t = entry['result'], entry['result']['telemetry']
        baseline = by[entry['suite'], 'baseline_r100']
        comparable = baseline['status'] == 'complete' and baseline['origin'] == entry['origin'] == 'retry'
        if comparable:
            bt = baseline['result']['telemetry']
            entry.update(delta_pp=d['score_percent'] - baseline['result']['score_percent'],
                population_model_speedup=bt['model_seconds']['mean'] / t['model_seconds']['mean'],
                fixed_32_token_speedup=bt['fixed_output_profile']['mean_seconds'] / t['fixed_output_profile']['mean_seconds'],
                wall_speedup=baseline['wall_hours'] / entry['wall_hours'])
            if entry['suite'] in ('tempcompass', 'videomme'):
                entry['paired'] = paired(entry['suite'], baseline, entry)
        metrics.append(dict(suite=entry['suite'], config=entry['config'], origin=entry['origin'], count=d['count'],
            score_percent=d['score_percent'], delta_pp=entry.get('delta_pp'), actual_retention=t['actual_retention'],
            mean_visual_tokens=t['input_visual_tokens']['mean'], mean_output_tokens=t['output_text_tokens']['mean'],
            mean_model_seconds=t['model_seconds']['mean'], population_model_speedup=entry.get('population_model_speedup'),
            fixed_32_token_speedup=entry.get('fixed_32_token_speedup'), model_hours=t['model_seconds']['total'] / 3600,
            wall_hours=entry['wall_hours'], max_peak_allocated_gib=t['peak_gpu_allocated_bytes']['max'] / 2**30,
            output_cutoffs=d['output_cutoffs'], adapted_questions=d.get('video_adaptation', {}).get('adapted_questions')))
    counts = collections.Counter(r['suite'] for r in entries if r['status'] == 'complete')
    total = sum(counts.values())
    new = sum(r['status'] == 'complete' and r['origin'] == 'retry' for r in entries)
    active = [r for r in entries if r['status'] == 'in progress / uncommitted']
    with (RECOVERY / '.matrix.lock').open() as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            controller_active = False
        except BlockingIOError:
            controller_active = True
    lines = ['## Video progress and token-reduction effects — ' + now.strftime('%Y-%m-%d'), '',
        'Snapshot at ' + now.strftime('%H:%M:%S Asia/Singapore') + '. CPU-only artifact review; the evaluation continues unchanged.', '',
        f'**QAT W4: {70 + total}/120 complete: 70/70 image jobs and {total}/50 video jobs.** '
        f'The video total is {new} new recovery completions plus {total - new} reused original passes; '
        f'{50 - total} video configurations remain uncommitted. Vanilla 2B/released BF16 comparisons remain pending.', '',
        table(['Suite', 'Complete / planned', 'Remaining'], [[s, f'{counts[s]}/10', 10 - counts[s]] for s in SUITES]), '']
    for entry in active:
        lines += [f'**Current attempt:** `{entry["suite"]}/{entry["config"]}` — '
            f'{entry["partial_questions"]}/{entry["expected_questions"]} completed question records '
            f'({100 * entry["partial_questions"] / entry["expected_questions"]:.1f}%). '
            'This is progress only; no partial benchmark score is reported.']
    lines += ['', 'The controller output lock is ' + ('held by the running controller' if controller_active else 'available') +
        '. Every reported completion passed artifact SHA-256, exit, validation and full-population telemetry checks. '
        'Completed QAT telemetry has ordered question indices and positive fused INT4 calls. '
        'Accuracy comparisons below also verify saved question identities and independently count per-question correctness.', '',
        '### TempCompass: all naive reduction settings completed', '',
        'All 1,580 questions completed for each configuration, with zero adapted clips and zero capped answers. '
        'These are matched-input-policy comparisons with deterministic local MC scoring, not API-judged official scores. '
        'The deltas are reduced minus baseline in percentage points.', '',
        table(['Configuration', 'Score %', 'Delta pp', 'Actual visual retained', 'Mean model s', 'Model speedup', '32-token speedup', 'Max allocated GiB'],
            [[r['config'], f"{r['result']['score_percent']:.3f}", f"{r['delta_pp']:+.3f}",
              f"{100*r['result']['telemetry']['actual_retention']:.1f}%",
              f"{r['result']['telemetry']['model_seconds']['mean']:.3f}", f"{r['population_model_speedup']:.2f}×",
              f"{r['fixed_32_token_speedup']:.2f}×", f"{r['result']['telemetry']['peak_gpu_allocated_bytes']['max']/2**30:.3f}"]
             for r in entries if r['suite'] == 'tempcompass' and r['status'] == 'complete' and r['origin'] == 'retry']), '',
        '**Observed TempCompass result:** uniform retention at 50% and 25% slightly exceeds baseline accuracy '
        '(+0.506 and +0.316 pp), while merge at 25% matches the baseline score. '
        'Uniform 25% retains about one quarter of the visual tokens and reduces mean full-population model time '
        'from 3.021 to 1.173 s/question (2.58×). Merge 25% is similar at 1.169 s/question (2.58×). '
        'At 10%, all methods lose accuracy; merge has the largest loss (−5.000 pp). '
        'Simple token reduction therefore has a useful 25–50% region on this suite, while aggressive merging has a substantial accuracy cost at 10%.', '',
        'The small positive score differences do not establish a statistically reliable accuracy improvement. '
        'Uniform 25% fixes 52 baseline errors and introduces 47 new errors (net +5); merge 25% fixes and introduces '
        '71 each. Equal aggregate scores do not mean identical answers.', '',
        '### VideoMME: completed comparable runs', '',
        'Population: 1,800 short+medium questions (900 each), excluding long videos; no subtitles/audio or API judge. '
        'The repaired baseline is already adaptive: 846/1,800 questions (47.0%) require a smaller spatial grid. '
        'Across all questions it retains 50.14% of the original uncapped policy’s total visual tokens. '
        '“100%” below means the full adaptive baseline, not the uncapped 576-token-per-group input.', '',
        table(['Configuration', 'Score %', 'Delta pp', 'Mean visual tokens', 'Mean model s', 'Model speedup', '32-token speedup', 'Model hours', 'Attempt hours', 'Max allocated GiB'],
            [[r['config'], f"{r['result']['score_percent']:.3f}", f"{r['delta_pp']:+.3f}",
              f"{r['result']['telemetry']['input_visual_tokens']['mean']:,.0f}",
              f"{r['result']['telemetry']['model_seconds']['mean']:.2f}", f"{r['population_model_speedup']:.2f}×",
              f"{r['fixed_32_token_speedup']:.2f}×", f"{r['result']['telemetry']['model_seconds']['total']/3600:.2f}",
              f"{r['wall_hours']:.2f}", f"{r['result']['telemetry']['peak_gpu_allocated_bytes']['max']/2**30:.2f}"]
             for r in entries if r['suite'] == 'videomme' and r['status'] == 'complete' and r['origin'] == 'retry']), '',
        '**Observed VideoMME result:** 50% resolution gives 61.222% versus 61.500% (−0.278 pp, net five fewer correct answers). '
        'Achieved visual retention is 45.66% because grids are patch-aligned. Mean model time falls 90.48 → 43.94 s '
        '(2.06×), attempt time 55.40 → 30.49 h (1.82×), and peak allocation 16.17 → 9.15 GiB (43.4% lower). '
        'Both runs generate two tokens per answer and have zero cap hits, so the speed difference does not come from shorter answers.', '']
    video_pair = by['videomme', 'resolution_r050'].get('paired')
    if video_pair:
        lines += [table(['Video duration', 'Questions', 'Baseline %', 'Resolution 50% %', 'Delta pp'],
            [[g, v['count'], f"{100*v['baseline_correct']/v['count']:.3f}",
              f"{100*v['reduced_correct']/v['count']:.3f}",
              f"{100*(v['reduced_correct']-v['baseline_correct'])/v['count']:+.3f}"]
             for g, v in video_pair['groups'].items()]), '',
            'The reduction fixes 69 baseline errors and introduces 74 new errors. The small aggregate change conceals '
            '143 changed correctness outcomes; it is not proof of equivalent predictions.', '']
    lines += ['### Why VideoMME takes days', '']
    base = by['videomme', 'baseline_r100']
    t = base['result']['telemetry']
    lines += [table(['Baseline measured component', 'Hours', 'Share of model time'],
        [[label, f"{t[key]['total']/3600:.3f}", f"{100*t[key]['total']/t['model_seconds']['total']:.2f}%"]
         for label, key in [('Decoder prefill', 'prefill_seconds'), ('Vision + embedding', 'vision_and_embedding_seconds'),
                            ('Reduction + host staging', 'reduction_seconds'), ('Autoregressive decode', 'decode_seconds')]]), '',
        'VideoMME’s baseline averages 78,494 visual plus 2,959 text tokens per question; its two-token answers cost '
        'little compared with prefilling that long context. Prefill consumes 41.15 of 45.24 measured model hours (91.0%). '
        'The 512-token prefill chunks and one-temporal-group vision chunks bound memory, but add repeated kernel/transfer '
        'and Python-loop work. All 1,800 questions are evaluated independently; repeated questions about the same clip '
        'do not share precomputed vision features or a prompt KV cache.', '',
        f'Model-only baseline p50/p95 latency is {base["p50_model_seconds"]:.2f}/{base["p95_model_seconds"]:.2f} s/question. '
        'The 55.40 h attempt time also includes about 10.16 h outside the recorded normal-answer model timers: '
        'media decoding/preprocessing, setup, profiling, I/O and other uninstrumented overhead. The residual cannot '
        'be attributed to any single component from these records. Attempt time is estimated from the attempt-directory '
        'timestamp to the exit.json modification time, and is affected by host scheduling/contention. '
        'Only one job runs at a time on GPU 1.', '',
        'Uniform/merge still encode the full adaptive spatial input before reducing decoder tokens, so they retain '
        'the full vision/preprocessing cost. Resolution reduction can save work in both vision and the decoder. '
        'The first three questions used for 32-token profiles are a small, nonrepresentative sample; their speedups '
        'also use longer outputs than the two-token benchmark answers. Keep fixed-output profile and full-population '
        'normal-answer speedups separate. No stable remaining-time estimate is inferred from the active partial run.', '',
        '### Reused original-policy results', '',
        table(['Suite', 'Configuration', 'Score %', 'Questions', 'Cap hits'],
            [[r['suite'], r['config'], f"{r['result']['score_percent']:.3f}", r['result']['count'], r['result']['output_cutoffs']]
             for r in entries if r['status'] == 'complete' and r['origin'] == 'original']), '',
        'These five passes are preserved and were not rerun. They use the original spatial/runtime policy, so no '
        'accuracy delta or timing speedup against the adaptive recovery baseline is assigned. In particular, '
        'VideoMME original resolution 25% at 61.722% cannot be read as an improvement over the repaired baseline '
        'under an identical input policy. Video-MMMU still lacks repaired full-population baselines.', '',
        '### Assessment so far', '',
        'Naive reduction can preserve most accuracy while materially reducing long-video prefill cost: TempCompass '
        'uniform/merge at 25% and VideoMME resolution at 50% are promising observed tradeoffs. This is not evidence '
        'that every suite tolerates the same retention. The completed image experiment already shows task-dependent '
        'losses, and TempCompass merge at 10% loses five points. VideoMME uniform/merge and repaired Video-MMMU '
        'results are still needed before ranking methods across video suites. At 25% retention, the image experiment '
        'favored resolution on all seven suites, whereas TempCompass favors uniform selection over resolution '
        '(61.646% versus 60.190%). Achieved token budgets differ slightly, as shown above; no single naive method '
        'is best everywhere. No INT4-versus-BF16 conclusion is '
        'available until matched baseline models are evaluated.', '',
        'Machine evidence: `reports/video_progress.json` contains verified completion provenance, paired correctness '
        'counts and the active attempt snapshot; `reports/video_progress.csv` contains the completed metrics. '
        'This synthesis does not modify the running evaluation, its pinned protocol, or completed artifacts.', '']
    text = '\n'.join(lines)
    report_dir = EXP / 'reports'
    report_dir.mkdir(exist_ok=True)
    (report_dir / 'video_progress.md').write_text('# QAT W4 video comparison\n\n' + text)
    (report_dir / 'video_progress.json').write_text(json.dumps(dict(snapshot=now.isoformat(),
        original_protocol_sha256=sha(ORIGINAL / 'protocol.json'), recovery_protocol_sha256=sha(RECOVERY / 'protocol.json'),
        controller_lock_held=controller_active, completed_video_jobs=total, completed_recovery_jobs=new,
        jobs=entries), indent=2) + '\n')
    with (report_dir / 'video_progress.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    update_block(EXP / 'experiment_report.md', text, historical_heading=True)
    update_block(EXP / 'experiment_README.md', f'Latest video review: **{now:%Y-%m-%d}**. QAT W4 has '
        f'**{70 + total}/120** completions, including **{total}/50 video jobs**. '
        'See the [current video comparison](reports/video_progress.md) for completed reductions, timing analysis and '
        'the active-attempt snapshot. Refresh with `python3 compile_video_progress.py` from this directory; no GPU is used.')
    print(json.dumps(dict(report=str(EXP / 'experiment_report.md'), completed_video=total,
                          new_recovery_completions=new, active=[dict(suite=r['suite'], config=r['config'],
                          questions=r['partial_questions']) for r in active]), indent=2))


if __name__ == '__main__':
    main()
