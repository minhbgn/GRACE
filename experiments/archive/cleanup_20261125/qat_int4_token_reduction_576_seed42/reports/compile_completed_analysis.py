"""CPU-only, read-only evaluation audit for the November 16 follow-up report.

Writes only completed_analysis_20261116.{json,csv} and completed_tables_20261116.md
beside this script. Never imports evaluation/model code or rewrites run artifacts.
"""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
OLD = EXP / 'results/full'
NEW = EXP / 'results_video_retry_gpu1/full'
OUT = Path(__file__).resolve().parent
IMAGES = ['hallusionbench', 'mmbench', 'scienceqa', 'ai2d', 'mmmu', 'seedbench_img', 'mmstar']
VIDEOS = ['tempcompass', 'videomme', 'videommmu_perception', 'videommmu_comprehension', 'videommmu_adaptation']
CONFIGS = ['baseline_r100'] + [f'{m}_r{r:03}' for m in ['resolution', 'uniform', 'merge'] for r in [50, 25, 10]]


def read(p):
    return json.loads(p.read_text())


def rows(p):
    if not p.exists():
        return []
    with p.open() as stream:
        return [json.loads(line) for line in stream if line.endswith('\n')]


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def table(headers, body):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in body])


def summarize(records, inputs=None):
    n = len(records)
    d = dict(count=n)
    for key in ['input_visual_tokens', 'input_text_tokens', 'output_text_tokens', 'model_seconds',
                'prefill_seconds', 'vision_and_embedding_seconds', 'reduction_seconds', 'decode_seconds']:
        d[key + '_sum'] = sum(r[key] for r in records)
        d[key + '_mean'] = d[key + '_sum'] / n
    d['input_total_tokens_sum'] = d['input_visual_tokens_sum'] + d['input_text_tokens_sum']
    d['input_total_tokens_mean'] = d['input_total_tokens_sum'] / n
    d['input_and_output_tokens_mean'] = (d['input_total_tokens_sum'] + d['output_text_tokens_sum']) / n
    d['text_only_rows'] = sum(r['input_visual_tokens'] == 0 for r in records)
    d['visual_tokens_per_model_second'] = d['input_visual_tokens_sum'] / d['model_seconds_sum']
    d['decoder_input_tokens_per_prefill_second'] = d['input_total_tokens_sum'] / d['prefill_seconds_sum']
    if inputs:
        durations = [r['sampling']['duration_seconds'] for r in inputs]
        d['source_seconds_sum'] = sum(durations)
        d['source_seconds_mean'] = sum(durations) / n
        d['sampled_frames_mean'] = sum(r['sampling']['sampled_frames'] for r in inputs) / n
        d['source_visual_rate_mean'] = sum(r['input_visual_tokens'] / sec for r, sec in zip(records, durations)) / n
        d['source_visual_rate_pooled'] = d['input_visual_tokens_sum'] / sum(durations)
        d['adapted_questions'] = sum(r.get('video_adaptation', {}).get('applied', False) for r in records)
        d['uncapped_visual_sum'] = sum(r.get('video_adaptation', {}).get('original_policy_visual_tokens',
                                       r['original_input_visual_tokens']) for r in records)
    return d


def timer_key(r):
    return tuple(r[k] for k in ['index', 'model_seconds', 'prefill_seconds', 'vision_and_embedding_seconds'])


def timer_origins(directory, final_rows):
    # Replay preserves the original timers exactly. Find their earliest saved source;
    # the final job's scheduling metadata cannot classify a restored prefix.
    origins = {}
    wanted = {timer_key(r) for r in final_rows}
    for attempt in sorted(directory.glob('attempt_*')):
        if not (attempt / 'job.json').exists():
            continue
        job = read(attempt / 'job.json')
        for r in rows(attempt / 'telemetry.jsonl'):
            key = timer_key(r)
            if key in wanted and key not in origins:
                schedule = job.get('scheduling', {})
                concurrency = schedule.get('parallel_evaluation', 1)
                origins[key] = dict(attempt=attempt.name, gpu=job.get('gpu'),
                                    workers_per_gpu=concurrency, legacy_serial=not bool(schedule))
    assert all(timer_key(r) in origins for r in final_rows)
    return [origins[timer_key(r)] for r in final_rows]


def main():
    source_checks = []
    proto = read(NEW / 'protocol.json')
    assert sha(OLD / 'protocol.json') == proto['source_protocol_sha256']
    for manifest, keys in [(NEW / 'protocol.json', ['source_code', 'repair_code']),
                           (NEW / 'parallel_implementation.json', ['code'])]:
        data = read(manifest)
        for key in keys:
            for name, expected in data[key].items():
                assert sha(ROOT / name) == expected, name
                source_checks.append(dict(file=name, sha256=expected))
    reuse = read(NEW / 'reuse_manifest.json')
    for item in reuse['jobs']:
        assert sha(Path(item['marker'])) == item['marker_sha256']
    jobs, raw, sampling, samples = [], {}, {}, {}
    for suite in IMAGES + VIDEOS:
        for config in CONFIGS:
            candidates = [(OLD, 'original')] if suite in IMAGES else [(NEW, 'recovery'), (OLD, 'original')]
            selected = next(((base, origin) for base, origin in candidates
                             if (base / 'qat_w4' / suite / config / 'COMPLETE.json').exists()), None)
            if not selected:
                jobs.append(dict(suite=suite, config=config, status='incomplete'))
                continue
            base, origin = selected
            directory = base / 'qat_w4' / suite / config
            marker = directory / 'COMPLETE.json'
            done = read(marker)
            attempt = directory / done['attempt']
            for name, expected in done['sha256'].items():
                assert sha(attempt / name) == expected, str(attempt / name)
            assert read(attempt / 'exit.json')['returncode'] == 0
            assert all(done[k] == v for k, v in read(attempt / 'validated.json').items())
            rr = rows(attempt / 'telemetry.jsonl')
            assert len(rr) == done['count']
            assert [r['index'] for r in rr] == list(range(len(rr)))
            assert all(r['fused_int4_calls'] > 0 for r in rr)
            assert all(r['decoder_layer_visual_tokens'] == [r['input_visual_tokens']] * 28 for r in rr)
            for key, value in done['telemetry'].items():
                if isinstance(value, dict) and 'total' in value:
                    assert math.isclose(sum(r[key] for r in rr), value['total'], rel_tol=1e-9, abs_tol=1e-7), key
                    assert math.isclose(sum(r[key] for r in rr)/len(rr), value['mean'], rel_tol=1e-9, abs_tol=1e-7)
            profiles = [sum(r['fixed_output_profile_seconds']) / len(r['fixed_output_profile_seconds'])
                        for r in rr if r['fixed_output_profile_seconds']]
            assert len(profiles) == 3
            assert math.isclose(sum(profiles)/3, done['telemetry']['fixed_output_profile']['mean_seconds'])
            runtime = read(attempt / 'runtime.json')
            assert runtime['integer_codes_verified'] and runtime['converted_modules'] == 196
            inputs = rows(attempt / 'video_inputs.jsonl') if suite in VIDEOS else []
            if inputs:
                assert len(inputs) == len(rr)
                for inp, r in zip(inputs, rr):
                    assert inp['doc_id'] == r['index']
            key = (suite, config)
            raw[key], sampling[key] = rr, inputs
            if suite in ['tempcompass', 'videomme']:
                ss = rows(attempt / 'samples.jsonl')
                samples[key] = ss
                assert len(ss) == len(rr)
                scores = [s['avg_accuracy']['rating'] if suite == 'tempcompass'
                          else s['videomme_perception_score']['score'] for s in ss]
                assert math.isclose(100*sum(scores)/len(scores), done['score_percent'], abs_tol=0.00051)
            entry = dict(suite=suite, config=config, status='complete', origin=origin,
                         marker=str(marker.relative_to(EXP)), marker_sha256=sha(marker),
                         attempt=str(attempt.relative_to(EXP)), sample_identity=done['sample_identity'],
                         score_percent=done['score_percent'], output_cutoffs=done['output_cutoffs'],
                         actual_retention=done['telemetry']['actual_retention'],
                         fixed_32_seconds=done['telemetry']['fixed_output_profile']['mean_seconds'],
                         peak_allocated_gib=done['telemetry']['peak_gpu_allocated_bytes']['max']/2**30,
                         **summarize(rr, inputs))
            if suite in VIDEOS and origin == 'recovery':
                origins = timer_origins(directory, rr)
                entry['timing_origins'] = dict(collections.Counter(
                    'legacy_serial' if x['legacy_serial'] else f"workers_per_gpu_{x['workers_per_gpu']}" for x in origins))
                entry['timing_source_attempts'] = dict(collections.Counter(x['attempt'] for x in origins))
                entry['single_worker_indices'] = [i for i,x in enumerate(origins) if x['workers_per_gpu'] == 1]
            jobs.append(entry)
    by = {(j['suite'], j['config']): j for j in jobs if j['status'] == 'complete'}
    for (suite, config), j in by.items():
        baseline = by.get((suite, 'baseline_r100'))
        if not baseline:
            continue
        assert j['sample_identity'] == baseline['sample_identity'], (suite, config)
        matched = suite in IMAGES or j['origin'] == baseline['origin']
        j['matched_input_policy'] = matched
        if matched:
            j['delta_pp'] = j['score_percent'] - baseline['score_percent']
            j['observed_model_time_ratio'] = baseline['model_seconds_mean']/j['model_seconds_mean']
            j['fixed_32_ratio'] = baseline['fixed_32_seconds']/j['fixed_32_seconds']
        if suite in ['tempcompass', 'videomme']:
            a, b = samples[suite, 'baseline_r100'], samples[suite, config]
            def correct(s):
                return s['avg_accuracy']['rating'] if suite == 'tempcompass' else s['videomme_perception_score']['score']
            assert [(s['doc_id'], s['doc_hash'], s['target']) for s in a] == [(s['doc_id'], s['doc_hash'], s['target']) for s in b]
            j['gained'] = sum(bool(correct(y)) and not correct(x) for x,y in zip(a,b))
            j['lost'] = sum(bool(correct(x)) and not correct(y) for x,y in zip(a,b))
            groups = collections.defaultdict(list)
            for i, s in enumerate(b):
                group = s['avg_accuracy']['dim'] if suite == 'tempcompass' else s['videomme_perception_score']['duration']
                groups[group].append(i)
            j['groups'] = {}
            for group, indices in groups.items():
                group_data = summarize([raw[suite,config][i] for i in indices], [sampling[suite,config][i] for i in indices])
                group_data['score_percent'] = 100*sum(correct(b[i]) for i in indices)/len(indices)
                j['groups'][group] = group_data
            if matched and suite == 'videomme':
                # Confirm the common temporal input and adaptive starting geometry.
                for arow, brow, at, bt in zip(sampling[suite,'baseline_r100'], sampling[suite,config],
                                             raw[suite,'baseline_r100'], raw[suite,config]):
                    assert arow['id'] == brow['id']
                    assert arow['sampling']['frame_indices'] == brow['sampling']['frame_indices']
                    for field in ['adaptive_baseline_visual_tokens', 'adaptive_baseline_height', 'adaptive_baseline_width']:
                        assert at['video_adaptation'][field] == bt['video_adaptation'][field]
                indices = j['single_worker_indices']
                if indices:
                    j['single_worker_paired'] = dict(count=len(indices),
                        model_time_ratio=sum(raw[suite,'baseline_r100'][i]['model_seconds'] for i in indices) /
                                         sum(raw[suite,config][i]['model_seconds'] for i in indices))
    baselines = [j for j in jobs if j['status'] == 'complete' and j['config'] == 'baseline_r100']
    aggregates = {}
    for name, group in [('images', [j for j in baselines if j['suite'] in IMAGES]),
                        ('available_videos', [j for j in baselines if j['suite'] in VIDEOS]), ('all_available', baselines)]:
        n = sum(j['count'] for j in group)
        aggregates[name] = dict(suites=len(group), count=n,
            weighted_visual_mean=sum(j['input_visual_tokens_sum'] for j in group)/n,
            weighted_input_total_mean=sum(j['input_total_tokens_sum'] for j in group)/n,
            equal_suite_visual_mean=sum(j['input_visual_tokens_mean'] for j in group)/len(group),
            equal_suite_input_total_mean=sum(j['input_total_tokens_mean'] for j in group)/len(group))
    result = dict(date='2026-11-16 Asia/Singapore', source_checks=source_checks,
                  original_protocol_sha256=sha(OLD/'protocol.json'), recovery_protocol_sha256=sha(NEW/'protocol.json'),
                  complete=sum(j['status']=='complete' for j in jobs), jobs=jobs, baseline_aggregates=aggregates)
    (OUT/'completed_analysis_20261116.json').write_text(json.dumps(result,indent=2)+'\n')
    flat = [{k:v for k,v in j.items() if not isinstance(v,(dict,list))} for j in jobs]
    keys = list(dict.fromkeys(k for j in flat for k in j))
    with (OUT/'completed_analysis_20261116.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys); writer.writeheader(); writer.writerows(flat)
    lines=[]
    def add(name, headers, body):
        lines.extend([f'<!-- {name} -->',table(headers,body),''])
    add('inventory',['Suite','Completed / 10'],[[s,f"{sum((s,c) in by for c in CONFIGS)}/10"] for s in IMAGES+VIDEOS])
    add('baseline_tokens',['Suite','Rows','Visual sum (exact)','Mean visual','Mean text','Mean total input','Mean input + output'],
        [[j['suite'],j['count'],f"{j['input_visual_tokens_sum']:,}",f"{j['input_visual_tokens_mean']:,.6f}",
          f"{j['input_text_tokens_mean']:,.6f}",f"{j['input_total_tokens_mean']:,.6f}",f"{j['input_and_output_tokens_mean']:,.6f}"] for j in baselines])
    add('baseline_throughput',['Suite','Mean model s/question','Visual tokens/model s','Input tokens/prefill s'],
        [[j['suite'],f"{j['model_seconds_mean']:.6f}",f"{j['visual_tokens_per_model_second']:.6f}",
          f"{j['decoder_input_tokens_per_prefill_second']:.6f}"] for j in baselines])
    video_baselines=[(j['suite'],j) for j in baselines if j['suite'] in VIDEOS]
    vm=by['videomme','baseline_r100']
    video_baselines += [('videomme '+g,v) for g,v in vm['groups'].items()]
    add('baseline_rates',['Video population','Mean source s','Mean frames','Mean visual tokens/source s','Pooled visual tokens/source s','Visual tokens/model s'],
        [[name,f"{j['source_seconds_mean']:.6f}",f"{j['sampled_frames_mean']:.6f}",f"{j['source_visual_rate_mean']:.6f}",
          f"{j['source_visual_rate_pooled']:.6f}",f"{j['visual_tokens_per_model_second']:.6f}"] for name,j in video_baselines])
    add('videomme_split_tokens',['Duration','Rows','Visual sum (exact)','Mean visual','Mean text','Mean input','Adapted rows'],
        [[g,j['count'],f"{j['input_visual_tokens_sum']:,}",f"{j['input_visual_tokens_mean']:,.6f}",f"{j['input_text_tokens_mean']:,.6f}",
          f"{j['input_total_tokens_mean']:,.6f}",j['adapted_questions']] for g,j in vm['groups'].items()])
    add('image_accuracy',['Suite']+[c.replace('resolution','res') for c in CONFIGS],
        [[s]+[f"{by[s,c]['score_percent']:.2f}" for c in CONFIGS] for s in IMAGES])
    for label,field in [('image_speed_natural','observed_model_time_ratio'),('image_speed_32','fixed_32_ratio')]:
        add(label,['Suite']+[c.replace('resolution','res') for c in CONFIGS[1:]],
            [[s]+[f"{by[s,c][field]:.3f}×" for c in CONFIGS[1:]] for s in IMAGES])
    for suite in ['tempcompass','videomme']:
        add(suite+'_results',['Configuration','Score %','Delta pp','Mean visual','Retained %','Model s','Observed time ratio','32-token ratio','Peak GiB'],
            [[c,f"{by[suite,c]['score_percent']:.3f}",f"{by[suite,c]['delta_pp']:+.3f}" if 'delta_pp' in by[suite,c] else 'policy differs',
              f"{by[suite,c]['input_visual_tokens_mean']:,.3f}",f"{100*by[suite,c]['actual_retention']:.3f}",
              f"{by[suite,c]['model_seconds_mean']:.3f}", f"{by[suite,c]['observed_model_time_ratio']:.3f}×" if 'delta_pp' in by[suite,c] else 'N/A',
              f"{by[suite,c]['fixed_32_ratio']:.3f}×" if 'delta_pp' in by[suite,c] else 'N/A', f"{by[suite,c]['peak_allocated_gib']:.3f}"] for c in CONFIGS])
        add(suite+'_accuracy_groups',['Configuration']+list(by[suite,'baseline_r100']['groups'])+['Fixes','New errors'],
            [[c]+[f"{v['score_percent']:.3f}" for v in by[suite,c]['groups'].values()]+[by[suite,c]['gained'],by[suite,c]['lost']] for c in CONFIGS])
    add('video_timing_origins',['Configuration','Original serial rows','One-worker rows','Shared-GPU rows','One-worker paired ratio'],
        [[c,j.get('timing_origins',{}).get('legacy_serial',0),j.get('timing_origins',{}).get('workers_per_gpu_1',0),
          sum(v for k,v in j.get('timing_origins',{}).items() if k not in ['legacy_serial','workers_per_gpu_1']),
          f"{j['single_worker_paired']['model_time_ratio']:.3f}×" if 'single_worker_paired' in j else 'N/A']
         for c in CONFIGS if (j:=by['videomme',c])['origin']=='recovery'])
    add('videommmu_results',['Split','Configuration','Origin','Score %','Cap hits','Mean visual','Model s','Observed ratio'],
        [[s,c,j['origin'],f"{j['score_percent']:.3f}",j['output_cutoffs'],f"{j['input_visual_tokens_mean']:,.3f}",f"{j['model_seconds_mean']:.3f}",
          f"{j['observed_model_time_ratio']:.3f}×" if 'delta_pp' in j else 'N/A']
         for s in VIDEOS[2:] for c in CONFIGS if (j:=by.get((s,c)))])
    add('all_token_counts',['Suite']+[c.replace('resolution','res') for c in CONFIGS],
        [[s]+[f"{by[s,c]['input_visual_tokens_mean']:,.3f}" if (s,c) in by else '—' for c in CONFIGS] for s in IMAGES+VIDEOS])
    (OUT/'completed_tables_20261116.md').write_text('\n'.join(lines))
    print(json.dumps(dict(complete=result['complete'],source_files_verified=len(source_checks),
                         aggregates=aggregates,output=str(OUT/'completed_analysis_20261116.json')),indent=2))


if __name__=='__main__':
    main()
