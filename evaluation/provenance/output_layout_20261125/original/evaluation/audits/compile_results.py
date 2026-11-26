#!/usr/bin/env python3
"""Audit and merge frozen evaluation results using only CPU and saved artifacts.

Consolidates the former matrix, progress and timing report compilers. The catalog
selects explicit completions, never directory timestamps. Predictions, markers,
protocols and evaluator sources are read-only. Output JSON/CSV is deterministic.
"""
import argparse
import os
import sys

import collections
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'experiments/qat_int4_token_reduction_576_seed42'
OLD = EXP / 'results/full'
NEW = EXP / 'results_video_retry_gpu0_scheduler_resume/full'
MATCHED = EXP / 'results_video_matched_exclusive/full'
OUT = ROOT / 'results/analysis'
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


def audit_matrix():
    source_checks = []
    catalog = read(ROOT / 'evaluation/protocols/result_catalog.json')
    proto = read(NEW / 'protocol.json')
    assert sha(OLD / 'protocol.json') == proto['source_protocol_sha256']
    for manifest, keys in [(NEW / 'protocol.json', ['source_code', 'repair_code']),
                           (NEW / 'parallel_implementation.json', ['code']),
                           (MATCHED / 'parallel_implementation.json', ['code'])]:
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
            selected = catalog['selections'][suite + '/' + config]
            origin = selected['origin']
            directory = ROOT / selected['directory']
            marker = directory / 'COMPLETE.json'
            assert sha(marker) == selected['marker_sha256'], f'Selected result changed: {marker}'
            done = read(marker)
            attempt = directory / done['attempt']
            job = read(attempt / 'job.json')
            assert job['limit'] is None and job['model'] == 'qat_w4'
            expected_count = dict(zip(IMAGES + VIDEOS,
                [1129, 4329, 4241, 3088, 900, 14233, 1500, 1580, 1800, 300, 300, 300]))[suite]
            assert done['count'] == expected_count, (suite, config, 'not a full population')
            for name, expected in done['sha256'].items():
                assert sha(attempt / name) == expected, str(attempt / name)
            assert read(attempt / 'exit.json')['returncode'] == 0
            assert all(done[k] == v for k, v in read(attempt / 'validated.json').items())
            rr = rows(attempt / 'telemetry.jsonl')
            assert len(rr) == done['count']
            assert [r['index'] for r in rr] == list(range(len(rr)))
            assert all(r['fused_int4_calls'] > 0 for r in rr)
            assert all(r['decoder_layer_visual_tokens'] == [r['input_visual_tokens']] * 28 for r in rr)
            assert all(len(r['generated_token_ids']) == r['output_text_tokens'] for r in rr)
            assert sum(r['generated_token_ids'][-1] != 151645 for r in rr) == done['output_cutoffs']
            for key, value in done['telemetry'].items():
                if isinstance(value, dict) and 'total' in value:
                    assert math.isclose(sum(r[key] for r in rr), value['total'], rel_tol=1e-9, abs_tol=1e-7), key
                    assert math.isclose(sum(r[key] for r in rr)/len(rr), value['mean'], rel_tol=1e-9, abs_tol=1e-7)
                    assert math.isclose(max(r[key] for r in rr), value['max'], rel_tol=1e-9, abs_tol=1e-7)
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
                         metric=done.get('metric', 'local_mc_accuracy' if suite == 'tempcompass'
                             else 'videomme_accuracy' if suite == 'videomme' else 'videommmu_mc_open_accuracy'),
                         model=job['model'],
                         execution=job.get('execution'), policy=job['policy'],
                         scheduling=job.get('scheduling', {}),
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
        matched = suite in IMAGES or (j['execution'] == baseline['execution'] and j['policy'] == baseline['policy'])
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
    assert len(by) == 120, 'Expected all 120 QAT configurations'
    result = dict(date='2026-11-25 Asia/Singapore', source_checks=source_checks,
                  original_protocol_sha256=sha(OLD/'protocol.json'), recovery_protocol_sha256=sha(NEW/'protocol.json'),
                  complete=sum(j['status']=='complete' for j in jobs), jobs=jobs, baseline_aggregates=aggregates)
    (OUT/'completed_analysis_20261125.json').write_text(json.dumps(result,indent=2)+'\n')
    flat = [{k:v for k,v in j.items() if not isinstance(v,(dict,list))} for j in jobs]
    keys = list(dict.fromkeys(k for j in flat for k in j))
    with (OUT/'completed_analysis_20261125.csv').open('w') as stream:
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
    (OUT/'completed_tables_20261125.md').write_text('\n'.join(lines))
    print(json.dumps(dict(complete=result['complete'],source_files_verified=len(source_checks),
                         aggregates=aggregates,output=str(OUT/'completed_analysis_20261125.json')),indent=2))



    return result


sys.path.insert(0, str(ROOT))
from evaluation.protocols.video_timing_subset import estimate, historical_fit

TIMING = EXP / 'results_videomme_timing50_seed42_phased_20261123'
STAMP = '20261125'


def audit_timing():
    analysis = read(OUT / f'completed_analysis_{STAMP}.json')
    full = {j['config']: j for j in analysis['jobs'] if j['suite'] == 'videomme'}
    protocol = read(TIMING / 'protocol.json')
    subset = read(TIMING / 'subset.json')
    selected = subset['samples']
    assert sha(TIMING / 'subset.json') == protocol['subset_sha256']
    assert sha(TIMING / 'reference/protocol.json') == protocol['reference_protocol_sha256']
    for name, expected in protocol['code'].items():
        assert sha(ROOT / name) == expected, name
    assert len(selected) == len({s['video_id'] for s in selected}) == 50
    saved = {r['config']: r for r in read(TIMING / 'extrapolation.json')['rows']}
    raw, entries = {}, []
    for config in CONFIGS:
        directory = TIMING / config
        done = read(directory / 'COMPLETE.json')
        attempt = directory / done['attempt']
        for name, expected in done['sha256'].items():
            assert sha(attempt / name) == expected, str(attempt / name)
        assert read(attempt / 'exit.json')['returncode'] == 0
        assert all(done[k] == v for k, v in read(attempt / 'validated.json').items())
        assert done['count'] == 50 and done['output_cutoffs'] == 0
        job = read(attempt / 'job.json')
        assert job['execution'] == protocol['execution']
        assert job['policy'] == protocol['policy']
        assert job['subset_sha256'] == protocol['subset_sha256']
        runtime = read(attempt / 'runtime.json')
        assert runtime['converted_modules'] == 196 and runtime['integer_codes_verified']
        rr, timings = rows(attempt / 'telemetry.jsonl'), rows(attempt / 'timing.jsonl')
        inputs, predictions = rows(attempt / 'video_inputs.jsonl'), rows(attempt / 'samples.jsonl')
        assert len(rr) == len(timings) == len(inputs) == len(predictions) == 50
        assert [r['index'] for r in rr] == list(range(50))
        ids = [s['id'] for s in selected]
        assert [r['id'] for r in inputs] == [r['id'] for r in timings] == ids
        assert [s['doc']['question_id'] for s in predictions] == ids
        assert [s['doc_id'] for s in predictions] == list(range(50))
        assert all(r['fused_int4_calls'] > 0 and
                   r['decoder_layer_visual_tokens'] == [r['input_visual_tokens']] * 28 for r in rr)
        for key, value in done['telemetry'].items():
            if isinstance(value, dict) and 'total' in value:
                assert math.isclose(sum(r[key] for r in rr), value['total'], rel_tol=1e-9)
        for r in timings:
            assert math.isclose(r['request_seconds'], r['preprocess_seconds'] +
                                r['staging_seconds'] + r['generate_call_seconds'])
        observations = rows(attempt / 'gpu_observations.jsonl')
        assert observations and all('error' not in o for o in observations)
        pids = {line.split(',')[0].strip() for o in observations
                for line in o['processes'].splitlines() if line.strip()}
        assert len(pids) == 1, 'Multiple GPU processes in exclusive timing run'
        assert read(attempt / 'warmup.json')['seconds'] > 0
        raw[config] = dict(model=dict(zip(ids, (r['model_seconds'] for r in rr))),
                           request={r['id']: r['request_seconds'] for r in timings})
        projected = {metric: estimate(selected, raw[config][metric], raw['baseline_r100'][metric])
                     for metric in ('model', 'request')}
        for metric in projected:
            assert projected[metric] == saved[config][metric], (config, metric, 'stale saved extrapolation')
        full_entry = full[config]
        marker = EXP / full_entry['marker']
        assert sha(marker) == full_entry['marker_sha256'], 'Full analysis is stale'
        full_attempt = EXP / full_entry['attempt']
        ft, fi, fs = (rows(full_attempt / name) for name in
                      ('telemetry.jsonl', 'video_inputs.jsonl', 'samples.jsonl'))
        assert len(ft) == len(fi) == len(fs) == 1800
        assert [s['doc_id'] for s in fs] == [r['index'] for r in ft] == list(range(1800))
        assert [s['doc']['question_id'] for s in fs] == [r['id'] for r in fi]
        population = {inp['id']: dict(seconds=t['model_seconds'], duration=s['doc']['duration'])
                      for t, inp, s in zip(ft, fi, fs)}
        full_index = {inp['id']: i for i, inp in enumerate(fi)}
        fit = historical_fit(selected, population)
        assert full_entry['execution'] == job['execution']
        # Timing runs disable the three fixed-output profiles only.
        assert {k:v for k,v in full_entry['policy'].items() if k != 'profile_samples'} == {
            k:v for k,v in job['policy'].items() if k != 'profile_samples'}
        for i, inp in enumerate(inputs):
            n = full_index[inp['id']]
            for key in ('frame_indices', 'height', 'width', 'grid'):
                assert fi[n]['sampling'][key] == inp['sampling'][key], (config, inp['id'], key)
            assert ft[n]['input_visual_tokens'] == rr[i]['input_visual_tokens']
            assert ft[n]['input_text_tokens'] == rr[i]['input_text_tokens']
            for key in ('original_policy_visual_tokens', 'adaptive_baseline_visual_tokens'):
                assert ft[n]['video_adaptation'][key] == rr[i]['video_adaptation'][key]
        recorded = full_entry['model_seconds_sum']
        error = projected['model']['full_seconds'] / recorded - 1
        shifts = {}
        for component in ('model_seconds', 'prefill_seconds', 'vision_and_embedding_seconds',
                          'reduction_seconds', 'decode_seconds'):
            old = sum(s['expansion_weight'] * ft[full_index[s['id']]][component] for s in selected)
            new = sum(s['expansion_weight'] * r[component] for s, r in zip(selected, rr))
            shifts[component] = dict(full_run_subset_expanded_hours=old/3600,
                                    new_subset_expanded_hours=new/3600, relative_change=new/old-1)
        entries.append(dict(config=config, marker=str((directory/'COMPLETE.json').relative_to(EXP)),
            marker_sha256=sha(directory/'COMPLETE.json'), attempt=str(attempt.relative_to(EXP)),
            gpu=job['gpu'], gpu_observations=len(observations), model=projected['model'],
            request=projected['request'], full_recorded_model_hours=recorded/3600,
            fresh_projection_relative_error=error, historical_sampling_backtest=fit,
            fresh_duration_relative_errors={d:projected['model']['by_duration'][d] /
                fit['by_duration'][d]['recorded_seconds']-1 for d in ('short','medium')},
            full_time_inside_fresh_sampling_ci=projected['model']['full_seconds_ci95'][0] <= recorded <=
                                               projected['model']['full_seconds_ci95'][1],
            timing_gpu_vs_full_gpu=[job['gpu'], read(full_attempt/'job.json')['gpu']],
            selected_prediction_changes=sum(predictions[i]['filtered_resps'] != fs[full_index[s['id']]]['filtered_resps']
                                            for i,s in enumerate(selected)),
            component_comparison=shifts))
    assert len({e['gpu'] for e in entries}) == 1
    result = dict(date='2026-11-25 Asia/Singapore', subset_sha256=sha(TIMING/'subset.json'),
        protocol_sha256=sha(TIMING/'protocol.json'), full_analysis_sha256=sha(OUT/f'completed_analysis_{STAMP}.json'),
        sample_questions=50, population_questions=1800, bootstrap_draws=4000, rows=entries)
    (OUT/f'timing_analysis_{STAMP}.json').write_text(json.dumps(result,indent=2)+'\n')
    flat = [dict(config=e['config'], full_recorded_model_hours=e['full_recorded_model_hours'],
                 model_hours=e['model']['full_seconds']/3600, request_hours=e['request']['full_seconds']/3600,
                 model_speedup=e['model']['speedup'], request_speedup=e['request']['speedup'],
                 error_vs_full=e['fresh_projection_relative_error'],
                 historical_sampling_error=e['historical_sampling_backtest']['relative_error']) for e in entries]
    with (OUT/f'timing_analysis_{STAMP}.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    print(table(['Configuration','Full model h','Projection h','Error','Same-run sampling error','New/old selected timing'],
        [[e['config'],f"{e['full_recorded_model_hours']:.3f}",f"{e['model']['full_seconds']/3600:.3f}",
          f"{e['fresh_projection_relative_error']:+.2%}",f"{e['historical_sampling_backtest']['relative_error']:+.2%}",
          f"{e['component_comparison']['model_seconds']['relative_change']:+.2%}"] for e in entries]))



    return result


def export_results(matrix, timing, destination):
    rows = []
    latest = destination / 'latest/qat_int4_token_reduction_576_seed42/qat_w4'
    for original in matrix['jobs']:
        row = dict(original, experiment='qat_int4_token_reduction_576_seed42', evidence='verified_artifacts')
        row['protocol'] = 'token_reduction_streamed_video_v1' if row['execution'] else 'token_reduction_original_v1'
        marker = EXP / row['marker']
        attempt = EXP / row['attempt']
        row['marker'] = str(marker.resolve().relative_to(ROOT))
        row['attempt'] = str(attempt.resolve().relative_to(ROOT))
        link = latest / row['suite'] / row['config']
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() != marker.parent.resolve():
                raise ValueError(f'Latest link conflicts with catalog: {link}')
        elif link.exists():
            raise FileExistsError(link)
        else:
            link.symlink_to(os.path.relpath(marker.parent.resolve(), link.parent), target_is_directory=True)
        row['latest'] = str(link.relative_to(ROOT)) if ROOT in link.parents else str(link)
        rows.append(row)
    # Historical scores remain explicitly separate from artifact-verified rows.
    history = read(ROOT / 'evaluation/protocols/historical_results.json')
    report_sources = []
    for source in history['sources']:
        path = ROOT / source['path']
        if sha(path) != source['sha256']:
            raise ValueError(f'Historical report changed: {path}')
        report_sources.append(source)
    timing_link = latest.parent / 'videomme_timing50_seed42'
    if not timing_link.exists():
        timing_link.symlink_to(os.path.relpath(TIMING.resolve(), timing_link.parent), target_is_directory=True)
    payload = dict(schema=1, frozen_date='2026-11-25 Asia/Singapore',
        catalog_sha256=sha(ROOT / 'evaluation/protocols/result_catalog.json'),
        verified_rows=len(rows), scored_inferences=sum(r['count'] for r in rows),
        rows=rows, timing=timing, historical_results=history,
        notes=['Historical scores without local predictions are report_only, never certified completions.',
               'Three Video-MMMU resolution_r010 rows use the original policy; deltas and speedups are withheld.',
               'Timing bootstrap intervals describe sampling within one execution; cross-run shifts can be larger.'])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'merged_results.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    flat = [{k:v for k,v in r.items() if not isinstance(v, (dict,list))}
            for r in [*rows, *history['rows']]]
    keys = list(dict.fromkeys(k for r in flat for k in r))
    with (destination / 'merged_results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat)
    lines = ['# Latest verified evaluation results', '',
        f"{len(rows)} full configurations; {sum(r['count'] for r in rows):,} scored inferences. "
        'See merged_results.json for provenance, timing projections and report-only historical results.', '',
        table(['Suite', *CONFIGS], [[suite] + [f"{next(r for r in rows if r['suite']==suite and r['config']==c)['score_percent']:.3f}" for c in CONFIGS] for suite in IMAGES+VIDEOS]), '',
        'Three Video-MMMU resolution_r010 rows use a different input policy. '
        'Scores are included; paired deltas and speedups are omitted.', '']
    (destination / 'SUMMARY.md').write_text('\n'.join(lines))
    print(f"Merged {len(rows)} verified rows, {len(history['rows'])} report-only rows and 10 timing configurations into {destination}")


def main():
    if not __debug__:
        raise SystemExit('Run without -O: audit assertions must remain enabled')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    destination = args.output.resolve()
    global OUT
    OUT = destination / 'analysis'
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = audit_matrix()
    timing = audit_timing()
    export_results(matrix, timing, destination)


if __name__ == '__main__':
    main()
