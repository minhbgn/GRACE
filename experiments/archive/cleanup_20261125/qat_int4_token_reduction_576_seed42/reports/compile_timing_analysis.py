"""Audit the completed timing sweep and compare it to the selected full runs.

CPU-only; imports the stdlib sampling estimator, never model/evaluation runners.
Writes dated report artifacts here, leaving saved runs unchanged.
Run compile_latest_analysis.py first.
"""
import csv
import json
import math
import sys
from pathlib import Path

from compile_completed_analysis import EXP, ROOT, OUT, CONFIGS, read, rows, sha, table

sys.path.insert(0, str(ROOT))
from evaluation.protocols.video_timing_subset import estimate, historical_fit

TIMING = EXP / 'results_videomme_timing50_seed42_phased_20261123'
STAMP = '20261125'


def main():
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
                      for t, inp, s in zip(ft, fi, fs, strict=True)}
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
            new = sum(s['expansion_weight'] * r[component] for s, r in zip(selected, rr, strict=True))
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


if __name__ == '__main__':
    main()
