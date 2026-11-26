"""CPU-only resolution backtest and comparison to newly measured projections."""
from pathlib import Path

from evaluation.protocols.video_timing_subset import historical_fit


def check(output):
    from evaluation.runners import run_videomme_timing_subset as timing
    matrix = timing.matrix
    subset = matrix.read(output / 'subset.json')
    samples = subset['samples']
    timing.summarize(output)
    projections = {r['config']: r for r in matrix.read(output / 'extrapolation.json')['rows']}
    results = []
    for config in ('resolution_r050', 'resolution_r025', 'resolution_r010'):
        for source in (timing.RECOVERY, timing.retry.SOURCE):
            directory = source / 'qat_w4/videomme' / config
            done = matrix.verified_completion(directory)
            if done:
                break
        if done is None:
            raise ValueError(f'Missing historical resolution reference: {config}')
        attempt = directory / done['attempt']
        telemetry = timing.jsonlines(attempt / 'telemetry.jsonl')
        inputs = timing.jsonlines(attempt / 'video_inputs.jsonl')
        predictions = timing.jsonlines(attempt / 'samples.jsonl')
        if len(telemetry) != 1800 or len(inputs) != 1800 or len(predictions) != 1800:
            raise ValueError('Historical full-population coverage mismatch')
        source_rows = {r['doc_id']: r['doc'] for r in predictions}
        population = {}
        historical_inputs = {}
        for measured, inp in zip(telemetry, inputs, strict=True):
            if measured['index'] != inp['doc_id']:
                raise ValueError('Historical question alignment mismatch')
            doc = source_rows[inp['doc_id']]
            if inp['id'] != doc['question_id']:
                raise ValueError('Historical question ID mismatch')
            population[inp['id']] = dict(seconds=measured['model_seconds'], duration=doc['duration'])
            historical_inputs[inp['id']] = inp['sampling']
        fit = historical_fit(samples, population)
        entry = dict(config=config, historical_source=str(directory),
            marker_sha256=matrix.sha(directory / 'COMPLETE.json'), historical_backtest=fit,
            new_run_status=projections[config]['status'], fresh_vs_full_status='pending')
        fresh = matrix.verified_completion(output / config)
        if fresh:
            new_attempt = output / config / fresh['attempt']
            new_inputs = timing.jsonlines(new_attempt / 'video_inputs.jsonl')
            job = matrix.read(new_attempt / 'job.json')
            old_job = matrix.read(attempt / 'job.json')
            policy_keys = ('attention', 'image_max_pixels', 'video_fps', 'max_new_tokens')
            execution_matches = job['execution'] == old_job.get('execution')
            policy_matches = all(job['policy'][k] == old_job['policy'][k] for k in policy_keys)
            geometry_matches = all(all(r['sampling'][k] == historical_inputs[r['id']][k]
                for k in ('frame_indices', 'height', 'width', 'grid')) for r in new_inputs)
            scheduling = done.get('scheduling', {})
            serial_reference = not scheduling or scheduling.get('standalone_timing') is True
            entry.update(execution_matches=execution_matches, policy_matches=policy_matches,
                         selected_geometry_matches=geometry_matches, historical_serial_reference=serial_reference)
            if execution_matches and policy_matches and geometry_matches and serial_reference:
                projected = projections[config]['model']['full_seconds']
                error = projected / fit['recorded_seconds'] - 1
                duration_checks = {}
                for duration in ('short', 'medium'):
                    duration_checks[duration] = projections[config]['model']['by_duration'][duration] / fit['by_duration'][duration]['recorded_seconds'] - 1
                passed = abs(error) <= .10 and all(abs(v) <= .15 for v in duration_checks.values())
                entry.update(fresh_vs_full_status='within_threshold' if passed else 'review_difference',
                    fresh_relative_error=error, fresh_duration_relative_errors=duration_checks)
            else:
                entry['fresh_vs_full_status'] = 'not_comparable_policy_execution_or_scheduling'
        results.append(entry)
    historical_pass = all(r['historical_backtest']['passed'] for r in results)
    complete = all(r['new_run_status'] == 'complete' for r in results)
    # r050 is the matched serial reference. r025/r010 have known policy differences.
    matched_pass = results[0]['fresh_vs_full_status'] == 'within_threshold'
    recommendation = ('ready_for_manual_review_before_uniform_merge' if historical_pass and complete and matched_pass
                      else 'resolution_runs_pending' if not complete else 'review_failed_or_unmatched_checks')
    report = dict(thresholds=dict(total_relative_error=.10, duration_relative_error=.15,
        historical_ci_relative_half_width=.20), rows=results, recommendation=recommendation,
        historical_sampling_checks_passed=historical_pass, resolution_runs_complete=complete,
        checker_sha256=matrix.sha(Path(__file__)), subset_sha256=matrix.sha(output / 'subset.json'),
        note='Historical backtests use the identical full recorded execution population. '
             'Fresh comparisons require matching execution, policy and selected geometry. '
             'Sampling-only intervals; thresholds are pragmatic screening criteria, not statistical guarantees. '
             'Uniform/merge are never dispatched by this check.')
    matrix.write(output / 'projection_check.json', report)
    lines = ['# Resolution projection check', '', report['note'], '',
        'Screening limits: overall error ≤10%; short and medium errors each ≤15%; '
        'historical bootstrap 95% interval relative half-width ≤20%. '
        'Reference time is summed model time, excluding startup/warmup; request-wall-time '
        'references are unavailable. A serial historical reference is inferred from its '
        'saved scheduling metadata, not independently proven for every instant.', '',
        '| Configuration | Full recorded model h | Historical subset error | Short error | Medium error | CI half-width | Backtest | New subset comparison |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    for row in results:
        fit = row['historical_backtest']
        comparison = row['fresh_vs_full_status']
        if 'fresh_relative_error' in row:
            comparison += f' ({row["fresh_relative_error"]:+.2%})'
        lines.append('| ' + ' | '.join([row['config'], f'{fit["recorded_seconds"]/3600:.3f}',
            f'{fit["relative_error"]:+.2%}', f'{fit["by_duration"]["short"]["relative_error"]:+.2%}',
            f'{fit["by_duration"]["medium"]["relative_error"]:+.2%}', f'{fit["relative_ci_half_width"]:.2%}',
            'pass' if fit['passed'] else 'review', comparison]) + ' |')
    lines += ['', 'Recommendation: **' + recommendation + '**.', '',
        'Review this report before explicitly starting the next phase. Keep the same subset '
        'for uniform/merge if accepted. If it fails, diagnose runtime shifts or workload gaps '
        'before changing the sample; trying seeds until one passes would overstate validation.']
    (output / 'projection_check.md').write_text('\n'.join(lines) + '\n')
    print(output / 'projection_check.md')
    print('Recommendation:', recommendation)
    return report
