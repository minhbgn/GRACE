"""Stratified video-cluster sampling and paired runtime extrapolation (CPU only)."""
from collections import Counter, defaultdict
import math
import random
import statistics


def phase_configs(value, configurations):
    """Select a nonempty, explicit phase without implicitly adding other jobs."""
    names = [f'{method}_r{round(ratio * 100):03d}' for method, ratio in configurations]
    requested = value.split(',') if value is not None else names
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(names):
        raise ValueError(f'Choose unique configuration names from {names}')
    return [config for name, config in zip(names, configurations) if name in requested]


def select_samples(population, seed=42):
    """Five random videos per workload quintile, one random question per video.

    Each duration has 300 three-question videos. Thus every question has the
    same inclusion probability (1/36), even though videos are the sampling unit.
    Workload uses original visual tokens, never measured time or correctness.
    """
    if Counter(r['duration'] for r in population) != {'short': 900, 'medium': 900}:
        raise ValueError('Expected 900 short and 900 medium questions')
    if len({r['id'] for r in population}) != 1800:
        raise ValueError('Duplicate question IDs')
    rng = random.Random(seed)
    selected, strata = [], []
    for duration in ('short', 'medium'):
        videos = defaultdict(list)
        for row in population:
            if row['duration'] == duration:
                videos[row['video_id']].append(row)
        if len(videos) != 300 or any(len(v) != 3 for v in videos.values()):
            raise ValueError('Expected 300 three-question videos per duration')
        ordered = sorted(videos, key=lambda v: (
            statistics.mean(r['original_visual_tokens'] for r in videos[v]), v))
        for quintile in range(5):
            members = ordered[quintile * 60:(quintile + 1) * 60]
            name = f'{duration}_q{quintile + 1}'
            strata.append(dict(name=name, duration=duration, population_questions=180,
                               population_videos=60, selected_questions=5,
                               min_original_visual_tokens=min(r['original_visual_tokens']
                                   for v in members for r in videos[v]),
                               max_original_visual_tokens=max(r['original_visual_tokens']
                                   for v in members for r in videos[v])))
            for video in rng.sample(members, 5):
                row = rng.choice(sorted(videos[video], key=lambda r: r['id']))
                selected.append(dict(row, stratum=name, expansion_weight=36))
    # Interleave duration groups so a changing clock/host load is less confounded.
    rng.shuffle(selected)
    return selected, strata


def estimate(samples, measurements, baseline=None, bootstrap=4000, seed=43):
    """Weighted expansion and within-stratum paired video bootstrap.

    Intervals describe sampling uncertainty conditional on this machine/run.
    They do not measure repeated-run or GPU contention variability.
    """
    ids = [r['id'] for r in samples]
    if set(ids) != set(measurements) or len(ids) != len(measurements):
        raise ValueError('Timing population differs from the pinned subset')
    if baseline is not None and set(baseline) != set(measurements):
        raise ValueError('Baseline timing population differs')
    for values in (measurements, baseline):
        if values is not None and any(not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError('Timings must be finite and positive')
    groups = defaultdict(list)
    for row in samples:
        groups[row['stratum']].append(row)

    def total(rows, values):
        return sum(r['expansion_weight'] * values[r['id']] for r in rows)

    expanded = total(samples, measurements)
    rng = random.Random(seed)
    totals, speedups = [], []
    for _ in range(bootstrap):
        draw = [rng.choice(group) for group in groups.values() for _ in group]
        value = total(draw, measurements)
        totals.append(value)
        if baseline is not None:
            speedups.append(total(draw, baseline) / value)

    def interval(values):
        values = sorted(values)
        return [values[int(.025 * (len(values) - 1))], values[int(.975 * (len(values) - 1))]]

    result = dict(full_seconds=expanded, mean_seconds=expanded / 1800,
                  full_seconds_ci95=interval(totals),
                  by_duration={d: total([r for r in samples if r['duration'] == d], measurements)
                               for d in ('short', 'medium')})
    if baseline is not None:
        result.update(speedup=total(samples, baseline) / expanded,
                      speedup_ci95=interval(speedups))
    return result


def historical_fit(samples, population, total_tolerance=.10, duration_tolerance=.15,
                   interval_tolerance=.20, bootstrap=4000):
    """Backtest the fixed sample inside one full recorded execution population."""
    if len(population) != 1800 or Counter(r['duration'] for r in population.values()) != {'short': 900, 'medium': 900}:
        raise ValueError('Need 1800 unique historical question timings, 900 per duration')
    if any(not math.isfinite(r['seconds']) or r['seconds'] <= 0 for r in population.values()):
        raise ValueError('Historical times must be positive and finite')
    if any(population[r['id']]['duration'] != r['duration'] for r in samples):
        raise ValueError('Historical duration labels differ')
    values = {r['id']: population[r['id']]['seconds'] for r in samples}
    result = estimate(samples, values, bootstrap=bootstrap)
    full = sum(r['seconds'] for r in population.values())
    error = result['full_seconds'] / full - 1
    by_duration = {}
    for duration in ('short', 'medium'):
        recorded = sum(r['seconds'] for r in population.values() if r['duration'] == duration)
        by_duration[duration] = dict(recorded_seconds=recorded,
            projected_seconds=result['by_duration'][duration],
            relative_error=result['by_duration'][duration] / recorded - 1)
    lo, hi = result['full_seconds_ci95']
    relative_half_width = (hi - lo) / (2 * result['full_seconds'])
    checks = dict(total_error=abs(error) <= total_tolerance,
        duration_errors=all(abs(r['relative_error']) <= duration_tolerance for r in by_duration.values()),
        interval_precision=relative_half_width <= interval_tolerance)
    return dict(recorded_seconds=full, projected_seconds=result['full_seconds'], relative_error=error,
        projected_seconds_ci95=[lo, hi], ci_contains_recorded=lo <= full <= hi,
        relative_ci_half_width=relative_half_width, by_duration=by_duration,
        checks=checks, passed=all(checks.values()))
