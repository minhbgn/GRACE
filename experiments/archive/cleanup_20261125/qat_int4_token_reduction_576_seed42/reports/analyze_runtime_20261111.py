"""CPU-only audit of saved timing evidence; never imports the evaluation runtime."""
import datetime as dt
import difflib
import hashlib
import json
from pathlib import Path
import statistics

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
OUT = EXP / 'results_video_retry_gpu1/full'
REPORTS = EXP / 'reports'
FIELDS = ('model_seconds', 'vision_and_embedding_seconds', 'reduction_seconds',
          'prefill_seconds', 'decode_seconds', 'input_visual_tokens',
          'input_text_tokens', 'output_text_tokens')

def read(path):
    return json.loads(path.read_text())

def rows(path):
    if not path.exists():
        return []
    # A live process may be midway through its last append.
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True)
            if line.endswith('\n')]

def means(records):
    return {key: statistics.mean(row[key] for row in records) for key in FIELDS}

def attempt(suite, config, name=None):
    directory = OUT / 'qat_w4' / suite / config
    return directory / (name or read(directory / 'COMPLETE.json')['attempt'])

source_checks = []
for file, groups in [('protocol.json', ['source_code', 'repair_code']),
                     ('parallel_implementation.json', ['code'])]:
    manifest = read(OUT / file)
    for group in groups:
        for name, expected in manifest[group].items():
            actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            assert actual == expected, name
            source_checks.append(dict(manifest=file, group=group, file=name, sha256=actual))

completed = []
tempcompass = []
verified_artifacts = 0
for marker in sorted((OUT / 'qat_w4').glob('*/*/COMPLETE.json')):
    done = read(marker)
    directory = marker.parent / done['attempt']
    for name, checksum in done['sha256'].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == checksum, directory / name
        verified_artifacts += 1
    records = rows(directory / 'telemetry.jsonl')
    assert len(records) == done['count']
    assert [r['index'] for r in records] == list(range(len(records)))
    if marker.parent.parent.name == 'tempcompass':
        job = read(directory / 'job.json')
        assert 'scheduling' not in job and 'extension_code' not in job
        assert done['count'] == 1580 and done['video_adaptation']['adapted_questions'] == 0
        start = dt.datetime.strptime(directory.name, 'attempt_%Y%m%dT%H%M%S_%fZ').replace(tzinfo=dt.timezone.utc)
        wall = (directory / 'exit.json').stat().st_mtime - start.timestamp()
        tempcompass.append(dict(config=marker.parent.name, attempt=str(directory.relative_to(ROOT)),
            score_percent=done['score_percent'], count=done['count'], sample_identity=done['sample_identity'],
            actual_retention=done['telemetry']['actual_retention'], means=means(records),
            approximate_wall_seconds=wall,
            fixed_output_profile=done['telemetry']['fixed_output_profile']))
    completed.append(dict(suite=marker.parent.parent.name, config=marker.parent.name, count=done['count']))
assert len({r['sample_identity'] for r in tempcompass}) == 1

legacy = attempt('videomme', 'uniform_r050', 'attempt_20261109T023755_583079Z')
parallel = attempt('videomme', 'uniform_r050', 'attempt_20261109T063011_815670Z')
old_rows = rows(legacy / 'telemetry.jsonl')
parallel_rows = rows(parallel / 'telemetry.jsonl')[:len(old_rows)]
old_inputs = rows(legacy / 'video_inputs.jsonl')
parallel_inputs = rows(parallel / 'video_inputs.jsonl')[:len(old_rows)]
assert len(old_rows) == 443 and old_inputs == parallel_inputs
for a, b in zip(old_rows, parallel_rows):
    for key in ('index', 'input_visual_tokens', 'input_text_tokens', 'output_text_tokens',
                'generated_token_ids', 'video_adaptation'):
        assert a[key] == b[key], (a['index'], key)
same_method = dict(count=len(old_rows), serial_attempt=str(legacy.relative_to(ROOT)),
    parallel_attempt=str(parallel.relative_to(ROOT)), serial=means(old_rows), parallel=means(parallel_rows),
    slowdown={key: means(parallel_rows)[key]/means(old_rows)[key] for key in FIELDS[:5]},
    sampling_and_generated_tokens_identical=True)

baseline_dir = attempt('videomme', 'baseline_r100')
resolution_dir = attempt('videomme', 'resolution_r050')
baseline = rows(baseline_dir / 'telemetry.jsonl')
resolution = rows(resolution_dir / 'telemetry.jsonl')
baseline_inputs = rows(baseline_dir / 'video_inputs.jsonl')
resolution_inputs = rows(resolution_dir / 'video_inputs.jsonl')
segments = []
live = []
for config, name, prefix in [
    ('uniform_r050', 'attempt_20261110T064111_096667Z', 926),
    ('uniform_r025', 'attempt_20261110T064111_150483Z', 984),
    ('uniform_r010', 'attempt_20261111T040346_447239Z', 1032),
]:
    directory = attempt('videomme', config, name)
    records = rows(directory / 'telemetry.jsonl')
    inputs = rows(directory / 'video_inputs.jsonl')[:len(records)]
    for index, (b, u) in enumerate(zip(baseline_inputs, inputs)):
        assert b['id'] == u['id'] and b['doc_id'] == u['doc_id']
        assert b['sampling']['frame_indices'] == u['sampling']['frame_indices']
        ri = resolution_inputs[index]
        assert b['id'] == ri['id'] and b['doc_id'] == ri['doc_id']
        assert b['sampling']['frame_indices'] == ri['sampling']['frame_indices']
        ba = baseline[index]['video_adaptation']
        ua = records[index]['video_adaptation']
        for key in ('adaptive_baseline_visual_tokens', 'adaptive_baseline_height', 'adaptive_baseline_width'):
            assert ba[key] == ua[key]
            assert ba[key] == resolution[index]['video_adaptation'][key]
    fresh = records[prefix:]
    segments.append(dict(config=config, start_index=prefix, stop_index_exclusive=len(records),
        questions=len(fresh), attempt=str(directory.relative_to(ROOT)),
        uniform=means(fresh), baseline=means(baseline[prefix:len(records)]),
        resolution50=means(resolution[prefix:len(records)]),
        checkpoint_replayed_prefix=prefix, prefix_means=means(records[:prefix]),
        same_frames_and_adaptive_baseline=True))
    checkpoint = directory / 'answers.checkpoint.jsonl'
    live.append(dict(config=config, answers=len(rows(checkpoint)),
        checkpoint_bytes=checkpoint.stat().st_size,
        last_answer_sgt=dt.datetime.fromtimestamp(checkpoint.stat().st_mtime,
            dt.timezone(dt.timedelta(hours=8))).isoformat(),
        complete=(directory.parent / 'COMPLETE.json').exists()))

temp_lookup = {row['config']: row for row in tempcompass}
temp_pairs = []
for suffix in ('050', '025', '010'):
    u, r = (temp_lookup[f'{method}_r{suffix}'] for method in ('uniform', 'resolution'))
    temp_pairs.append(dict(nominal_retention=int(suffix)/100,
        uniform_to_resolution_model_ratio=u['means']['model_seconds']/r['means']['model_seconds'],
        component_differences={key: u['means'][key]-r['means'][key] for key in FIELDS[:5]},
        visual_token_ratio=u['means']['input_visual_tokens']/r['means']['input_visual_tokens'],
        score_difference_pp=u['score_percent']-r['score_percent']))

result = dict(snapshot_sgt=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
    source_checks=source_checks, verified_completion_artifacts=verified_artifacts,
    completed_retry_jobs=completed, tempcompass=tempcompass, tempcompass_pairs=temp_pairs,
    videomme_same_method_serial_parallel=same_method, fresh_serial_segments=segments,
    answer_snapshots=live,
    caveats=['Per-question model timers exclude video decoding/preprocessing, harness and answer fsync.',
        'Completed Uniform25 aggregates mix 984 historical answers with 816 fresh serial answers.',
        'Segment comparisons match questions but were measured on different dates, not a randomized A/B.',
        'Approximate wall times use attempt-name start and exit.json filesystem modification time.',
        'A live answer may finish after this snapshot. No workload or policy was changed.'])
(REPORTS / 'runtime_analysis_20261111.json').write_text(json.dumps(result, indent=2)+'\n')
old_path = ROOT / 'evaluation/runners/run_token_reduction_video_retry.py'
new_path = ROOT / 'evaluation/runners/run_parallel_video_evaluation.py'
(REPORTS / 'serial_vs_parallel_controller_20261111.diff').write_text(''.join(difflib.unified_diff(
    old_path.read_text().splitlines(keepends=True), new_path.read_text().splitlines(keepends=True),
    fromfile=str(old_path.relative_to(ROOT)), tofile=str(new_path.relative_to(ROOT)))))
print(json.dumps(dict(source_files_verified=len(source_checks), completion_artifacts_verified=verified_artifacts,
    tempcompass_pairs=temp_pairs, answer_snapshots=live), indent=2))
