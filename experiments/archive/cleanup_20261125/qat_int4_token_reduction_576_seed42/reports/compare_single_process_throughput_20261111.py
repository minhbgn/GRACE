"""Compare fresh medium VideoMME timings with pre-parallel TempCompass evidence."""
import datetime as dt
import hashlib
import json
from pathlib import Path

REPORTS = Path(__file__).resolve().parent
EXP = REPORTS.parent
ROOT = EXP.parents[1]
OUT = EXP / 'results_video_retry_gpu1/full'

def read(path):
    return json.loads(path.read_text())

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith('\n')]

comparisons = []
for suffix, prefix in [('050', 926), ('025', 984), ('010', 1032)]:
    config = 'uniform_r' + suffix
    pair = {}
    for suite in ('tempcompass', 'videomme'):
        directory = OUT / 'qat_w4' / suite / config
        marker = directory / 'COMPLETE.json'
        attempt = directory / read(marker)['attempt'] if marker.exists() else sorted(directory.glob('attempt_*'))[-1]
        start_index = prefix if suite == 'videomme' else 0
        telemetry = rows(attempt/'telemetry.jsonl')[start_index:]
        inputs = rows(attempt/'video_inputs.jsonl')[start_index:start_index+len(telemetry)]
        total = lambda key: sum(row[key] for row in telemetry)
        count = len(telemetry)
        started = dt.datetime.strptime(attempt.name, 'attempt_%Y%m%dT%H%M%S_%fZ').replace(tzinfo=dt.timezone.utc).timestamp()
        ended = (attempt/'exit.json').stat().st_mtime if (attempt/'exit.json').exists() else dt.datetime.now(dt.timezone.utc).timestamp()
        pair[suite] = dict(attempt=str(attempt.relative_to(ROOT)), start_index=start_index,
            count=count, approximate_attempt_wall_seconds=ended-started,
            fresh_questions_per_hour=count*3600/(ended-started),
            means={key:total(key)/count for key in ('model_seconds','prefill_seconds',
                'vision_and_embedding_seconds','input_visual_tokens','input_text_tokens','output_text_tokens')},
            sampled_frames_per_question=sum(row['sampling']['sampled_frames'] for row in inputs)/count,
            input_pixels_per_question=sum(row['sampling']['total_pixels'] for row in inputs)/count,
            prefill_input_tokens_per_second=(total('input_visual_tokens')+total('input_text_tokens'))/total('prefill_seconds'))
    comparisons.append(dict(configuration=config, **pair,
        prefill_rate_ratio=pair['videomme']['prefill_input_tokens_per_second']/pair['tempcompass']['prefill_input_tokens_per_second']))

checks = []
for filename, groups in [('protocol.json',['source_code','repair_code']), ('parallel_implementation.json',['code'])]:
    for group in groups:
        for name, expected in read(OUT/filename)[group].items():
            actual = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            assert expected == actual, name
            checks.append(dict(file=name,sha256=actual))

tc = OUT/'qat_w4/tempcompass/uniform_r050'
old = read(tc/read(tc/'COMPLETE.json')['attempt']/'job.json')
current = read(OUT/'qat_w4/videomme/uniform_r050/attempt_20261110T064111_096667Z/job.json')
differences = {key:dict(old=old.get(key),current=current.get(key)) for key in sorted(set(old)|set(current)) if old.get(key)!=current.get(key)}
assert set(differences) == {'suite','gpu','controller_pid','extension_code','resume_provenance','scheduling'}

result = dict(snapshot_sgt=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
    comparisons=comparisons, verified_source_hashes=checks, job_differences=differences,
    live_measurement=read(REPORTS/'live_throughput_20261111.json'),
    wrapper_microbenchmark=read(REPORTS/'checkpoint_overhead_20261111.json'),
    limitations=['Cross-suite tokens/second is a workload-normalized consistency check, not a matched-input A/B equivalence test.',
        'Historical TempCompass wall throughput includes setup/profiling; fresh VideoMME attempt averages include setup/replay time in denominator, but exclude replayed answers from numerator.',
        'One worker per GPU still means two workers sharing host CPU resources.',
        'Wrapper microbenchmark uses dummy generation with real telemetry/sampling payload; it is not an in-worker profiler.'])
(REPORTS/'single_process_throughput_20261111.json').write_text(json.dumps(result,indent=2)+'\n')
for pair in comparisons:
    print(pair['configuration'], json.dumps({suite:{key:pair[suite][key] for key in ('count','fresh_questions_per_hour','sampled_frames_per_question','prefill_input_tokens_per_second','means')} for suite in ('tempcompass','videomme')}))
print('Verified source hashes:',len(checks),'Job-option differences:',list(differences))
