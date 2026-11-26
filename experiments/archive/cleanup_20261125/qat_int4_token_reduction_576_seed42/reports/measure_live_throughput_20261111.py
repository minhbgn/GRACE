"""Observe new durable answers without touching workers or CUDA."""
import argparse
import datetime as dt
import json
from pathlib import Path
import time

SGT = dt.timezone(dt.timedelta(hours=8))
FIELDS = ('model_seconds', 'vision_and_embedding_seconds', 'reduction_seconds',
          'prefill_seconds', 'decode_seconds', 'input_visual_tokens', 'input_text_tokens')

def now():
    return dt.datetime.now(SGT).isoformat()

def summary(state, elapsed):
    events = state['events']
    result = {key: state[key] for key in ('configuration', 'attempt', 'gpu', 'starting_answers')}
    result.update(new_answers=len(events), ending_answers=state['starting_answers']+len(events),
                  fixed_window_questions_per_hour=len(events)*3600/elapsed if elapsed else None)
    if events:
        result['new_answer_means'] = {key: sum(e['telemetry'][key] for e in events)/len(events) for key in FIELDS}
    if len(events) >= 2:
        duration = events[-1]['observed_monotonic'] - events[0]['observed_monotonic']
        result.update(complete_interarrival_cycles=len(events)-1,
            interarrival_elapsed_seconds=duration,
            interarrival_seconds_per_question=duration/(len(events)-1),
            interarrival_questions_per_hour=3600*(len(events)-1)/duration,
            interarrival_model_seconds_per_question=sum(e['telemetry']['model_seconds'] for e in events[1:])/(len(events)-1),
            interarrival_cycle_seconds=[b['observed_monotonic']-a['observed_monotonic'] for a,b in zip(events,events[1:])])
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=600)
    args = parser.parse_args()
    reports = Path(__file__).resolve().parent
    root = reports.parent / 'results_video_retry_gpu1/full/qat_w4/videomme'
    states = []
    started_sgt = now()
    started = time.monotonic()
    for config in ('uniform_r050', 'uniform_r010'):
        attempt = sorted((root/config).glob('attempt_*'))[-1]
        job = json.loads((attempt/'job.json').read_text())
        assert job['scheduling']['parallel_evaluation'] == 1
        file = attempt/'answers.checkpoint.jsonl'
        stream = file.open('rb')
        prefix = stream.read()
        assert not prefix or prefix.endswith(b'\n'), 'Begin between committed answers; rerun monitor'
        states.append(dict(configuration=config, attempt=str(attempt), gpu=job['gpu'],
                           starting_answers=len(prefix.splitlines()), events=[], stream=stream, buffer=b''))
    next_update = 0
    output = reports/'live_throughput_20261111.json'
    while True:
        elapsed = time.monotonic()-started
        for state in states:
            state['buffer'] += state['stream'].read()
            lines = state['buffer'].split(b'\n')
            state['buffer'] = lines.pop()
            for line in lines:
                row = json.loads(line)
                assert row['index'] == state['starting_answers'] + len(state['events'])
                state['events'].append(dict(index=row['index'], observed_sgt=now(),
                    observed_monotonic=time.monotonic(), telemetry=row['telemetry'],
                    video_input=row['video_input']))
        if elapsed >= next_update or elapsed >= args.seconds:
            result = dict(started_sgt=started_sgt, ended_sgt=now(), elapsed_seconds=elapsed,
                target_seconds=args.seconds, complete=elapsed >= args.seconds,
                summaries=[summary(s,elapsed) for s in states],
                events={s['configuration']:s['events'] for s in states},
                method='New complete checkpoint lines observed every second; excludes all pre-existing/replayed answers. Fixed-window counts have boundary uncertainty. Interarrival rates exclude the first partial question; observation resolution about one second.')
            output.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(elapsed_seconds=round(elapsed,1),
                counts={s['configuration']:len(s['events']) for s in states},complete=result['complete'])),flush=True)
            next_update = elapsed + 60
        if elapsed >= args.seconds:
            break
        time.sleep(min(1,args.seconds-elapsed))
    for state in states:
        state['stream'].close()

if __name__ == '__main__':
    main()
