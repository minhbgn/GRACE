#!/usr/bin/env python3
"""Measure saved video token rates, with optional partial-population rates (CPU)."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
from pathlib import Path
import statistics
import sys
from zoneinfo import ZoneInfo

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.paths import GRACE_ROOT
from evaluation.protocols.video_protocol import SUITES, digest, load_suite
from evaluation.protocols.video_population import VIDEOMME_SHORT_MEDIUM
from evaluation.runners.run_video_matrix import Experiment, completed, read, sha, summary_from_rows
from evaluation.workers.video_reuse import compatible_policy

DEFAULT = GRACE_ROOT / 'experiments/released_bf16_init_32k_b128_p589824_t4096_seed42'
TOKEN_KEYS = ('input_visual_tokens', 'input_text_tokens', 'input_tokens',
              'output_text_tokens', 'total_text_tokens', 'total_tokens')


def prediction_rows(path, allow_partial=False):
    """A running worker can be observed during its final JSONL write."""
    rows, partial = [], False
    with path.open('rb') as stream:
        snapshot = stream.read(os.fstat(stream.fileno()).st_size)
        for line in snapshot.decode().splitlines(keepends=True):
            if allow_partial and not line.endswith('\n'):
                partial = True
                break
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if not allow_partial or line.endswith('\n'):
                    raise
                partial = True
    return rows, partial


def measured_rows(attempt, job, rows=None):
    if rows is None:
        rows, _ = prediction_rows(attempt / 'predictions.jsonl')
    output = []
    for row, source in zip(rows, job['rows']):
        duration = row['sampling']['duration_seconds']
        elapsed = row['elapsed_seconds']
        if any(not math.isfinite(x) or x <= 0 for x in (duration, elapsed)):
            raise ValueError(f'Invalid time denominator: {attempt}/{row["id"]}')
        sampling = row['sampling']
        if not math.isclose(duration, sampling['source_frames'] / sampling['source_fps'], rel_tol=1e-12):
            raise ValueError('Saved duration disagrees with source frame count/rate')
        tokens = dict(row['tokens'])
        tokens['total_text_tokens'] = tokens['input_text_tokens'] + tokens['output_text_tokens']
        tokens['total_tokens'] = tokens['input_tokens'] + tokens['output_text_tokens']
        item = dict(method=job['method'], suite=job['suite'], id=row['id'],
                    video_id=source['video_id'], category=row['category'],
                    duration_seconds=duration, elapsed_seconds=elapsed, **tokens)
        for denominator, seconds in [('video', duration), ('elapsed', elapsed)]:
            item.update({f'{k}_per_{denominator}_second': tokens[k] / seconds for k in TOKEN_KEYS})
        output.append(item)
    return output


def validate_partial_rows(rows, job):
    """Validate the saved ordered prefix without implying full-run certification."""
    source_rows = job['rows']
    if not rows or len(rows) > len(source_rows):
        raise ValueError('Invalid partial population size')
    if [r['id'] for r in rows] != [r['id'] for r in source_rows[:len(rows)]]:
        raise ValueError('Partial predictions are not a unique ordered source prefix')
    for row, source in zip(rows, source_rows):
        if any(row[k] != source[k] for k in ('gold', 'options', 'question_type', 'category')):
            raise ValueError('Partial source metadata mismatch')
        if row['source_identity'] != digest(source):
            raise ValueError('Partial source identity mismatch')
        inputs = {k: row['tokens'][k] for k in TOKEN_KEYS[:3]}
        if row['input_identity'] != digest(dict(source=source, sampling=row['sampling'],
                tokens=inputs, input_ids_sha256=row['input_ids_sha256'])):
            raise ValueError('Partial input identity mismatch')
        cap = job['protocol']['max_new_tokens']
        if row['generation']['max_new_tokens'] != cap:
            raise ValueError('Partial generation cap mismatch')
        if row['tokens']['input_tokens'] + cap > row['context_limit']:
            raise ValueError('Partial input exceeded context')
    # Validates scores, token identities/accounting and generation telemetry.
    summary = summary_from_rows(rows)
    if summary['generation_cutoffs']:
        raise ValueError('Partial population contains non-EOS generations')


def aggregate(rows):
    """Primary rate is ratio of sums; also retain the mean of per-question rates."""
    n = len(rows)
    result = dict(questions=n, unique_videos=len({r['video_id'] for r in rows}),
                  video_seconds=sum(r['duration_seconds'] for r in rows),
                  elapsed_seconds=sum(r['elapsed_seconds'] for r in rows),
                  mean_video_seconds=statistics.mean(r['duration_seconds'] for r in rows))
    for key in TOKEN_KEYS:
        total = sum(r[key] for r in rows)
        result[f'{key}_total'] = total
        result[f'{key}_mean'] = total / n
        for denominator, seconds in [('video', result['video_seconds']),
                                     ('elapsed', result['elapsed_seconds'])]:
            name = f'{key}_per_{denominator}_second'
            values = sorted(r[name] for r in rows)
            result[name] = total / seconds
            result[name + '_question_mean'] = statistics.mean(values)
            result[name + '_median'] = statistics.median(values)
            result[name + '_p95'] = values[math.ceil(.95 * n) - 1]
    return result


def audit_pair(pair, policy, expected, smoke=False, partial_over=None):
    attempts = sorted(pair.glob('attempt_*'))
    entry = dict(method=pair.parent.name, suite=pair.name, status='incomplete',
                 expected_questions=expected, path=str(pair), attempts=[], measurement_scope='none')
    marker = pair / 'COMPLETE.json'
    for attempt in attempts:
        if not (attempt / 'job.json').exists():
            continue  # A live parent may have just created the attempt directory.
        job = read(attempt / 'job.json')
        if job['protocol'] != policy:
            raise ValueError(f'Attempt policy mismatch: {attempt}')
        if (job['method'], job['suite']) != (entry['method'], entry['suite']):
            raise ValueError(f'Job method/suite mismatch: {attempt}')
        if len(job['rows']) != expected or len({r['id'] for r in job['rows']}) != expected:
            raise ValueError(f'Incorrect source population: {attempt}')
        predictions = attempt / 'predictions.jsonl'
        observed, partial = prediction_rows(predictions, allow_partial=True) if predictions.exists() else ([], False)
        record = dict(attempt=attempt.name, path=str(attempt), observed_questions=len(observed),
                      expected_questions=expected, status='incomplete', partial_final_line=partial,
                      summary_present=(attempt / 'summary.json').exists())
        for filename in ('failure.json', 'blocked.json'):
            if (attempt / filename).exists():
                record[filename[:-5]] = read(attempt / filename)
        entry['attempts'].append(record)
    if not marker.exists():
        entry['observed_questions_latest_attempt'] = entry['attempts'][-1]['observed_questions'] if entry['attempts'] else 0
        entry['coverage_fraction'] = entry['observed_questions_latest_attempt'] / expected
        if partial_over is None or entry['coverage_fraction'] <= partial_over:
            return entry, []
        # Use the last already-read snapshot, never concatenate retries or reread
        # a growing predictions file after deciding whether it exceeds the cutoff.
        try:
            validate_partial_rows(observed, job)
        except ValueError as error:
            entry.update(measurement_scope='invalid_partial', partial_validation_error=str(error),
                         attempt=attempt.name, source_population_sha256=digest(job['rows']),
                         prediction_snapshot_sha256=digest(observed), job_sha256=sha(attempt / 'job.json'),
                         non_eos_questions=[r.get('id') for r in observed
                                            if r.get('generation', {}).get('finish_reason') != 'eos'])
            return entry, []
        measured = measured_rows(attempt, job, observed)
        entry.update(measurement_scope='partial', attempt=attempt.name,
                     source_population_sha256=digest(job['rows']),
                     prediction_snapshot_sha256=digest(observed),
                     job_sha256=sha(attempt / 'job.json'),
                     metrics=aggregate(measured))
        entry['categories'] = {c: aggregate([r for r in measured if r['category'] == c])
                               for c in sorted({r['category'] for r in measured})}
        return entry, measured
    done = read(marker)
    attempt = pair / done['attempt']
    job = read(attempt / 'job.json')
    required = {'job.json', 'predictions.jsonl', 'summary.json', 'processor.json'}
    if not required <= set(done['sha256']):
        raise ValueError(f'Incomplete completion hashes: {marker}')
    # Reuse the evaluator's full scoring, ordered coverage, EOS, context, token,
    # input identity, summary and artifact-hash verification. No weights loaded.
    completed(pair, job['rows'], policy)
    measured = measured_rows(attempt, job)
    if len(measured) != expected:
        raise ValueError(f'Incomplete measurement population: {attempt}')
    entry.update(status='complete', measurement_scope='full', coverage_fraction=1.0,
                 attempt=done['attempt'], smoke=smoke,
                 source_population_sha256=digest(job['rows']),
                 completion_marker_sha256=sha(marker), artifact_sha256=done['sha256'],
                 metrics=aggregate(measured))
    entry['categories'] = {c: aggregate([r for r in measured if r['category'] == c])
                           for c in sorted({r['category'] for r in measured})}
    for record in entry['attempts']:
        if record['attempt'] == done['attempt']:
            record['status'] = 'complete'
    return entry, measured


def save_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def table(entries, denominator='video'):
    lines = [f'| Model | Task | Questions | Visual / {denominator} s | Text input / {denominator} s | '
             f'Total input / {denominator} s | Text output / {denominator} s | Status |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for entry in entries:
        metrics = entry.get('metrics')
        rates = [f'{metrics[f"{key}_per_{denominator}_second"]:.3f}' if metrics else '—'
                 for key in TOKEN_KEYS[:4]]
        count = str(metrics['questions']) if entry['status'] == 'complete' else (
            f'{entry["observed_questions_latest_attempt"]}/{entry["expected_questions"]} '
            f'({100 * entry["coverage_fraction"]:.1f}%)')
        status = 'incomplete (partial rates)' if entry['measurement_scope'] == 'partial' else entry['status']
        if entry['measurement_scope'] == 'invalid_partial':
            reason = 'non-EOS' if 'non-EOS' in entry['partial_validation_error'] else 'validation failed'
            status = f'incomplete ({reason}; rates withheld)'
        suite = entry['suite'] + (' (short + medium)' if entry.get('population') == 'short_medium' else '')
        lines.append('| ' + ' | '.join([entry['method'], suite, count, *rates, status]) + ' |')
    return lines


MEAN_KEYS = ('input_visual_tokens', 'input_text_tokens', 'output_text_tokens',
             'total_text_tokens', 'input_tokens', 'total_tokens')
QUESTION_KEYS = ('question_only_tokens', 'question_with_options_tokens', 'task_prompt_tokens')


def question_payloads(suite, raw, source):
    """Keep question stems, answer choices and evaluation instructions distinct."""
    if suite == 'tempcompass':
        with_options = raw['question']
        stem = re.split(r'(?m)^[A-J]\.\s+', with_options, maxsplit=1)[0].rstrip()
    else:
        stem = raw['question']
        with_options = stem
        if source['question_type'] == 'multiple-choice':
            with_options += '\n' + '\n'.join(f'{key}. {value}' for key, value in source['options'].items())
    return stem, with_options, source['prompt']


def add_question_metrics(metrics, rows):
    for key in QUESTION_KEYS:
        metrics[key + '_total'] = sum(row[key] for row in rows)
        metrics[key + '_mean'] = metrics[key + '_total'] / len(rows)


def audit_question_lengths(experiment, matrices, entries, questions, historical):
    """CPU-only annotation tokenization; no video decoding or model loading."""
    import pandas as pd
    from tokenizers import Tokenizer
    exp = Experiment(experiment / 'experiment.json')
    examples, provenance = [], {}
    for suite, (folder, expected) in SUITES.items():
        rows, info = load_suite(suite)
        if info['status'] == 'unavailable':
            continue
        manifests = [read(matrix / 'datasets' / (suite + '.json')) for matrix in matrices
                     if (matrix / 'datasets' / (suite + '.json')).exists()]
        if not manifests or any(m['annotation_sha256'] != info['annotation_sha256'] for m in manifests):
            raise ValueError(f'Question annotations differ from frozen evaluation data: {suite}')
        raw = pd.concat([pd.read_parquet(path) for path in sorted(info['annotation_sha256'])], ignore_index=True)
        if len(rows) != expected or len(raw) != expected:
            raise ValueError(f'Unexpected question population: {suite}')
        for source, annotation in zip(rows, raw.to_dict('records')):
            examples.append(dict(suite=suite, id=source['id'], category=source['category'],
                                 payloads=question_payloads(suite, annotation, source)))
        provenance[suite] = info['annotation_sha256']
    payloads = [text for example in examples for text in example['payloads']]
    tokenizer_paths, methods_by_hash = {}, {}
    for method, path in exp.paths.items():
        file = path / 'tokenizer.json'
        checksum = sha(file)
        tokenizer_paths[checksum] = file
        methods_by_hash.setdefault(checksum, []).append(method)
    reference = exp.paths['vanilla_2b'] / 'tokenizer.json'
    reference_hash = sha(reference)
    counts = None
    for checksum, path in tokenizer_paths.items():
        tokenizer = Tokenizer.from_file(str(path))
        lengths = [len(encoding.ids) for encoding in tokenizer.encode_batch(payloads, add_special_tokens=False)]
        if counts is None:
            counts = lengths
        elif lengths != counts:
            raise ValueError('Model tokenizers produce different question lengths; report per-tokenizer groups instead')
    lookup = {}
    for index, example in enumerate(examples):
        example.update(zip(QUESTION_KEYS, counts[3 * index:3 * index + 3]))
        del example['payloads']
        lookup[example['suite'], example['id']] = example
    measured_by_pair = {}
    for question in questions:
        source = lookup[question['suite'], question['id']]
        question.update({key: source[key] for key in QUESTION_KEYS})
        measured_by_pair.setdefault((question['method'], question['suite']), []).append(question)
    for entry in entries:
        if 'metrics' not in entry:
            continue
        observed = measured_by_pair[entry['method'], entry['suite']]
        add_question_metrics(entry['metrics'], observed)
        for category, metrics in entry['categories'].items():
            add_question_metrics(metrics, [q for q in observed if q['category'] == category])
    for entry in historical:
        if 'metrics' not in entry:
            continue
        job = read(Path(entry['path']) / entry['attempt'] / 'job.json')
        observed = [lookup['videomme', row['id']] for row in job['rows'][:entry['metrics']['questions']]]
        add_question_metrics(entry['metrics'], observed)
        for category, metrics in entry['categories'].items():
            add_question_metrics(metrics, [q for q in observed if q['category'] == category])
    groups = []
    baseline = {entry['suite']: entry for entry in entries if entry['method'] == 'vanilla_2b'}
    legacy_baseline = next((e for e in historical if e['method'] == 'vanilla_2b'), None)
    for suite in SUITES:
        for category in (('short', 'medium', 'long') if suite == 'videomme' else (None,)):
            selected = [e for e in examples if e['suite'] == suite and (category is None or e['category'] == category)]
            if not selected:
                continue
            metrics = dict(suite=suite, category=category, questions=len(selected))
            add_question_metrics(metrics, selected)
            reference_entry = legacy_baseline if category == 'long' else baseline.get(suite)
            if reference_entry and reference_entry['status'] == 'complete':
                full = reference_entry.get('categories', {}).get(category) if category else reference_entry['metrics']
                if full and full['questions'] == len(selected):
                    metrics['full_text_input_mean_reference'] = full['input_text_tokens_mean']
                    metrics['video_chat_residual_mean_reference'] = (full['input_text_tokens_mean']
                                                                    - metrics['task_prompt_tokens_mean'])
                    metrics['reference_model'] = 'vanilla_2b'
            groups.append(metrics)
    return dict(reference_tokenizer=str(reference), reference_tokenizer_sha256=reference_hash,
                tokenizers_verified=methods_by_hash, all_payload_counts_equal=True,
                annotations_sha256=provenance, groups=groups, questions=examples)


def question_length_table(groups):
    lines = ['| Benchmark | Video length | Questions | Mean question only | Mean question + options | '
             'Mean task prompt | Mean full text input (Vanilla 2B) | Video/chat residual |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for group in groups:
        values = [f'{group[key + "_mean"]:,.2f}' for key in QUESTION_KEYS]
        values += [f'{group[key]:,.2f}' if key in group else '—' for key in
                   ('full_text_input_mean_reference', 'video_chat_residual_mean_reference')]
        lines.append('| ' + ' | '.join([group['suite'], group['category'] or 'all',
                                      str(group['questions']), *values]) + ' |')
    return lines


def mean_cells(metrics):
    return [f'{metrics[key + "_mean"]:,.2f}' if metrics else '—' for key in MEAN_KEYS]


def mean_status(entry):
    if entry['measurement_scope'] == 'invalid_partial':
        return 'incomplete (validation failed; means withheld)'
    return 'incomplete (observed prefix)' if entry['measurement_scope'] == 'partial' else entry['status']


def average_table(entries):
    lines = ['| Model | Benchmark | Questions | Mean question only | Mean visual input | Mean text input | Mean text output | '
             'Mean total text | Mean total input | Mean all tokens | Status |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for entry in entries:
        metrics = entry.get('metrics')
        count = metrics['questions'] if metrics else entry.get('observed_questions_latest_attempt', 0)
        suite = entry['suite'] + (' (short + medium)' if entry.get('population') == 'short_medium' else '')
        question_mean = (f'{metrics["question_only_tokens_mean"]:,.2f}'
                         if metrics and 'question_only_tokens_mean' in metrics else '—')
        lines.append('| ' + ' | '.join([entry['method'], suite, f'{count}/{entry["expected_questions"]}',
                                      question_mean, *mean_cells(metrics), mean_status(entry)]) + ' |')
    return lines


def duration_average_table(entries, *, historical_long=False):
    lines = ['| Model | Video length | Questions | Mean question only | Mean visual input | Mean text input | Mean text output | '
             'Mean total text | Mean total input | Mean all tokens | Status |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for entry in entries:
        if entry['suite'] != 'videomme':
            continue
        for category in (('long',) if historical_long else ('short', 'medium', 'long')):
            metrics = entry.get('categories', {}).get(category)
            excluded = category == 'long' and entry.get('population') == 'short_medium'
            if historical_long and not metrics:
                continue
            count = str(metrics['questions']) + '/900' if metrics else ('excluded' if excluded else '—/900')
            status = ('excluded from current protocol' if excluded else mean_status(entry))
            if not metrics and not excluded and entry['measurement_scope'] != 'invalid_partial':
                status = 'no validated observations'
            if historical_long:
                status = 'historical; excluded from current scores; ' + status
            question_mean = (f'{metrics["question_only_tokens_mean"]:,.2f}'
                             if metrics and 'question_only_tokens_mean' in metrics else '—')
            lines.append('| ' + ' | '.join([entry['method'], category, count,
                                          question_mean, *mean_cells(metrics), status]) + ' |')
    return lines


def collect_historical_videomme(matrices, methods, selected_policy, partial_over):
    """Audit original-population evidence separately; never mix it into current means."""
    choices = {method: [] for method in methods}
    for matrix in matrices:
        manifest = read(matrix / 'protocol.json')
        path = matrix / 'datasets/videomme.json'
        if manifest['policy'].get('videomme_population') or not path.exists():
            continue
        if not compatible_policy(manifest['policy'], selected_policy):
            continue
        dataset = read(path)
        expected = expected_population('videomme', dataset, manifest['policy'])
        for method in methods:
            entry, _ = audit_pair(matrix / method / 'videomme', manifest['policy'], expected,
                                  partial_over=partial_over)
            if entry.get('source_population_sha256', dataset['sample_identity']) != dataset['sample_identity']:
                raise ValueError(f'Historical VideoMME dataset mismatch: {method}')
            entry.update(matrix=str(matrix), population='full', report_scope='historical_excluded',
                         policy_version=manifest['policy']['version'])
            choices[method].append(entry)
    def rank(entry):
        count = entry.get('metrics', {}).get('questions', entry.get('observed_questions_latest_attempt', 0))
        return (entry['status'] == 'complete', count, entry.get('attempt', ''))
    return [max(choices[method], key=rank) for method in methods if choices[method]]


def expected_population(suite, dataset, policy):
    expected = SUITES[suite][1]
    if suite == 'videomme' and policy.get('videomme_population'):
        if (policy['videomme_population'] != VIDEOMME_SHORT_MEDIUM
                or dataset.get('population') != VIDEOMME_SHORT_MEDIUM):
            raise ValueError('Unknown or inconsistent VideoMME subset')
        expected = VIDEOMME_SHORT_MEDIUM['selected_questions']
    if dataset['questions'] != expected:
        raise ValueError(f'Unexpected selected suite size: {suite}')
    return expected


def collect_runs(matrices, methods, partial_over):
    """Choose one intact attempt per pair across compatible disjoint matrices."""
    manifests = {matrix: read(matrix / 'protocol.json') for matrix in matrices}
    first = manifests[matrices[0]]
    if any(m['limit'] is not None or not compatible_policy(m['policy'], first['policy'])
           or m['experiment'] != first['experiment'] for m in manifests.values()):
        raise ValueError('Matrices must share a compatible full-run policy and experiment')
    suites = sorted({p.stem for matrix in matrices for p in (matrix / 'datasets').glob('*.json')})
    if not suites:
        raise ValueError('No saved suite manifests')
    entries, questions, candidates = [], [], []
    for method in methods:
        for suite in suites:
            choices, frozen = [], None
            reference = next((read(matrix / 'datasets' / (suite + '.json')) for matrix in matrices
                              if (matrix / 'datasets' / (suite + '.json')).exists()), None)
            selected_expected = expected_population(suite, reference, first['policy'])
            for matrix in matrices:
                manifest_path = matrix / 'datasets' / (suite + '.json')
                if not manifest_path.exists():
                    continue
                dataset = read(manifest_path)
                if dataset.get('status') != 'ready':
                    continue
                expected = expected_population(suite, dataset, manifests[matrix]['policy'])
                if expected != selected_expected:
                    continue  # Never mix original 2700-question VideoMME into the 1800-question table.
                if frozen is not None and frozen != dataset['sample_identity']:
                    raise ValueError(f'Matrices have different frozen dataset identities: {suite}')
                frozen = dataset['sample_identity']
                entry, measured = audit_pair(matrix / method / suite, manifests[matrix]['policy'], expected,
                                             partial_over=partial_over)
                if entry.get('source_population_sha256', frozen) != frozen:
                    raise ValueError(f'Frozen dataset identity mismatch: {method}/{suite}')
                entry.update(matrix=str(matrix), policy_version=manifests[matrix]['policy']['version'],
                             population=dataset.get('population', {}).get('name', 'full'))
                candidates.append(entry)
                choices.append((entry, measured))
            if not choices:
                continue
            def rank(candidate):
                entry = candidate[0]
                count = entry['metrics']['questions'] if 'metrics' in entry else entry['observed_questions_latest_attempt']
                return (entry['status'] == 'complete', count, entry.get('attempt', ''))
            entry, measured = max(choices, key=rank)
            entries.append(entry)
            for row in measured:
                row.update(run_status=entry['status'], measurement_scope=entry['measurement_scope'],
                           coverage_fraction=entry['coverage_fraction'], attempt=entry['attempt'],
                           matrix=entry['matrix'], population=entry['population'])
            questions.extend(measured)
    return entries, questions, candidates, first['policy']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=DEFAULT)
    parser.add_argument('--matrix', type=Path, action='append',
                        help='Full/ directory; repeat to combine compatible concurrent matrices')
    parser.add_argument('--output-prefix', type=Path, required=True)
    parser.add_argument('--include-partial-over', type=float,
                        help='Also measure incomplete latest attempts with coverage strictly above this fraction (e.g. 0.5)')
    args = parser.parse_args()
    if args.include_partial_over is not None and not 0 <= args.include_partial_over < 1:
        parser.error('--include-partial-over must be in [0, 1)')
    experiment = args.experiment.resolve()
    defaults = [experiment / 'video_evaluation' / name / 'full' for name in
                ('video_resilient_v3_f185cede8bfd', 'concurrent_grace_w4_onwards')]
    defaults = [p for p in defaults if (p / 'protocol.json').exists()]
    legacy = experiment / 'video_evaluation/video_unsaturated_v2_3a74c7231e27/full'
    if defaults and (legacy / 'protocol.json').exists():
        defaults.append(legacy)
    matrices = list(dict.fromkeys(p.resolve() for p in (args.matrix or defaults or
                    [experiment / 'video_evaluation/video_unsaturated_v2_3a74c7231e27/full'])))
    methods = list(read(experiment / 'experiment.json')['models'])
    entries, questions, candidates, policy = collect_runs(matrices, methods, args.include_partial_over)
    historical_videomme = (collect_historical_videomme(matrices, methods, policy, args.include_partial_over)
                          if policy.get('videomme_population') else [])
    question_lengths = audit_question_lengths(experiment, matrices, entries, questions, historical_videomme)

    # Retry matrices can expose the same underlying methods through symlinks.
    # Record the aliases but do not count them as independent completed runs.
    aliases = []
    for path in sorted((experiment / 'video_evaluation').rglob('protocol.json')):
        if path.parent.resolve() in matrices or read(path)['policy'] != policy:
            continue
        for matrix in matrices:
            shared = [m for m in methods if (path.parent / m).exists()
                      and (path.parent / m).resolve() == (matrix / m).resolve()]
            if shared:
                aliases.append(dict(matrix=str(path.parent), canonical_matrix=str(matrix), shared_methods=shared))

    # Historical smoke completions are useful inventory, never full benchmarks.
    smoke_entries = []
    for path in sorted((experiment / 'video_evaluation').rglob('protocol.json')):
        smoke_manifest = read(path)
        if smoke_manifest['limit'] is None:
            continue
        for marker in sorted(path.parent.glob('*/*/COMPLETE.json')):
            entry, _ = audit_pair(marker.parent, smoke_manifest['policy'], smoke_manifest['limit'], smoke=True)
            entry['matrix'] = str(path.parent)
            smoke_entries.append(entry)

    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    timestamp_local = dt.datetime.fromisoformat(timestamp).astimezone(ZoneInfo('Asia/Singapore')).strftime('%Y-%m-%d %H:%M:%S SGT')
    report = dict(generated_at_utc=timestamp, generated_at_singapore=timestamp_local, matrix=str(matrices[0]),
                  auditor_sha256=sha(Path(__file__)),
                  matrices=[str(p) for p in matrices], policy=policy,
                  source_policies={str(p): read(p / 'protocol.json')['policy'] for p in matrices},
                  completed_runs=sum(e['status'] == 'complete' for e in entries),
                  measured_partial_runs=sum(e['measurement_scope'] == 'partial' for e in entries),
                  invalid_partial_runs=sum(e['measurement_scope'] == 'invalid_partial' for e in entries),
                  partial_coverage_threshold=args.include_partial_over,
                  expected_runs=len(entries), definitions=dict(
                      primary_rate='sum(tokens) / sum(source video duration seconds), per question evaluation',
                      elapsed_rate='sum(tokens) / sum(saved elapsed_seconds), end-to-end worker processing rate',
                      question_mean='arithmetic mean of individual question token/time ratios',
                      input_text='all nonvisual input IDs, including timestamps, vision boundaries and chat template',
                      elapsed_scope='video probing/extraction/decoding/preprocessing, transfer, generation, scoring; first question includes model loading; excludes worker startup and final summary',
                      output_text='generated token IDs including EOS',
                      question_only='annotation question stem only; no answer options, task instructions, timestamps or chat template; add_special_tokens=False',
                      question_with_options='annotation question plus answer choices, without task instructions or video/chat text',
                      task_prompt='full source prompt text, including question/options and evaluation instructions; without video/chat wrapper',
                      video_chat_residual='Vanilla 2B full text input minus independently tokenized task prompt; includes timestamps, vision boundaries, chat formatting and BPE boundary differences',
                      mean_visual_input='sum(input visual tokens) / evaluated questions',
                      mean_total_text='mean(text input + generated text output), including EOS',
                      mean_total_input='mean(visual input + text input)',
                      mean_all_tokens='mean(visual input + text input + generated text output)',
                      incomplete='rates only if latest-attempt coverage strictly exceeds partial_coverage_threshold; otherwise no rates',
                      partial='validated observed prefix of latest attempt only; remains incomplete; no extrapolation or merging retries',
                      invalid_partial='saved coverage is shown, but rates withheld after validation failure, including non-EOS generations',
                      selection='one certified completion, otherwise the largest latest-attempt prefix across compatible matrices; never concatenate retries',
                      prediction_snapshot_sha256='canonical JSON digest of parsed prediction rows used for a partial measurement, excluding any unfinished final JSONL record'),
                  retry_aliases=aliases, runs=entries, candidate_runs=candidates,
                  question_token_analysis=question_lengths,
                  historical_videomme_runs=historical_videomme, historical_smoke_runs=smoke_entries)
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    flat = [dict(method=e['method'], suite=e['suite'], status=e['status'],
                 matrix=e['matrix'], population=e['population'], policy_version=e['policy_version'],
                 validation_error=e.get('partial_validation_error', ''),
                 measurement_scope=e['measurement_scope'], coverage_fraction=e['coverage_fraction'],
                 expected_questions=e['expected_questions'],
                 observed_questions=e['metrics']['questions'] if 'metrics' in e else e['observed_questions_latest_attempt'],
                 attempt=e.get('attempt', ''), **e.get('metrics', {})) for e in entries]
    save_csv(prefix.with_suffix('.csv'), flat)
    save_csv(prefix.parent / (prefix.name + '_questions.csv'), questions)
    categories = [dict(method=e['method'], suite=e['suite'], status=e['status'], report_scope='current',
                       matrix=e['matrix'], population=e['population'],
                       measurement_scope=e['measurement_scope'], category=c, **metrics)
                  for e in entries for c, metrics in e.get('categories', {}).items()]
    categories.extend(dict(method=e['method'], suite='videomme', status=e['status'],
                           matrix=e['matrix'], population='full', report_scope='historical_long_excluded',
                           measurement_scope=e['measurement_scope'], category='long', **e['categories']['long'])
                      for e in historical_videomme if 'long' in e.get('categories', {}))
    save_csv(prefix.parent / (prefix.name + '_categories.csv'), categories)
    attempts_by_path = {str(Path(a['path']).resolve()): dict(method=e['method'], suite=e['suite'],
                       matrix=e['matrix'], population=e['population'], **a)
                       for e in candidates for a in e['attempts']}
    attempts = list(attempts_by_path.values())
    save_csv(prefix.parent / (prefix.name + '_attempts.csv'),
             [{k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for k, v in a.items()}
              for a in attempts])
    video_questions = {g['category']: g for g in question_lengths['groups'] if g['suite'] == 'videomme'}
    short, long = video_questions.get('short', {}), video_questions.get('long', {})
    length_comparison = ''
    if 'full_text_input_mean_reference' in short and 'full_text_input_mean_reference' in long:
        length_comparison = (f'VideoMME question stems do get slightly longer: '
            f'{short["question_only_tokens_mean"]:.2f} tokens for short versus '
            f'{long["question_only_tokens_mean"]:.2f} for long. Including choices and evaluation instructions, '
            f'the task prompt grows from {short["task_prompt_tokens_mean"]:.2f} to '
            f'{long["task_prompt_tokens_mean"]:.2f} tokens, while full text input grows from '
            f'{short["full_text_input_mean_reference"]:,.2f} to '
            f'{long["full_text_input_mean_reference"]:,.2f}. Longer questions therefore explain only a small '
            'part of the text-input growth; the video/chat residual dominates.')
    lines = ['# Saved video token rates', '',
             f'Updated {timestamp_local}. Validated **{report["completed_runs"]}/{len(entries)}** selected-population model/suite pairs.', '',
             f'Also measured **{report["measured_partial_runs"]} incomplete pairs** with latest-attempt '
             f'coverage strictly above {100 * args.include_partial_over:g}%.' if args.include_partial_over is not None
             else 'Partial-run measurement is disabled.', '',
             f'**{report["invalid_partial_runs"]} partial pairs failed validation; their coverage is shown with rates withheld.**', '',
             'Source matrices: ' + ', '.join(f'`{p.relative_to(experiment)}`' if p.is_relative_to(experiment)
                                            else f'`{p}`' for p in matrices) + '.', '',
             '**VideoMME now means the 1,800 short/medium questions; the 900 long questions are excluded.** '
             'A complete VideoMME row covers this declared subset, not the original 2,700 questions.'
             if policy.get('videomme_population') else 'VideoMME uses the original 2,700-question population.', '',
             'The existing filename is retained for IDE links; the update time above identifies this snapshot. '
             'Each model/suite is counted once across matrices. Prefer a certified completion; otherwise use '
             'the largest saved prefix from one latest attempt. Retries and directories are never concatenated.', '',
             'Compatible older non-VideoMME results remain eligible when they preserve evaluated progress. '
             'The original 2,700-question VideoMME population is excluded from this combined table. '
             'JSON/CSV record the source matrix for every selected row.', '',
             'All completed runs passed the existing evaluator’s artifact hashes, selected ordered question coverage, '
             'source/input identities, exact token accounting, saved-score/summary checks, context bounds, and EOS checks. '
             'Selected source populations also match the frozen dataset manifests. No inference was run.', '',
             '## Average tokens per evaluated question', '',
             'These are arithmetic means of saved per-question token counts, **not tokens per second**. '
             'Visual input counts the video tokens. Text input includes question/options, timestamps and chat markers; '
             'text output includes generated IDs and EOS. **Total text = text input + text output; '
             'total input = visual input + text input; all tokens = visual input + total text.** '
             'Columns are rounded independently to two decimal places. '
             'Repeated questions for the same video are separate evaluations. Partial means use only the observed '
             'validated prefix; invalid or unobserved populations have no means.', '',
             *average_table(entries), '',
             '## Question length versus full text input', '',
             '**Question only** counts the annotation question stem, excluding choices, task instructions, '
             'timestamps and chat formatting. **Question + options** adds answer choices; **task prompt** also '
             'includes evaluation instructions. These full-annotation means cover all 1,580 TempCompass '
             'questions, all 300 questions in each Video-MMMU split, and 900 questions per VideoMME length. '
             'They do not require completed inference. The per-model tables above instead use the observed '
             'validated questions. CPU tokenization uses `add_special_tokens=False`; every model tokenizer '
             'was checked to produce identical counts for all three payloads.', '',
             *question_length_table(question_lengths['groups']), '', length_comparison, '',
             '**Why text input scales with video length:** `input_counts()` in '
             '`evaluation/protocols/video_protocol.py` reconstructs one timestamp and vision boundary pair '
             'for every temporal video patch group: `<12.0 seconds><|vision_start|>[visual tokens]<|vision_end|>`. '
             'With 2 fps sampling and two frames per temporal patch, there is approximately one such group '
             'per video second. Longer videos therefore add many nonvisual text tokens, even when the question '
             'stays short. Larger timestamp values can also need more tokenizer pieces. No subtitles or audio '
             'are being counted. The residual column is full text input minus the separately tokenized task '
             'prompt; it includes timestamps, vision boundaries, chat markers and BPE boundary differences. '
             'The full-input reference uses certified complete Vanilla 2B results; the long row is historical '
             'and excluded from current scores.', '',
             '## VideoMME average tokens by video length', '',
             'Each length has 900 questions in the original population. Current runs evaluate short and medium only. '
             'Partial rows use only observed questions in that length; missing means are not zero or estimates. '
             'The status describes the parent model/suite run, even if one length already has 900 saved answers.', '',
             *duration_average_table(entries), '',
             '### Historical long videos (excluded from current scores)', '',
             'The long-video means below come only from validated older 2,700-question attempts. '
             'They are separate evidence and are **not included** in the current benchmark means or completion count. '
             'Partial historical means cover only the observed long questions; models with no validated long '
             'observations have no historical row. Source attempts and hashes are recorded in JSON.', '',
             *duration_average_table(historical_videomme, historical_long=True), '',
             '## Tokens per source video second', '',
             '**Each rate is sum(tokens) / sum(video duration in seconds).** '
             'Repeated videos count once per evaluated question because their tokens are processed again. '
             'These duration-weighted rates normalize token use for clip length; they do not measure inference speed. '
             'CSV/JSON also retain the arithmetic mean, median and p95 of per-question rates.', '',
             'Text input includes the question/options, timestamps, vision boundary tokens and chat template. '
             'The timestamp/boundary sequence grows with video length even though subtitles and audio are disabled. '
             'Visual-token rates also depend on processed frame geometry and the sampling/temporal patch policy.', '',
             'Rows labelled **incomplete (partial rates)** use only the latest attempt’s saved questions; '
             'they remain incomplete. Other incomplete rows have no rates. **100% saved coverage without a valid '
             'completion certificate remains incomplete.** Counts may change after this snapshot. '
             'Retries are never concatenated. Partial rows passed ordered-prefix, source/input identity, '
             'token accounting, saved-score, context and EOS checks, and their selected source manifests match '
             'the frozen datasets. The parsed prediction snapshot has a separate digest; it has no full-run '
             'completion certificate. Any unfinished final JSONL record is excluded.', '',
             '**Partial populations can be biased by question order.** These are observed-prefix '
             'rates, not estimates for the missing questions. VideoMME visits short, then medium videos'
             + ('.' if policy.get('videomme_population') else ', then long videos.'), '',
             *table(entries), '', '## VideoMME duration groups', '',
             '| Model | Group | Questions | Mean video seconds | Mean visual tokens | Mean text input tokens | '
             'Visual / video s | Text input / video s | Total input / video s |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for entry in entries:
        if entry['suite'] != 'videomme' or entry['status'] != 'complete':
            continue
        for category in ('short', 'medium', 'long'):
            if category not in entry['categories']:
                continue
            m = entry['categories'][category]
            values = [entry['method'], category, str(m['questions'])] + [f'{m[k]:.3f}' for k in
                ('mean_video_seconds', 'input_visual_tokens_mean', 'input_text_tokens_mean',
                 'input_visual_tokens_per_video_second', 'input_text_tokens_per_video_second', 'input_tokens_per_video_second')]
            lines.append('| ' + ' | '.join(values) + ' |')
        if 'long' in entry['categories']:
            long = entry['categories']['long']
            share = 100 * long['input_tokens_total'] / entry['metrics']['input_tokens_total']
            lines.extend(['', f'The final 900 long-video questions account for **{share:.2f}%** of '
                          'this complete run’s total input tokens.', ''])
    lines.extend(['', '## Tokens per elapsed processing second', '',
                  'Denominator: sum of saved per-question `elapsed_seconds`. It includes video extraction/probing, '
                  'decoding/preprocessing, CPU/GPU transfer, generation and scoring. The first question includes '
                  'model loading; worker startup and final summary are excluded. This is an observed end-to-end '
                  'processing rate, not isolated prefill or decode throughput. Shared-machine conditions and '
                  'per-attempt initialization costs affect comparisons. Imported predictions retain the original '
                  'inference times, not the CPU import duration.', '',
                  *table([e for e in entries if e['status'] == 'complete'], 'elapsed'), '',
                  '## Partial runs above the coverage threshold', '',
                  'Rates below describe the saved prefix of the latest attempt, with the same ratio-of-sums '
                  'definition. They are excluded from the complete-run count.', '',
                  *table([e for e in entries if e['measurement_scope'] == 'partial']), '',
                  *table([e for e in entries if e['measurement_scope'] == 'partial'], 'elapsed'), '',
                  '## Partial runs with validation failures', '',
                  'These saved predictions do not qualify for uncensored token-rate measurements. '
                  'Their raw coverage does not certify a complete result.', '',
                  *table([e for e in entries if e['measurement_scope'] == 'invalid_partial']), '',
                  *[f'- `{e["method"]}/{e["suite"]}`: {e["partial_validation_error"]}.'
                    for e in entries if e['measurement_scope'] == 'invalid_partial'], '',
                  '## Historical smoke runs', '',
                  'One-question source-resolution smoke runs are separate previews, excluded from the full-run count. '
                  'Their rates use the same definitions but a different spatial policy.', '', *table(smoke_entries), '',
                  '## Files', '',
                  f'- `{prefix.name}.json`: definitions, validation hashes, all run states, duration/category statistics, retry aliases and smoke results.',
                  f'- `{prefix.name}.csv`: all benchmark pairs; status and measurement scope distinguish complete, partial and unmeasured runs.',
                  f'- `{prefix.name}_questions.csv`: exact counts and both rates for current measured questions, labelled by run status and scope.',
                  f'- `{prefix.name}_categories.csv`: current category means/rates and separately labelled historical long-video means/rates.',
                  f'- `{prefix.name}_attempts.csv`: all attempt states, observed coverage and saved failure/block evidence.', ''])
    prefix.with_suffix('.md').write_text('\n'.join(lines))
    print(f'Validated {report["completed_runs"]}/{len(entries)} selected-population runs; '
          f'{report["measured_partial_runs"]} measured partial runs; '
          f'{report["invalid_partial_runs"]} invalid partial runs; '
          f'{len(questions)} question evaluations; {len(attempts)} attempts; {len(smoke_entries)} historical smoke runs.')
    print(prefix.with_suffix('.md'))


if __name__ == '__main__':
    main()
