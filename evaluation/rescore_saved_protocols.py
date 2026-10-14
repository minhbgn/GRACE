"""CPU-only audit of saved native replay answers; never edits replay artifacts.

MMBench uses the pinned harness's deterministic answer extraction and CircularEval.
Unresolved extraction is reported as a bound, with a conservative exact-match
score (unresolved = incorrect). No API calls or random answer fallback are used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd

from paper_acceptance import PAPER, assess
from run_lmms_matrix import verified_completion

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def circular(rows, options, evaluator):
    indices = [int(r['index']) for r in rows]
    if len(indices) != len(set(indices)) or set(indices) != set(options):
        raise ValueError('MMBench rows must cover every TSV rotation exactly once')
    groups = defaultdict(list)
    expected_groups = defaultdict(set)
    for idx in options:
        expected_groups[idx % 1_000_000].add(idx)
    for row in rows:
        idx = int(row['index'])
        meta = options[idx]
        if row['gold'] != meta['answer']:
            raise ValueError(f'MMBench gold mismatch: {idx}')
        pred = evaluator.prefetch_answer({**meta, 'prediction': row['raw']})
        groups[idx % 1_000_000].append({
            'index': idx, 'gold': row['gold'], 'extracted': pred or None,
            'raw': row['raw'],
        })
    evidence = []
    for base, rotations in sorted(groups.items()):
        expected = expected_groups[base]
        if base not in options:
            raise ValueError(f'Missing base question: {base}')
        if {r['index'] for r in rotations} != expected:
            raise ValueError(f'Incomplete circular group: {base}')
        wrong = any(r['extracted'] is not None and r['extracted'] != r['gold'] for r in rotations)
        unresolved = any(r['extracted'] is None for r in rotations)
        status = 'incorrect' if wrong else 'unresolved' if unresolved else 'correct'
        evidence.append({'base_index': base, 'status': status, 'rotations': rotations})
    counts = Counter(g['status'] for g in evidence)
    lower = 100 * counts['correct'] / len(evidence)
    upper = 100 * (counts['correct'] + counts['unresolved']) / len(evidence)
    return {
        'rows': len(rows), 'groups': len(evidence), 'group_counts': dict(counts),
        'groups_with_fewer_rows_than_option_columns': sum(
            len(g['rotations']) < sum(pd.notna(options[g['base_index']][ch]) for ch in 'ABCD')
            for g in evidence),
        'rotation_count_histogram': dict(Counter(len(g['rotations']) for g in evidence)),
        'unresolved_rows': sum(r['extracted'] is None for g in evidence for r in g['rotations']),
        'exact_match_circular_percent': lower,
        'judge_completion_bounds_percent': [lower, upper],
        'fully_resolved_group_score': counts['unresolved'] == 0,
        'group_evidence': evidence,
    }


def hallusion(rows):
    qgroups, fgroups = defaultdict(list), defaultdict(list)
    for r in rows:
        prefix = (r['category'] or '_unknown', r['subcategory'] or '_unknown', r['set_id'])
        qgroups[prefix + (r['question_id'],)].append(r['correct'])
        if r['has_image']:
            fgroups[prefix + (r['figure_id'],)].append(r['correct'])
    return {'aAcc': 100 * sum(r['correct'] for r in rows) / len(rows),
            'qAcc': 100 * sum(all(v) for v in qgroups.values()) / len(qgroups),
            'fAcc': 100 * sum(all(v) for v in fgroups.values()) / len(fgroups),
            'rows': len(rows), 'q_groups': len(qgroups), 'f_groups': len(fgroups)}


def star(rows):
    groups = {key: defaultdict(list) for key in ('category', 'l2_category')}
    for row in rows:
        for key in groups:
            groups[key][row[key]].append(row['correct'])
    return {'overall_percent': 100 * sum(r['correct'] for r in rows) / len(rows),
            **{key + '_macro_percent': 100 * sum(sum(v) / len(v) for v in cats.values()) / len(cats)
               for key, cats in groups.items()},
            'category_counts': {key: {k: len(v) for k, v in cats.items()} for key, cats in groups.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Output must live outside the immutable native replay tree.
    if args.output.resolve().is_relative_to(args.full_root.resolve()):
        parser.error('--output must be outside --full-root')
    source = ROOT / 'third_party/lmms-eval-v0.7.1/lmms_eval/tasks/mmbench/mmbench_evals.py'
    spec = importlib.util.spec_from_file_location('offline_mmbench', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.eval_logger.disable('offline_mmbench')
    evaluator = module.MMBench_Evaluator()

    def forbidden(*args, **kwargs):
        raise RuntimeError('Network/judge/random extraction fallback forbidden in offline scoring')

    evaluator._post_request = forbidden
    evaluator.extract_answer_from_item = forbidden
    tsv = ROOT / 'evaluation/MMBench/mmbench_dev_en_20231003.tsv'
    options = pd.read_csv(tsv, sep='\t', usecols=['index', 'A', 'B', 'C', 'D', 'answer'])
    if options['index'].duplicated().any():
        raise ValueError('Duplicate TSV indices')
    options = options.set_index('index').to_dict('index')
    report = {'protocol': 'saved_answers_circular_and_metric_audit_v1',
              'judge_calls': 0, 'inference_runs': 0,
              'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in (source, tsv, Path(__file__).resolve())},
              'hallusion_policy': 'Archived GRACE rule correctness; all 1129 rows for aAcc/qAcc, image rows for fAcc. Not GPT-judged.',
              'mmbench_policy': 'All rotations supplied in the pinned TSV must be correct; unresolved extraction fails conservative exact-match score; bounds cover possible judge resolution.',
              'models': {}}
    for model in PAPER:
        result = {'inputs': {}}
        for suite in ('hallusionbench', 'mmbench', 'mmstar', 'mmmu'):
            job = args.full_root / model / suite
            marker = verified_completion(job)
            if marker is None:
                raise ValueError(f'Invalid completion/artifact hashes: {job}')
            path = job / marker['attempt'] / 'results.json'
            payload = json.loads(path.read_text())
            rows = payload['results']
            if suite != 'mmmu' and any(
                r['correct'] != bool(r['pred'] and r['pred'] == r['gold']) for r in rows
            ):
                raise ValueError(f'Invalid saved correctness/errors: {path}')
            if any(r['error'] is not None for r in rows):
                raise ValueError(f'Inference errors: {path}')
            result['inputs'][suite] = {'path': str(path.resolve()), 'sha256': digest(path)}
            if suite == 'mmbench':
                result[suite] = circular(rows, options, evaluator)
                result[suite]['acceptance_at_bounds'] = [assess(model, suite, value) for value in result[suite]['judge_completion_bounds_percent']]
            elif suite == 'hallusionbench':
                result[suite] = hallusion(rows)
                for metric in ('aAcc', 'qAcc', 'fAcc'):
                    if abs(result[suite][metric] - 100 * payload['summary'][metric]) > 1e-9:
                        raise ValueError(f'Saved HallusionBench {metric} mismatch')
                result[suite]['acceptance'] = {k: assess(model, suite, result[suite][k]) for k in ('aAcc', 'qAcc', 'fAcc')}
            elif suite == 'mmstar':
                result[suite] = star(rows)
            else:
                mc = [r for r in rows if r['question_type'].startswith('multi')]
                op = [r for r in rows if r['question_type'] == 'open']
                if len(mc) + len(op) != len(rows):
                    raise ValueError('Unknown MMMU question type')
                result[suite] = {'mc_total': len(mc), 'mc_correct': sum(r['correct'] for r in mc),
                                 'open_total': len(op), 'open_correct': sum(r['correct'] for r in op),
                                 'overall_percent': 100 * sum(r['correct'] for r in rows) / len(rows),
                                 'maximum_open_rescoring_gain_pp': 100 * sum(not r['correct'] for r in op) / len(rows),
                                 'ceiling_if_all_open_correct_percent': 100 * (sum(r['correct'] for r in mc) + len(op)) / len(rows)}
        report['models'][model] = result
    report['hallusion_mean_absolute_gap_pp'] = {
        metric: sum(abs(r['hallusionbench'][metric] - PAPER[m]['hallusionbench']) for m, r in report['models'].items()) / len(PAPER)
        for metric in ('aAcc', 'qAcc', 'fAcc')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for model, result in report['models'].items():
        print(model, json.dumps({suite: {k: v for k, v in data.items() if k != 'group_evidence'}
                                 for suite, data in result.items() if suite != 'inputs'}))
    print('HallusionBench mean absolute gap:', report['hallusion_mean_absolute_gap_pp'])


if __name__ == '__main__':
    main()
