"""Offline saved-answer diagnostics; does not modify scores, gates or generation."""
import ast
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[1]
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(ROOT / 'evaluation'))
from run_lmms_matrix import verified_completion

prior = json.loads((HERE / 'saved_answer_analysis.json').read_text())
full = EXPERIMENT / 're_evaluation/full'
scorer = ROOT / 'third_party/lmms-eval-v0.7.1/lmms_eval/tasks/mmbench/mmbench_evals.py'
spec = importlib.util.spec_from_file_location('offline_mmbench', scorer)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
evaluator = module.MMBench_Evaluator()
def no_network(*args, **kwargs):
    raise RuntimeError('Judging/network requests forbidden in offline diagnosis')
evaluator._post_request = no_network
module.eval_logger.remove()
tsv = ROOT / 'evaluation/MMBench/mmbench_dev_en_20231003.tsv'
options = pd.read_csv(tsv, sep='\t', usecols=['index', 'A', 'B', 'C', 'D']).set_index('index').to_dict('index')
star_source = ROOT / 'third_party/lmms-eval-v0.7.1/lmms_eval/tasks/mmstar/utils.py'
namespace = {}
node = next(n for n in ast.parse(star_source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'exact_match')
exec(compile(ast.Module(body=[node], type_ignores=[]), str(star_source), 'exec'), namespace)
def percent(n, d):
    return 100 * n / d if d else None

def hall_metrics(rows):
    gold = Counter(r['gold'] for r in rows)
    pred = Counter(r['pred'] if r['pred'] in ('0', '1') else 'invalid' for r in rows)
    confusion = {g: dict(Counter(r['pred'] if r['pred'] in ('0', '1') else 'invalid' for r in rows if r['gold'] == g)) for g in ('0', '1')}
    classes = {}
    for label in ('0', '1'):
        tp = confusion[label].get(label, 0)
        precision = percent(tp, pred[label])
        recall = percent(tp, gold[label])
        classes[label] = {'precision': precision, 'recall': recall, 'f1': 2*precision*recall/(precision+recall) if precision and recall else 0}
    return {'n': len(rows), 'gold_counts': dict(gold), 'prediction_counts': dict(pred), 'confusion_gold_by_prediction': confusion,
            'accuracy': percent(sum(r['correct'] for r in rows), len(rows)), 'majority_baseline': percent(max(gold.values()), len(rows)),
            'balanced_accuracy': sum(x['recall'] for x in classes.values())/2, 'macro_f1': sum(x['f1'] for x in classes.values())/2, 'classes': classes}

report = {'scope': 'Offline diagnostics from completed saved predictions. No inference, judge, random guessing, gate edits or resolution overrides.',
          'label_key': {'0': 'No', '1': 'Yes'},
          'reference_scorer_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (scorer, star_source)},
          'models': {}, 'mmmu_mc_paired_against_bf16': {}}
all_mmmu = {}
for method in prior['models']:
    result = {}
    for suite in ('hallusionbench', 'mmbench', 'mmmu', 'mmstar'):
        job = full / method / suite
        marker = verified_completion(job)
        assert marker is not None, (method, suite)
        path = job / marker['attempt'] / 'results.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == prior['models'][method]['suites'][suite]['result_sha256']
        payload = json.loads(path.read_text())
        rows = payload['results']
        assert all(r['error'] is None for r in rows)
        if suite == 'hallusionbench':
            result[suite] = {'all': hall_metrics(rows), 'image_only': hall_metrics([r for r in rows if r['has_image']]),
                            'text_only': hall_metrics([r for r in rows if not r['has_image']]),
                            'qAcc': 100*payload['summary']['qAcc'], 'fAcc': 100*payload['summary']['fAcc']}
        elif suite == 'mmbench':
            groups = defaultdict(list)
            conflicts = []
            unresolved = []
            resolved_correct = 0
            for r in rows:
                extracted = evaluator.prefetch_answer({**options[int(r['index'])], 'prediction': r['raw']})
                groups[int(r['index']) % 1000000].append((r, extracted))
                resolved_correct += bool(extracted == r['gold'])
                if not extracted: unresolved.append(r['index'])
                elif extracted != r['pred']: conflicts.append({'index': r['index'], 'raw': r['raw'], 'old': r['pred'], 'reference_prefetch': extracted, 'gold': r['gold']})
            lower = sum(all(p and p == r['gold'] for r,p in group) for group in groups.values())
            upper = sum(all(not p or p == r['gold'] for r,p in group) for group in groups.values())
            result[suite] = {'rows': len(rows), 'groups': len(groups), 'unresolved_rows': len(unresolved), 'unresolved_indices': unresolved,
                             'resolved_parser_conflicts': conflicts, 'circular_lower_percent': percent(lower, len(groups)),
                             'circular_upper_percent': percent(upper, len(groups)), 'potentially_judge_dependent_groups': upper-lower,
                             'row_lower_percent': percent(resolved_correct, len(rows)),
                             'saved_choice_circular_percent': prior['models'][method]['suites'][suite]['circular_saved_choice_percent']}
        elif suite == 'mmstar':
            categories = defaultdict(list)
            native_categories = defaultdict(list)
            changes = []
            for r in rows:
                score = namespace['exact_match'](r['raw'], r['gold'])
                categories[r['l2_category']].append(score)
                native_categories[r['l2_category']].append(r['correct'])
                if bool(score) != r['correct']:
                    changes.append({'index': r['index'], 'raw': r['raw'], 'gold': r['gold'], 'native_correct': r['correct'], 'reference_correct': bool(score)})
            result[suite] = {'native_micro': 100*payload['summary']['overall_acc'],
                             'native_l2_macro': 100*sum(sum(v)/len(v) for v in native_categories.values())/len(native_categories),
                             'reference_exact_micro': 100*sum(sum(v) for v in categories.values())/len(rows),
                             'reference_exact_l2_macro': 100*sum(sum(v)/len(v) for v in categories.values())/len(categories),
                             'correct_to_wrong': sum(r['native_correct'] for r in changes), 'wrong_to_correct': sum(r['reference_correct'] for r in changes),
                             'examples': changes[:15]}
        else:
            all_mmmu[method] = {r['id']: r for r in rows if r['question_type'].startswith('multi')}
            result[suite] = {'by_question_type': payload['summary']['by_question_type'], 'by_subject': payload['summary']['by_subject']}
    report['models'][method] = result
for method, rows in all_mmmu.items():
    base = all_mmmu['released_bf16']
    assert set(rows) == set(base)
    assert all(rows[k]['gold'] == base[k]['gold'] for k in rows)
    report['mmmu_mc_paired_against_bf16'][method] = {
        'same_prediction': sum(rows[k]['pred'] == base[k]['pred'] for k in rows),
        'bf16_correct_other_wrong': sum(base[k]['correct'] and not rows[k]['correct'] for k in rows),
        'bf16_wrong_other_correct': sum(not base[k]['correct'] and rows[k]['correct'] for k in rows)}
(HERE / 'four_suite_analysis.json').write_text(json.dumps(report, indent=2)+'\n')
for method, r in report['models'].items():
    print(method, json.dumps({'hall': r['hallusionbench']['all'], 'mmbench': {k:v for k,v in r['mmbench'].items() if k not in ('unresolved_indices','resolved_parser_conflicts')}, 'mmstar': {k:v for k,v in r['mmstar'].items() if k != 'examples'}, 'mmmu_types': r['mmmu']['by_question_type']}))
print('MMMU paired:', report['mmmu_mc_paired_against_bf16'])
