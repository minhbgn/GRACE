"""Deterministic local scoring v1. No inference, random fallback or judge calls.

The pinned MMMU open functions are loaded without importing its judge setup.
MMBench calls only the same prefetch rules used in the validated offline audit.
See README_local_development.md for deliberate local MC parsing conventions.
"""
from __future__ import annotations

import ast
from collections import Counter
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / 'third_party/lmms-eval-v0.7.1/lmms_eval/tasks'
SCORER = 'deterministic_local_v1'
METRICS = {
    'mmbench': 'circular_accuracy_all_supplied_rotations',
    'hallusionbench': 'rule_aAcc_all_image_and_text_rows',
    'mmstar': 'overall_accuracy_1500',
    'mmmu': 'mc_and_open_micro_accuracy_validation',
    'scienceqa': 'accuracy_test_all_image_and_text_rows',
    'ai2d': 'accuracy_test', 'seedbench_img': 'accuracy_image_subset',
}


def choice(response, valid='ABCD', options=None):
    """Return (label, extraction method); never consult the gold answer.

    Explicit answer markers > leading label > unique uppercase standalone label
    > exact unique option text. Conflicting explicit labels remain unresolved.
    Lowercase is accepted only in an unambiguous answer position.
    """
    valid = ''.join(valid)
    if not valid:
        return '', 'unresolved'
    text = re.sub(r'[*`]', '', response).strip()
    # Ignore complete reasoning blocks; an unfinished block has no final answer.
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    if '<think>' in text:
        return '', 'unfinished_thinking'
    token = rf'\(?([{valid}{valid.lower()}])\)?(?![\w])'
    marked = re.findall(r'(?i:\b(?:final\s+answer|correct\s+(?:answer|choice|option)|answer|choice)\s*(?:is\s*|:|=)\s*)' + token, text)
    marked += re.findall(r'\\boxed\{\s*([' + valid + valid.lower() + r'])\s*\}', text)
    marked += re.findall(r'(?i:<answer>)\s*' + token + r'\s*(?i:</answer>)', text)
    if marked:
        labels = {m.upper() for m in marked}
        return (next(iter(labels)), 'explicit') if len(labels) == 1 else ('', 'conflicting_explicit')
    leading = re.match(r'^\s*\(?([' + valid + valid.lower() + r'])\)?(?=\s*$|[.:,)\-])', text)
    if leading:
        return leading.group(1).upper(), 'leading'
    # Do not uppercase prose: the article "a" must never become option A.
    labels = set(re.findall(r'(?<!\w)([' + valid + r'])(?!\w)', text))
    if len(labels) == 1:
        label = next(iter(labels))
        # A at the beginning of a sentence may simply be an article.
        if not (label == 'A' and re.match(r'^A\s+\w', text)):
            return label, 'standalone'
    if options:
        normalize = lambda value: ' '.join(str(value).casefold().strip(' \n\t.,').split())
        matches = [key for key, value in options.items() if normalize(value) == normalize(text)]
        if len(matches) == 1:
            return matches[0], 'option_text_exact'
    return '', 'ambiguous' if labels else 'unresolved'


def yesno(response):
    text = re.sub(r'[*`]', '', response).strip()
    match = re.match(r"(?i)^(?:(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*)?(yes|no)\b", text)
    return ('1' if match[1].lower() == 'yes' else '0', 'explicit_yesno') if match else ('', 'unresolved')


@lru_cache(maxsize=1)
def open_helpers():
    path = PINNED / 'mmmu/utils.py'
    names = {'eval_open', 'parse_open_response', 'normalize_str', 'check_is_number', 'extract_numbers'}
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError('Pinned MMMU open helpers changed')
    namespace = {'re': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def open_answer(response, gold):
    helpers = open_helpers()
    if not response.strip():
        return [], False
    # Preserve candidate types; sort only to make JSON evidence byte-stable.
    candidates = sorted(helpers['parse_open_response'](response), key=lambda v: (type(v).__name__, str(v)))
    return candidates, bool(helpers['eval_open'](gold, candidates))


@lru_cache(maxsize=1)
def mmbench_evaluator():
    path = PINNED / 'mmbench/mmbench_evals.py'
    spec = importlib.util.spec_from_file_location('local_mmbench_rules', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.eval_logger.disable('local_mmbench_rules')
    evaluator = module.MMBench_Evaluator()
    def forbidden(*args, **kwargs):
        raise RuntimeError('Judge/network/random fallback forbidden in local scoring')
    evaluator._post_request = forbidden
    evaluator.extract_answer_from_item = forbidden
    return evaluator


def generation_record(ids, eos_token_id, cap):
    ids = [int(i) for i in ids]
    eos = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
    stopped_eos = bool(ids and ids[-1] in eos)
    reason = 'eos' if stopped_eos else 'length' if len(ids) >= cap else 'other'
    return dict(token_ids=ids, generated_tokens=len(ids), max_new_tokens=cap,
                finish_reason=reason, hit_token_cap=len(ids) >= cap,
                truncated=reason == 'length')


def score_rows(suite, rows, metadata=None):
    """Authoritative corrected row scoring; metadata maps IDs to dataset records."""
    metadata = metadata or {}
    for row in rows:
        raw = row['raw']
        method = 'inference_error'
        pred = ''
        correct = False
        if not row.get('error'):
            if suite == 'mmbench':
                meta = metadata[int(row['index'])]
                if row['gold'] != meta['answer']:
                    raise ValueError('MMBench gold mismatch')
                pred = mmbench_evaluator().prefetch_answer({**meta, 'prediction': raw}) or ''
                method = 'pinned_mmbench_prefetch' if pred else 'unresolved'
            elif suite == 'hallusionbench':
                pred, method = yesno(raw)
            elif suite == 'mmmu':
                meta = metadata[row['id']]
                if row['question_type'].startswith('multi'):
                    opts = meta['options']
                    opts = ast.literal_eval(opts) if isinstance(opts, str) else list(opts)
                    mapping = dict(zip('ABCDEFGHIJ', opts))
                    pred, method = choice(raw, mapping, mapping)
                else:
                    pred, correct = open_answer(raw, meta['answer'])
                    method = 'mmmu_open_typed_candidates' if pred else 'unresolved'
            else:
                count = row.get('n_choices', row.get('n_options', 4)) or 4
                pred, method = choice(raw, 'ABCDEFGHIJ'[:count])
                if suite in ('scienceqa', 'ai2d'):
                    pred = ord(pred) - ord('A') if pred else -1
            if not (suite == 'mmmu' and not row['question_type'].startswith('multi')):
                correct = pred not in ('', -1) and pred == row['gold']
        row.update(pred=pred, correct=bool(correct), extraction_method=method,
                   unresolved=not row.get('error') and pred in ('', -1, []))
    return rows


def load_metadata(suite, args):
    import pandas as pd
    if suite == 'mmbench':
        data = pd.read_csv(args.tsv, sep='\t', usecols=['index', 'A', 'B', 'C', 'D', 'answer'])
        if data['index'].duplicated().any():
            raise ValueError('Duplicate MMBench source IDs')
        return data.set_index('index').to_dict('index')
    if suite == 'mmmu':
        records = {}
        for path in sorted(Path(args.data_dir).glob(f'*/{args.split}-*.parquet')):
            for row in pd.read_parquet(path, columns=['id', 'options', 'answer']).to_dict('records'):
                if row['id'] in records:
                    raise ValueError('Duplicate MMMU source IDs')
                records[row['id']] = row
        return records
    return {}


def corrected_summary(suite, rows, metadata=None, partial=False):
    n = len(rows)
    summary = dict(scorer=SCORER, primary_metric=METRICS[suite],
                   parse_fail=sum(r['unresolved'] for r in rows),
                   unresolved=sum(r['unresolved'] for r in rows),
                   errors=sum(bool(r.get('error')) for r in rows),
                   generation_unknown=sum(r.get('generation') is None for r in rows),
                   finish_reasons=dict(Counter((r.get('generation') or {}).get('finish_reason', 'unknown') for r in rows)),
                   truncated=sum((r.get('generation') or {}).get('truncated', False) for r in rows),
                   hit_token_cap=sum((r.get('generation') or {}).get('hit_token_cap', False) for r in rows),
                   generated_tokens=sum((r.get('generation') or {}).get('generated_tokens', 0) for r in rows),
                   primary_score=sum(r['correct'] for r in rows) / n if n else 0.0)
    if suite == 'mmbench':
        # Reuse the already validated grouping and completeness implementation.
        from rescore_saved_protocols import circular
        if partial:
            summary.update(primary_score=None, circular=None, partial=True)
        else:
            evidence = circular(rows, metadata, mmbench_evaluator())
            summary.update(primary_score=evidence['exact_match_circular_percent'] / 100,
                           circular=evidence, row_accuracy=sum(r['correct'] for r in rows) / n)
    if suite == 'hallusionbench':
        summary['population'] = dict(total=n, image=sum(r['has_image'] for r in rows),
                                     text=sum(not r['has_image'] for r in rows))
    return summary


def finalize_rows(args, suite, rows):
    if not args.local_scoring:
        return {}
    meta = load_metadata(suite, args)
    score_rows(suite, rows, meta)
    return corrected_summary(suite, rows, meta, partial=bool(args.limit))


@lru_cache(maxsize=None)
def _row_metadata(suite, tsv, data_dir, split):
    from types import SimpleNamespace
    return load_metadata(suite, SimpleNamespace(tsv=tsv, data_dir=data_dir, split=split))


def prepare_row(args, suite, row):
    """Persist corrected predictions in shards as well as the merged artifact."""
    if args.local_scoring:
        meta = _row_metadata(suite, getattr(args, 'tsv', None),
                             getattr(args, 'data_dir', None), getattr(args, 'split', None))
        score_rows(suite, [row], meta)
    return row
