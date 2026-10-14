"""CPU-only diagnostics; preserves all original benchmark scores and artifacts."""
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[1]
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(ROOT / 'evaluation'))
from run_lmms_matrix import read, sha, verified_completion, write
from run_upstream_matrix import validate, source_check
from check_re_evaluation_gate import REFERENCES
from constrained_experiment import Experiment
from paper_acceptance import PAPER
from transformers import AutoTokenizer

# These pure scoring helpers need no model or judge. Load the exact pinned
# helper functions without importing the task module's unused API client.
source = ROOT / 'third_party/lmms-eval-v0.7.1/lmms_eval/tasks/mmmu/utils.py'
keep = {'check_is_number', 'normalize_str', 'extract_numbers', 'parse_open_response', 'eval_open'}
module = ast.parse(source.read_text())
namespace = {'re': re}
exec(compile(ast.Module(body=[n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in keep], type_ignores=[]), str(source), 'exec'), namespace)

# Intentionally conservative diagnostic extraction, not a replacement scorer.
# A leading labelled answer is counted only with punctuation, parentheses or
# end-of-response. No fallback scans inside words; unparsed remains unparsed.
def explicit_leading(raw):
    match = re.match(r'^\s*(?:\(([A-J])\)|([A-J])(?:\s*[:.)]|\s*$))', raw)
    return (match[1] or match[2]) if match else None

assert explicit_leading('C: A forest') == 'C'
assert explicit_leading('To solve for the missing account balance,') is None
assert explicit_leading('(B) 3') == 'B'

exp = Experiment(EXPERIMENT / 'experiment.json')
full = EXPERIMENT / 're_evaluation/full'
protocol = read(full / 'protocol.json')
assert protocol['limit'] is None and protocol['source'] == source_check()
report = {'scope': 'Saved-response diagnostics only. Alternative scores are not new official runs.',
          'original_protocol_sha256': sha(full / 'protocol.json'),
          'open_parser_source_sha256': sha(source), 'models': {}, 'examples': [], 'verified_pairs': 0}
for method in REFERENCES:
    print('Analyzing', method, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(exp.paths[method], local_files_only=True)
    provenance = read(full / method / 'model_provenance.json')
    previous = read(EXPERIMENT / 'lmms_evaluation/full' / method / 'model_provenance.json')
    model = {'repository': provenance['repository'], 'revision': provenance['revision'],
             'matches_previously_hub_audited_file_hashes': provenance['files_sha256'] == previous['files_sha256'],
             'processor': read(full / method / 'processor.json'), 'suites': {}}
    for suite in ('hallusionbench', 'mmbench', 'scienceqa', 'ai2d', 'mmmu', 'seedbench_img', 'mmstar'):
        job = full / method / suite
        done = verified_completion(job)
        if done is None: raise ValueError(f'Incomplete {method}/{suite}')
        payload = read(job / done['attempt'] / 'results.json')
        verified = validate(payload, suite, exp.paths[method], None)
        assert verified['score_percent'] == done['score_percent']
        report['verified_pairs'] += 1
        rows, summary = payload['results'], payload['summary']
        result = {'n': len(rows), 'score': verified['score_percent'], 'summary': summary,
                  'result_sha256': sha(job / done['attempt'] / 'results.json')}
        if suite == 'mmbench':
            groups = defaultdict(list)
            for row in rows: groups[int(row['index']) % 1000000].append(row)
            bases = {int(row['index']) for row in rows if int(row['index']) < 1000000}
            assert bases == set(groups)
            result.update(circular_groups=len(groups), circular_group_sizes=dict(Counter(map(len, groups.values()))),
                          circular_saved_choice_percent=100 * sum(all(r['correct'] for r in group) for group in groups.values()) / len(groups))
        if suite == 'hallusionbench':
            result['raw_answer_counts'] = dict(Counter(r['raw'] for r in rows))
            result['modality'] = {str(flag): {'n': len(subset), 'score': 100 * sum(r['correct'] for r in subset) / len(subset)}
                                  for flag in (True, False) if (subset := [r for r in rows if r['has_image'] == flag])}
        if suite in ('mmmu', 'mmstar'):
            mc = [r for r in rows if suite == 'mmstar' or r['question_type'].startswith('multi')]
            lengths = [len(ids) for ids in tokenizer([r['raw'] for r in mc], add_special_tokens=False)['input_ids']]
            cap = 8 if suite == 'mmmu' else 64
            # Re-encoding decoded strings is a proxy; original token IDs and
            # finish reasons were not saved, so exact truncation is not proven.
            at_cap = [row for row, n in zip(mc, lengths) if n >= cap]
            changes = [(r, explicit_leading(r['raw'])) for r in mc
                       if explicit_leading(r['raw']) is not None and explicit_leading(r['raw']) != r['pred']]
            net = sum(letter == r['gold'] for r, letter in changes) - sum(r['correct'] for r, letter in changes)
            result.update(mc_n=len(mc), mc_cap=cap, mc_single_letter=sum(bool(re.fullmatch('[A-J]', r['raw'].strip())) for r in mc),
                          mc_reencoded_at_cap=len(at_cap),
                          at_cap_reasoning_starts=sum(r['raw'].startswith(('To ', 'The ', 'We ', 'Let')) for r in at_cap),
                          leading_choice_conflicts=len(changes), leading_choice_net_correct_change=net,
                          leading_choice_only_rescore_percent=(sum(r['correct'] for r in rows) + net) / len(rows) * 100,
                          mc_no_standalone_uppercase_choice=sum(not re.search(r'(?<!\w)[A-J](?!\w)', r['raw']) for r in mc))
            report['examples'].extend({'method': method, 'suite': suite, 'kind': 'leading_choice_conflict',
                         'id': r.get('id', r.get('index')), 'raw': r['raw'], 'old_pred': r['pred'],
                         'leading_pred': letter, 'gold': r['gold']} for r, letter in changes[:5])
            report['examples'].extend({'method': method, 'suite': suite, 'kind': 'cap_proxy_reasoning',
                         'id': r.get('id', r.get('index')), 'raw': r['raw'], 'pred': r['pred'], 'gold': r['gold']}
                         for r in [r for r in at_cap if r['raw'].startswith(('To ', 'The ', 'We ', 'Let'))][:5])
            if suite == 'mmmu':
                opens = [r for r in rows if not r['question_type'].startswith('multi')]
                new = [namespace['eval_open'](r['gold'], namespace['parse_open_response'](r['raw'])) for r in opens]
                result['open'] = {'n': len(opens), 'original_correct': sum(r['correct'] for r in opens),
                                  'official_helper_correct': sum(new),
                                  'candidate_preserving_overall_percent': 100 * (sum(r['correct'] for r in mc) + sum(new)) / len(rows)}
        if suite == 'scienceqa': result['modality'] = summary['by_modality']
        model['suites'][suite] = result
    report['models'][method] = model
write(HERE / 'saved_answer_analysis.json', report)
for method, model in report['models'].items():
    print(method, 'previous_hashes_match=', model['matches_previously_hub_audited_file_hashes'])
    for suite in ('mmmu','mmstar'):
        r = model['suites'][suite]
        print(suite, {k: v for k,v in r.items() if k not in ('summary','result_sha256')})
    print('mmbench_circular=', model['suites']['mmbench']['circular_saved_choice_percent'])
print('Saved', HERE / 'saved_answer_analysis.json')
