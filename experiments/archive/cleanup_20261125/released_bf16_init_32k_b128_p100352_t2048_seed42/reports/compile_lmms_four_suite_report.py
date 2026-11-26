"""Audit the completed four-suite round and derive report data without inference.
Run with reproduction/GRACE/.venv-lmms-eval/bin/python.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / 'evaluation'))
from run_lmms_matrix import TASKS, validate_result, verified_completion

SUITES = ['scienceqa', 'ai2d', 'seedbench', 'mmstar']
METHODS = list(json.loads((EXP / 'experiment.json').read_text())['models'])
FULL = EXP / 'lmms_evaluation/full'
OUT = EXP / 'reports'

def read(p):
    return json.loads(p.read_text())

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def average(v):
    return sum(v) / len(v)

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])

scores, sample_data, hashes, counts, categories, audits = {}, {}, {}, {}, {}, {}
identities, datasets = {}, {}
protocol = read(FULL / 'protocol.json')
for method in METHODS:
    scores[method], sample_data[method], categories[method] = {}, {}, {}
    for suite in SUITES:
        directory = FULL / method / suite
        done = verified_completion(directory)
        assert done, (method, suite, 'missing completion')
        attempt = directory / done['attempt']
        result, job = read(attempt / 'results.json'), read(attempt / 'job.json')
        assert job['method'] == method and job['suite'] == suite and job['limit'] is None
        assert job['protocol'] == protocol['protocol']
        dataset = read(attempt / 'dataset.json')
        assert dataset == read(FULL / 'dataset_fingerprints' / f'{suite}.json')
        assert dataset == datasets.setdefault(suite, dataset)
        judge = read(attempt / 'judge_audit.json')
        assert judge == {'calls': 0, 'failures': [], 'random_fallbacks': 0}
        samples = [json.loads(line) for line in (attempt / 'samples.jsonl').open()]
        valid = validate_result(result, samples, suite, None)
        assert valid == {k: done[k] for k in valid}
        ids = [(s['doc_id'], s['doc'], s['target']) for s in samples]
        identity = hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()
        assert identity == identities.setdefault(suite, identity), (method, suite, 'documents changed')
        values, groups, group_scores = [], defaultdict(list), defaultdict(list)
        for s in samples:
            if suite == 'seedbench':
                if 'seed_image' not in s:
                    continue
                r = s['seed_image']
                value = float(r['pred'].lower().strip() == r['answer'].lower().strip())
                group = 'all'
            elif suite == 'mmstar':
                value = s['average']['score']
                group = s['average']['l2_category']
                group_scores[s['doc']['category']].append(s)
            else:
                value, group = s['exact_match'], 'all'
            assert value in (0, 1), value
            values.append((s['doc_id'], value, group))
            groups[group].append(value)
        recomputed = average([average(v) for v in groups.values()]) * 100
        assert abs(recomputed - valid['score_percent']) < 1e-9, (method, suite)
        scores[method][suite] = recomputed
        sample_data[method][suite] = values
        counts[suite] = {'evaluated': len(samples), 'scored': len(values),
                         'groups': {k: len(v) for k, v in groups.items()}}
        if suite == 'mmstar':
            categories[method] = {k: v * 100 for k, v in result['results']['mmstar'].items()
                                  if k.endswith(',none') and '_stderr' not in k and k != 'average,none'}
            categories[method]['micro_accuracy'] = average([v for _, v, _ in values]) * 100
        for name in done['sha256']:
            hashes[str((attempt / name).relative_to(EXP))] = sha(attempt / name)
        hashes[str((directory / 'COMPLETE.json').relative_to(EXP))] = sha(directory / 'COMPLETE.json')
    scores[method]['mean4'] = average(list(scores[method].values()))

# Paired bootstrap of correctness differences; fixed L2 strata preserve MMStar's macro metric.
rng = np.random.default_rng(42)
pairs = {}
for left, right in [('grace_w4','qat_w4'), ('grace_w8','qat_w8'),
                    ('released_grace_w4','released_grace_w8'),
                    ('grace_w4','released_grace_w4'), ('grace_w8','released_grace_w8')]:
    key = f'{left}_minus_{right}'
    pairs[key] = {}
    for suite in SUITES:
        a, b = sample_data[left][suite], sample_data[right][suite]
        assert [(i,g) for i,_,g in a] == [(i,g) for i,_,g in b]
        diffs = defaultdict(list)
        for (_, va, g), (_, vb, _) in zip(a,b):
            diffs[g].append(va-vb)
        boots = np.zeros(20000)
        for group in diffs.values():
            n = len(group)
            c = Counter(group)
            draws = rng.multinomial(n, [c[-1]/n, c[0]/n, c[1]/n], size=20000)
            boots += (draws[:,2] - draws[:,0]) / n / len(diffs) * 100
        pairs[key][suite] = {'delta_pp': scores[left][suite] - scores[right][suite],
                             'bootstrap_95_ci_pp': np.quantile(boots,[0.025,0.975]).tolist(),
                             'wins': sum(v == 1 for g in diffs.values() for v in g),
                             'losses': sum(v == -1 for g in diffs.values() for v in g)}
    pairs[key]['mean4_delta_pp'] = scores[left]['mean4']-scores[right]['mean4']

# Published Table 6 (arXiv v5), ordered as ScienceQA, AI2D, SEED, MMStar.
paper = dict(zip(['vanilla_8b_teacher','vanilla_2b','released_bf16','released_grace_w8','released_grace_w4'],
                 [[85.0,85.7,77.5,70.9],[81.4,76.9,71.2,58.3],[86.2,81.3,76.7,67.3],
                  [85.3,80.4,75.9,66.5],[84.3,79.5,75.1,65.8]]))
published = {m:{**dict(zip(SUITES,vals)), 'mean4':average(vals)} for m,vals in paper.items()}
old = {}
for method in METHODS:
    old[method] = {}
    for suite in SUITES:
        path = EXP / 'evaluation' / method / f'{"seedbench_img" if suite == "seedbench" else suite}.json'
        data = read(path)['summary']
        old[method][suite] = (data['by_modality']['image']['acc'] if suite == 'scienceqa' else data['overall_acc']) * 100
        hashes[str(path.relative_to(EXP))] = sha(path)
    old[method]['mean4'] = average(list(old[method].values()))

for path in [Path(__file__).resolve(), FULL/'protocol.json', FULL/'datasets.lock.json']:
    hashes[str(path.relative_to(EXP))] = sha(path)
for method in METHODS:
    path = FULL / method / 'model_provenance.json'
    hashes[str(path.relative_to(EXP))] = sha(path)

report = {'scope':'Four-suite descriptive mean only; seven-suite mean unavailable',
          'date':'2026-09-25', 'scores_percent':scores, 'published_table6_percent':published,
          'paper_url':'https://arxiv.org/html/2601.22709v5#A1.T6',
          'old_image_aligned_percent':old, 'paired':pairs, 'mmstar_categories_percent':categories,
          'counts':counts, 'saved_document_sha256':identities, 'datasets':datasets,
          'bootstrap':{'replicates':20000, 'seed':42, 'method':'paired correctness; fixed L2 strata for MMStar',
                       'limitations':'Per-benchmark sample uncertainty only; no training-seed variability, no multiple-testing adjustment, no four-suite mean confidence interval'},
          'source_sha256':hashes}
(OUT/'lmms_four_suite_analysis.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
with (OUT/'lmms_four_suite_scores.csv').open('w') as f:
    writer=csv.writer(f);writer.writerow(['model',*SUITES,'mean4'])
    for m in METHODS: writer.writerow([m,*[scores[m][s] for s in SUITES],scores[m]['mean4']])
sections = {}
sections['scores'] = table(['Model','SQA-IMG','AI2D','SEED-IMG','MMStar','Mean-4'],
                          [[m,*[f'{scores[m][s]:.2f}' for s in SUITES],f'{scores[m]["mean4"]:.2f}'] for m in METHODS])
sections['paper'] = table(['Model','SQA paper → ours','AI2D paper → ours','SEED paper → ours','MMStar paper → ours','Paper Mean-4','Our Mean-4','Δ Mean-4'],
    [[m,*[f'{published[m][s]:.1f} → {scores[m][s]:.2f}' for s in SUITES],f'{published[m]["mean4"]:.3f}', f'{scores[m]["mean4"]:.2f}',f'{scores[m]["mean4"]-published[m]["mean4"]:+.2f}'] for m in published])
sections['old'] = table(['Model','Old SQA-IMG','New SQA-IMG','Old Mean-4','New Mean-4','Δ Mean-4'],
    [[m,f'{old[m]["scienceqa"]:.2f}',f'{scores[m]["scienceqa"]:.2f}', f'{old[m]["mean4"]:.2f}',f'{scores[m]["mean4"]:.2f}',f'{scores[m]["mean4"]-old[m]["mean4"]:+.2f}'] for m in METHODS])
sections['pairs'] = table(['Contrast (left − right)',*SUITES,'Δ Mean-4'],
    [[k,*[f'{v[s]["delta_pp"]:+.2f}' for s in SUITES],f'{v["mean4_delta_pp"]:+.2f}'] for k,v in pairs.items()])
sections['ci'] = table(['Bits','Benchmark','GRACE-only correct / QAT-only correct','Δ pp','Paired 95% interval'],
    [[bits,s,f'{v["wins"]} / {v["losses"]}',f'{v["delta_pp"]:+.2f}',f'[{v["bootstrap_95_ci_pp"][0]:+.2f}, {v["bootstrap_95_ci_pp"][1]:+.2f}]']
     for bits in ['w4','w8'] for s,v in pairs[f'grace_{bits}_minus_qat_{bits}'].items() if s in SUITES])
sections['categories'] = table(['MMStar category','W4 GRACE − QAT','W8 GRACE − QAT','Teacher − BF16 init'],
    [[k.split(',')[0],f'{categories["grace_w4"][k]-categories["qat_w4"][k]:+.2f}',f'{categories["grace_w8"][k]-categories["qat_w8"][k]:+.2f}',f'{categories["vanilla_8b_teacher"][k]-categories["released_bf16"][k]:+.2f}'] for k in categories['grace_w4'] if k != 'micro_accuracy'])
(OUT/'lmms_four_suite_tables.md').write_text('\n\n'.join('## '+k+'\n\n'+v for k,v in sections.items())+'\n')
print(json.dumps({'verified_jobs':40,'source_files_hashed':len(hashes),'counts':counts,'scores':scores,'pairs':pairs},indent=2))
