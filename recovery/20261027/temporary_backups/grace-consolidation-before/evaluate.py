#!/usr/bin/env python3
"""Matched seven-suite evaluation of ten explicit checkpoint rows, resumable by suite."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from common import (HERE, ROOT, REPOS, check_processor_compatibility, model_paths,
                    pin, provenance, read, sha, validate_export)
sys.path.insert(0, str(ROOT / 'evaluation'))
from summarize_constrained_evals import SUITES, ID_FIELDS, load_results

EXPECTED = dict(hallusionbench=1129, mmbench=4329, scienceqa=4241, ai2d=3088,
                mmmu=900, seedbench_img=14233, mmstar=1500)
METHODS = ('vanilla_2b', 'vanilla_8b_teacher', 'sft_2b', 'released_bf16',
           'released_grace_w4', 'released_grace_w8', 'qat_w4', 'grace_w4', 'qat_w8', 'grace_w8')


def suites():
    data = Path(os.environ.get('DATA_ROOT', ROOT / 'evaluation')).resolve()
    def loc(env, default):
        return str(Path(os.environ.get(env, default)).resolve())
    science = Path(loc('SCIENCEQA_META', data / 'ScienceQA/data'))
    seed = Path(loc('SEED_ROOT', data / 'SEEDBench/data'))
    return {
        'hallusionbench': ('HallusionBench/eval_hallusion.py', 8, ['--data_dir', loc('HALLUSION_DIR', data / 'HallusionBench/data')]),
        'mmbench': ('MMBench/eval_mmbench.py', 8, ['--tsv', loc('MMBENCH_TSV', data / 'MMBench/mmbench_dev_en_20231003.tsv')]),
        'scienceqa': ('ScienceQA/eval_scienceqa.py', 8, ['--problems', loc('SCIENCEQA_PROBLEMS', science / 'problems.json'), '--splits', loc('SCIENCEQA_SPLITS', science / 'pid_splits.json'), '--image_root', loc('SCIENCEQA_IMAGE_ROOT', science / 'images'), '--split', 'test']),
        'ai2d': ('AI2D/eval_ai2d.py', 8, ['--data_dir', loc('AI2D_DIR', data / 'AI2D/data')]),
        'mmmu': ('mmmu/eval_mmmu.py', 32, ['--data_dir', loc('MMMU_DIR', data / 'mmmu/data'), '--split', 'validation']),
        'seedbench_img': ('SEEDBench/eval_seedbench.py', 8, ['--json_path', loc('SEED_JSON', seed / 'SEED-Bench.json'), '--image_dir', loc('SEED_IMAGE_DIR', seed / 'SEED-Bench-image')]),
        'mmstar': ('MMStar/eval_mmstar.py', 8, ['--tsv', loc('MMSTAR_TSV', data / 'MMStar/MMStar.tsv')]),
    }


def command(method, suite, paths, out, nproc, attn):
    script, tokens, arguments = suites()[suite]
    return [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={nproc}',
            str(ROOT / 'evaluation' / script), '--model_path', str(paths[method]),
            '--processor_path', str(paths[method]), '--output', str(out / method / f'{suite}.json'),
            '--attn', attn, '--max_new_tokens', str(tokens), '--min_pixels', '4096',
            '--max_pixels', '100352', '--fixed_generation', *arguments]


def validate_result(path, suite, model, attn):
    payload = read(path)
    summary, results = payload['summary'], payload['results']
    n = len(results)
    if n != EXPECTED[suite] or summary['total'] != n:
        raise ValueError(f'{path}: expected {EXPECTED[suite]} questions, got {n}')
    identities = {tuple(row[key] for key in ID_FIELDS[suite]): row['gold'] for row in results}
    if len(identities) != n:
        raise ValueError(f'{path}: duplicate question identities')
    correct = sum(bool(row['correct']) for row in results)
    if correct != summary['correct'] or not math.isclose(correct / n, summary[SUITES[suite][1]], abs_tol=1e-12):
        raise ValueError(f'{path}: prediction/summary mismatch')
    expected_policy = dict(version=1, min_pixels=4096, max_pixels=100352,
                           max_new_tokens=suites()[suite][1], do_sample=False, num_beams=1,
                           repetition_penalty=1.0, context='no truncation; native model context', attention=attn)
    if suite == 'mmmu':
        expected_policy['multiple_choice_max_new_tokens'] = 8
    if summary.get('evaluation_protocol') != expected_policy:
        raise ValueError(f'{path}: evaluation protocol mismatch')
    size = summary['image_processor_size']
    if size['shortest_edge'] != 4096 or size['longest_edge'] != 100352:
        raise ValueError(f'{path}: image budget mismatch')
    for key in ('model_path', 'processor_path'):
        if Path(summary[key]).resolve() != model.resolve():
            raise ValueError(f'{path}: {key} mismatch')
    return payload


def dataset_fingerprint():
    """Hash question assets; inventory external image bytes by size and mtime."""
    files = {}
    image_inventory = {}
    for suite, (_, _, args) in suites().items():
        for flag, value in zip(args[::2], args[1::2]):
            if flag == '--split':
                continue
            path = Path(value)
            if flag in ('--image_dir', '--image_root'):
                image_inventory[str(path)] = {str(p.relative_to(path)): [p.stat().st_size, p.stat().st_mtime_ns]
                                             for p in sorted(path.rglob('*')) if p.is_file()}
            elif path.is_file():
                files[str(path)] = sha(path)
            else:
                pattern = '*/validation-*.parquet' if suite == 'mmmu' else '*.parquet'
                for p in sorted(path.glob(pattern)):
                    files[str(p)] = sha(p)
    image_hash = __import__('hashlib').sha256(json.dumps(image_inventory, sort_keys=True).encode()).hexdigest()
    return dict(question_assets_sha256=files, image_inventory_sha256=image_hash,
                image_inventory_policy='relative path, byte size and mtime_ns; image pixels are not hashed')


def protocol(paths, attn):
    scripts = [ROOT / 'evaluation' / item[0] for item in suites().values()]
    scripts += [ROOT / 'evaluation/matched_protocol.py', Path(__file__), HERE / 'common.py',
                ROOT / 'evaluation/summarize_constrained_evals.py']
    return dict(version=1, models={k: dict(model=str(v), processor=str(v)) for k, v in paths.items()},
                min_pixels=4096, max_pixels=100352, context='no truncation; native model context',
                do_sample=False, num_beams=1, repetition_penalty=1.0, attention=attn,
                suites={s: dict(script=x[0], max_new_tokens=x[1], multiple_choice_max_new_tokens=8, arguments=x[2], expected_n=EXPECTED[s])
                        for s, x in suites().items()},
                implementation_sha256={str(p): sha(p) for p in scripts}, datasets=dataset_fingerprint())


def summarize(out):
    policy = read(out / 'protocol.json')
    for method in METHODS:
        read(out / method / 'provenance.json')
        for suite in SUITES:
            validate_result(out / method / f'{suite}.json', suite,
                            Path(policy['models'][method]['model']), policy['attention'])
    data = load_results(out, METHODS)  # also checks matching identities AND gold across all ten rows
    labels = list(SUITES) + ['mean']
    scores = {m: {s: 100 * data[m, s][metric] for s, (_, metric) in SUITES.items()} for m in METHODS}
    for m in METHODS:
        scores[m]['mean'] = sum(scores[m].values()) / 7
    def table(headers, rows):
        return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                         ['| ' + ' | '.join(str(x).replace('|', '\\|').replace('\n', ' ') for x in row) + ' |' for row in rows])
    rows = [[m, *[f'{scores[m][s]:.2f}' for s in labels]] for m in METHODS]
    report = ['# Released-BF16 comparison',
              'Validated all 70 merged results, expected counts, unique identities, matching gold labels, model/processor paths and evaluation policy.',
              table(['Model', *labels], rows),
              'Scores are percentages. The unweighted seven-suite mean is descriptive, not an official leaderboard metric. '
              'sft_2b is the previous vanilla-initialized local SFT export; the four new low-bit runs start independently from released_bf16.']
    delta_rows = []
    for bits in (4, 8):
        grace = f'grace_w{bits}'
        pairs = [(grace, f'qat_w{bits}', 'distillation gain'),
                 (f'released_grace_w{bits}', grace, 'gap to full method'),
                 ('released_bf16', grace, 'gap to BF16 initialization'),
                 (grace, 'vanilla_2b', 'gain over vanilla 2B'),
                 (grace, 'vanilla_8b_teacher', 'difference from teacher'),
                 (grace, 'sft_2b', 'difference from previous SFT')]
        for left, right, name in pairs:
            delta_rows.append([f'W{bits}: {name} ({left} minus {right})',
                               *[f'{scores[left][s] - scores[right][s]:+.2f}' for s in labels]])
    report += ['## Differences (percentage points)', table(['Comparison', *labels], delta_rows)]
    diagnostic = []
    breakdown = []
    for suite in SUITES:
        for method in METHODS:
            summary = data[method, suite]
            diagnostic.append([method, suite, summary['row_errors'], summary.get('parse_fail', 'not reported')])
            if suite == 'hallusionbench':
                for metric in ('qAcc', 'fAcc'):
                    breakdown.append([method, suite, metric, summary[f'{metric}_total_groups'], f'{100 * summary[metric]:.2f}'])
            for field in sorted(k for k in summary if k.startswith('by_')):
                for category, group in summary[field].items():
                    breakdown.append([method, suite, f'{field}/{category}', group['total'], f'{100 * group["acc"]:.2f}'])
    report += ['## Errors and parse failures', table(['Model', 'Suite', 'Errors', 'Parse failures'], diagnostic),
               'Failed parses remain in the denominator. MMMU uses the existing local mixed-answer scorer.',
               '## Additional metrics and categories', table(['Model', 'Suite', 'Category/metric', 'N', 'Score (%)'], breakdown),
               'Acceptance thresholds remain for supervisor discussion. BF16 storage of quantized weights does not measure packed W4/W8 memory or speed.']
    (out / 'comparison.md').write_text('\n\n'.join(report) + '\n')
    for name, header, values in [('comparison.csv', ['model', *labels], rows),
                                 ('deltas.csv', ['comparison', *labels], delta_rows)]:
        with (out / name).open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(values)
    print(f'Validated comparison written to {out / "comparison.md"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--check', action='store_true', help='CPU preflight, no output writes')
    mode.add_argument('--summarize', action='store_true', help='Validate all 70 saved results and write tables')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods):
        parser.error('Duplicate result IDs')
    experiment = Path(os.environ.get('EXPERIMENT_DIR', HERE)).resolve()
    out = Path(os.environ.get('OUT_DIR', experiment / 'evaluation')).resolve()
    paths = model_paths(experiment)
    attn = os.environ.get('ATTN', 'flash_attention_2')
    nproc = int(os.environ.get('NPROC', '2'))
    if nproc < 1 or attn not in ('flash_attention_2', 'sdpa', 'eager'):
        parser.error('Invalid NPROC or ATTN')
    if args.summarize:
        summarize(out)
        return
    if args.dry_run:
        for method in args.methods:
            for suite in SUITES:
                print(shlex.join(command(method, suite, paths, out, nproc, attn)))
        return
    print('Checking checkpoint completeness, provenance, processors and dataset paths on CPU...', flush=True)
    records = {}
    for method in args.methods:
        path = paths[method]
        if method == 'sft_2b':
            validate_export(path)
        elif method.startswith(('qat_', 'grace_')):
            validate_export(path, int(method[-1]), paths['released_bf16'])
        records[method] = provenance(path, REPOS.get(method), released_student=method == 'released_bf16')
    check_processor_compatibility([paths['released_bf16'], *[paths[m] for m in args.methods]])
    # Reuse the existing thorough ScienceQA/SEED image and seven-suite path preflight.
    env = dict(os.environ, PYTHON=sys.executable, METHODS='baseline',
               BASELINE_MODEL=str(paths[args.methods[0]]), BASELINE_PROCESSOR=str(paths[args.methods[0]]))
    subprocess.run(['bash', str(ROOT / 'evaluation/run_all_constrained_evals.sh'), '--check'], env=env, check=True)
    policy = protocol(paths, attn)
    if args.check:
        if (out / 'protocol.json').exists() and read(out / 'protocol.json') != policy:
            raise ValueError('Existing evaluation protocol differs; use a new OUT_DIR')
        print('Preflight passed. No evaluation or GPU allocation performed.')
        return
    # One writer per output directory; protocol and model hashes cannot drift on resume.
    sys.path.insert(0, str(ROOT / 'qwen-vl-finetune/scripts'))
    from run_constrained import lock
    failures = []
    with lock(out / '.evaluation.lock'):
        pin(out / 'protocol.json', policy)
        for method in args.methods:
            pin(out / method / 'provenance.json', records[method])
            for suite in SUITES:
                result = out / method / f'{suite}.json'
                if result.exists():
                    try:
                        validate_result(result, suite, paths[method], attn)
                        print(f'Skip complete: {method}/{suite}', flush=True)
                        continue
                    except (ValueError, KeyError, TypeError, OSError) as exc:
                        print(f'Retry invalid merged result: {exc}', flush=True)
                print(f'Evaluating {method}/{suite}', flush=True)
                try:
                    subprocess.run(command(method, suite, paths, out, nproc, attn), check=True)
                    validate_result(result, suite, paths[method], attn)
                except (subprocess.CalledProcessError, ValueError, KeyError, TypeError, OSError) as exc:
                    failures.append(f'{method}/{suite}: {exc}')
        if failures:
            raise ValueError('Evaluation failures (rerun to retry):\n' + '\n'.join(failures))
        if all((out / m / f'{s}.json').exists() for m in METHODS for s in SUITES):
            summarize(out)
        else:
            print('Selected rows complete; full comparison waits for all ten rows.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f'ERROR: {exc}')
