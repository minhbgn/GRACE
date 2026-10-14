#!/usr/bin/env python3
"""Run the shared local evaluators for a configured checkpoint matrix, resumable by suite."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'qwen-vl-finetune/scripts'))
from constrained_experiment import Experiment, check_processor_compatibility, pin, read, sha
from summarize_constrained_evals import SUITES, EXPECTED, validate_result, summarize_matrix


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
    image_hash = hashlib.sha256(json.dumps(image_inventory, sort_keys=True).encode()).hexdigest()
    return dict(question_assets_sha256=files, image_inventory_sha256=image_hash,
                image_inventory_policy='relative path, byte size and mtime_ns; image pixels are not hashed')


def protocol(spec, attn):
    paths = spec.paths
    scripts = [ROOT / 'evaluation' / item[0] for item in suites().values()]
    scripts += [ROOT / 'evaluation/matched_protocol.py', Path(__file__), ROOT / 'qwen-vl-finetune/scripts/constrained_experiment.py',
                ROOT / 'evaluation/summarize_constrained_evals.py']
    return dict(version=1, report_title=spec.spec.get('report_title', 'Checkpoint comparison'),
                report_note=spec.spec.get('report_note', ''), comparisons=spec.spec['comparisons'], experiment_config_sha256=sha(spec.config_path), models={k: dict(model=str(v), processor=str(v)) for k, v in paths.items()},
                min_pixels=4096, max_pixels=100352, context='no truncation; native model context',
                do_sample=False, num_beams=1, repetition_penalty=1.0, attention=attn,
                suites={s: dict(script=x[0], max_new_tokens=x[1], multiple_choice_max_new_tokens=8, arguments=x[2], expected_n=EXPECTED[s])
                        for s, x in suites().items()},
                implementation_sha256={str(p): sha(p) for p in scripts}, datasets=dataset_fingerprint())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--check', action='store_true', help='CPU preflight, no output writes')
    mode.add_argument('--summarize', action='store_true', help='Validate all configured saved results and write tables')
    parser.add_argument('--config', required=True, type=Path, help='Experiment model matrix JSON')
    parser.add_argument('--methods', nargs='+', help='Result IDs from the experiment configuration')
    args = parser.parse_args()
    spec = Experiment(args.config)
    methods = tuple(spec.models)
    args.methods = args.methods or methods
    if any(m not in spec.models for m in args.methods):
        parser.error('Unknown result ID; configured IDs: ' + ', '.join(methods))
    if len(set(args.methods)) != len(args.methods):
        parser.error('Duplicate result IDs')
    experiment = spec.directory
    out = Path(os.environ.get('OUT_DIR', experiment / 'evaluation')).resolve()
    paths = spec.paths
    attn = os.environ.get('ATTN', 'flash_attention_2')
    nproc = int(os.environ.get('NPROC', '2'))
    if nproc < 1 or attn not in ('flash_attention_2', 'sdpa', 'eager'):
        parser.error('Invalid NPROC or ATTN')
    if args.summarize:
        summarize_matrix(out)
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
        if spec.models[method].get('kind') in ('export', 'trained'):
            spec.validate_export(method)
        records[method] = spec.provenance(method)
    check_processor_compatibility([paths[spec.student], *[paths[m] for m in args.methods]])
    # Reuse the existing thorough ScienceQA/SEED image and seven-suite path preflight.
    env = dict(os.environ, PYTHON=sys.executable, METHODS='baseline',
               BASELINE_MODEL=str(paths[args.methods[0]]), BASELINE_PROCESSOR=str(paths[args.methods[0]]))
    subprocess.run(['bash', str(ROOT / 'evaluation/run_all_constrained_evals.sh'), '--check'], env=env, check=True)
    policy = protocol(spec, attn)
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
        if all((out / m / f'{s}.json').exists() for m in methods for s in SUITES):
            summarize_matrix(out)
        else:
            print('Selected rows complete; full comparison waits for all configured rows.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f'ERROR: {exc}')
