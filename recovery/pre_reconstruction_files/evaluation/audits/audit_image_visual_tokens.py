#!/usr/bin/env python3
"""CPU-only reconstruction of native-processor visual tokens in the seven image suites.

Uses the real evaluators' loaders/image selection and the installed processor's
patch-count utility. No weights, generation, CUDA allocation or pixel overrides.
Counts image occurrences per question, rather than deduplicating source images.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

from run_constrained_matrix_evals import ROOT, suites
from summarize_constrained_evals import EXPECTED


def load_worker(suite, script):
    spec = importlib.util.spec_from_file_location('token_audit_' + suite, ROOT / 'evaluation' / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_rows(suite, worker, args):
    if suite == 'scienceqa':
        return worker.load_data(args.problems, args.splits, args.split)
    if suite == 'seedbench_img':
        rows, _ = worker.load_data(args.json_path)
        return [r for r in rows if str(r.get('data_type', '')).lower() == 'image']
    if suite == 'mmmu':
        frame = worker.load_data(args.data_dir, args.split)
    else:
        frame = worker.load_data(args.tsv if suite in ('mmbench', 'mmstar') else args.data_dir)
    return frame.to_dict(orient='records')


def row_images(suite, worker, args, row):
    if suite == 'mmmu':
        return worker.collect_images(row)
    if suite == 'scienceqa':
        path = worker.resolve_image_path(row, args.split, args.image_root)
        # The evaluator permits questions with no image, but missing declared
        # assets must not silently lower this audit's token average.
        if row.get('image') and path is None:
            raise FileNotFoundError(f"ScienceQA image missing: {row['pid']}")
        return [Image.open(path).convert('RGB')] if path else []
    if suite == 'seedbench_img':
        path = worker.resolve_image_path(row, args.image_dir)
        if path is None:
            raise FileNotFoundError(f"SEED image missing: {row.get('data_id')}")
        return [Image.open(path).convert('RGB')]
    image = (worker._decode_image_b64(row['image']) if suite in ('mmbench', 'mmstar')
             else worker._decode_image(row.get('image')))
    if image is None and suite != 'hallusionbench':
        raise ValueError(f'{suite}: missing required image')
    return [] if image is None else [image]


def distribution(values):
    ordered = sorted(values)
    n = len(ordered)
    return dict(count=n, total=sum(ordered), mean=sum(ordered) / n if n else None,
                min=ordered[0] if n else None, max=ordered[-1] if n else None,
                p50=ordered[math.ceil(n * .50) - 1] if n else None,
                p95=ordered[math.ceil(n * .95) - 1] if n else None)


def summarize(per_question, per_image, ceiling):
    return dict(per_question=distribution(per_question), per_image=distribution(per_image),
                text_only_questions=sum(t == 0 for t in per_question),
                images_at_ceiling=sum(t == ceiling for t in per_image),
                fraction_images_at_ceiling=(sum(t == ceiling for t in per_image) / len(per_image)
                                            if per_image else None),
                questions_above_single_image_ceiling=sum(t > ceiling for t in per_question))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processor', required=True, type=Path, help='An exported model directory')
    parser.add_argument('--suites', help='Comma-separated suite IDs; default: all seven')
    parser.add_argument('--limit', type=int, help='First N questions per suite (preview only)')
    parser.add_argument('--verify-first', type=int, default=2,
                        help='Verify patch counts against actual image_grid_thw for N image questions per suite')
    parser.add_argument('--output', type=Path, help='Optional summary JSON; existing files are protected')
    args = parser.parse_args()
    if (args.limit is not None and args.limit <= 0) or args.verify_first < 0:
        parser.error('--limit must be positive; --verify-first must be nonnegative')
    if args.output and args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    definitions = suites()
    chosen = args.suites.split(',') if args.suites else list(definitions)
    if len(set(chosen)) != len(chosen) or any(s not in definitions for s in chosen):
        parser.error('Unknown/duplicate suite; choices: ' + ','.join(definitions))
    processor_path = args.processor.resolve()
    processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
    image_processor = processor.image_processor
    if not image_processor.do_resize or not hasattr(image_processor, 'get_number_of_image_patches'):
        raise ValueError('This audit requires the native resizing Qwen image processor with patch-count utility')
    merge_area = image_processor.merge_size ** 2
    ceiling = image_processor.size['longest_edge'] // (image_processor.patch_size ** 2 * merge_area)
    configs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for name in ('processor_config.json', 'preprocessor_config.json')
               if (p := processor_path / name).is_file()}
    report = dict(processor_path=str(processor_path), processor_config_sha256=configs,
                  transformers_version=transformers.__version__, image_size=dict(image_processor.size),
                  patch_size=image_processor.patch_size, merge_size=image_processor.merge_size,
                  per_image_ceiling=ceiling, limit_per_suite=args.limit,
                  measurement='CPU reconstruction from current source images and native processor; not saved inference telemetry',
                  averaging='Question-weighted; text-only questions contribute zero; repeated image occurrences count each time',
                  suites={})
    all_questions, all_images = [], []
    print('Suite             Questions  Images  Mean/question  Mean/image  Max/image  At ceiling', flush=True)
    for suite in chosen:
        script, _, flags = definitions[suite]
        data_args = SimpleNamespace(**{flag[2:]: value for flag, value in zip(flags[::2], flags[1::2])})
        worker = load_worker(suite, script)
        rows = source_rows(suite, worker, data_args)
        if len(rows) != EXPECTED[suite]:
            raise ValueError(f'{suite}: source population {len(rows)} differs from expected {EXPECTED[suite]}')
        if args.limit:
            rows = rows[:args.limit]
        question_tokens, image_tokens, verified = [], [], 0
        for row in tqdm(rows, desc=suite):
            images = row_images(suite, worker, data_args, row)
            try:
                patches = [image_processor.get_number_of_image_patches(im.height, im.width, images_kwargs={})
                           for im in images]
                if any(p % merge_area for p in patches):
                    raise ValueError('Image patch count is not divisible by the merge area')
                counts = [p // merge_area for p in patches]
                if images and verified < args.verify_first:
                    grids = image_processor(images=images, return_tensors='pt')['image_grid_thw'].tolist()
                    actual = [math.prod(grid) // merge_area for grid in grids]
                    if counts != actual:
                        raise ValueError(f'{suite}: patch utility {counts} disagrees with actual grids {actual}')
                    verified += 1
                question_tokens.append(sum(counts))
                image_tokens.extend(counts)
            finally:
                for image in images:
                    image.close()
        stats = summarize(question_tokens, image_tokens, ceiling)
        stats['questions_verified_against_processor_grids'] = verified
        report['suites'][suite] = stats
        all_questions.extend(question_tokens)
        all_images.extend(image_tokens)
        mean_image = stats['per_image']['mean']
        print(f"{suite:17} {len(question_tokens):9} {len(image_tokens):7} "
              f"{stats['per_question']['mean']:14.2f} "
              f"{mean_image if mean_image is not None else 0:11.2f} "
              f"{str(stats['per_image']['max']):>10} {stats['images_at_ceiling']:11}", flush=True)
        del rows, worker
    report['aggregate'] = summarize(all_questions, all_images, ceiling)
    print('Aggregate:\n' + json.dumps(report['aggregate'], indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
        print(f'Saved: {args.output}', flush=True)


if __name__ == '__main__':
    main()
