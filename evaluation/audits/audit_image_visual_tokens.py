#!/usr/bin/env python3
"""CPU-only reconstruction of native-processor visual tokens in the seven image suites.

Uses the real evaluators' loaders/image selection and the installed processor's
patch-count utility. No weights, generation, CUDA allocation or pixel overrides.
Counts image occurrences per question, rather than deduplicating source images.
"""
from __future__ import annotations

# Support both direct script execution and `python -m evaluation.<group>.<module>`.
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.paths import EVALUATION_ROOT, source_file

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

from evaluation.runners.run_constrained_matrix_evals import ROOT, suites
from evaluation.audits.summarize_constrained_evals import EXPECTED


IMAGE_LABELS = dict(scienceqa='ScienceQA', ai2d='AI2D', mmmu='MMMU', mmstar='MMStar',
                    seedbench_img='SEEDBench image', hallusionbench='HallusionBench', mmbench='MMBench')


def image_prompt_and_count(suite, worker, args, row):
    """Match evaluator prompts/image selection without preprocessing image pixels."""
    if suite == 'scienceqa':
        path = worker.resolve_image_path(row, args.split, args.image_root)
        if row.get('image') and path is None:
            raise FileNotFoundError(f'ScienceQA image missing: {row["pid"]}')
        return worker.format_prompt(row), int(path is not None)
    if suite == 'mmmu':
        indices = {int(match.group(1)) for match in worker.IMAGE_TAG_RE.finditer(str(row.get('question', '')))}
        count = sum(worker._is_present(row.get(f'image_{index}')) for index in indices)
        if not count and worker._is_present(row.get('image_1')):
            count = 1
        if str(row.get('question_type') or 'multiple-choice').lower().startswith('multi'):
            options = worker._parse_options(row.get('options'))
            if not options:
                raise ValueError('MMMU multiple-choice row has no options')
            prompt = worker.format_prompt_mc(row['question'], options)
        else:
            prompt = worker.format_prompt_open(row['question'])
        return prompt, count
    if suite == 'hallusionbench':
        value = row.get('image')
        present = (bool(value.get('bytes')) if isinstance(value, dict) else
                   bool(value) if isinstance(value, (bytes, bytearray)) else isinstance(value, Image.Image))
        return worker.format_prompt(row['question']), int(present)
    if suite == 'mmbench':
        return worker.format_prompt(row['question'], row.get('hint'), worker.collect_options(row)), 1
    options = list(row['options']) if suite == 'ai2d' else worker.collect_options(row)
    return worker.format_prompt(row['question'], options), 1


def image_text_tokens(processor, prompt, image_count):
    content = [{'type': 'image'} for _ in range(image_count)] + [{'type': 'text', 'text': prompt}]
    rendered = processor.apply_chat_template([{'role': 'user', 'content': content}],
                                            tokenize=False, add_generation_prompt=True)
    ids = processor.tokenizer(rendered, add_special_tokens=False)['input_ids']
    placeholders = ids.count(processor.image_token_id)
    if placeholders != image_count:
        raise ValueError('Unexpected image placeholders in reconstructed prompt')
    return len(ids) - placeholders


def master_video_rows(report):
    """Use a complete reference population, never extrapolate an ordered prefix."""
    wanted = [('tempcompass', 'TempCompass', 1580, None),
              ('videomme', 'VideoMME short', 900, 'short'),
              ('videomme', 'VideoMME medium', 900, 'medium'),
              ('videommmu_perception', 'Video-MMMU Perception', 300, None),
              ('videommmu_comprehension', 'Video-MMMU Comprehension', 300, None),
              ('videommmu_adaptation', 'Video-MMMU Adaptation', 300, None)]
    if report['policy'].get('fps') != 2 or report['policy'].get('longest_side') != 256:
        raise ValueError('Master video profile requires the declared 2 fps / 256 px policy')
    result = []
    for suite, label, count, category in wanted:
        choices = [e for e in report['runs'] if e['suite'] == suite and e['status'] == 'complete']
        choices.sort(key=lambda e: e['method'] != 'vanilla_2b')
        selected = None
        for entry in choices:
            metrics = entry.get('categories', {}).get(category, {}) if category else entry.get('metrics', {})
            if metrics.get('questions') == count:
                selected = entry, metrics
                break
        if selected is None:
            raise ValueError(f'No complete reference population for {label}; cannot treat a partial prefix as full')
        entry, metrics = selected
        comparisons = []
        for other in choices:
            values = other.get('categories', {}).get(category, {}) if category else other.get('metrics', {})
            if values.get('questions') == count:
                comparisons.append(dict(method=other['method'],
                    visual_mean_difference=values['input_visual_tokens_mean'] - metrics['input_visual_tokens_mean'],
                    text_mean_difference=values['input_text_tokens_mean'] - metrics['input_text_tokens_mean']))
        result.append(dict(suite=suite + ('_' + category if category else ''), label=label, modality='video',
            questions=count, mean_visual_input_tokens=metrics['input_visual_tokens_mean'],
            mean_text_input_tokens=metrics['input_text_tokens_mean'], reference_model=entry['method'],
            measurement='whole-population saved reference; projection to models using the same input policy',
            source_matrix=entry['matrix'], source_attempt=entry['attempt'],
            source_completion_sha256=entry['completion_marker_sha256'],
            cross_model_checks=comparisons))
    return result


def create_master_report(args):
    from evaluation.runners.run_video_matrix import read, sha
    cache = read(args.visual_cache)
    processor_path = args.processor.resolve()
    if cache['limit_per_suite'] is not None or Path(cache['processor_path']).resolve() != processor_path:
        raise ValueError('Visual cache must cover whole suites for the selected image processor')
    for name, checksum in cache['processor_config_sha256'].items():
        if sha(processor_path / name) != checksum:
            raise ValueError(f'Image processor changed: {name}')
    processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
    if processor.image_processor.size != cache['image_size']:
        raise ValueError('Cached image size differs from the current processor')
    image_matrix = args.visual_cache.parent.parent / 'local_development/local_native_greedy256_v1/full'
    frozen = read(image_matrix / 'protocol.json')['datasets']['question_assets_sha256']
    for file, checksum in frozen.items():
        if sha(Path(file)) != checksum:
            raise ValueError(f'Image question assets changed: {file}')
    definitions, results = suites(), []
    for suite, label in IMAGE_LABELS.items():
        script, _, flags = definitions[suite]
        data_args = SimpleNamespace(**{flag[2:]: value for flag, value in zip(flags[::2], flags[1::2])})
        worker = load_worker(suite, script)
        rows = source_rows(suite, worker, data_args)
        stats = cache['suites'][suite]
        if len(rows) != EXPECTED[suite] or stats['per_question']['count'] != len(rows):
            raise ValueError(f'Image suite coverage mismatch: {suite}')
        total_text, total_images, text_only = 0, 0, 0
        for row in tqdm(rows, desc='text ' + suite):
            prompt, count = image_prompt_and_count(suite, worker, data_args, row)
            total_text += image_text_tokens(processor, prompt, count)
            total_images += count
            text_only += count == 0
        if total_images != stats['per_image']['count'] or text_only != stats['text_only_questions']:
            raise ValueError(f'Image selection differs from the cached visual audit: {suite}')
        results.append(dict(suite=suite, label=label, modality='image', questions=len(rows),
            mean_visual_input_tokens=stats['per_question']['mean'], mean_text_input_tokens=total_text / len(rows),
            reference_model='qat_w4', reference_processor=str(processor_path),
            measurement='whole-population CPU reconstruction; visual counts from the validated processor cache',
            text_input_tokens_total=total_text, images=total_images, text_only_questions=text_only,
            evaluator_sha256=sha(ROOT / 'evaluation' / script)))
        print(f'{label}: visual={results[-1]["mean_visual_input_tokens"]:.2f}, text={total_text / len(rows):.2f}', flush=True)
    video = read(args.video_report)
    results.extend(master_video_rows(video))
    prefix = args.master_output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).astimezone(ZoneInfo('Asia/Singapore')).strftime('%Y-%m-%d %H:%M:%S SGT')
    report = dict(generated_at_singapore=timestamp, auditor_sha256=sha(Path(__file__)), rows=results,
        definitions=dict(averaging='per evaluated question over the full suite; each image/video occurrence counts',
            visual='merged visual input tokens', text='all nonvisual input tokens; question/options/hints, instructions, chat and visual boundaries; video includes timestamps',
            outputs='generated answer tokens excluded; these depend on the model',
            projection='input-profile projection across models requires identical preprocessing, tokenizer and chat/prompt policies; never extrapolate a duration-ordered partial prefix'),
        image_policy=dict(processor=str(processor_path), image_size=cache['image_size'],
                          max_visual_tokens_per_image=cache['per_image_ceiling'],
                          processor_config_sha256=cache['processor_config_sha256']),
        video_policy=video['policy'], excluded_suites=[dict(suite='videomme_long', questions=900,
            reason='Excluded by the current evaluation policy; not included in the master table')],
        sources=dict(image_visual_cache=str(args.visual_cache.resolve()), image_visual_cache_sha256=sha(args.visual_cache),
                     image_matrix=str(image_matrix.resolve()), image_protocol_sha256=sha(image_matrix / 'protocol.json'),
                     video_report=str(args.video_report.resolve()), video_report_sha256=sha(args.video_report)))
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with prefix.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['suite', 'label', 'modality', 'questions',
            'mean_visual_input_tokens', 'mean_text_input_tokens', 'reference_model', 'measurement'])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in results)
    lines = ['# Master image and video suite input-token table', '', f'Updated {timestamp}.', '',
        '**Means are tokens per question over the whole declared suite**, including text-only image-suite questions '
        'with zero visual tokens and repeated media used by different questions. **Text means input text**, including '
        'question/options, instructions, chat/vision markers, and video timestamps. Generated answers are excluded.', '',
        '| Suite | Modality | Questions | Mean visual input tokens | Mean text input tokens |',
        '| --- | --- | ---: | ---: | ---: |']
    lines += ['| ' + ' | '.join([row['label'], row['modality'], f'{row["questions"]:,}',
              f'{row["mean_visual_input_tokens"]:,.2f}', f'{row["mean_text_input_tokens"]:,.2f}']) + ' |' for row in results]
    lines += ['', '**Excluded:** VideoMME long (900 questions). Short and medium are separate 900-question suites above.', '',
        'Image reference: QAT W4 native processor, 4,096–589,824 pixels, **576 merged visual tokens maximum per image**. '
        'Multi-image questions can exceed 576. The whole-population visual audit is reused; full text inputs are '
        'reconstructed on CPU with the real suite prompt builders and native chat template. '
        'MMBench counts all 4,329 evaluation rotations, not 1,164 base questions.', '',
        'Video reference: complete Vanilla 2B runs, **2 fps / 256 px longest side**, without frame-count clipping. '
        'All six included video populations already have complete reference coverage: **no extrapolation from missing '
        'questions is needed**. These counts can be used as projections for other models under the same input policy.', '',
        'The values are reference input profiles, not universal model-independent constants. Image native processor '
        'budgets can differ across checkpoints; different tokenizers/chat templates can change text input. '
        'Generated answer length is model-dependent and is not projected here.', '',
        'Reconstruction and aggregation only; no model weights loaded and no inference run.', '',
        f'Source image audit: `{args.visual_cache.name}`. Source video audit: `{args.video_report.name}`. '
        'The companion JSON records reference policies, source checksums, completion provenance and cross-model checks.', '']
    prefix.with_suffix('.md').write_text('\n'.join(lines))
    print(prefix.with_suffix('.md'))


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
    parser.add_argument('--master-output-prefix', type=Path, help='Create a combined full-suite image/video table')
    parser.add_argument('--visual-cache', type=Path, help='Whole-suite visual JSON from this image auditor')
    parser.add_argument('--video-report', type=Path, help='Validated combined video-token report JSON')
    args = parser.parse_args()
    if args.master_output_prefix:
        if not args.visual_cache or not args.video_report or args.limit or args.suites:
            parser.error('Master mode requires --visual-cache and --video-report, and uses all whole suites')
        create_master_report(args)
        return
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
