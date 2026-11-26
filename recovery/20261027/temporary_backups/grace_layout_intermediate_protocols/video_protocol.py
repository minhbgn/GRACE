"""Offline video data adapters and an auditable, unsaturated input policy."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / 'evaluation/video_benchmarks'
SUITES = {'tempcompass': ('TempCompass', 1580), 'videomme': ('VideoMME', 2700),
          'videommmu_perception': ('VideoMMMU', 300),
          'videommmu_comprehension': ('VideoMMMU', 300),
          'videommmu_adaptation': ('VideoMMMU', 300)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def inventory(base):
    """Index extracted assets and ZIP members without bulk extraction."""
    assets = {}
    for path in sorted(base.rglob('*.mp4')):
        if 'question_only' not in str(path.relative_to(base)):
            assets[path.stem] = {'path': str(path.resolve()), 'size': path.stat().st_size,
                                 'mtime_ns': path.stat().st_mtime_ns}
    for path in sorted(base.glob('*.zip')):
        if 'question_only' in path.name:
            continue
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                if entry.filename.endswith('.mp4') and Path(entry.filename).stem not in assets:
                    assets[Path(entry.filename).stem] = {
                        'archive': str(path.resolve()), 'member': entry.filename,
                        'size': entry.file_size, 'crc': entry.CRC,
                        'archive_size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
    return assets


def load_suite(suite, data_root=DEFAULT_DATA):
    import pandas as pd
    folder, expected = SUITES[suite]
    base = Path(data_root) / folder / 'data'
    if suite == 'tempcompass':
        shards = sorted((base / 'multi-choice').glob('test-*.parquet'))
    elif suite == 'videomme':
        shards = sorted((base / 'videomme').glob('test-*.parquet'))
    else:
        shards = sorted((base / suite.split('_')[1].capitalize()).glob('test-*.parquet'))
    # Empty suites are excluded, partially downloaded suites are errors.
    if not shards:
        return [], {'status': 'unavailable', 'reason': 'No local annotation shards', 'base': str(base)}
    frame = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    if len(frame) != expected:
        raise ValueError(f'{suite}: expected {expected} questions, found {len(frame)}')
    assets = inventory(base)
    rows = []
    for index, row in enumerate(frame.to_dict('records')):
        if suite == 'tempcompass':
            video = str(row['video_id'])
            question = row['question']
            options = dict(re.findall(r'^([A-J])\.\s*(.+)$', question, re.M))
            gold = row['answer'].split('.')[0].strip()
            uid, category, kind = str(index), row['dim'], 'multiple-choice'
        elif suite == 'videomme':
            video = row['videoID']
            options = dict(zip('ABCDEFGHIJ', [re.sub(r'^[A-J]\.\s*', '', x) for x in row['options']]))
            question = row['question'] + '\n' + '\n'.join(f'{k}. {v}' for k, v in options.items())
            gold, uid, category, kind = row['answer'], row['question_id'], row['duration'], 'multiple-choice'
        else:
            video = row['id']
            uid, category, kind = video, row['qa_type'], row['question_type']
            gold = row['answer']
            options = dict(zip('ABCDEFGHIJ', [re.sub(r'^[A-J]\.\s*', '', x) for x in row['options']]))
            question = row['question']
            if kind == 'multiple-choice':
                question += '\n' + '\n'.join(f'{k}. {v}' for k, v in options.items())
            if suite.endswith('adaptation'):
                question = ('Watch and learn the video content, then apply what you learned to answer '
                            'the following question. The question image is at the end of the video.\n' + question)
            else:
                question += '\nPlease ignore the Quiz question in the last frame of the video.'
        if kind == 'multiple-choice':
            if not options or gold not in options:
                raise ValueError(f'{suite}/{uid}: invalid choices/gold')
            question += '\nAnswer with only the option letter.'
        else:
            question += '\nAnswer briefly.'
        rows.append(dict(id=str(uid), video_id=video, asset=assets.get(video), prompt=question,
                         options=options, gold=gold, question_type=kind, category=str(category)))
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError(f'{suite}: duplicate question IDs')
    missing = sorted({r['video_id'] for r in rows if not r['asset'] or not r['asset']['size']})
    return rows, dict(status='ready' if not missing else 'incomplete', questions=len(rows),
                      videos=len({r['video_id'] for r in rows}), missing_videos=missing,
                      annotation_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in shards},
                      sample_identity=digest(rows))


@contextmanager
def local_video(asset, scratch=None):
    if 'path' in asset:
        yield asset['path']
    else:
        with tempfile.TemporaryDirectory(prefix='grace-video-', dir=scratch) as directory:
            path = Path(directory) / 'video.mp4'
            with zipfile.ZipFile(asset['archive']) as archive, archive.open(asset['member']) as source, path.open('wb') as target:
                shutil.copyfileobj(source, target)
            yield str(path)


def sampling(total_frames, source_fps, fps):
    """Uniform coverage of the entire clip, including both endpoints; no frame cap."""
    if total_frames <= 0 or source_fps <= 0 or not math.isfinite(source_fps):
        raise ValueError('Invalid video metadata')
    n = total_frames if fps == 0 else min(total_frames, max(1, math.ceil(total_frames / source_fps * fps)))
    if n == 1:
        return [0]
    return [round(i * (total_frames - 1) / (n - 1)) for i in range(n)]


def geometry(frames, height, width, patch, merge, temporal, longest_side=None):
    factor = patch * merge
    scale = min(1, longest_side / max(height, width)) if longest_side else 1
    h, w = max(factor, round(height * scale / factor) * factor), max(factor, round(width * scale / factor) * factor)
    grid = [math.ceil(frames / temporal), h // patch, w // patch]
    return dict(height=h, width=w, grid=grid, visual_tokens=math.prod(grid) // merge**2,
                # Qwen3-VL's longest_edge is TOTAL pixels over all padded frames.
                total_pixels=grid[0] * temporal * h * w)


def probe(path, policy, processor):
    import decord
    reader = decord.VideoReader(path, ctx=decord.cpu(0), num_threads=1)
    total, fps = len(reader), float(reader.get_avg_fps())
    first = reader[0]
    height, width = first.shape[:2]
    indices = sampling(total, fps, policy['fps'])
    vp = processor.video_processor
    shape = geometry(len(indices), height, width, vp.patch_size, vp.merge_size,
                     vp.temporal_patch_size, policy['longest_side'])
    return dict(source_frames=total, source_fps=fps, source_height=height, source_width=width,
                duration_seconds=total / fps, frame_indices=indices, sampled_frames=len(indices), **shape)


def prompt_text(processor, row):
    return processor.apply_chat_template([{'role': 'user', 'content': [
        {'type': 'video'}, {'type': 'text', 'text': row['prompt']}]}],
        tokenize=False, add_generation_prompt=True)


def input_counts(processor, row, meta):
    """Exact text count without materializing millions of repeated visual IDs."""
    times = processor._calculate_timestamps(list(meta['frame_indices']), meta['source_fps'],
                                           processor.video_processor.temporal_patch_size)
    units = ''.join(f'<{t:.1f} seconds>{processor.vision_start_token}{processor.video_token}{processor.vision_end_token}'
                    for t in times)
    text = prompt_text(processor, row).replace(
        processor.vision_start_token + processor.video_token + processor.vision_end_token, units, 1)
    ids = processor.tokenizer(text, truncation=False)['input_ids']
    placeholders = ids.count(processor.video_token_id)
    if placeholders != meta['grid'][0]:
        raise ValueError('Timestamp/placeholder reconstruction mismatch')
    text_tokens = len(ids) - placeholders
    return dict(input_visual_tokens=meta['visual_tokens'], input_text_tokens=text_tokens,
                input_tokens=text_tokens + meta['visual_tokens'])


def build_inputs(processor, row, path, meta):
    import decord
    import torch
    import torch.nn.functional as F
    from transformers.video_utils import VideoMetadata
    vp = processor.video_processor
    reader = decord.VideoReader(path, ctx=decord.cpu(0), num_threads=1)
    # Resize uint8 frames individually before stacking to bound CPU memory.
    frames = []
    for index in meta['frame_indices']:
        frame = torch.from_numpy(reader[index].asnumpy()).permute(2, 0, 1)
        if list(frame.shape[-2:]) != [meta['height'], meta['width']]:
            frame = F.interpolate(frame[None].float(), size=(meta['height'], meta['width']),
                                  mode='bicubic', align_corners=False, antialias=True)[0].clamp(0, 255).round().to(torch.uint8)
        frames.append(frame)
    video = torch.stack(frames)
    metadata = VideoMetadata(total_num_frames=meta['source_frames'], fps=meta['source_fps'],
                             frames_indices=list(meta['frame_indices']))
    # The bound is above the entire actual input; disabling resize also prevents
    # adaptive Qwen budgets from covertly reducing long videos' resolution.
    vp.size = {'shortest_edge': 1, 'longest_edge': meta['total_pixels'] + 1}
    inputs = processor(text=[prompt_text(processor, row)], videos=[video],
                       videos_kwargs={'do_sample_frames': False, 'do_resize': False,
                                      'video_metadata': [metadata]},
                       padding=False, truncation=False, return_tensors='pt')
    actual_grid = inputs['video_grid_thw'][0].tolist()
    if actual_grid != meta['grid']:
        raise ValueError(f'Unexpected processor grid: {actual_grid} != {meta["grid"]}')
    ids = inputs['input_ids'][0]
    visual = int((ids == processor.video_token_id).sum())
    actual = dict(input_visual_tokens=visual, input_text_tokens=len(ids) - visual, input_tokens=len(ids))
    if actual != input_counts(processor, row, meta):
        raise ValueError('Actual processor tokens disagree with preflight')
    return inputs, actual
