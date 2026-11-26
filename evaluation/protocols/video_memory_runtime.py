"""Bound video vision/prefill working memory without changing sampled inputs.

Qwen3-VL vision attention separates temporal patch groups. Encode those groups
independently, stage merged/DeepStack features on the host, and use causal cached
decoder prefill chunks. The original runtime remains unchanged for old results.
"""
from __future__ import annotations

import time
import types
from types import SimpleNamespace

from evaluation.protocols.token_reduction_runtime import (
    MeasuredGenerator, bounded_grid, compact, prepare_features,
)


class HostVideoInputs(dict):
    """Keep flattened video pixels on CPU until each temporal group's transfer."""

    def to(self, device):
        return type(self)({key: (value if key == 'pixel_values_videos' else value.to(device))
                           if hasattr(value, 'to') else value for key, value in self.items()})


def adaptive_video_geometry(processor, row, meta, policy, method, ratio, decoder_token_cap):
    """Choose a shared duration-aware baseline before the requested reduction.

    Reserve the exact prompt/timestamp tokens and the full generation cap. The
    same per-clip adaptive baseline is used across all repaired configurations;
    it is independent of the requested retention/method. Frames never change.
    """
    import math
    from evaluation.protocols.video_protocol import geometry, input_counts

    vp = processor.video_processor
    factor = vp.patch_size * vp.merge_size
    height, width = bounded_grid(meta['source_height'], meta['source_width'],
                                 policy['image_max_pixels'] // factor**2, factor)
    original = dict(meta, **geometry(meta['sampled_frames'], height, width, vp.patch_size,
                                    vp.merge_size, vp.temporal_patch_size))
    counts = input_counts(processor, row, original)
    groups = original['grid'][0]
    visual_budget = decoder_token_cap - policy['max_new_tokens'] - counts['input_text_tokens']
    if visual_budget < groups:
        raise ValueError('All temporal groups plus exact text/output reserve exceed the declared decoder cap')
    adaptive_h, adaptive_w = bounded_grid(height, width, visual_budget // groups, factor)
    adaptive_tokens = groups * (adaptive_h // factor) * (adaptive_w // factor)
    adaptation = dict(applied=(adaptive_h, adaptive_w) != (height, width),
        original_policy_visual_tokens=original['visual_tokens'],
        adaptive_baseline_visual_tokens=adaptive_tokens,
        original_height=height, original_width=width,
        adaptive_baseline_height=adaptive_h, adaptive_baseline_width=adaptive_w,
        exact_text_tokens=counts['input_text_tokens'], output_reserve=policy['max_new_tokens'],
        decoder_token_cap=decoder_token_cap, temporal_groups=groups,
        frame_policy='unchanged 2 fps, full duration', requested_method=method, requested_retention=ratio)
    final_h, final_w = adaptive_h, adaptive_w
    if method == 'resolution' and ratio < 1:
        final_h, final_w = bounded_grid(adaptive_h, adaptive_w,
            math.ceil((adaptive_h // factor) * (adaptive_w // factor) * ratio), factor)
    selected = dict(meta, **geometry(meta['sampled_frames'], final_h, final_w,
                                    vp.patch_size, vp.merge_size, vp.temporal_patch_size))
    selected['adaptation'] = adaptation
    return selected, adaptation


def adaptive_video_inputs(processor, row, path, policy, method, ratio, decoder_token_cap):
    from evaluation.protocols.video_protocol import probe, build_inputs

    meta = probe(path, dict(fps=policy['video_fps'], longest_side=0), processor)
    selected, adaptation = adaptive_video_geometry(processor, row, meta, policy, method, ratio, decoder_token_cap)
    inputs, counts = build_inputs(processor, row, path, selected)
    if counts['input_tokens'] + policy['max_new_tokens'] > decoder_token_cap:
        raise ValueError('Actual processor sequence exceeds declared decoder cap')
    inputs['_baseline_visual_tokens'] = adaptation['adaptive_baseline_visual_tokens']
    inputs['_video_adaptation'] = adaptation
    return HostVideoInputs(inputs), selected


def stream_video_features(original, visual, pixels, grid, group_chunk=1):
    """Preserve clip/time/spatial order and every DeepStack feature.

    The vision encoder's attention cu_seqlens partition at each temporal group;
    its positions are spatial. Chunking at that boundary preserves its graph.
    Only merged outputs are retained; unused unmerged last_hidden_state is freed.
    """
    import torch

    if group_chunk < 1 or pixels.device.type != 'cpu':
        raise ValueError('Streamed vision requires host pixels and a positive group chunk')
    device = next(visual.parameters()).device
    dtype = next(visual.parameters()).dtype
    groups = grid.detach().cpu().tolist()
    if sum(t * h * w for t, h, w in groups) != len(pixels):
        raise ValueError('Video pixels/grid coverage mismatch')
    merged_clips, deep_parts = [], None
    offset = 0
    for temporal, height, width in groups:
        clip_parts = []
        for start in range(0, temporal, group_chunk):
            count = min(group_chunk, temporal - start)
            size = count * height * width
            chunk = pixels[offset:offset + size].to(device=device, dtype=dtype)
            chunk_grid = grid.new_tensor([[count, height, width]])
            result = original(chunk, chunk_grid, return_dict=True)
            clip_parts.append(torch.cat(result.pooler_output, dim=0).cpu())
            if deep_parts is None:
                deep_parts = [[] for _ in result.deepstack_features]
            for parts, features in zip(deep_parts, result.deepstack_features, strict=True):
                parts.append(features.cpu())
            offset += size
            del result, chunk
        merged_clips.append(torch.cat(clip_parts, dim=0).to(device))
    return SimpleNamespace(pooler_output=tuple(merged_clips),
                           deepstack_features=[torch.cat(parts, dim=0).to(device) for parts in deep_parts or []])


def install_streamed_vision(model, group_chunk=1):
    core = model.model
    original = core.get_video_features

    def streamed(self, pixel_values_videos, video_grid_thw, **kwargs):
        if set(kwargs) - {'return_dict'}:
            raise ValueError(f'Unsupported streamed vision arguments: {sorted(kwargs)}')
        return stream_video_features(original, self.visual, pixel_values_videos,
                                     video_grid_thw, group_chunk)

    core.get_video_features = types.MethodType(streamed, core)


def host_features(features):
    embeddings, positions, mask, deepstack = features
    return (embeddings.cpu(), positions.cpu(), mask.cpu(), [level.cpu() for level in deepstack])


def decoder_greedy_chunked(model, features, cap, eos_ids, fixed=False, chunk_size=512):
    """Causal prefill with original M-RoPE and exact aligned DeepStack slices.

    The complete KV cache remains on GPU: this bounds intermediate activations,
    not context memory. There is no input truncation, frame dropping or KV eviction.
    """
    import torch

    embeddings, positions, mask, deepstack = features
    if chunk_size < 1 or cap < 1 or embeddings.shape[0] != 1:
        raise ValueError('Positive chunk/output sizes and batch one are required')
    count = embeddings.shape[1]
    if count + cap > model.config.text_config.max_position_embeddings:
        raise ValueError('Reduced decoder sequence + output reserve exceeds native context')
    device = model.model.get_input_embeddings().weight.device
    sync = torch.cuda.synchronize if device.type == 'cuda' else lambda: None
    text_positions = torch.arange(count, device=positions.device)[None, None]
    position_ids = torch.cat([text_positions, positions], dim=0)
    next_position = int(positions.max()) + 1
    visual_offset = 0
    cache = None
    sync()
    started = time.perf_counter()
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        visual_count = int(mask[:, start:stop].sum())
        local_deepstack = [level[visual_offset:visual_offset + visual_count].to(device) for level in deepstack]
        outputs = model.model.language_model(
            inputs_embeds=embeddings[:, start:stop].to(device),
            position_ids=position_ids[..., start:stop].to(device),
            attention_mask=torch.ones((1, stop), dtype=torch.long, device=device),
            past_key_values=cache, use_cache=True,
            visual_pos_masks=mask[:, start:stop].to(device),
            deepstack_visual_embeds=local_deepstack if visual_count else None,
            return_dict=True)
        cache = outputs.past_key_values
        visual_offset += visual_count
        del local_deepstack
    if visual_offset != int(mask.sum()):
        raise ValueError('Prefill visual coverage mismatch')
    token = model.lm_head(outputs.last_hidden_state[:, -1:, :]).argmax(-1)
    del outputs
    sync()
    prefill_seconds = time.perf_counter() - started
    ids = [token[0, 0]]
    for step in range(1, cap):
        if not fixed and int(token[0, 0]) in eos_ids:
            break
        outputs = model.model.language_model(
            inputs_embeds=model.model.get_input_embeddings()(token),
            position_ids=torch.tensor([count + step - 1, *([next_position + step - 1] * 3)],
                                      device=device)[:, None, None],
            attention_mask=torch.ones((1, count + step), dtype=torch.long, device=device),
            past_key_values=cache, use_cache=True, return_dict=True)
        token = model.lm_head(outputs.last_hidden_state[:, -1:, :]).argmax(-1)
        cache = outputs.past_key_values
        ids.append(token[0, 0])
        del outputs
    sync()
    total = time.perf_counter() - started
    return torch.stack(ids)[None], dict(prefill_seconds=prefill_seconds,
        decode_seconds=total - prefill_seconds, decoder_seconds=total,
        final_kv_tokens=cache.get_seq_length())


class MemoryMeasuredGenerator(MeasuredGenerator):
    """Keep original scoring/profiling/telemetry, with separately pinned execution."""

    def once(self, inputs, cap, fixed=False):
        import torch

        torch.cuda.synchronize()
        started = time.perf_counter()
        features = prepare_features(self.model, inputs)
        torch.cuda.synchronize()
        encoded = time.perf_counter()
        original = int(features[2].sum())
        ratio = self.job['retention'] if self.job['method'] in ('uniform', 'merge') else 1.
        features = compact(*features, ratio, self.job['method'])
        retained = int(features[2].sum())
        # Drop full GPU features before allocating a growing decoder KV cache.
        features = host_features(features)
        torch.cuda.synchronize()
        reduced = time.perf_counter()
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        generated, timing = decoder_greedy_chunked(
            self.model, features, cap, eos, fixed,
            chunk_size=self.job['execution']['prefill_chunk_tokens'])
        torch.cuda.synchronize()
        record = dict(original_input_visual_tokens=original, input_visual_tokens=retained,
            input_text_tokens=inputs['input_ids'].numel() - original,
            output_text_tokens=generated.numel(), vision_and_embedding_seconds=encoded - started,
            reduction_seconds=reduced - encoded, model_seconds=time.perf_counter() - started,
            execution=self.job['execution'], video_adaptation=inputs['_video_adaptation'],
            retention_vs_original_policy=retained / inputs['_video_adaptation']['original_policy_visual_tokens'],
            **timing)
        return generated, record
