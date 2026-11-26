"""Qwen3-VL batch-one reduction before decoder layer 0; no training or hidden fallback."""
from __future__ import annotations
import inspect
import json
import math
from pathlib import Path
import time
import types


def bounded_grid(height, width, budget, factor=32):
    """Aspect-preserving grid with <= budget merged tokens and >= one token."""
    budget = max(1, int(budget))
    scale = min(1., math.sqrt(budget * factor * factor / (height * width)))
    h = max(1, math.floor(height * scale / factor))
    w = max(1, math.floor(width * scale / factor))
    while h * w > budget:
        if h >= w:
            h -= 1
        else:
            w -= 1
    return h * factor, w * factor


def group_plan(mask, ratio, method):
    """Keep text and all temporal groups. Ratios round up per visual group."""
    positions = mask.nonzero(as_tuple=False).flatten().tolist()
    spans = []
    for p in positions:
        if not spans or p != spans[-1][-1] + 1:
            spans.append([])
        spans[-1].append(p)
    mapping = {p: [p] for p in range(len(mask)) if not bool(mask[p])}
    for span in spans:
        count = max(1, math.ceil(len(span) * ratio))
        for i in range(count):
            lo, hi = i * len(span) // count, (i + 1) * len(span) // count
            group = span[lo:hi]
            representative = group[(len(group) - 1) // 2]
            mapping[representative] = group if method == 'merge' else [representative]
    return [(p, mapping[p]) for p in sorted(mapping)]


def compact(embeds, positions, visual_mask, deepstack, ratio, method):
    import torch
    if not 0 < ratio <= 1 or method not in ('uniform', 'merge', 'baseline', 'resolution'):
        raise ValueError('Invalid reduction')
    if ratio == 1 or not bool(visual_mask.any()):
        return embeds, positions, visual_mask, deepstack
    plan = group_plan(visual_mask[0], ratio, method)
    anchors = torch.tensor([p for p, _ in plan], device=embeds.device)
    out = torch.stack([embeds[0, group].mean(0) for _, group in plan])[None]
    reduced_mask = visual_mask[:, anchors]
    # Apply exactly the same averaging to each aligned DeepStack feature tensor.
    visual_indices = visual_mask[0].nonzero(as_tuple=False).flatten().tolist()
    rank = {p: i for i, p in enumerate(visual_indices)}
    visual_groups = [[rank[p] for p in group] for anchor, group in plan if anchor in rank]
    reduced_deep = [torch.stack([level[group].mean(0) for group in visual_groups]) for level in deepstack]
    return out, positions[..., anchors], reduced_mask, reduced_deep


def prepare_features(model, inputs):
    """Compute full visual features and original M-RoPE before compacting the sequence."""
    import torch
    ids = inputs['input_ids']
    if ids.shape[0] != 1 or not bool(inputs.get('attention_mask', torch.ones_like(ids)).all()):
        raise ValueError('This experiment requires batch one without padding')
    core = model.model
    embeddings = core.get_input_embeddings()(ids)
    masks, stacks = [], []
    for pixels_key, grid_key, token, getter in (
        ('pixel_values', 'image_grid_thw', model.config.image_token_id, core.get_image_features),
        ('pixel_values_videos', 'video_grid_thw', model.config.video_token_id, core.get_video_features)):
        if inputs.get(pixels_key) is None:
            continue
        result = getter(inputs[pixels_key], inputs[grid_key], return_dict=True)
        features = torch.cat(result.pooler_output, dim=0).to(embeddings)
        mask = ids == token
        if int(mask.sum()) != len(features):
            raise ValueError('Visual placeholder/feature mismatch')
        embeddings[mask] = features
        masks.append(mask)
        stacks.append(result.deepstack_features)
    visual_mask = (ids == model.config.image_token_id) | (ids == model.config.video_token_id)
    if len(stacks) == 2:
        deepstack = []
        for image_level, video_level in zip(*stacks):
            level = image_level.new_zeros(int(visual_mask.sum()), image_level.shape[-1])
            level[masks[0][visual_mask]] = image_level
            level[masks[1][visual_mask]] = video_level.to(level)
            deepstack.append(level)
    else:
        deepstack = stacks[0] if stacks else []
    kwargs = {k: inputs[k] for k in ('image_grid_thw', 'video_grid_thw', 'attention_mask') if k in inputs}
    if 'mm_token_type_ids' in inspect.signature(core.get_rope_index).parameters:
        kwargs['mm_token_type_ids'] = inputs.get('mm_token_type_ids')
    positions, _ = core.get_rope_index(ids, **kwargs)
    return embeddings, positions, visual_mask, deepstack


def decoder_greedy(model, features, cap, eos_ids, fixed=False):
    """Explicit cache loop: short physical KV sequence, original multimodal coordinates."""
    import torch
    embeddings, positions, mask, deepstack = features
    count = embeddings.shape[1]
    if count + cap > model.config.text_config.max_position_embeddings:
        raise ValueError('Reduced decoder sequence + output reserve exceeds native context')
    # Preserve 3-D coordinates, use compact text positions for causal attention.
    text_positions = torch.arange(count, device=embeddings.device)[None, None]
    position_ids = torch.cat([text_positions, positions], dim=0)
    next_position = int(positions.max()) + 1
    ids, cache = [], None
    prefill_seconds = 0.
    sync = torch.cuda.synchronize if embeddings.is_cuda else lambda: None
    sync(); started = time.perf_counter()
    for step in range(cap):
        attention_mask = torch.ones((1, count + step), dtype=torch.long, device=embeddings.device)
        outputs = model.model.language_model(inputs_embeds=embeddings, position_ids=position_ids,
            attention_mask=attention_mask, past_key_values=cache, use_cache=True,
            visual_pos_masks=mask if step == 0 else None,
            deepstack_visual_embeds=deepstack if step == 0 and deepstack else None,
            return_dict=True)
        token = model.lm_head(outputs.last_hidden_state[:, -1:, :]).argmax(-1)
        cache = outputs.past_key_values
        ids.append(token[0, 0])
        if step == 0:
            sync(); prefill_seconds = time.perf_counter() - started
        if not fixed and int(token[0, 0]) in eos_ids:
            break
        embeddings = model.model.get_input_embeddings()(token)
        position_ids = torch.tensor([count + step, *([next_position + step] * 3)],
                                   device=embeddings.device)[:, None, None]
    sync()
    total = time.perf_counter() - started
    return torch.stack(ids)[None], dict(prefill_seconds=prefill_seconds,
        decode_seconds=total - prefill_seconds, decoder_seconds=total,
        final_kv_tokens=cache.get_seq_length())


def load_model(path, int4, attention):
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(path), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation=attention).eval()
    audit = dict(runtime='BF16', fused_calls=0, converted_modules=0)
    if int4:
        from qwenvl.quantize import awq_linear
        from qwenvl.quantize.qat_to_awq import pack_qat_linear, verify_pack, set_module
        if awq_linear.awq_ext is None:
            raise RuntimeError('Real INT4 kernels required: awq_ext is unavailable; no fallback permitted')
        sidecar = torch.load(Path(path) / 'qat_quantized_weights.bin', map_location='cpu', weights_only=True)
        names = sorted(k[:-len('.log_scale')] for k in sidecar if k.endswith('.log_scale'))
        expected = model.config.qat['num_replaced']
        if not names or len(names) != expected:
            raise ValueError('QAT sidecar module coverage mismatch')
        for name in names:
            weight, scale = sidecar[name + '.weight'], sidecar[name + '.log_scale']
            packed = pack_qat_linear(weight, scale, sidecar.get(name + '.bias'), 4, 128)
            if verify_pack(packed, weight, scale, 4, 128) != 0:
                raise ValueError(f'INT4 integer-code conversion mismatch: {name}')
            def fused(self, x):
                shape, dtype = x.shape[:-1] + (self.out_features,), x.dtype
                if not x.is_cuda:
                    raise RuntimeError('Strict INT4 requires CUDA tensors')
                x2d = x.to(torch.float16).reshape(-1, self.in_features).contiguous()
                # Always fused GEMM, including long prefill. Never dense dequant+matmul.
                chunks = [awq_linear.awq_ext.gemm_forward_cuda(part.contiguous(), self.qweight,
                          self.scales, self.qzeros, 8) for part in x2d.split(256)]
                audit['fused_calls'] += len(chunks)
                out = torch.cat(chunks, dim=0)
                if self.bias is not None:
                    out = out + self.bias
                return out.reshape(shape).to(dtype)
            packed.forward = types.MethodType(fused, packed)
            set_module(model, name, packed)
        del sidecar
        audit.update(runtime='packed INT4; fused GEMM only; FP16 activations inside kernel',
                     converted_modules=len(names), scale_storage='FP16', integer_codes_verified=True)
    return model.to('cuda:0'), audit


class MeasuredGenerator:
    def __init__(self, model, job, attempt, audit):
        self.model, self.job, self.attempt, self.audit = model, job, Path(attempt), audit
        self.index = 0

    def once(self, inputs, cap, fixed=False):
        import torch
        policy = self.job['policy']
        torch.cuda.synchronize(); start = time.perf_counter()
        features = prepare_features(self.model, inputs)
        torch.cuda.synchronize(); encoded = time.perf_counter()
        original = int(features[2].sum())
        effective_ratio = self.job['retention'] if self.job['method'] in ('uniform','merge') else 1.
        features = compact(*features, effective_ratio, self.job['method'])
        torch.cuda.synchronize(); reduced = time.perf_counter()
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        generated, timing = decoder_greedy(self.model, features, cap, eos, fixed)
        torch.cuda.synchronize()
        token_count = inputs['input_ids'].numel()
        record = dict(original_input_visual_tokens=original,
            input_visual_tokens=int(features[2].sum()),
            input_text_tokens=token_count - original, output_text_tokens=generated.numel(),
            vision_and_embedding_seconds=encoded - start, reduction_seconds=reduced - encoded,
            model_seconds=time.perf_counter() - start, **timing)
        return generated, record

    def __call__(self, *args, **inputs):
        import torch
        if args:
            raise ValueError('Use keyword inputs')
        cap = inputs.pop('max_new_tokens', self.job['policy']['max_new_tokens'])
        for k in ('do_sample','num_beams','temperature','pad_token_id','eos_token_id','use_cache'):
            inputs.pop(k, None)
        if self.index < self.job['policy']['profile_samples']:
            for _ in range(self.job['policy']['profile_warmup']):
                self.once(inputs, self.job['policy']['profile_tokens'], fixed=True)
            profile = [self.once(inputs, self.job['policy']['profile_tokens'], fixed=True)[1]['model_seconds']
                       for _ in range(self.job['policy']['profile_repeats'])]
        else:
            profile = []
        torch.cuda.reset_peak_memory_stats()
        before = self.audit['fused_calls']
        generated, record = self.once(inputs, cap)
        if self.job['model'] == 'qat_w4' and self.audit['fused_calls'] == before:
            raise RuntimeError('No fused INT4 execution observed')
        record.update(index=self.index, fixed_output_profile_seconds=profile,
            fixed_output_profile_tokens=self.job['policy']['profile_tokens'],
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            fused_int4_calls=self.audit['fused_calls'] - before,
            generated_token_ids=generated[0].tolist(), max_new_tokens=cap)
        self.index += 1
        with (self.attempt / 'telemetry.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n'); stream.flush()
        # Existing image/video workers slice at the original prompt length.
        return torch.cat([inputs['input_ids'], generated], dim=1)


class BudgetProcessor:
    """Native image preprocessing at the shared training ceiling, per-image resolution targets."""
    def __init__(self, processor, method, ratio, max_pixels=589824):
        self.processor, self.method, self.ratio = processor, method, ratio
        image = processor.image_processor
        image.size = dict(shortest_edge=4096, longest_edge=max_pixels)
        image.min_pixels, image.max_pixels = 4096, max_pixels

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, *args, **kwargs):
        baseline = self.processor(*args, **kwargs)
        if self.method != 'resolution' or self.ratio == 1 or 'image_grid_thw' not in baseline:
            return baseline
        images = kwargs.get('images')
        if not images or len(images) != len(baseline['image_grid_thw']):
            raise ValueError('Resolution reduction expects one flat PIL image per grid entry')
        factor = self.image_processor.patch_size * self.image_processor.merge_size
        resized = []
        for image, grid in zip(images, baseline['image_grid_thw'].tolist()):
            _, gh, gw = grid
            height, width = gh * self.image_processor.patch_size, gw * self.image_processor.patch_size
            h, w = bounded_grid(height, width, math.ceil(gh * gw / self.image_processor.merge_size**2 * self.ratio), factor)
            from PIL import Image
            resized.append(image.resize((w, h), Image.Resampling.BICUBIC))
        kwargs['images'] = resized
        kwargs['images_kwargs'] = dict(kwargs.get('images_kwargs', {}), do_resize=False)
        reduced = self.processor(*args, **kwargs)
        reduced['_baseline_visual_tokens'] = int(sum(g.prod().item() for g in baseline['image_grid_thw']) / self.image_processor.merge_size**2)
        return reduced


def video_inputs(processor, row, path, policy, method, ratio):
    from evaluation.protocols.video_protocol import probe, geometry, build_inputs
    meta = probe(path, dict(fps=policy['video_fps'], longest_side=0), processor)
    vp = processor.video_processor
    factor = vp.patch_size * vp.merge_size
    h, w = bounded_grid(meta['source_height'], meta['source_width'], policy['image_max_pixels'] // factor**2, factor)
    baseline_tokens = math.ceil(meta['sampled_frames'] / vp.temporal_patch_size) * (h // factor) * (w // factor)
    if method == 'resolution' and ratio < 1:
        h, w = bounded_grid(h, w, math.ceil((h // factor) * (w // factor) * ratio), factor)
    meta.update(geometry(meta['sampled_frames'], h, w, vp.patch_size, vp.merge_size, vp.temporal_patch_size))
    inputs, counts = build_inputs(processor, row, path, meta)
    inputs['_baseline_visual_tokens'] = baseline_tokens
    return inputs, meta
