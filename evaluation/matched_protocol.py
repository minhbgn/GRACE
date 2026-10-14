"""Opt-in common image and decoding policy for local comparison evaluations.

Old launchers retain their existing behavior unless --fixed_generation is supplied.
"""


def add_protocol_arguments(parser):
    parser.add_argument('--local_scoring', action='store_true', help='Validated deterministic local scoring v1')
    parser.add_argument('--mc_max_new_tokens', type=int, default=8, help='MMMU MC cap; explicitly version changes')
    parser.add_argument('--min_pixels', type=int, default=None)
    parser.add_argument('--max_pixels', type=int, default=None)
    parser.add_argument('--fixed_generation', action='store_true',
                        help='Greedy, one beam, no inherited checkpoint decoding penalties')


def apply_protocol(args, model, processor):
    if args.local_scoring and (args.min_pixels is not None or args.max_pixels is not None):
        raise ValueError('Corrected local protocol requires native processors; no pixel overrides')
    if args.max_new_tokens <= 0 or args.mc_max_new_tokens <= 0:
        raise ValueError('Generation caps must be positive')
    if (args.min_pixels is None) != (args.max_pixels is None):
        raise ValueError('Specify both --min_pixels and --max_pixels')
    if args.min_pixels is not None:
        if not 0 < args.min_pixels <= args.max_pixels:
            raise ValueError('Invalid evaluation pixel limits')
        image = processor.image_processor
        image.size = dict(image.size, shortest_edge=args.min_pixels, longest_edge=args.max_pixels)
        # Older processor versions also use these attributes.
        image.min_pixels, image.max_pixels = args.min_pixels, args.max_pixels
    if args.fixed_generation:
        from transformers import GenerationConfig
        model.generation_config = GenerationConfig(
            do_sample=False, num_beams=1, repetition_penalty=1.0,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            use_cache=True,
        )


def protocol_metadata(args, **suite_settings):
    if not args.fixed_generation:
        return {}
    return {'evaluation_protocol': dict(
        version=1, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
        repetition_penalty=1.0, context='no truncation; native model context',
        attention=args.attn, **suite_settings,
    )}
