"""CPU-only artifact checks for the released-BF16 experiment."""
import hashlib
import json
import os
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROJECT = ROOT / 'qwen-vl-finetune'
PREVIOUS = ROOT / 'experiments/constrained_32k_b128_p100352_t2048_seed42'
REPOS = {
    'vanilla_2b': 'Qwen/Qwen3-VL-2B-Instruct',
    'vanilla_8b_teacher': 'Qwen/Qwen3-VL-8B-Instruct',
    'released_bf16': 'ForeverBlue/Qwen3-VL-2B-GRACE-BF16',
    'released_grace_w4': 'ForeverBlue/Qwen3-VL-2B-GRACE-W4G128',
    'released_grace_w8': 'ForeverBlue/Qwen3-VL-2B-GRACE-W8G128',
}
# Pinned released student already downloaded on this machine. Verify actual bytes,
# not just the directory name or the model's self-reported config.
BF16_REVISION = '82f9a267340f65a3619be0e19daf6582d3998a5f'
BF16_WEIGHTS_SHA256 = '4b522cda73d4ebd792e97a89cbca51091287b65c87e705be53f56c035715bfd3'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def pin(path, data):
    path = Path(path)
    if path.exists():
        if read(path) != data:
            raise ValueError(f'Pinned artifact changed: {path}; use a fresh output directory')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
        temp.replace(path)


def model_paths(experiment):
    paths = {key: ROOT / 'models' / repo.split('/')[-1] for key, repo in REPOS.items()}
    paths['released_bf16'] = Path(os.environ.get('BASE_MODEL_DIR', paths['released_bf16'])).resolve()
    paths['vanilla_8b_teacher'] = Path(os.environ.get('TEACHER_MODEL_PATH', paths['vanilla_8b_teacher'])).resolve()
    paths['sft_2b'] = Path(os.environ.get('SFT_MODEL_DIR', PREVIOUS / 'runs/sft')).resolve()
    for bits in (4, 8):
        for method in ('qat', 'grace'):
            paths[f'{method}_w{bits}'] = experiment / f'w{bits}g128/runs/{method}'
    return paths


def checkpoint_files(path):
    config = read(path / 'config.json')
    if config.get('model_type') != 'qwen3_vl' or config.get('quantization_config'):
        raise ValueError(f'{path}: expected unpacked Qwen3-VL research weights, not a packed backend')
    indexes = list(path.glob('model.safetensors.index.json'))
    names = set(read(indexes[0])['weight_map'].values()) if indexes else {'model.safetensors'}
    for name in names:
        if not (path / name).is_file():
            raise ValueError(f'Missing model shard: {path / name}')
        with (path / name).open('rb') as handle:
            header_size = struct.unpack('<Q', handle.read(8))[0]
            if header_size > 100 * 1024 * 1024:
                raise ValueError(f'Invalid safetensors header: {path / name}')
            header = json.loads(handle.read(header_size))
        for key, tensor in header.items():
            if key != '__metadata__' and (key.endswith(('.qweight', '.qzeros')) or
                    (key.endswith('.weight') and tensor['dtype'] not in ('BF16', 'F16', 'F32'))):
                raise ValueError(f'Unsupported packed/non-floating weight: {path / name}: {key}')
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        if not (path / name).is_file():
            raise ValueError(f'Missing processor asset: {path / name}')
    if not any((path / n).is_file() for n in ('processor_config.json', 'preprocessor_config.json')):
        raise ValueError(f'Missing processor config: {path}')
    return sorted(set(names) | {p.name for p in path.iterdir()
                               if p.is_file() and p.suffix in ('.json', '.jinja', '.md')})


def provenance(path, repo=None, released_student=False):
    files = checkpoint_files(path)
    hashes = {name: sha(path / name) for name in files}
    revisions = set()
    for name in files:
        meta = path / '.cache/huggingface/download' / (name + '.metadata')
        if meta.exists():
            lines = meta.read_text().splitlines()
            revisions.add(lines[0])
            # HF LFS etags are SHA256; ordinary files have Git blob hashes.
            if len(lines[1]) == 64 and lines[1] != hashes[name]:
                raise ValueError(f'Download hash mismatch: {path / name}')
    if released_student and hashes.get('model.safetensors') != BF16_WEIGHTS_SHA256:
        raise ValueError('Student is not the pinned released GRACE-BF16 checkpoint')
    if released_student and revisions and revisions != {BF16_REVISION}:
        raise ValueError(f'Unexpected released-BF16 revision: {revisions}')
    if len(revisions) > 1:
        raise ValueError(f'Mixed downloaded revisions: {path}: {revisions}')
    return dict(path=str(path), repository=repo, revision=next(iter(revisions), None),
                files_sha256=hashes, config=read(path / 'config.json'),
                loading_format='unpacked floating-point safetensors, Transformers Qwen3VL',
                generation_config=read(path / 'generation_config.json') if (path / 'generation_config.json').exists() else None,
                identity_verification='pinned weight SHA256' if released_student else 'local download metadata and explicit repository mapping')


def check_processor_compatibility(paths):
    from transformers import AutoProcessor
    reference = None
    for path in dict.fromkeys(paths):
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        tokenizer = processor.tokenizer
        signature = (tokenizer.get_vocab(), tokenizer.special_tokens_map,
                     processor.chat_template, processor.image_processor.patch_size,
                     processor.image_processor.merge_size)
        if reference is not None and signature != reference:
            raise ValueError(f'Incompatible vocabulary, special tokens, template or image grid: {path}')
        reference = signature


def validate_export(path, bits=None, student=None):
    checkpoint_files(path)
    summary = read(path / 'run_summary.json')
    if not (path / 'EXPORT_COMPLETE').is_file() or summary.get('complete') is not True or summary.get('profile') is not False:
        raise ValueError(f'Not a completed full-training export: {path}')
    recipe = read(path / 'launch_recipe.json')
    config = recipe['comparison']['config']
    if summary['global_step'] != config['max_steps'] or summary['global_step'] != 256:
        raise ValueError(f'Expected exactly 256 updates: {path}')
    if bits:
        model_config = read(path / 'config.json')
        if config['qat_bits'] != bits or model_config.get('qat', {}).get('bits') != bits:
            raise ValueError(f'Wrong exported bit width: {path}')
        if model_config['qat'].get('group_size') != 128 or not model_config.get('qat_lora', {}).get('merged'):
            raise ValueError(f'Expected merged G128 LoRA export: {path}')
        if model_config['qat'].get('num_replaced', 0) <= 0:
            raise ValueError(f'No quantized modules: {path}')
        if Path(config['model_path']).resolve() != student.resolve():
            raise ValueError(f'Wrong training initialization: {path}')
        if config.get('student_weights_sha256') != BF16_WEIGHTS_SHA256:
            raise ValueError(f'Missing released-student provenance: {path}')
