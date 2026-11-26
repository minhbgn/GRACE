"""Shared experiment configuration and CPU-only checkpoint validation."""
import hashlib
import json
import os
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'qwen-vl-finetune'


class Experiment:
    """Resolve declarative model mappings; relative asset paths start at GRACE."""
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.spec = read(self.config_path)
        if self.spec.get('schema') != 1:
            raise ValueError('Unsupported experiment configuration schema')
        self.directory = Path(os.environ.get('EXPERIMENT_DIR', self.config_path.parent)).resolve()
        self.models = self.spec['models']
        self.student = self.spec['student']
        self.teacher = self.spec['teacher']
        self.bits = tuple(self.spec['bit_widths'])
        if not self.bits or len(set(self.bits)) != len(self.bits) or any(b not in (4, 8) for b in self.bits):
            raise ValueError('bit_widths must contain unique supported widths (4, 8)')
        self.paths = {}
        for name, model in self.models.items():
            base = self.directory if model.get('relative_to') == 'experiment' else ROOT
            path = base / model['path']
            self.paths[name] = Path(os.environ.get(model.get('environment', ''), path)).resolve()
        for name in (self.student, self.teacher):
            if name not in self.models:
                raise ValueError(f'Missing model mapping: {name}')
        for pair in self.spec['comparisons']:
            if pair['left'] not in self.models or pair['right'] not in self.models:
                raise ValueError(f'Unknown comparison model: {pair}')

    def provenance(self, name):
        model = self.models[name]
        return provenance(self.paths[name], model.get('repository'),
                          expected_weights_sha256=model.get('weights_sha256'),
                          expected_revision=model.get('revision'))

    def validate_export(self, name):
        model = self.models[name]
        validate_export(self.paths[name], bits=model.get('bits'),
                        student=self.paths[self.student],
                        expected_steps=model.get('expected_steps'),
                        expected_student_sha256=self.models[self.student].get('weights_sha256'))


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


def provenance(path, repo=None, *, expected_weights_sha256=None, expected_revision=None):
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
    if expected_weights_sha256 and hashes.get('model.safetensors') != expected_weights_sha256:
        raise ValueError('Student is not the pinned initialization checkpoint')
    if expected_revision and revisions and revisions != {expected_revision}:
        raise ValueError(f'Unexpected checkpoint revision: {revisions}')
    if len(revisions) > 1:
        raise ValueError(f'Mixed downloaded revisions: {path}: {revisions}')
    return dict(path=str(path), repository=repo, revision=next(iter(revisions), None),
                files_sha256=hashes, config=read(path / 'config.json'),
                loading_format='unpacked floating-point safetensors, Transformers Qwen3VL',
                generation_config=read(path / 'generation_config.json') if (path / 'generation_config.json').exists() else None,
                identity_verification='pinned weight SHA256' if expected_weights_sha256 else 'local download metadata and explicit repository mapping')


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


def validate_export(path, bits=None, student=None, *, expected_steps=None, expected_student_sha256=None):
    checkpoint_files(path)
    summary = read(path / 'run_summary.json')
    if not (path / 'EXPORT_COMPLETE').is_file() or summary.get('complete') is not True or summary.get('profile') is not False:
        raise ValueError(f'Not a completed full-training export: {path}')
    recipe = read(path / 'launch_recipe.json')
    config = recipe['comparison']['config']
    if summary['global_step'] != config['max_steps'] or (expected_steps is not None and summary['global_step'] != expected_steps):
        raise ValueError(f'Unexpected completed update count: {path}')
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
        if expected_student_sha256 and config.get('student_weights_sha256') != expected_student_sha256:
            raise ValueError(f'Missing released-student provenance: {path}')
