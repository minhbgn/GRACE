"""Deterministic single-image subset and strict CPU-side input validation.

The manifest stores complete conversations, not truncated image token streams.
Every method reads the same records. Bad records raise; there is no retry with
another index. No CUDA operations are used during preparation or loading.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset

from .data_processor import IMAGE_TOKEN_INDEX, VIDEO_TOKEN_INDEX, preprocess_qwen_visual
from .rope2d import get_rope_index_3

SCHEMA_VERSION = 1


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def processor_fingerprint(model_path):
    root = Path(model_path)
    names = ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
             "processor_config.json", "chat_template.jinja")
    return {name: file_hash(root / name) for name in names if (root / name).exists()}


def image_path(sample):
    image = sample.get("image")
    if isinstance(image, list):
        if len(image) != 1:
            raise ValueError("Exactly one image is required")
        image = image[0]
    if not isinstance(image, str) or not image or sample.get("video"):
        raise ValueError("Exactly one image and no video are required")
    return image


def validate_input(item, max_length, max_pixels, merge_size=2, patch_size=16):
    ids, labels = item["input_ids"], item["labels"]
    if ids.ndim != 2 or ids.shape[0] != 1 or not 1 < ids.shape[1] <= max_length:
        raise ValueError(f"Sequence shape {tuple(ids.shape)} exceeds limit {max_length}")
    if labels.shape != ids.shape or not (labels[:, 1:] != -100).any():
        raise ValueError("No supervised next-token targets")
    grid = item.get("image_grid_thw")
    if grid is None or tuple(grid.shape) != (1, 3) or int(grid[0, 0]) != 1:
        raise ValueError("Expected one still-image grid")
    patches = int(grid.prod())
    # Validate the processed grid, not merely the requested processor settings.
    if patches * patch_size**2 > max_pixels:
        raise ValueError("Processed image exceeds pixel budget")
    expected = patches // merge_size**2
    if expected < 4 or int((ids == IMAGE_TOKEN_INDEX).sum()) != expected:
        raise ValueError("Incomplete image tokens or fewer than four RCKA tokens")
    if (ids == VIDEO_TOKEN_INDEX).any():
        raise ValueError("Video tokens are not allowed")
    pixels = item.get("pixel_values")
    if pixels is None or pixels.shape[0] != patches or not torch.isfinite(pixels).all():
        raise ValueError("Invalid image tensor")


def preprocess(sample, processor, max_length, max_pixels):
    item = preprocess_qwen_visual([sample], processor, truncate=False)
    merge = processor.image_processor.merge_size
    validate_input(item, max_length, max_pixels, merge, processor.image_processor.patch_size)
    item["position_ids"], _ = get_rope_index_3(
        merge, item["input_ids"], image_grid_thw=item["image_grid_thw"]
    )
    item["attention_mask"] = [item["input_ids"].shape[1]]
    return item


def token_hash(item):
    return json_hash({k: item[k].tolist() for k in ("input_ids", "labels", "image_grid_thw")})


def fit_complete_turns(sample, processor, max_length, max_pixels):
    sample = deepcopy(sample)
    turns = sample.get("conversations", [])
    if not turns or len(turns) % 2 or any(
        turn.get("from") != ("human" if i % 2 == 0 else "gpt")
        for i, turn in enumerate(turns)
    ):
        raise ValueError("Expected alternating complete human/gpt turns")
    while turns:
        item = preprocess_qwen_visual([sample], processor, truncate=False)
        if item["input_ids"].shape[1] <= max_length:
            # Also constructs positional indices, exactly as training will.
            return sample, preprocess(sample, processor, max_length, max_pixels)
        del turns[-2:]
    raise ValueError("First complete question/answer pair exceeds token budget")


def source_name(sample):
    # Source stratification only; task labels are not present consistently.
    parts = Path(image_path(sample)).parts
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0]


def allocate_quotas(counts, total):
    population = sum(counts.values())
    if total <= 0 or total > population:
        raise ValueError(f"Requested {total} examples from {population} eligible records")
    quotas = {key: total * count // population for key, count in counts.items()}
    order = sorted(counts, key=lambda key: (-(total * counts[key] % population), key))
    for key in order[:total - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def prepare_manifest(annotation, data_root, processor, model_path, count, seed,
                     max_length, max_pixels, min_pixels, output):
    output = Path(output)
    recipe = dict(schema=SCHEMA_VERSION, annotation_sha256=file_hash(annotation),
                  data_root=str(Path(data_root).resolve()), count=count, seed=seed,
                  max_length=max_length, max_pixels=max_pixels, min_pixels=min_pixels,
                  processor=processor_fingerprint(model_path))
    if output.exists():
        manifest = json.loads(output.read_text())
        if manifest["recipe"] != recipe:
            raise ValueError("Subset recipe changed; use a new EXPERIMENT_DIR")
        if json_hash(manifest["records"]) != manifest["records_sha256"]:
            raise ValueError("Subset manifest checksum mismatch")
        print(f"Reusing validated subset: {output}", flush=True)
        return manifest
    annotations = json.loads(Path(annotation).read_text())
    groups = defaultdict(list)
    excluded = Counter()
    for index, sample in enumerate(annotations):
        try:
            groups[source_name(sample)].append((index, sample))
        except (ValueError, TypeError, AttributeError):
            excluded["not_single_image"] += 1
    quotas = allocate_quotas({key: len(value) for key, value in groups.items()}, count)
    records = []
    for source in sorted(groups):
        pool = groups[source]
        random.Random(f"{seed}:{source}").shuffle(pool)
        accepted = 0
        for index, original in pool:
            if accepted == quotas[source]:
                break
            sample = dict(original, data_path=recipe["data_root"])
            try:
                path = Path(recipe["data_root"]) / image_path(sample)
                stat = path.stat()
                fitted, item = fit_complete_turns(sample, processor, max_length, max_pixels)
                records.append(dict(index=index, source=source, sample=fitted,
                                    length=item["input_ids"].shape[1],
                                    token_sha256=token_hash(item),
                                    image_size=stat.st_size, image_mtime_ns=stat.st_mtime_ns))
                accepted += 1
                if len(records) % 256 == 0:
                    print(f"Validated {len(records)}/{count}; rejected={dict(excluded)}", flush=True)
            except (OSError, ValueError, RuntimeError, KeyError, IndexError) as error:
                excluded[type(error).__name__ + ": " + str(error)[:100]] += 1
        if accepted != quotas[source]:
            raise ValueError(f"Source {source}: only {accepted} valid examples; need {quotas[source]}. "
                             "Reduce SUBSET_SIZE; no silent source redistribution. " + str(excluded))
    random.Random(seed).shuffle(records)
    manifest = dict(recipe=recipe, records=records, records_sha256=json_hash(records),
                    source_counts=quotas, rejected=dict(excluded))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False) + "\n")
    temporary.replace(output)
    return manifest


class ConstrainedDataset(Dataset):
    def __init__(self, manifest_path, processor, max_length, max_pixels, profile=False):
        manifest = json.loads(Path(manifest_path).read_text())
        self.records = manifest["records"]
        if json_hash(self.records) != manifest["records_sha256"]:
            raise ValueError("Subset manifest checksum mismatch")
        self.processor = processor
        self.max_length, self.max_pixels = max_length, max_pixels
        if profile:
            # Repeated largest records exercise the actual envelope on both ranks.
            largest = sorted(self.records, key=lambda record: record["length"], reverse=True)[:8]
            self.records = (largest * (len(self.records) // len(largest) + 1))[:len(self.records)]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        sample = record["sample"]
        stat = (Path(sample["data_path"]) / image_path(sample)).stat()
        if (stat.st_size, stat.st_mtime_ns) != (record["image_size"], record["image_mtime_ns"]):
            raise ValueError(f"Image changed since validation: {image_path(sample)}")
        item = preprocess(sample, self.processor, self.max_length, self.max_pixels)
        if token_hash(item) != record["token_sha256"]:
            raise ValueError(f"Preprocessing changed for annotation {record['index']}")
        return item
