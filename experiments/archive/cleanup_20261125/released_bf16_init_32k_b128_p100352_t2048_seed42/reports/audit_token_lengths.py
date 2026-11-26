"""CPU-only audit of retained training lengths, text tokens and removed turns."""
import csv
import hashlib
import json
from pathlib import Path
import sys

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / 'qwen-vl-finetune'))
import numpy as np
import torch
from transformers import AutoProcessor
from qwenvl.data.data_processor import _build_messages, IMAGE_TOKEN_INDEX, update_processor_pixels
from qwenvl.data.constrained_data import json_hash, processor_fingerprint, preprocess, token_hash
from qwenvl.train.argument import DataArguments


def annotation_records(path, digest):
    """Stream a JSON array using stdlib, keeping one buffer and one row in memory."""
    decoder = json.JSONDecoder()
    with path.open('rb') as source:
        # Binary chunk boundaries can split UTF-8 characters.
        import codecs
        utf8 = codecs.getincrementaldecoder('utf-8')()
        buffer, pos, started, eof = '', 0, False, False
        while True:
            while pos < len(buffer) and buffer[pos].isspace():
                pos += 1
            if not started and pos < len(buffer):
                assert buffer[pos] == '['
                started = True
                pos += 1
                continue
            if pos < len(buffer) and buffer[pos] == ',':
                pos += 1
                continue
            if pos < len(buffer) and buffer[pos] == ']':
                # Finish hashing trailing whitespace too.
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(block)
                return
            if pos < len(buffer):
                try:
                    item, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    if eof:
                        raise
                else:
                    pos = end
                    yield item
                    continue
            if eof:
                raise ValueError('Unexpected end of annotation JSON')
            block = source.read(1024 * 1024)
            digest.update(block)
            eof = not block
            buffer = buffer[pos:] + utf8.decode(block, final=eof)
            pos = 0


def describe(values):
    x = np.asarray(values)
    return dict(count=len(x), minimum=int(x.min()), mean=float(x.mean()),
                median=float(np.median(x)), p90=float(np.percentile(x, 90)),
                p95=float(np.percentile(x, 95)), p99=float(np.percentile(x, 99)), maximum=int(x.max()))


def main():
    torch.set_num_threads(2)
    manifest = json.loads((EXP / 'subset.json').read_text())
    records = manifest['records']
    assert json_hash(records) == manifest['records_sha256']
    model = ROOT / 'models/Qwen3-VL-2B-GRACE-BF16'
    assert processor_fingerprint(model) == manifest['recipe']['processor']
    proc = AutoProcessor.from_pretrained(model, local_files_only=True)
    proc.tokenizer.model_max_length = 2048
    update_processor_pixels(proc, DataArguments(max_pixels=100352, min_pixels=4096))
    rows = []
    for start in range(0, len(records), 512):
        batch = records[start:start+512]
        rendered = [proc.apply_chat_template(_build_messages(r['sample'], Path(r['sample']['data_path'])),
                                            tokenize=False) for r in batch]
        encoded = proc.tokenizer(rendered, add_special_tokens=False, truncation=False)['input_ids']
        for r, ids in zip(batch, encoded):
            assert ids.count(IMAGE_TOKEN_INDEX) == 1
            # Rendering contains one image placeholder; expansion adds only image tokens.
            nonvisual = len(ids) - 1
            visual = r['length'] - nonvisual
            assert 4 <= visual <= 98, (r['index'], visual)
            rows.append(dict(index=r['index'], total=r['length'], nonvisual=nonvisual,
                             visual=visual, retained_turn_pairs=len(r['sample']['conversations'])//2))
        if len(rows) % 8192 == 0:
            print(f'Text tokenized: {len(rows)}/{len(records)}', flush=True)
    # Check against actual image preprocessing, token IDs and labels saved in the manifest.
    checks = sorted(set(range(0, len(records), 1024)) | set(sorted(range(len(records)),
                    key=lambda i: records[i]['length'], reverse=True)[:16]))
    for i in checks:
        item = preprocess(records[i]['sample'], proc, 2048, 100352)
        assert token_hash(item) == records[i]['token_sha256'], records[i]['index']
        assert item['input_ids'].shape[1] == rows[i]['total']
        assert int((item['input_ids'] == IMAGE_TOKEN_INDEX).sum()) == rows[i]['visual']
    print(f'Full image/token-hash spot checks passed: {len(checks)}', flush=True)
    selected = {r['index']: r for r in records}
    trimmed, changed_other, digest, seen = [], [], hashlib.sha256(), set()
    annotation = ROOT / 'ShareGPT4V/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json'
    for index, original in enumerate(annotation_records(annotation, digest)):
        if index not in selected:
            continue
        seen.add(index)
        before = original['conversations']
        after = selected[index]['sample']['conversations']
        if before != after:
            if before[:len(after)] == after and len(before) > len(after):
                trimmed.append(dict(index=index, original_turn_pairs=len(before)//2,
                                    retained_turn_pairs=len(after)//2))
            else:
                changed_other.append(index)
    assert digest.hexdigest() == manifest['recipe']['annotation_sha256']
    assert seen == selected.keys() and not changed_other
    totals = np.array([r['total'] for r in rows])
    nonvisual = np.array([r['nonvisual'] for r in rows])
    result = dict(date='2026-10-05', subset_records_sha256=manifest['records_sha256'],
        definition='Nonvisual = all input tokens except expanded image_pad tokens; includes user text, assistant answers, role/turn delimiters and vision boundary tokens. No padding. Not just supervised answer tokens.',
        method='Retained sequence lengths from checksum-verified manifest. Render/tokenize every conversation with fingerprint-verified original processor; remove its one image placeholder for nonvisual count; infer visual count by subtraction. Full image preprocessing and saved token hashes verified on longest 16 plus every 1024th row.',
        count=len(rows), full_preprocess_checks=len(checks),
        total=describe(totals), nonvisual=describe(nonvisual), visual=describe([r['visual'] for r in rows]),
        total_at_least={str(k):int((totals>=k).sum()) for k in [1024,1536,1800,1950,2000,2048]},
        longest_records=sorted(rows,key=lambda r:r['total'],reverse=True)[:16],
        max_nonvisual_record=max(rows,key=lambda r:r['nonvisual']),
        retained_conversations_trimmed=len(trimmed), trimmed_details=trimmed,
        removed_turn_pairs=sum(r['original_turn_pairs']-r['retained_turn_pairs'] for r in trimmed),
        preparation_rejections=manifest['rejected'],
        hypothetical_all_images_576=dict(count_over_2048=int((nonvisual+576>2048).sum()),
          maximum_total=int((nonvisual+576).max()),
          note='Upper-bound scenario for these already-retained conversations only; actual dynamic visual counts can be smaller. Not a re-preparation of the original annotation pool.'))
    out = EXP / 'reports'
    (out/'token_lengths.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'token_lengths.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
