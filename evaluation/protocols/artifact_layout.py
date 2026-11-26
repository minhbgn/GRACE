"""Current artifact locations and explicit resolution of immutable old records.

Historical JSON and completion hashes are never rewritten for a directory move.
Only the recorded relocation map is applied when reading a saved artifact path.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / 'experiments/qat_int4_token_reduction_576_seed42'
OUTPUTS = EXPERIMENT / 'outputs'
REPORTS = EXPERIMENT / 'reports'
REPOSITORY_REPORTS = ROOT / 'reports'
LAYOUT = ROOT / 'evaluation/provenance/output_layout_20261125'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resolve_artifact(path):
    """Resolve a repository-relative or historical absolute artifact location."""
    path = Path(path)
    if path.is_absolute():
        try:
            name = path.relative_to(ROOT).as_posix()
        except ValueError:
            return path
    else:
        name = path.as_posix()
    prefixes = json.loads((LAYOUT / 'paths.json').read_text())['prefixes']
    for old in sorted(prefixes, key=len, reverse=True):
        if name == old or name.startswith(old + '/'):
            return ROOT / (prefixes[old] + name[len(old):])
    return ROOT / name


def verify_source(name, expected):
    """Verify current bytes or an explicitly recorded pre-relocation source.

For a migrated source, both the archived original and the registered current
implementation must match. An unrecorded later edit remains an error. Fresh
protocols always hash current source bytes directly, including this resolver.
"""
    actual = sha(ROOT / name)
    if actual == expected:
        return True
    sources = json.loads((LAYOUT / 'sources.json').read_text())
    entry = sources.get(name)
    if (entry and entry['original_sha256'] == expected and
            entry.get('current_sha256') == actual and sha(ROOT / entry['snapshot']) == expected):
        return True
    raise ValueError(f'Pinned source changed outside the recorded output relocation: {name}')
