"""Import certified video predictions into a compatible declared population."""
import datetime as dt
import json
from pathlib import Path
import shutil

from evaluation.runners.run_video_matrix import (
    completed, digest, pin, read, sha, summary_from_rows, validate_attempt, write)


def compatible_policy(old, new):
    # These fields describe orchestration/loading documentation or population,
    # rather than sampling, prompts, generation or scoring. Row identities are
    # checked separately, and imported predictions retain their original policy.
    metadata = {'version', 'cuda_allocator', 'loading_format', 'videomme_population'}
    versions = {'video_unsaturated_v2', 'video_resilient_v3'}
    return (old.get('version') in versions and new.get('version') in versions
            and {k: v for k, v in old.items() if k not in metadata}
            == {k: v for k, v in new.items() if k not in metadata})


def reuse_completed(source_root, job_dir, job, identity, model_provenance):
    """Copy validated predictions; return False if this pair has no source marker.

    Never modify the source, reuse smoke as full, accept missing coverage, or
    transplant a COMPLETE marker without validating its referenced artifacts.
    """
    source_root = Path(source_root).resolve()
    source_dir = source_root / job['method'] / job['suite']
    if not (source_dir / 'COMPLETE.json').exists():
        return False
    source_identity = read(source_root / 'protocol.json')
    if source_identity['limit'] is not None:
        raise ValueError(f'Reuse requires a full-population source, not smoke: {source_root}')
    if source_identity['experiment'] != identity['experiment']:
        raise ValueError(f'Reuse experiment mismatch: {source_root}')
    if not compatible_policy(source_identity['policy'], job['protocol']):
        raise ValueError(f'Reuse inference policy mismatch: {source_root}')
    # The legacy computation remains byte-identical; the new parent runner adds
    # population selection and supervision. Refuse altered processing/scoring.
    required = {'run_video_matrix.py', 'video_protocol.py', 'video_worker.py',
                'local_scoring.py', 'paths.py'}
    required |= {'video_worker_resilient.py', 'video_runtime.py'} & set(source_identity['code'])
    if any(source_identity['code'].get(k) != identity['code'].get(k) for k in required):
        raise ValueError(f'Reuse computation code mismatch: {source_root}')
    if any(identity['environment'].get(k) != value for k, value in source_identity['environment'].items()):
        raise ValueError(f'Reuse environment mismatch: {source_root}')
    if read(source_dir.parent / 'model_provenance.json') != model_provenance:
        raise ValueError(f'Reuse model provenance mismatch: {source_dir}')
    marker = read(source_dir / 'COMPLETE.json')
    required_files = {'job.json', 'predictions.jsonl', 'summary.json', 'processor.json'}
    if not required_files <= set(marker['sha256']):
        raise ValueError(f'Reuse completion marker lacks required hashes: {source_dir}')
    # Check job.json before trusting its population for full source validation.
    source_attempt = source_dir / marker['attempt']
    if sha(source_attempt / 'job.json') != marker['sha256']['job.json']:
        raise ValueError(f'Reuse job changed: {source_attempt}')
    original = read(source_attempt / 'job.json')
    if any(original[k] != job[k] for k in ('method', 'suite', 'model_path')):
        raise ValueError(f'Reuse job identity mismatch: {source_attempt}')
    if original['protocol'] != source_identity['policy']:
        raise ValueError(f'Reuse source policy mismatch: {source_attempt}')
    completed(source_dir, original['rows'], original['protocol'])
    by_id = {r['id']: r for r in original['rows']}
    if len(by_id) != len(original['rows']) or any(by_id.get(r['id']) != r for r in job['rows']):
        raise ValueError(f'Reuse question/source identity mismatch: {source_attempt}')
    predictions = {r['id']: r for r in
                   (json.loads(line) for line in (source_attempt / 'predictions.jsonl').read_text().splitlines())}
    selected = [predictions[r['id']] for r in job['rows']]
    attempt = job_dir / ('attempt_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    attempt.mkdir(parents=True)
    write(attempt / 'job.json', job)
    with (attempt / 'predictions.jsonl').open('w') as stream:
        for row in selected:
            stream.write(json.dumps(row) + '\n')
    shutil.copyfile(source_attempt / 'processor.json', attempt / 'processor.json')
    write(attempt / 'summary.json', summary_from_rows(selected))
    write(attempt / 'reuse.json', dict(source_attempt=str(source_attempt),
          source_completion_sha256=sha(source_dir / 'COMPLETE.json'),
          source_protocol=original['protocol'], source_run_identity=digest(source_identity),
          source_questions=len(original['rows']), selected_questions=len(selected),
          note='Imported saved predictions; no model inference. Processor metadata is from the source run.'))
    summary = validate_attempt(attempt, job['rows'], job['protocol'])
    pin(job_dir / 'COMPLETE.json', dict(attempt=attempt.name, summary=summary,
        sha256={name: sha(attempt / name) for name in required_files | {'reuse.json'}}))
    print(f'Reused complete: {job["method"]}/{job["suite"]}; '
          f'{len(selected)}/{len(original["rows"])} saved predictions from {source_attempt}', flush=True)
    return True
