#!/usr/bin/env python3
"""Check complete, validated local reference results before the LoRA stage."""
import argparse
import math
from pathlib import Path

from paper_acceptance import assess
from run_lmms_matrix import Experiment, read, sha, verified_completion, write
from run_upstream_matrix import SUITES, TOKENS, source_check, validate

REFERENCES = ('vanilla_2b', 'vanilla_8b_teacher', 'released_bf16',
              'released_grace_w4', 'released_grace_w8')
LORA_MODELS = ('qat_w4', 'grace_w4', 'qat_w8', 'grace_w8')


def check_gate(output, model_paths, *, accept_barely=False):
    """Recompute scores from hashed full results; never trust a stale summary."""
    full = output / 'full'
    protocol = read(full / 'protocol.json')
    if protocol.get('limit') is not None or 'limit' not in protocol:
        raise ValueError('Stage two requires full results, not a smoke run')
    if protocol.get('source') != source_check() or protocol.get('tokens') != TOKENS:
        raise ValueError('Stage-one protocol is not the pinned upstream local protocol')
    accepted = {'pass', 'barely_acceptable'} if accept_barely else {'pass'}
    checks = []
    identities = {}
    for method in REFERENCES:
        for suite in SUITES:
            job = full / method / suite
            done = verified_completion(job)
            if done is None:
                checks.append({'method': method, 'suite': suite, 'status': 'missing'})
                continue
            required = {'command.json', 'results.json', 'validated.json'}
            if not required.issubset(done['sha256']):
                raise ValueError(f'Missing artifact hashes: {method}/{suite}')
            payload = read(job / done['attempt'] / 'results.json')
            verified = validate(payload, suite, model_paths[method], None)
            if not math.isclose(verified['score_percent'], done['score_percent'], abs_tol=1e-9):
                raise ValueError(f'Completion score disagrees with predictions: {method}/{suite}')
            if any(verified[key] != done[key] for key in ('count', 'sample_identity')):
                raise ValueError(f'Completion sample provenance differs: {method}/{suite}')
            identity = (verified['count'], verified['sample_identity'])
            if suite in identities and identities[suite] != identity:
                raise ValueError(f'Question identities/labels differ across models: {suite}')
            identities[suite] = identity
            checks.append({'method': method, 'suite': suite,
                           **assess(method, suite, verified['score_percent']),
                           'completion_sha256': sha(job / 'COMPLETE.json')})
    failures = [row for row in checks if row['status'] not in accepted]
    return {'passed': not failures, 'expected_pairs': 35,
            'completed_pairs': sum(row['status'] != 'missing' for row in checks),
            'tolerance_pp': 5 if accept_barely else 3, 'checks': checks,
            'protocol_sha256': sha(full / 'protocol.json'),
            'note': 'All seven local scorers are unjudged; no 8-point judge allowance. Numerical agreement alone does not prove paper-protocol identity.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--accept-barely', action='store_true')
    args = parser.parse_args()
    exp = Experiment(args.experiment)
    report = check_gate(args.output, exp.paths, accept_barely=args.accept_barely)
    write(args.output / 'stage1_gate.json', report)
    lines = ['# Reference-stage acceptance', '',
             f"Gate: {'PASS' if report['passed'] else 'BLOCKED'}. "
             f"Complete: {report['completed_pairs']}/35; tolerance: ±{report['tolerance_pp']} pp per pair.",
             '', report['note'], '',
             '| Model | Suite | Paper % | Local % | Delta pp | Status |',
             '| --- | --- | ---: | ---: | ---: | --- |']
    for row in report['checks']:
        values = ['—' if row.get(key) is None else f'{row[key]:.2f}'
                  for key in ('paper_percent', 'score_percent', 'delta_pp')]
        lines.append('| ' + ' | '.join([row['method'], row['suite'], *values, row['status']]) + ' |')
    (args.output / 'stage1_gate.md').write_text('\n'.join(lines) + '\n')
    print(lines[2])
    print(f"Details: {args.output / 'stage1_gate.md'}")
    raise SystemExit(0 if report['passed'] else 2)


if __name__ == '__main__':
    main()
