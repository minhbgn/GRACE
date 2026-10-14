#!/usr/bin/env python3
"""CPU-only validation/rescoring of archived native responses, never fresh baselines."""
import argparse
import copy
from pathlib import Path

from run_lmms_matrix import read, sha, verified_completion, write
from run_local_development import REFERENCES, SUITES, ID_FIELDS, metadata, source_samples
from local_scoring import PINNED, SCORER, score_rows, corrected_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-root', type=Path, required=True)
    p.add_argument('--circular-reference', type=Path, required=True,
                   help='Previously validated metric_protocol_audit.json')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.resolve().is_relative_to(args.full_root.resolve()):
        p.error('Audit output must be outside the immutable replay root')
    reference = read(args.circular_reference)
    report = dict(scorer=SCORER, inference_runs=0, judge_calls=0, models={},
                  note='Archived-response scoring validation only. Old token IDs/finish reasons are unknown. These are not fresh local_native_greedy256_v1 baselines.',
                  circular_reference_sha256=sha(args.circular_reference))
    sources = [Path(__file__), Path(__file__).with_name('local_scoring.py'),
               Path(__file__).with_name('rescore_saved_protocols.py'),
               PINNED / 'mmmu/utils.py', PINNED / 'mmbench/mmbench_evals.py']
    report['implementation_sha256'] = {str(path): sha(path) for path in sources}
    for model in REFERENCES:
        report['models'][model] = {}
        for suite in SUITES:
            job = args.full_root / model / suite
            done = verified_completion(job)
            if not done:
                raise ValueError(f'Incomplete archived run: {job}')
            path = job / done['attempt'] / 'results.json'
            payload = read(path)
            old = payload['results']
            actual = [(tuple(r[k] for k in ID_FIELDS[suite]), r['gold']) for r in old]
            if sorted(actual, key=str) != sorted(source_samples(suite), key=str):
                raise ValueError(f'Archived dataset coverage differs: {job}')
            if any(r.get('error') for r in old):
                raise ValueError(f'Archived inference errors: {job}')
            rows = score_rows(suite, copy.deepcopy(old), metadata(suite))
            summary = corrected_summary(suite, rows, metadata(suite))
            if suite == 'mmbench':
                for key in ('rows', 'groups', 'group_counts', 'unresolved_rows',
                            'exact_match_circular_percent', 'judge_completion_bounds_percent'):
                    if summary['circular'][key] != reference['models'][model]['mmbench'][key]:
                        raise ValueError(f'MMBench differs from validated audit: {model}/{key}')
                summary['validated_circular_parity'] = True
                del summary['circular']['group_evidence']
            changed = []
            unresolved = []
            for before, after in zip(old, rows):
                identity = {k: after[k] for k in ID_FIELDS[suite]}
                if before['pred'] != after['pred'] or before['correct'] != after['correct']:
                    changed.append(dict(identity=identity, raw=after['raw'], gold=after['gold'],
                                        old_pred=before['pred'], new_pred=after['pred'],
                                        old_correct=before['correct'], new_correct=after['correct'],
                                        extraction_method=after['extraction_method']))
                if after['unresolved']:
                    unresolved.append(dict(identity=identity, raw=after['raw'], gold=after['gold'],
                                           extraction_method=after['extraction_method']))
            report['models'][model][suite] = dict(input=str(path), input_sha256=sha(path),
                total=len(rows), summary=summary, changed=changed, unresolved_outputs=unresolved)
            print(f'{model}/{suite}: score={100 * summary["primary_score"]:.3f}%, changed={len(changed)}, unresolved={summary["unresolved"]}, generation_unknown={summary["generation_unknown"]}', flush=True)
    write(args.output, report)
    print(f'PASS: 35 saved pairs; MMBench matches all five validated circular results. Report: {args.output}')


if __name__ == '__main__':
    main()
