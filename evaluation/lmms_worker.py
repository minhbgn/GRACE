#!/usr/bin/env python3
"""One isolated model/task evaluation; called by run_lmms_matrix.py."""
import functools
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from run_lmms_matrix import JUDGED, TASKS, local_mmmu_judge_guard, pin, read, source_check, task_config, validate_result, write


def main():
    job_path = Path(sys.argv[1]).resolve()
    attempt = job_path.parent
    job = read(job_path)
    suite, protocol = job['suite'], job['protocol']
    output = Path(job['output_root'])
    os.environ.update(API_TYPE='openai', MODEL_VERSION=protocol['judges']['mmbench'],
                      OPENAI_API_URL=protocol['judge_endpoint'], NUM_IMAGE='1')
    source_check()
    from lmms_eval import evaluator
    from lmms_eval.tasks import TaskManager
    from lmms_eval.utils import handle_non_serializable
    cfg = task_config(suite, protocol)
    cfg['dataset_path'] = job['dataset']['repo']
    cfg['dataset_kwargs'] = {**cfg.get('dataset_kwargs', {}), 'revision': job['dataset']['revision'], 'token': None}
    # Preserve task-specific prompts, filters, scoring, stop strings and token caps.
    cfg['generation_kwargs'] = {**cfg.get('generation_kwargs', {}),
                                'temperature': 0, 'do_sample': False, 'num_beams': 1}
    audit = {'calls': 0, 'failures': [], 'random_fallbacks': 0}
    if suite in JUDGED:
        import requests
        original_post = requests.post

        def guarded_post(url, *args, **kwargs):
            if url != protocol['judge_endpoint']:
                return original_post(url, *args, **kwargs)
            audit['calls'] += 1
            try:
                response = original_post(url, *args, **kwargs)
                response.raise_for_status()
                content = response.json()['choices'][0]['message']['content']
                if not isinstance(content, str) or not content.strip():
                    raise ValueError('Empty judge response')
                return response
            except Exception as error:
                # Save types only: HTTP errors may contain credentials from custom endpoints.
                audit['failures'].append(type(error).__name__)
                raise
        requests.post = guarded_post
        if suite == 'hallusionbench':
            # YAML !function imports use fresh modules, not canonical sys.modules.
            for metric in cfg['metric_list']:
                namespace = metric['aggregation'].__globals__
                namespace['evaluate_by_chatgpt'] = functools.partial(
                    namespace['evaluate_by_chatgpt'], gpt_model=protocol['judges'][suite])
        else:
            from lmms_eval.tasks.mmbench import mmbench_evals
            original_randint = mmbench_evals.rd.randint
            # Retain the harness's fallback behavior but never certify a score that used it.
            def audited_randint(*args, **kwargs):
                audit['random_fallbacks'] += 1
                return original_randint(*args, **kwargs)
            mmbench_evals.rd = SimpleNamespace(randint=audited_randint)

    original_get_tasks = evaluator.get_task_dict

    def recorded_get_tasks(*args, **kwargs):
        tasks = original_get_tasks(*args, **kwargs)
        task = tasks[TASKS[suite][0]]
        provenance = {'source': job['dataset'], 'splits': {
            name: {'rows': len(split), 'fingerprint': split._fingerprint}
            for name, split in task.dataset.items()}, 'evaluation_rows': len(task.eval_docs)}
        write(attempt / 'dataset.json', provenance)
        pin(output / 'dataset_fingerprints' / f'{suite}.json', provenance)
        return tasks
    evaluator.get_task_dict = recorded_get_tasks
    model_args = {'pretrained': job['model_path'], 'device_map': 'cuda:0',
                  **protocol['model_args']}
    if any(',' in str(v) for v in model_args.values()):
        raise ValueError('Model arguments cannot contain commas')
    try:
        with local_mmmu_judge_guard(suite):
            results = evaluator.simple_evaluate(
                model='qwen3_vl', model_args=','.join(f'{k}={v}' for k, v in model_args.items()),
                tasks=[cfg], task_manager=TaskManager(model_name='qwen3_vl'),
                num_fewshot=0, batch_size=1, device='cuda:0', limit=job['limit'],
                bootstrap_iters=protocol['bootstrap_iters'], log_samples=True, force_simple=True,
                random_seed=protocol['seed'], numpy_random_seed=protocol['seed'],
                torch_random_seed=protocol['seed'], fewshot_random_seed=protocol['seed'],
                cli_args=SimpleNamespace(output_path=str(attempt), process_with_media=False))
        samples = results.pop('samples')[TASKS[suite][0]]
        (attempt / 'results.json').write_text(json.dumps(results, indent=2, default=handle_non_serializable) + '\n')
        with (attempt / 'samples.jsonl').open('w') as f:
            for sample in samples:
                f.write(json.dumps(sample, default=handle_non_serializable) + '\n')
        if audit['failures'] or audit['random_fallbacks']:
            raise RuntimeError('Judge failure or random fallback: outputs retained but not accepted; inspect judge_audit.json')
        validated = validate_result(results, samples, suite, job['limit'])
        pin(output / 'sample_identities' / f'{suite}.json',
            {k: validated[k] for k in ('count', 'sample_identity')})
        write(attempt / 'validated.json', validated)
    finally:
        write(attempt / 'judge_audit.json', audit)


if __name__ == '__main__':
    main()
