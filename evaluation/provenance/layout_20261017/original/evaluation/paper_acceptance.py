"""Published Qwen3 Table 6 targets; tolerances assess results, never tune them."""
import math

ORDER = ('hallusionbench', 'mmbench', 'scienceqa', 'ai2d', 'mmmu', 'seedbench', 'mmstar')
PAPER = {name: dict(zip(ORDER, values)) for name, values in {
    'vanilla_2b': [51.4, 78.4, 81.4, 76.9, 53.4, 71.2, 58.3],
    'vanilla_8b_teacher': [61.1, 84.5, 85.0, 85.7, 69.6, 77.5, 70.9],
    'released_bf16': [66.9, 86.4, 86.2, 81.3, 72.1, 76.7, 67.3],
    'released_grace_w4': [65.4, 84.6, 84.3, 79.5, 70.5, 75.1, 65.8],
    'released_grace_w8': [66.1, 85.5, 85.3, 80.4, 71.3, 75.9, 66.5],
}.items()}
SOURCE = 'https://arxiv.org/html/2601.22709v5#A1.T6'


def assess(method, suite, score, *, judged=False, smoke=False):
    suite = 'seedbench' if suite == 'seedbench_img' else suite
    target = PAPER.get(method, {}).get(suite)
    if target is None:
        return {'status': 'no_published_target'}
    if score is None:
        return {'paper_percent': target, 'status': 'missing'}
    if smoke:
        return {'paper_percent': target, 'status': 'smoke_only'}
    if not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError('Score must be a finite percentage')
    delta = score - target
    gap = abs(delta)
    status = ('pass' if gap <= 8 + 1e-9 else 'fail') if judged else (
        'pass' if gap <= 3 + 1e-9 else 'barely_acceptable' if gap <= 5 + 1e-9 else 'fail')
    return {'paper_percent': target, 'score_percent': score, 'delta_pp': delta,
            'status': status, 'preferred_tolerance_pp': 8 if judged else 3,
            'maximum_tolerance_pp': 8 if judged else 5}
