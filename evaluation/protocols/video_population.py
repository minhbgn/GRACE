"""Explicit population selection for reduced VideoMME evaluation."""
from collections import Counter

from evaluation.protocols.video_protocol import digest


VIDEOMME_SHORT_MEDIUM = dict(name='short_medium', source_questions=2700,
                            selected_questions=1800, excluded_questions=900,
                            excluded_category='long')


def select_population(suite, rows, info, skip_videomme_long=False):
    """Validate the known 900-question long suffix before excluding it.

    Selection precedes smoke limits and asset readiness checks. Missing assets
    belonging only to excluded questions cannot block this declared subset.
    """
    if not skip_videomme_long or suite != 'videomme' or info['status'] == 'unavailable':
        return rows, info
    kept, excluded = rows[:1800], rows[1800:]
    if (len(rows) != 2700 or len({r['id'] for r in rows}) != 2700
            or Counter(r['category'] for r in kept) != {'short': 900, 'medium': 900}
            or len(excluded) != 900 or any(r['category'] != 'long' for r in excluded)):
        raise ValueError('VideoMME: expected 2700 unique questions with exactly 900 long questions at the end')
    missing = sorted({r['video_id'] for r in kept if not r['asset'] or not r['asset']['size']})
    report = dict(info, source_population=info,
                  population=dict(VIDEOMME_SHORT_MEDIUM),
                  status='incomplete' if missing else 'ready',
                  questions=len(kept), videos=len({r['video_id'] for r in kept}),
                  missing_videos=missing, sample_identity=digest(kept),
                  excluded_ids=[r['id'] for r in excluded],
                  excluded_sample_identity=digest(excluded))
    return kept, report
