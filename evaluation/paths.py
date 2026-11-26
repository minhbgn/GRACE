"""Stable asset roots and source lookup for the organized evaluation package."""
from pathlib import Path

EVALUATION_ROOT = Path(__file__).resolve().parent
GRACE_ROOT = EVALUATION_ROOT.parent
SOURCE_PATHS = {'audit_image_visual_tokens.py': 'audits/audit_image_visual_tokens.py',
 'audit_local_scoring.py': 'audits/audit_local_scoring.py',
 'check_re_evaluation_gate.py': 'audits/check_re_evaluation_gate.py',
 'download_video_benchmarks.py': 'data_tools/download_video_benchmarks.py',
 'download_video_benchmarks.sh': 'data_tools/download_video_benchmarks.sh',
 'lmms_worker.py': 'workers/lmms_worker.py',
 'local_scoring.py': 'protocols/local_scoring.py',
 'matched_protocol.py': 'protocols/matched_protocol.py',
 'paper_acceptance.py': 'protocols/paper_acceptance.py',
 'paths.py': 'paths.py',
 'prepare_lmms_eval.sh': 'data_tools/prepare_lmms_eval.sh',
 'rescore_saved_protocols.py': 'audits/rescore_saved_protocols.py',
 'run_all_constrained_evals.sh': 'runners/run_all_constrained_evals.sh',
 'run_constrained_matrix_evals.py': 'runners/run_constrained_matrix_evals.py',
 'run_lmms_matrix.py': 'runners/run_lmms_matrix.py',
 'run_local_development.py': 'runners/run_local_development.py',
 'run_upstream_matrix.py': 'runners/run_upstream_matrix.py',
 'run_video_matrix.py': 'runners/run_video_matrix.py',
 'summarize_constrained_evals.py': 'audits/summarize_constrained_evals.py',
 'video_population.py': 'protocols/video_population.py',
 'video_protocol.py': 'protocols/video_protocol.py',
 'video_runtime.py': 'workers/video_runtime.py',
 'video_worker.py': 'workers/video_worker.py'}


def source_file(name):
    """Resolve an owned source filename for subprocess launch and provenance."""
    return EVALUATION_ROOT / SOURCE_PATHS[name]
