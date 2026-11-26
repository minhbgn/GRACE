"""Relocated evidence stays verifiable without cluttering experiment roots."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.protocols import artifact_layout as layout


class ArtifactLayoutTests(unittest.TestCase):
    def test_old_relative_and_absolute_paths_resolve_to_named_runs(self):
        name = 'experiments/qat_int4_token_reduction_576_seed42/'
        old = name + 'results_video_matched_exclusive/full/qat_w4/videomme/resolution_r025/COMPLETE.json'
        new = layout.ROOT / (name + 'outputs/video_matched_exclusive/full/qat_w4/videomme/resolution_r025/COMPLETE.json')
        self.assertEqual(layout.resolve_artifact(old), new)
        self.assertEqual(layout.resolve_artifact(layout.ROOT / old), new)
        self.assertTrue(new.exists())

    def test_longest_prefix_routes_individual_reports_and_diagnostics(self):
        old = 'results/qat_int4_token_reduction_576_seed42/evidence/reports/'
        figure = layout.resolve_artifact(old + 'accuracy_retention_20261125.png')
        diagnostic = layout.resolve_artifact(old + 'runtime_host_snapshot_20261111.json')
        self.assertEqual(figure.parent, layout.REPORTS / 'figures')
        self.assertEqual(diagnostic.parent, layout.OUTPUTS / 'diagnostics')
        self.assertTrue(figure.exists())
        self.assertTrue(diagnostic.exists())

    def test_only_registered_current_and_original_source_bytes_are_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            current = root / 'worker.py'
            archived = root / 'old_worker.py'
            archived.write_text('old source')
            current.write_text('registered relocated source')
            old_hash = hashlib.sha256(archived.read_bytes()).hexdigest()
            new_hash = hashlib.sha256(current.read_bytes()).hexdigest()
            (root / 'sources.json').write_text(json.dumps({'worker.py':dict(
                original_sha256=old_hash, current_sha256=new_hash, snapshot=archived.name)}))
            with patch.object(layout, 'ROOT', root), patch.object(layout, 'LAYOUT', root):
                self.assertTrue(layout.verify_source('worker.py', old_hash))
                self.assertTrue(layout.verify_source('worker.py', new_hash))
                current.write_text('unregistered edit')
                with self.assertRaisesRegex(ValueError, 'outside the recorded output relocation'):
                    layout.verify_source('worker.py', old_hash)

    def test_surface_markdown_and_git_ignore_contract(self):
        self.assertEqual({p.name for p in layout.EXPERIMENT.glob('*.md')}, {'README.md','REPORT.md'})
        self.assertFalse(list(layout.EXPERIMENT.glob('results*')))
        raw = layout.OUTPUTS / 'future_run/predictions.jsonl'
        report = layout.REPORTS / 'metrics/completed_analysis_20261125.json'
        for path, expected in [(raw,0), (report,1)]:
            result = subprocess.run(['git','check-ignore','--no-index',str(path)], cwd=layout.ROOT,
                                    text=True,capture_output=True)
            self.assertEqual(result.returncode, expected, result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
