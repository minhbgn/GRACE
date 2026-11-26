"""Relocation contracts: entry points, asset roots and historical provenance."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.paths import EVALUATION_ROOT, GRACE_ROOT, SOURCE_PATHS, source_file


class LayoutTests(unittest.TestCase):
    def test_registered_sources_exist_and_root_stays_small(self):
        for name in SOURCE_PATHS:
            self.assertTrue(source_file(name).is_file(), name)
        self.assertEqual({p.name for p in EVALUATION_ROOT.glob('*.py')}, {'__init__.py', 'paths.py'})
        self.assertEqual(list(EVALUATION_ROOT.glob('*.sh')), [])
        self.assertEqual(list(EVALUATION_ROOT.glob('README*')), [])

    def test_original_sources_match_snapshot_checksums(self):
        archive = EVALUATION_ROOT / 'provenance/layout_20261017'
        manifest = json.loads((archive / 'manifest.json').read_text())
        for old, entry in manifest['files'].items():
            source = archive / 'original' / old
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), entry['sha256'], old)
        # Preserve the hashes referenced by the pre-layout GPU validation report.
        report = GRACE_ROOT / 'experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/reports/video_evaluation_validation.json'
        if report.exists():
            checksums = json.loads(report.read_text())['source']['code_sha256']
            for name, expected in checksums.items():
                source = archive / 'original/evaluation' / name
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), expected, name)

    def test_direct_entry_points_work_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, PYTHONPATH='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            for name in ('run_video_matrix.py', 'run_local_development.py', 'run_lmms_matrix.py',
                         'run_upstream_matrix.py', 'download_video_benchmarks.py'):
                with self.subTest(source=name):
                    result = subprocess.run([sys.executable, str(source_file(name)), '--help'],
                        cwd=directory, env=env, text=True, capture_output=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_asset_and_archive_roots_did_not_move(self):
        from evaluation.protocols.video_protocol import DEFAULT_DATA
        from evaluation.runners.run_upstream_matrix import ARCHIVE, source_check
        self.assertEqual(DEFAULT_DATA, EVALUATION_ROOT / 'video_benchmarks')
        self.assertEqual(ARCHIVE, EVALUATION_ROOT / 'provenance/upstream_20260925')
        source_check()  # Validate the untouched upstream archive's own manifest.

    def test_old_smoke_artifact_checksums_still_match(self):
        output = GRACE_ROOT / 'experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/video_evaluation'
        markers = list(output.glob('video_unsaturated_v1_*/smoke_1/*/*/COMPLETE.json'))
        if not markers:
            self.skipTest('Local historical smoke artifacts are unavailable')
        for marker in markers:
            done = json.loads(marker.read_text())
            for name, checksum in done['sha256'].items():
                artifact = marker.parent / done['attempt'] / name
                self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), checksum)

    def test_documentation_index_links_resolve(self):
        index = EVALUATION_ROOT / 'docs/README.md'
        for target in re.findall(r'\]\(([^)]+)\)', index.read_text()):
            self.assertTrue((index.parent / target).exists(), target)

    def test_shell_launchers_parse(self):
        paths = list((EVALUATION_ROOT / 'runners').glob('*.sh'))
        paths += list((EVALUATION_ROOT / 'data_tools').glob('*.sh'))
        paths += list((GRACE_ROOT / 'experiments').glob('*/run_*evaluation.sh'))
        for path in paths:
            result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
