"""Offline subprocess checks: tiny archives, fake HF CLI, local ntfy receiver."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "download_video_benchmarks.sh"
sys.path.insert(0, str(SCRIPT.parent))
import download_video_benchmarks as downloader


class SpaceChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.runner = downloader.Runner(SimpleNamespace(
            root=self.root, min_free_gb=None, datasets=downloader.DEFAULT_DATASETS))
        self.runner.log = Mock()

    def test_checks_only_current_dataset_and_credits_its_partial_files(self):
        destination = self.root / "VideoMMMU/data"
        with patch.object(downloader, "disk_snapshot", return_value=((0, 0, 0), 10 * downloader.GIB)) as snapshot:
            with patch.object(downloader.shutil, "disk_usage", return_value=SimpleNamespace(free=21 * downloader.GIB)):
                self.runner.check_space("VideoMMMU", destination)
                snapshot.assert_called_once_with(destination)
                with self.assertRaisesRegex(downloader.DownloadError, "LongVideoBench"):
                    self.runner.check_space("LongVideoBench", self.root / "LongVideoBench/data")

    def test_finishes_small_dataset_then_rejects_next_before_download(self):
        destination = self.root / "VideoMMMU/data"
        (destination / "Perception").mkdir(parents=True)
        (destination / "Perception/test-000.parquet").write_bytes(b"fixture")
        for index in range(7):
            (destination / f"discipline{index}.zip").write_bytes(b"fixture")
        self.runner.args.max_workers = 1
        self.runner.run = Mock()
        with patch.object(downloader.shutil, "disk_usage", return_value=SimpleNamespace(free=171 * downloader.GIB)):
            with self.assertRaisesRegex(downloader.DownloadError, "Insufficient disk space for LongVideoBench"):
                self.runner.datasets()
        downloads = [call.args[0] for call in self.runner.run.call_args_list if ": download " in call.args[0]]
        self.assertEqual(len(downloads), 1)
        self.assertTrue(downloads[0].startswith("VideoMMMU:"))
        self.assertTrue(any(call.args == ("VideoMMMU: completed",) for call in self.runner.log.call_args_list))
        self.assertTrue((destination / "discipline0.zip").exists())

    def test_large_dataset_rejected_before_any_hf_command(self):
        self.runner.args.datasets = ["MLVU"]
        self.runner.run = Mock()
        with patch.object(downloader.shutil, "disk_usage", return_value=SimpleNamespace(free=171 * downloader.GIB)):
            with self.assertRaisesRegex(downloader.DownloadError, "Insufficient disk space for MLVU"):
                self.runner.datasets()
        self.runner.run.assert_not_called()


class Downloads(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "data with spaces"
        self.bin = self.base / "bin"
        self.bin.mkdir()
        self.messages = []
        messages = self.messages
        self.reject = False
        owner = self

        class Receiver(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"])).decode()
                messages.append((dict(self.headers), body))
                self.send_response(503 if owner.reject else 200)
                self.end_headers()

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}")
        self.command = ["bash", str(SCRIPT), "--root", str(self.root),
                        "--ntfy-url", f"http://127.0.0.1:{self.server.server_port}",
                        "--topic", "offline-test", "--min-free-gb", "0",
                        "--poll-seconds", "0.05", "--stall-minutes", "0.01"]

    def fake_hf(self, body):
        path = self.bin / "hf"
        path.write_text(f"#!{sys.executable}\n" + body)
        path.chmod(0o755)

    def run_script(self, *extra):
        return subprocess.run(self.command + list(extra), env=self.env,
                              capture_output=True, text=True, timeout=15)

    def test_all_archives_resume_and_replace_partial_file(self):
        # TAR video archive split into precisely 31 pieces (aa..be).
        lvb = self.root / "LongVideoBench/data"
        lvb.mkdir(parents=True)
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            member = tarfile.TarInfo("videos/example.mp4")
            member.size = 5
            archive.addfile(member, io.BytesIO(b"video"))
        data = stream.getvalue()
        suffixes = ["a" + chr(c) for c in range(97, 123)] + ["b" + chr(c) for c in range(97, 102)]
        for index, suffix in enumerate(suffixes):
            (lvb / f"videos.tar.part.{suffix}").write_bytes(data[index * len(data) // 31:(index + 1) * len(data) // 31])
        with tarfile.open(lvb / "subtitles.tar", "w") as archive:
            member = tarfile.TarInfo("subtitles/example.srt")
            archive.addfile(member, io.BytesIO())
        (lvb / "lvb_val.json").write_text("[]")
        for task in range(1, 8):
            target = self.root / f"MLVU/data/MLVU/video/{task}_task"
            target.mkdir(parents=True)
            (target / "example.mp4").write_bytes(b"video")
            annotations = self.root / "MLVU/data/MLVU/json"
            annotations.mkdir(exist_ok=True)
            (annotations / f"{task}_task.json").write_text("[]")
        mmmu = self.root / "VideoMMMU/data"
        (mmmu / "Perception").mkdir(parents=True)
        (mmmu / "Perception/test-000.parquet").write_bytes(b"fixture")
        for index in range(7):
            with zipfile.ZipFile(mmmu / f"discipline{index}.zip", "w") as archive:
                archive.writestr(f"discipline{index}/example.mp4", "complete")
        partial = mmmu / "discipline0/example.mp4"
        partial.parent.mkdir()
        partial.write_text("partial")
        self.fake_hf("print('Reusing existing downloaded files')\n")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(partial.read_text(), "complete")
        self.assertEqual((lvb / "videos/example.mp4").read_bytes(), b"video")
        mtime = partial.stat().st_mtime_ns
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(partial.stat().st_mtime_ns, mtime)
        self.assertIn("skipping completed extraction", result.stdout)
        self.assertEqual(self.messages[-1][0]["Title"], "Video downloads complete")

    def test_failure_reason_and_token_redaction(self):
        self.fake_hf("import sys\nprint('No space left on device hf_secret123', file=sys.stderr)\nsys.exit(1)\n")
        result = self.run_script()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.messages[-1][0]["Title"], "Video download FAILED")
        self.assertIn("No space left on device", self.messages[-1][1])
        self.assertNotIn("hf_secret123", self.messages[-1][1])

    def test_stall_kills_process_and_alerts_despite_spinner(self):
        self.fake_hf("import time\nwhile True:\n print('Waiting...', flush=True)\n time.sleep(0.03)\n")
        result = self.run_script()
        self.assertEqual(result.returncode, 1)
        self.assertIn("Stalled", self.messages[-1][1])

    def test_child_sigkill_is_reported_as_possible_oom(self):
        self.fake_hf("import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n")
        result = self.run_script()
        self.assertEqual(result.returncode, 1)
        self.assertIn("possible OOM", self.messages[-1][1])

    def test_sigterm_stops_child_and_alerts(self):
        pid_file = self.base / "child.pid"
        self.fake_hf(f"import os, time\nfrom pathlib import Path\nPath({str(pid_file)!r}).write_text(str(os.getpid()))\ntime.sleep(60)\n")
        process = subprocess.Popen(self.command + ["--stall-minutes", "10"], env=self.env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 5
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(pid_file.exists())
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=10)
            self.assertEqual(process.returncode, 1)
            self.assertIn("SIGTERM", self.messages[-1][1])
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid_file.read_text()), 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def test_preflight_space_failure_and_dry_run(self):
        self.fake_hf("raise RuntimeError('HF should not run')\n")
        result = self.run_script("--min-free-gb", "1000000000")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Insufficient disk space", self.messages[-1][1])
        self.messages.clear()
        result = self.run_script("--dry-run", "--datasets", "MLVU")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self.messages, [])
        self.assertIn("MLVU/MVLU", result.stdout)
        self.assertNotIn("longvideobench/LongVideoBench", result.stdout)
        self.assertNotIn("lmms-eval/VideoMMMU", result.stdout)

    def test_failed_notification_is_queued_and_replayed(self):
        self.reject = True
        result = self.run_script("--test-notification")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Alert queued", result.stdout)
        self.reject = False
        self.messages.clear()
        result = self.run_script("--test-notification")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.messages), 2)  # old alert, then current test


if __name__ == "__main__":
    unittest.main()
