#!/usr/bin/env python3
"""Download LongVideoBench validation, MLVU Dev MC, and Video-MMMU sequentially.

Accept all three HF access gates, then run `hf auth login` first.
Run via download_video_benchmarks.sh, preferably inside tmux.
"""

import argparse
from collections import deque
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request


GIB = 1024**3
SOURCES = [
    ("LongVideoBench", "longvideobench/LongVideoBench",
     "60d1c89c1919a198b73be39c2babb213b29d6a5c",
     ["README.md", "lvb_val.json", "validation-*.parquet", "subtitles.tar", "videos.tar.part.*"]),
    ("MLVU", "MLVU/MVLU", "06ddc388aa34746b3abba77972b7a7dfd977f7a3",
     ["README.md", "MLVU/json/[1-7]_*.json", "MLVU/video/[1-7]_*/*"]),
    ("VideoMMMU", "lmms-eval/VideoMMMU", "d1c35ac933123d79e877b7f1b9506afb0309cf1b",
     ["README.md", "*/test-*.parquet", "*.zip"]),
]
# Conservative peak disk budgets including retained archives, extraction and
# working headroom. Based on the pinned inventories in README_datasets.md.
DISK_BUDGET_GIB = {"LongVideoBench": 320, "MLVU": 300, "VideoMMMU": 30}
DEFAULT_DATASETS = ["VideoMMMU", "LongVideoBench", "MLVU"]


class DownloadError(RuntimeError):
    pass


def redact(text):
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "NTFY_TOKEN"):
        if os.environ.get(key):
            text = text.replace(os.environ[key], "[REDACTED]")
    return re.sub(r"\bhf_[A-Za-z0-9]+", "[REDACTED]", text)


def disk_snapshot(root):
    """Observe partial files too; CLI spinners alone do not count as progress."""
    count = size = latest = allocated = 0
    for base, _, names in os.walk(root):
        for name in names:
            try:
                stat = (Path(base) / name).stat()
            except FileNotFoundError:  # HF may rename a completed partial file.
                continue
            count += 1
            size += stat.st_size
            allocated += stat.st_blocks * 512
            latest = max(latest, stat.st_mtime_ns)
    return (count, size, latest), allocated


def io_snapshot(pgid):
    """Include child disk I/O (e.g. Xet staging outside the destination)."""
    result = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != pgid:
                continue
            counters = dict(line.split(":", 1) for line in (entry / "io").read_text().splitlines())
            result[(entry.name, fields[19])] = (
                int(counters["read_bytes"]) + int(counters["write_bytes"])
            )
        except (OSError, ValueError, IndexError, KeyError):
            continue
    return result


class Runner:
    def __init__(self, args):
        self.args = args
        self.process = None
        self.reader = None
        self.stage = "preflight"
        self.tail = deque(maxlen=30)
        self.log_file = None
        self.log_path = None
        root_id = hashlib.sha256(str(args.root).encode()).hexdigest()[:16]
        self.pending = Path(tempfile.gettempdir()) / f"video-download-{os.getuid()}-{root_id}-pending_ntfy.jsonl"

    def log(self, message):
        line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {redact(message)}"
        try:
            print(line, flush=True)
        except OSError:
            pass
        if self.log_file:
            try:
                self.log_file.write(line + "\n")
                self.log_file.flush()
            except OSError as error:
                # Disk-full errors must not prevent the failure notification.
                try:
                    print(f"Log write failed: {error}", file=sys.stderr, flush=True)
                except OSError:
                    pass

    def post(self, alert):
        headers = {"Title": alert["title"], "Priority": alert["priority"],
                   "Tags": "warning" if alert["priority"] == "high" else "white_check_mark",
                   "Content-Type": "text/plain; charset=utf-8"}
        if os.environ.get("NTFY_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["NTFY_TOKEN"]
        request = urllib.request.Request(alert["url"], data=alert["body"].encode(),
                                         headers=headers, method="POST")
        # Bound retries so notification failures cannot hang job cleanup.
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    response.read()
                return True
            except (OSError, ValueError) as error:
                self.log(f"ntfy attempt {attempt + 1}/3 failed: {error}")
        return False

    def notify(self, title, body, priority="high"):
        alert = {"title": title, "priority": priority,
                 "url": self.args.ntfy_url.rstrip("/") + "/" + self.args.topic,
                 # Stay below ntfy's 4 KiB body limit, including UTF-8 text.
                 "body": redact(body).encode()[-3800:].decode("utf-8", errors="replace")}
        if self.post(alert):
            self.log(f"ntfy sent: {title}")
            return True
        try:
            with self.pending.open("a") as stream:
                stream.write(json.dumps(alert) + "\n")
            self.log(f"Alert queued for next run: {self.pending}")
        except OSError as error:
            self.log(f"Could not persist undelivered alert: {error}")
        return False

    def retry_pending(self):
        if not self.pending.exists():
            return
        alerts = [json.loads(line) for line in self.pending.read_text().splitlines() if line]
        remaining = []
        # If the service is down, keep the rest without a long retry per alert.
        for index, alert in enumerate(alerts):
            if not self.post(alert):
                remaining = alerts[index:]
                break
        self.pending.write_text("".join(json.dumps(alert) + "\n" for alert in remaining))

    def stop(self):
        if self.process is None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # A shell/pipeline leader can exit before all descendants stop.
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait()
        self.process = None
        if self.reader:
            self.reader.join(timeout=5)

    def run(self, stage, command, destination):
        self.stage = stage
        self.tail.clear()
        self.log(stage)
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        process = self.process

        def drain():
            while True:
                chunk = os.read(process.stdout.fileno(), 4096)
                if not chunk:
                    break
                message = redact(chunk.decode("utf-8", errors="replace"))
                self.tail.append(message)
                try:
                    sys.stdout.write(message)
                    sys.stdout.flush()
                    self.log_file.write(message)
                    self.log_file.flush()
                except OSError:
                    # Keep draining the child even if the log disk fills up.
                    pass

        reader = threading.Thread(target=drain, daemon=True)
        self.reader = reader
        reader.start()
        previous, _ = disk_snapshot(destination)
        previous_io = io_snapshot(process.pid)
        progress_at = time.monotonic()
        while process.poll() is None:
            try:
                process.wait(timeout=self.args.poll_seconds)
            except subprocess.TimeoutExpired:
                current, _ = disk_snapshot(destination)
                current_io = io_snapshot(process.pid)
                io_progress = any(value > previous_io.get(pid, 0) for pid, value in current_io.items())
                if current != previous or io_progress:
                    progress_at = time.monotonic()
                previous, previous_io = current, current_io
                if time.monotonic() - progress_at >= self.args.stall_minutes * 60:
                    self.stop()
                    reader.join(timeout=5)
                    raise DownloadError(f"Stalled: no data-file changes or child disk I/O for "
                                        f"{self.args.stall_minutes:g} minutes; possible network hang.")
        reader.join(timeout=5)
        status = process.returncode
        self.process = None
        if status:
            hint = ""
            if status in (-9, 137):
                hint = " Process was killed (SIGKILL); possible OOM or external kill, not confirmed."
            raise DownloadError(f"Command exited with status {status}.{hint}")

    def download_command(self, name, repo, revision, patterns):
        command = ["hf", "download", repo, "--repo-type", "dataset", "--revision", revision,
                   "--max-workers", str(self.args.max_workers)]
        # Repeat --include: the CLI installed here uses one pattern per option.
        for pattern in patterns:
            command.extend(["--include", pattern])
        return command + ["--local-dir", str(self.args.root / name / "data")]

    def extract(self, name, revision, destination, label, archives, command):
        marker = destination / f".extracted-{label}-{revision}"
        signature = json.dumps([(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in archives])
        if marker.exists() and marker.read_text() == signature:
            self.log(f"{name}: skipping completed extraction {label}")
            return
        self.run(f"{name}: extract {label}", command, destination)
        # Only a successful extraction creates the marker. Interrupted extraction
        # runs again, overwriting partial files (including ZIP members).
        marker.write_text(signature)

    def check_space(self, name, destination):
        self.stage = f"{name}: disk-space preflight"
        self.tail.clear()
        _, existing = disk_snapshot(destination)
        free = shutil.disk_usage(destination).free
        budget = DISK_BUDGET_GIB[name] if self.args.min_free_gb is None else self.args.min_free_gb
        needed = max(0, budget * GIB - existing)
        self.log(f"{name}: {free / GIB:.1f} GiB free; {existing / GIB:.1f} GiB already allocated "
                 f"for this dataset; {needed / GIB:.1f} GiB additional space budgeted "
                 f"({budget:g} GiB total including extraction).")
        if free < needed:
            raise DownloadError(f"Insufficient disk space for {name}: {free / GIB:.1f} GiB free, "
                                f"need about {needed / GIB:.1f} GiB additional space. "
                                "Stopped before downloading this dataset; earlier completed datasets "
                                "are retained. Use --datasets to select another suite or --root to "
                                "choose a larger filesystem.")

    def datasets(self):
        sources = {source[0]: source for source in SOURCES}
        for selected in self.args.datasets:
            name, repo, revision, patterns = sources[selected]
            destination = self.args.root / name / "data"
            destination.mkdir(parents=True, exist_ok=True)
            self.check_space(name, destination)
            self.run(f"{name}: download {repo}@{revision}",
                     self.download_command(name, repo, revision, patterns), destination)
            self.stage = f"{name}: check downloaded files"
            if name == "LongVideoBench":
                parts = [destination / f"videos.tar.part.{suffix}"
                         for suffix in [*("a" + chr(c) for c in range(97, 123)),
                                        *("b" + chr(c) for c in range(97, 102))]]
                required = [destination / "lvb_val.json", destination / "subtitles.tar", *parts]
                if any(not p.is_file() or p.stat().st_size == 0 for p in required):
                    raise DownloadError("Missing/empty LongVideoBench annotations, subtitles, or one of 31 TAR parts.")
                self.extract(name, revision, destination, "videos", parts,
                             ["bash", "-o", "pipefail", "-c",
                              'destination=$1; shift; cat -- "$@" | tar -xf - -C "$destination"',
                              "extract-lvb", str(destination), *map(str, parts)])
                archive = destination / "subtitles.tar"
                self.extract(name, revision, destination, "subtitles", [archive],
                             ["tar", "-xf", str(archive), "-C", str(destination)])
            elif name == "MLVU":
                for task in range(1, 8):
                    annotations = list(destination.glob(f"MLVU/json/{task}_*.json"))
                    videos = list(destination.glob(f"MLVU/video/{task}_*/*.mp4"))
                    if not annotations or not videos:
                        raise DownloadError(f"Missing MLVU task {task} annotations or MP4 files.")
            else:
                archives = sorted(destination.glob("*.zip"))
                if len(archives) != 7 or not list(destination.glob("*/test-*.parquet")):
                    raise DownloadError("Expected seven Video-MMMU ZIPs and test annotations.")
                for archive in archives:
                    self.extract(name, revision, destination, archive.stem, [archive],
                                 ["unzip", "-o", "-q", str(archive), "-d", str(destination)])
            self.log(f"{name}: completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get(
        "VIDEO_BENCH_ROOT", Path(__file__).resolve().parent / "video_benchmarks")))
    parser.add_argument("--topic", default=os.environ.get("NTFY_TOPIC", "download-videobenchmarks-status"))
    parser.add_argument("--ntfy-url", default=os.environ.get("NTFY_URL", "https://ntfy.sh"))
    parser.add_argument("--stall-minutes", type=float, default=30,
                        help="Stop and alert after this long without file/disk-I/O progress (default: 30)")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--datasets", nargs="+", choices=list(DISK_BUDGET_GIB), default=DEFAULT_DATASETS,
                        help="Suites to download in order (default: VideoMMMU LongVideoBench MLVU)")
    parser.add_argument("--min-free-gb", type=float, default=None,
                        help="Override each selected dataset's total GiB budget, minus its existing files; "
                        "0 disables checks (defaults: VideoMMMU=30, LongVideoBench=320, MLVU=300)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan; no writes, network, or notifications")
    parser.add_argument("--test-notification", action="store_true", help="Only send a test ntfy message, then exit")
    args = parser.parse_args()
    if (args.stall_minutes <= 0 or args.poll_seconds <= 0 or args.max_workers < 1
            or (args.min_free_gb is not None and args.min_free_gb < 0)):
        parser.error("Intervals/workers must be positive and the disk budget nonnegative")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.topic):
        parser.error("ntfy topic must be 1-64 letters, digits, underscores, or hyphens")
    if urllib.parse.urlparse(args.ntfy_url).scheme not in ("http", "https"):
        parser.error("--ntfy-url must be an HTTP(S) server URL")
    args.root = args.root.expanduser().resolve()
    args.datasets = list(dict.fromkeys(args.datasets))
    runner = Runner(args)
    if args.dry_run:
        sources = {source[0]: source for source in SOURCES}
        for name in args.datasets:
            source = sources[name]
            budget = DISK_BUDGET_GIB[name] if args.min_free_gb is None else args.min_free_gb
            print(f"{name}: check {budget:g} GiB budget minus this dataset's existing files before downloading.")
            print(shlex.join(runner.download_command(*source)))
        print("Extract TAR/ZIP archives for selected datasets; retain completed datasets if a later one cannot fit.")
        print(f"ntfy: {args.ntfy_url}/{args.topic}; stall timeout: {args.stall_minutes:g} minutes")
        return 0

    # Fall back to /tmp when the data disk is full. The alert queue always lives
    # in /tmp so that a disk-full failure can still be recorded for the next run.
    try:
        log_dir = args.root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        runner.log_path = log_dir / f"download-{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}.log"
        runner.log_file = runner.log_path.open("a", buffering=1)
    except OSError:
        runner.log_file = tempfile.NamedTemporaryFile(mode="a", prefix="video-download-", suffix=".log", delete=False)
        runner.log_path = Path(runner.log_file.name)
    # Serialize runs against the same data root to prevent conflicting extraction.
    lock_path = Path(tempfile.gettempdir()) / ("video-download-" + hashlib.sha256(
        str(args.root).encode()).hexdigest()[:16] + ".lock")
    lock_file = lock_path.open("a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        runner.log("Another downloader is already running against this root.")
        runner.log_file.close()
        lock_file.close()
        return 1

    def interrupted(signum, _frame):
        raise DownloadError(f"Interrupted by {signal.Signals(signum).name}.")

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        runner.log(f"Root: {args.root}; log: {runner.log_path}")
        runner.retry_pending()
        if args.test_notification:
            return 0 if runner.notify("Video download test", f"ntfy test from {socket.gethostname()}.", "default") else 1
        for executable in ("hf", "tar", "unzip", "bash", "cat"):
            if not shutil.which(executable):
                raise DownloadError(f"Missing executable: {executable}; see README_datasets.md for setup.")
        runner.datasets()
        runner.notify("Video downloads complete", f"Host: {socket.gethostname()}\n"
                      f"Completed: {', '.join(args.datasets)}.\n"
                      f"Root: {args.root}\nLog: {runner.log_path}", "default")
        runner.log("Selected datasets completed. Annotation-to-video validation is still required before evaluation.")
        return 0
    except (DownloadError, OSError, ValueError, MemoryError) as error:
        # Ignore further stop signals while cleaning up and attempting the alert.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        runner.stop()
        details = "".join(runner.tail).replace("\r", "\n")[-2300:]
        message = (f"Host: {socket.gethostname()}\nStage: {runner.stage}\nError: {error}\n"
                   f"Log: {runner.log_path}\nRecent output:\n{details}")
        runner.log(message)
        runner.notify("Video download FAILED", message)
        return 1
    finally:
        runner.stop()
        try:
            runner.log_file.close()
        except OSError:
            pass
        lock_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
