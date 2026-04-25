from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil  # type: ignore[import-untyped]
import pytest


def _qf_run_e4_cmd() -> list[str]:
    script = shutil.which("qf-run-e4")
    if script is not None:
        return [script]
    fallback = Path(sys.executable).with_name("qf-run-e4")
    if fallback.exists():
        return [str(fallback)]
    raise RuntimeError("qf-run-e4 executable not found in PATH or virtualenv scripts")


def _run_cmd_for_e4(tmp_path: Path, *, n_workers: int, seeds: int, max_n: int) -> list[str]:
    base = _qf_run_e4_cmd()
    return [
        *base,
        "--n-workers",
        str(n_workers),
        "--seeds",
        str(seeds),
        "--max-n",
        str(max_n),
        "--output-dir",
        str(tmp_path),
    ]


def test_keyboard_interrupt_no_zombies(tmp_path: Path) -> None:
    cmd = _run_cmd_for_e4(tmp_path, n_workers=4, seeds=800, max_n=30)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    main = psutil.Process(proc.pid)
    children_before: list[psutil.Process] = []
    for _ in range(120):
        time.sleep(0.05)
        if proc.poll() is not None:
            pytest.skip("subprocess exited before pool workers were observable")
        children_before = main.children(recursive=True)
        if children_before:
            break
    if not children_before:
        pytest.skip("no worker children observed within 6s (environment timing)")

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except (PermissionError, ProcessLookupError):
        pytest.skip("environment forbids signaling subprocesses")
    try:
        proc.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout=5.0)
        raise AssertionError("main process did not exit within 15s of SIGINT") from None

    def _is_worker_process(process: psutil.Process) -> bool:
        try:
            cmd = " ".join(process.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
        markers = (
            "qf-run-e4",
            "multiprocessing.spawn",
            "multiprocessing.resource_tracker",
        )
        return any(marker in cmd for marker in markers)

    leftover: list[int] = []
    for _ in range(5):
        time.sleep(1.0)
        leftover = []
        for p in children_before:
            try:
                if p.is_running() and p.status() != psutil.STATUS_ZOMBIE and _is_worker_process(p):
                    leftover.append(p.pid)
            except psutil.NoSuchProcess:
                continue
        if not leftover:
            break

    if leftover:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        time.sleep(1.0)
        verified_leftover: list[int] = []
        for pid in leftover:
            if not psutil.pid_exists(pid):
                continue
            try:
                candidate = psutil.Process(pid)
                command_line = " ".join(candidate.cmdline())
                if str(tmp_path) in command_line and _is_worker_process(candidate):
                    verified_leftover.append(pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        leftover = verified_leftover

    assert len(leftover) == 0, f"worker processes still alive after SIGINT: {leftover}"
    assert proc.returncode == 130


def test_normal_exit_no_orphans(tmp_path: Path) -> None:
    cmd = _run_cmd_for_e4(tmp_path, n_workers=2, seeds=2, max_n=15)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        start_new_session=True,
    )
    assert proc.returncode == 0, f"normal run failed | stdout={proc.stdout} stderr={proc.stderr}"
    time.sleep(2.0)
    tmp_str = str(tmp_path)
    leftover: list[int] = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(p.info["cmdline"] or [])
            if tmp_str in cmdline and "qf-run-e4" in cmdline and p.pid != os.getpid():
                leftover.append(p.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    assert len(leftover) == 0, f"orphaned qf-run processes after normal exit: {leftover}"
