"""Logging configuration for experiment drivers and worker processes.

The main process owns concrete handlers (file and stdout), and worker processes
forward records via :class:`logging.handlers.QueueHandler`.
"""

from __future__ import annotations

import logging
import logging.handlers
import multiprocessing as mp
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_FORMAT = "%(asctime)s.%(msecs)03dZ | %(levelname)-7s | %(name)s | pid=%(process)d | %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def configure_main_process_logging(
    name: str,
    output_dir: Path,
    *,
    file_level: int = logging.DEBUG,
    stream_level: int = logging.INFO,
    truncate: bool = True,
) -> tuple[Path, logging.handlers.QueueListener, Any]:
    """Configure main-process handlers and a worker-log forwarding queue.

    Args:
        name: Short experiment identifier (e.g. ``"e1_aut_scaling"``). Becomes
            the log filename stem under ``output_dir``.
        output_dir: Directory where the ``.log`` file is written. Created if
            absent.
        file_level: Minimum level captured by the file handler.
        stream_level: Minimum level captured by the stdout handler.
        truncate: If ``True`` (default) the log file is opened in ``"w"`` mode
            so that each experiment run starts with a fresh file. Set to
            ``False`` to append.

    Returns:
        ``(log_path, queue_listener, log_queue)`` where ``log_queue`` should be
        passed to worker initializer and the listener must be started before
        scheduling pool tasks.

    Notes:
        Only the namespace ``quantum_foundations`` is configured; third-party
        libraries (numpy, networkx) keep their default loggers unless they
        emit a record under ``quantum_foundations.*``.

        The function is idempotent: repeated calls remove previous handlers
        attached by this function (marked with ``_qf_managed``) before
        re-installing fresh ones.
    """
    root = logging.getLogger("quantum_foundations")
    root.setLevel(min(file_level, stream_level))

    for handler in list(root.handlers):
        if getattr(handler, "_qf_managed", False):
            root.removeHandler(handler)
            handler.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = (output_dir / f"{name}.log").resolve()
    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    mode = "w" if truncate else "a"
    file_handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(fmt)
    file_handler._qf_managed = True  # type: ignore[attr-defined]
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(stream_level)
    stream_handler.setFormatter(fmt)
    stream_handler._qf_managed = True  # type: ignore[attr-defined]

    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    log_queue = mp.get_context("spawn").Queue(-1)
    listener = logging.handlers.QueueListener(
        log_queue,
        file_handler,
        stream_handler,
        respect_handler_level=True,
    )

    started = datetime.now(UTC).isoformat()
    root.info(
        "main process logging configured | experiment=%s | log_file=%s | started=%s",
        name,
        str(log_path),
        started,
    )
    return log_path, listener, log_queue


def configure_worker_logging(log_queue: Any) -> None:
    """Initializer for process-pool workers.

    Replaces managed handlers with a single QueueHandler forwarding records to
    the main process QueueListener.
    """
    root = logging.getLogger("quantum_foundations")
    root.setLevel(logging.DEBUG)
    root.propagate = False
    root.handlers = [logging.handlers.QueueHandler(log_queue)]
    root.info("worker process logging configured | pid=%d", mp.current_process().pid)


def configure_experiment_logging(
    name: str,
    output_dir: Path,
    *,
    file_level: int = logging.DEBUG,
    stream_level: int = logging.INFO,
    truncate: bool = True,
) -> Path:
    """Backward-compatible wrapper returning only log path."""
    log_path, _listener, _queue = configure_main_process_logging(
        name,
        output_dir,
        file_level=file_level,
        stream_level=stream_level,
        truncate=truncate,
    )
    return log_path
