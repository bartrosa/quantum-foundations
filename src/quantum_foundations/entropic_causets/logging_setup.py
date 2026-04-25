"""Logging configuration for entropic_causets experiments and library code.

Two-channel design:

* File handler: full DEBUG-level log persisted to ``<output_dir>/<name>.log``.
* Stream handler: INFO-level mirror to stdout (warnings/errors are still emitted at their levels).

Library code (``causet.py``, ``automorphism.py``, ``entropy.py``) does **not** call
``logging.basicConfig`` or attach handlers. Library code only obtains loggers via
``logging.getLogger(__name__)`` and emits records. Handlers are configured at the
experiment or CLI entry point by calling :func:`configure_experiment_logging`.
"""
from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

_FORMAT = (
    "%(asctime)s.%(msecs)03dZ | %(levelname)-7s | %(name)s | "
    "pid=%(process)d | %(message)s"
)
_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def configure_experiment_logging(
    name: str,
    output_dir: Path,
    *,
    file_level: int = logging.DEBUG,
    stream_level: int = logging.INFO,
    truncate: bool = True,
) -> Path:
    """Attach file + stdout handlers to the ``quantum_foundations`` root logger.

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
        Absolute path to the configured ``.log`` file (resolved relative to cwd
        as passed in ``output_dir`` / ``name``).

    Notes:
        Only the namespace ``quantum_foundations`` is configured; third-party
        libraries (numpy, networkx) keep their default loggers unless they
        emit a record under ``quantum_foundations.*``.

        The function is idempotent: repeated calls remove previous handlers
        attached by this function (marked with ``_qf_managed``) before
        re-installing fresh ones.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = (output_dir / f"{name}.log").resolve()

    root = logging.getLogger("quantum_foundations")
    root.setLevel(min(file_level, stream_level))

    for handler in list(root.handlers):
        if getattr(handler, "_qf_managed", False):
            root.removeHandler(handler)
            handler.close()

    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    mode = "w" if truncate else "a"
    file_handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(fmt)
    file_handler._qf_managed = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(stream_level)
    stream_handler.setFormatter(fmt)
    stream_handler._qf_managed = True  # type: ignore[attr-defined]
    root.addHandler(stream_handler)

    started = datetime.now(UTC).isoformat()
    root.info(
        "logging configured | experiment=%s | log_file=%s | started=%s",
        name,
        str(log_path),
        started,
    )
    return log_path
