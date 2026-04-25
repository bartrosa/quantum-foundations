from __future__ import annotations

import logging
from pathlib import Path

from quantum_foundations.entropic_causets.causet import build_stratified_game_causet
from quantum_foundations.entropic_causets.experiments import e1_aut_scaling
from quantum_foundations.entropic_causets.logging_setup import configure_experiment_logging


def test_configure_experiment_logging_creates_file(tmp_path: Path) -> None:
    log_path = configure_experiment_logging("test_exp", tmp_path, truncate=True)
    assert log_path.exists()
    assert log_path.suffix == ".log"
    text = log_path.read_text(encoding="utf-8")
    assert "main process logging configured" in text
    assert len(text) > 20


def test_library_debug_reaches_file_after_configure(tmp_path: Path) -> None:
    configure_experiment_logging("test_debug", tmp_path, truncate=True)
    _ = build_stratified_game_causet(n_target=5, max_rank=3, sample_rate=0.5, seed=1)
    text = (tmp_path / "test_debug.log").read_text(encoding="utf-8")
    assert "build_stratified" in text or "built causet" in text


def test_second_configure_replaces_handlers_not_duplicates(tmp_path: Path) -> None:
    configure_experiment_logging("idem", tmp_path, truncate=True)
    root = logging.getLogger("quantum_foundations")
    n1 = len([h for h in root.handlers if getattr(h, "_qf_managed", False)])
    configure_experiment_logging("idem", tmp_path, truncate=True)
    n2 = len([h for h in root.handlers if getattr(h, "_qf_managed", False)])
    assert n1 == n2 == 2  # file + stream


def test_e1_run_produces_sizable_log_file(tmp_path: Path) -> None:
    configure_experiment_logging("e1_aut_scaling", tmp_path)
    e1_aut_scaling.run(n_workers=1, seeds=8, ns=(8, 9))
    log_file = tmp_path / "e1_aut_scaling.log"
    assert log_file.stat().st_size > 1000


def test_truncate_false_appends(tmp_path: Path) -> None:
    p1 = configure_experiment_logging("append", tmp_path, truncate=True)
    size1 = p1.stat().st_size
    _ = configure_experiment_logging("append", tmp_path, truncate=False)
    assert (tmp_path / "append.log").stat().st_size >= size1
