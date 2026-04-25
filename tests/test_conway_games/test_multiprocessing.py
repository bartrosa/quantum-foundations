from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from quantum_foundations.conway_games.experiments import e4_conway_aut_scaling
from quantum_foundations.entropic_causets.logging_setup import (
    configure_main_process_logging,
    configure_worker_logging,
)
from quantum_foundations.entropic_causets.parallel import TaskDiagnostic

logger = logging.getLogger("quantum_foundations.tests.multiprocessing")


def _log_test_worker(i: int) -> None:
    logger.info("worker_log_test_msg_%d", i)


def _slow_e4_task(
    task: e4_conway_aut_scaling.E4Task,
) -> tuple[e4_conway_aut_scaling.E4Row, TaskDiagnostic]:
    time.sleep(1.0)
    row = e4_conway_aut_scaling.E4Row(
        n=task.n,
        seed=task.seed,
        n_actual=task.n,
        log_aut_pos=0.0,
        log_aut_twin_canonical=0.0,
        log_aut_twin_local=0.0,
        log_aut_outcome=0.0,
        log_rank_factorial=0.0,
        log_n_factorial=0.0,
        s_a_pos=0.0,
        s_a_twin_canonical=0.0,
        s_a_twin_local=0.0,
        s_a_outcome=0.0,
        s_a_rank=0.0,
        delta_max_pos=0.0,
        delta_pos_twin_canonical=0.0,
        delta_pos_twin_local=0.0,
        delta_canonical_local=0.0,
        delta_local_outcome=0.0,
        twin_canonical_class_count=0,
        twin_local_class_count=0,
        outcome_class_count=0,
        time_s=1.0,
        backend=task.backend,
    )
    diag = TaskDiagnostic(
        f"slow:n={task.n}:s={task.seed}",
        "ok",
        1.0,
        tuple(),
    )
    return row, diag


def _interrupting_as_completed(*args: object, **kwargs: object) -> list[object]:
    raise KeyboardInterrupt()


def test_pool_smoke_multiworker(tmp_path: Path) -> None:
    log_path, listener, log_queue = configure_main_process_logging("test_e4", tmp_path)
    listener.start()
    try:
        result = e4_conway_aut_scaling.run(
            n_workers=2,
            seeds=2,
            ns=(10,),
            log_queue=log_queue,
        )
    finally:
        try:
            listener.stop()
        except Exception:
            pass
    assert result is not None
    assert log_path.exists()
    text = log_path.read_text(encoding="utf-8")
    assert "scheduling" in text
    assert "task complete" in text
    assert "shutting down pool" in text


def test_keyboard_interrupt_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path, listener, log_queue = configure_main_process_logging("test_e4_interrupt", tmp_path)
    listener.start()
    monkeypatch.setattr(e4_conway_aut_scaling, "as_completed", _interrupting_as_completed)
    monkeypatch.setattr(
        os,
        "_exit",
        lambda code: (_ for _ in ()).throw(SystemExit(code)),
    )
    try:
        with pytest.raises(SystemExit) as exc:
            e4_conway_aut_scaling.run(n_workers=2, seeds=2, ns=(10,), log_queue=log_queue)
        assert exc.value.code == 130
        time.sleep(0.5)
        assert "KeyboardInterrupt | 2 submitted, 0 completed, 2 pending" in log_path.read_text(
            encoding="utf-8"
        )
    finally:
        try:
            listener.stop()
        except Exception:
            pass


def test_per_task_timeout_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path, listener, log_queue = configure_main_process_logging("test_e4_timeout", tmp_path)
    listener.start()
    monkeypatch.setattr(e4_conway_aut_scaling, "_task", _slow_e4_task)
    try:
        result = e4_conway_aut_scaling.run(
            n_workers=2,
            seeds=2,
            ns=(10,),
            log_queue=log_queue,
            task_timeout_s=0.1,
        )
    finally:
        try:
            listener.stop()
        except Exception:
            pass
    assert len(result.rows) == 0
    assert "task TIMEOUT" in log_path.read_text(encoding="utf-8")


def test_worker_logs_reach_main(tmp_path: Path) -> None:
    log_path, listener, log_queue = configure_main_process_logging("test_worker_log", tmp_path)
    listener.start()
    try:
        with ProcessPoolExecutor(
            max_workers=2,
            mp_context=mp.get_context("spawn"),
            initializer=configure_worker_logging,
            initargs=(log_queue,),
        ) as pool:
            futures = [pool.submit(_log_test_worker, i) for i in range(4)]
            for f in futures:
                f.result(timeout=10)
    finally:
        listener.stop()
    text = log_path.read_text(encoding="utf-8")
    for i in range(4):
        assert f"worker_log_test_msg_{i}" in text
