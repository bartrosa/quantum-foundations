"""E4: Conway |Aut| scaling with full decomposition reporting."""

from __future__ import annotations

import csv
import json
import logging
import math
import multiprocessing as mp
import os
import signal
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, cast

from quantum_foundations.conway_games.causet import build_random_conway_causet
from quantum_foundations.conway_games.entropy import entropy_decomposition_conway
from quantum_foundations.entropic_causets.logging_setup import configure_worker_logging
from quantum_foundations.entropic_causets.parallel import TaskDiagnostic, spawn_task_seeds

logger = logging.getLogger(__name__)
PER_TASK_TIMEOUT_S = 600.0


@dataclass(frozen=True, slots=True)
class E4Task:
    n: int
    seed: int
    backend: str


@dataclass(frozen=True, slots=True)
class E4Row:
    n: int
    seed: int
    n_actual: int
    log_aut_pos: float
    log_aut_twin_canonical: float
    log_aut_twin_local: float
    log_aut_outcome: float
    log_rank_factorial: float
    log_n_factorial: float
    s_a_pos: float
    s_a_twin_canonical: float
    s_a_twin_local: float
    s_a_outcome: float
    s_a_rank: float
    delta_max_pos: float
    delta_pos_twin_canonical: float
    delta_pos_twin_local: float
    delta_canonical_local: float
    delta_local_outcome: float
    twin_canonical_class_count: int
    twin_local_class_count: int
    outcome_class_count: int
    time_s: float
    backend: str


@dataclass(frozen=True, slots=True)
class E4Result:
    rows: tuple[E4Row, ...]


def _task(t: E4Task) -> tuple[E4Row, TaskDiagnostic]:
    t0 = time.perf_counter()
    task_id = f"e4:n={t.n}:s={t.seed}"
    logger.info("[%s] task started", task_id)
    # Keep events-per-rank roughly stable across N during sweeps.
    c = build_random_conway_causet(n_target=t.n, max_rank=max(2, t.n // 4), seed=t.seed)
    d = entropy_decomposition_conway(c, backend=t.backend)
    dt = time.perf_counter() - t0
    row = E4Row(
        n=t.n,
        seed=t.seed,
        n_actual=c.n,
        log_aut_pos=d.log_aut_pos,
        log_aut_twin_canonical=d.log_aut_twin_canonical,
        log_aut_twin_local=d.log_aut_twin_local,
        log_aut_outcome=d.log_aut_outcome,
        log_rank_factorial=d.log_rank_factorial,
        log_n_factorial=d.log_n_factorial,
        s_a_pos=d.s_a_pos,
        s_a_twin_canonical=d.s_a_twin_canonical,
        s_a_twin_local=d.s_a_twin_local,
        s_a_outcome=d.s_a_outcome,
        s_a_rank=d.s_a_rank,
        delta_max_pos=d.delta_max_pos,
        delta_pos_twin_canonical=d.delta_pos_twin_canonical,
        delta_pos_twin_local=d.delta_pos_twin_local,
        delta_canonical_local=d.delta_canonical_local,
        delta_local_outcome=d.delta_local_outcome,
        twin_canonical_class_count=len(d.twin_canonical_class_sizes),
        twin_local_class_count=len(d.twin_local_class_sizes),
        outcome_class_count=len(d.outcome_class_sizes),
        time_s=dt,
        backend=t.backend,
    )
    logger.info("[%s] task complete | elapsed=%.2fs", task_id, dt)
    return row, TaskDiagnostic(task_id, "ok", dt, tuple())


def _flush_logging_handlers() -> None:
    root = logging.getLogger("quantum_foundations")
    for handler in root.handlers:
        try:
            handler.flush()
        except Exception:
            continue


def _shutdown_pool_hard(pool: ProcessPoolExecutor, *, interrupted: bool) -> None:
    processes = getattr(pool, "_processes", None) or {}
    procs = list(processes.values())
    if not interrupted:
        logger.info("shutting down pool | wait=True cancel_futures=False")
        pool.shutdown(wait=True)
        return

    logger.warning("hard pool shutdown | sending SIGKILL to all workers")
    logger.info("shutting down pool | wait=False cancel_futures=True")
    if not procs:
        logger.warning("no worker processes found in pool._processes")
        pool.shutdown(wait=False, cancel_futures=True)
        return
    killed_pids: list[int] = []
    for p in procs:
        if p.is_alive():
            try:
                p.kill()
                if p.pid is not None:
                    killed_pids.append(int(p.pid))
            except Exception:
                continue
    pool.shutdown(wait=False, cancel_futures=True)
    if killed_pids:
        logger.warning("SIGKILL sent to workers: %s", killed_pids)
    _kill_descendant_processes(os.getpid())
    time.sleep(0.2)
    still_alive = [int(p.pid) for p in procs if p.is_alive() and p.pid is not None]
    if still_alive:
        logger.error(
            "SIGKILL did not reap workers in 200ms: %s — continuing exit anyway.",
            still_alive,
        )


def _kill_descendant_processes(root_pid: int) -> None:
    """Best-effort SIGKILL for all descendants of *root_pid* via /proc."""
    ppid_by_pid: dict[int, int] = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        status_path = os.path.join(entry.path, "status")
        try:
            with open(status_path, encoding="utf-8") as handle:
                pid = int(entry.name)
                ppid = -1
                for line in handle:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                        break
                if ppid >= 0:
                    ppid_by_pid[pid] = ppid
        except Exception:
            continue

    descendants: set[int] = set()
    frontier = [root_pid]
    while frontier:
        parent = frontier.pop()
        for pid, ppid in ppid_by_pid.items():
            if ppid == parent and pid not in descendants:
                descendants.add(pid)
                frontier.append(pid)

    for pid in sorted(descendants, reverse=True):
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            continue


def run(
    *,
    n_workers: int = max(1, (os.cpu_count() or 1) - 1),
    seeds: int = 30,
    ns: tuple[int, ...] = (15, 20, 30, 50, 70, 100),
    global_seed: int = 20260425,
    backend: str = "auto",
    log_queue: Any | None = None,
    task_timeout_s: float = PER_TASK_TIMEOUT_S,
) -> E4Result:
    child = spawn_task_seeds(global_seed, len(ns) * seeds)
    tasks: list[E4Task] = []
    i = 0
    for n in ns:
        for _ in range(seeds):
            tasks.append(E4Task(n=n, seed=child[i], backend=backend))
            i += 1
    logger.info(
        "scheduling %d tasks across %d workers | mp_context=spawn",
        len(tasks),
        n_workers,
    )
    rows: list[E4Row] = []
    mp_ctx = mp.get_context("spawn")
    pool = ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=mp_ctx,
        initializer=(
            cast(Callable[[], object], partial(configure_worker_logging, log_queue))
            if log_queue is not None
            else None
        ),
    )
    future_to_task: dict[Any, E4Task] = {}
    start_times: dict[str, float] = {}
    failed = 0
    timed_out = 0
    interrupted = False
    force_kill = False
    completed = 0
    try:
        try:
            for task in tasks:
                fut = pool.submit(_task, task)
                future_to_task[fut] = task
                start_times[f"e4:n={task.n}:s={task.seed}"] = time.perf_counter()
            logger.info("all %d tasks submitted to pool", len(future_to_task))
            pending = set(future_to_task.keys())
            while pending:
                now = time.perf_counter()
                for fut in list(pending):
                    task = future_to_task[fut]
                    task_id = f"e4:n={task.n}:s={task.seed}"
                    age = now - start_times[task_id]
                    if age > task_timeout_s:
                        pending.remove(fut)
                        timed_out += 1
                        force_kill = True
                        fut.cancel()
                        logger.error(
                            "[e4:n=%d:s=%d] task TIMEOUT after %.0fs",
                            task.n,
                            task.seed,
                            task_timeout_s,
                        )
                try:
                    for fut in as_completed(list(pending), timeout=0.5):
                        if fut not in pending:
                            continue
                        pending.remove(fut)
                        task = future_to_task[fut]
                        task_id = f"e4:n={task.n}:s={task.seed}"
                        try:
                            row, _diag = fut.result(timeout=task_timeout_s)
                            rows.append(row)
                        except FutureTimeoutError:
                            timed_out += 1
                            logger.error("[%s] TIMEOUT after %ds", task_id, int(task_timeout_s))
                        except Exception:
                            failed += 1
                            logger.exception("[%s] FAILED", task_id)
                        completed += 1
                        if completed % max(1, len(tasks) // 10) == 0:
                            logger.info(
                                "progress | %d / %d tasks complete (failed=%d, timed_out=%d)",
                                completed,
                                len(tasks),
                                failed,
                                timed_out,
                            )
                except FutureTimeoutError:
                    pass
        except KeyboardInterrupt:
            interrupted = True
            logger.warning(
                "KeyboardInterrupt | %d submitted, %d completed, %d pending",
                len(future_to_task),
                completed,
                len(future_to_task) - completed,
            )
            for fut in future_to_task:
                try:
                    fut.cancel()
                except Exception:
                    pass
        logger.info("all tasks done | ok=%d failed=%d timed_out=%d", len(rows), failed, timed_out)
        if timed_out > 0:
            logger.warning(
                "timed out tasks detected (%d) | forcing hard shutdown of pool workers",
                timed_out,
            )
    finally:
        _shutdown_pool_hard(pool, interrupted=interrupted or force_kill)
    if interrupted:
        _flush_logging_handlers()
        os._exit(130)
    return E4Result(rows=tuple(sorted(rows, key=lambda r: (r.n, r.seed))))


def write_outputs(result: E4Result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = [f.name for f in E4Row.__dataclass_fields__.values()]
    with (output_dir / "e4_conway_aut_scaling.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(cols)
        for row in result.rows:
            w.writerow([getattr(row, c) for c in cols])
    valid = [r for r in result.rows if not math.isnan(r.log_aut_pos)]
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "n_rows": len(result.rows),
        "mean_log_aut_twin_canonical": (
            sum(r.log_aut_twin_canonical for r in valid) / len(valid)
        )
        if valid
        else 0.0,
        "mean_log_aut_twin_local": (sum(r.log_aut_twin_local for r in valid) / len(valid))
        if valid
        else 0.0,
    }
    (output_dir / "e4_conway_aut_scaling.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
