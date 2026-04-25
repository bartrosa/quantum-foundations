"""E6: hierarchy-collapse rates for Conway entropy decomposition."""

from __future__ import annotations

import csv
import json
import logging
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
PER_TASK_TIMEOUT_S = 120.0


@dataclass(frozen=True, slots=True)
class E6Task:
    n: int
    seed: int
    backend: str


@dataclass(frozen=True, slots=True)
class E6Row:
    n: int
    seed: int
    delta_max_pos: float
    delta_pos_twin_canonical: float
    delta_pos_twin_local: float
    delta_canonical_local: float
    delta_local_outcome: float
    delta_local_outcome_normalized: float
    log_n_factorial: float
    time_s: float


@dataclass(frozen=True, slots=True)
class E6Result:
    rows: tuple[E6Row, ...]


def _task(t: E6Task) -> tuple[E6Row, TaskDiagnostic]:
    t0 = time.perf_counter()
    task_id = f"e6:n={t.n}:s={t.seed}"
    logger.info("[%s] task started", task_id)
    # Keep events-per-rank roughly stable across N during sweeps.
    c = build_random_conway_causet(n_target=t.n, max_rank=max(2, t.n // 4), seed=t.seed)
    d = entropy_decomposition_conway(c, backend=t.backend)
    dt = time.perf_counter() - t0
    denom = d.log_n_factorial if abs(d.log_n_factorial) > 1e-12 else 1.0
    row = E6Row(
        n=t.n,
        seed=t.seed,
        delta_max_pos=d.delta_max_pos,
        delta_pos_twin_canonical=d.delta_pos_twin_canonical,
        delta_pos_twin_local=d.delta_pos_twin_local,
        delta_canonical_local=d.delta_canonical_local,
        delta_local_outcome=d.delta_local_outcome,
        delta_local_outcome_normalized=d.delta_local_outcome / denom,
        log_n_factorial=d.log_n_factorial,
        time_s=dt,
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
    ns: tuple[int, ...] = (20, 30, 50, 70, 100),
    global_seed: int = 20260425,
    backend: str = "auto",
    log_queue: Any | None = None,
    task_timeout_s: float = PER_TASK_TIMEOUT_S,
) -> E6Result:
    child = spawn_task_seeds(global_seed, len(ns) * seeds)
    tasks: list[E6Task] = []
    i = 0
    for n in ns:
        for _ in range(seeds):
            tasks.append(E6Task(n=n, seed=child[i], backend=backend))
            i += 1
    logger.info("scheduling %d tasks across %d workers | mp_context=spawn", len(tasks), n_workers)
    rows: list[E6Row] = []
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
    future_to_task: dict[Any, E6Task] = {}
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
                start_times[f"e6:n={task.n}:s={task.seed}"] = time.perf_counter()
            pending = set(future_to_task.keys())
            logger.info("all %d tasks submitted to pool", len(future_to_task))
            while pending:
                now = time.perf_counter()
                for fut in list(pending):
                    task = future_to_task[fut]
                    task_id = f"e6:n={task.n}:s={task.seed}"
                    if (now - start_times[task_id]) > task_timeout_s:
                        pending.remove(fut)
                        timed_out += 1
                        force_kill = True
                        fut.cancel()
                        logger.error(
                            "[e6:n=%d:s=%d] task TIMEOUT after %.0fs",
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
                        task_id = f"e6:n={task.n}:s={task.seed}"
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
    finally:
        _shutdown_pool_hard(pool, interrupted=interrupted or force_kill)
    if interrupted:
        _flush_logging_handlers()
        os._exit(130)
    logger.info("all tasks done | ok=%d failed=%d timed_out=%d", len(rows), failed, timed_out)
    return E6Result(rows=tuple(sorted(rows, key=lambda r: (r.n, r.seed))))


def write_outputs(result: E6Result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = [f.name for f in E6Row.__dataclass_fields__.values()]
    with (output_dir / "e6_hierarchy_collapse.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(cols)
        for row in result.rows:
            w.writerow([getattr(row, c) for c in cols])
    payload = {"timestamp": datetime.now(UTC).isoformat(), "n_rows": len(result.rows)}
    (output_dir / "e6_hierarchy_collapse.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
