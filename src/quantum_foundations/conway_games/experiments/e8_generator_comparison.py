"""E8: comparative run across all Conway causet generators (entropy + outcomes)."""

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
from statistics import mean
from typing import Any, cast

from quantum_foundations.conway_games.canonical_games import (
    make_balanced_binary_tree,
    make_down,
    make_integer,
    make_nimber,
    make_star,
    make_switch,
    make_up,
)
from quantum_foundations.conway_games.causet import (
    ConwayCauset,
    build_coupled_pool_conway_causet,
    build_random_conway_causet,
)
from quantum_foundations.conway_games.entropy import entropy_decomposition_conway
from quantum_foundations.conway_games.outcome import outcome_histogram
from quantum_foundations.entropic_causets.logging_setup import configure_worker_logging
from quantum_foundations.entropic_causets.parallel import TaskDiagnostic

logger = logging.getLogger(__name__)

DEFAULT_N_VALUES_STOCHASTIC: tuple[int, ...] = (15, 20, 30, 50)
DEFAULT_N_SEEDS_STOCHASTIC = 30
DEFAULT_PER_TASK_TIMEOUT_S = 60.0
DEFAULT_RECURSIVE_MAX_N = 50


@dataclass(frozen=True, slots=True)
class GeneratorTask:
    generator: str
    params: dict[str, Any]
    n_target: int
    seed_or_index: int


@dataclass(frozen=True, slots=True)
class E8RunSpec:
    gtask: GeneratorTask
    backend: str
    recursive_max_n: int


@dataclass(frozen=True, slots=True)
class E8Row:
    generator: str
    params: str
    n_target: int
    n_actual: int
    seed_or_index: int
    log_aut_pos: float
    log_aut_twin_canonical: float
    log_aut_twin_local: float
    log_aut_outcome: float
    delta_canonical_local: float
    delta_pos_twin_canonical: float
    delta_pos_twin_local: float
    delta_local_outcome: float
    twin_canonical_class_count: int
    twin_local_class_count: int
    outcome_class_count: int
    count_L_shallow: int
    count_R_shallow: int
    count_eq_shallow: int
    count_fuzzy_shallow: int
    count_L_recursive: int
    count_R_recursive: int
    count_eq_recursive: int
    count_fuzzy_recursive: int
    recursive_skipped: bool
    time_s: float
    backend: str


@dataclass(frozen=True, slots=True)
class E8Result:
    rows: tuple[E8Row, ...]


def enumerate_tasks(
    n_values: tuple[int, ...],
    n_seeds: int,
) -> list[GeneratorTask]:
    """Build the full list of comparison tasks."""
    tasks: list[GeneratorTask] = []

    for n in n_values:
        for seed in range(n_seeds):
            tasks.append(
                GeneratorTask(
                    generator="random_default",
                    params={
                        "sample_rate_left": 0.3,
                        "sample_rate_right": 0.3,
                        "overlap_rate": None,
                    },
                    n_target=n,
                    seed_or_index=seed,
                )
            )
            tasks.append(
                GeneratorTask(
                    generator="random_overlap_05",
                    params={"sample_rate_left": 0.3, "sample_rate_right": 0.3, "overlap_rate": 0.5},
                    n_target=n,
                    seed_or_index=seed,
                )
            )
            for po in (0.0, 0.5, 1.0):
                tasks.append(
                    GeneratorTask(
                        generator=f"coupled_pool_{int(po * 10):02d}",
                        params={"pool_overlap": po, "sample_rate": 0.3},
                        n_target=n,
                        seed_or_index=seed,
                    )
                )

    for k in (3, 5, 10, 20):
        tasks.append(
            GeneratorTask(
                generator="nimber",
                params={"k": k},
                n_target=k + 1,
                seed_or_index=k,
            )
        )
    for n in (3, 5, 10, 20):
        tasks.append(
            GeneratorTask(
                generator="integer",
                params={"n": n},
                n_target=n + 1,
                seed_or_index=n,
            )
        )
    for d in (2, 3, 4):
        tasks.append(
            GeneratorTask(
                generator="balanced_tree",
                params={"depth": d},
                n_target=2 ** (d + 1) - 1,
                seed_or_index=d,
            )
        )
    for name, n in (("up", 3), ("down", 3), ("star", 2)):
        tasks.append(
            GeneratorTask(
                generator=name,
                params={},
                n_target=n,
                seed_or_index=0,
            )
        )
    for a in (1, 2, 3, 5):
        tasks.append(
            GeneratorTask(
                generator="switch",
                params={"a": a, "b": 0},
                n_target=a + 2,
                seed_or_index=a,
            )
        )

    return tasks


def build_causet(task: GeneratorTask) -> ConwayCauset:
    """Dispatch to the appropriate builder based on generator name."""
    if task.generator in ("random_default", "random_overlap_05"):
        c = build_random_conway_causet(
            n_target=task.n_target,
            max_rank=max(2, task.n_target // 4),
            seed=task.seed_or_index,
            **task.params,
        )
    elif task.generator.startswith("coupled_pool_"):
        c = build_coupled_pool_conway_causet(
            task.n_target,
            max_rank=max(2, task.n_target // 4),
            seed=task.seed_or_index,
            **task.params,
        )
    elif task.generator == "nimber":
        c = make_nimber(task.params["k"])
    elif task.generator == "integer":
        c = make_integer(task.params["n"])
    elif task.generator == "balanced_tree":
        c = make_balanced_binary_tree(task.params["depth"])
    elif task.generator == "up":
        c = make_up()
    elif task.generator == "down":
        c = make_down()
    elif task.generator == "star":
        c = make_star()
    elif task.generator == "switch":
        c = make_switch(task.params["a"], task.params.get("b", 0))
    else:
        msg = f"Unknown generator: {task.generator!r}"
        raise ValueError(msg)
    return c


def _format_params(p: dict[str, Any]) -> str:
    return json.dumps(p, sort_keys=True)


def _row_task_id(t: GeneratorTask) -> str:
    return f"e8:gen={t.generator}:n={t.n_target}:s={t.seed_or_index}"


def _task(spec: E8RunSpec) -> tuple[E8Row, TaskDiagnostic]:
    t = spec.gtask
    task_id = _row_task_id(t)
    logger.info("[%s] task started", task_id)
    t0 = time.perf_counter()
    c = build_causet(t)
    d = entropy_decomposition_conway(c, backend=spec.backend)
    params_str = _format_params(t.params)
    hs = outcome_histogram(c, recursive=False)
    if c.n > spec.recursive_max_n:
        hr = {"L": 0, "R": 0, "=": 0, "||": 0}
        rec_skip = True
    else:
        hr = outcome_histogram(c, recursive=True)
        rec_skip = False
    dt = time.perf_counter() - t0
    row = E8Row(
        generator=t.generator,
        params=params_str,
        n_target=t.n_target,
        n_actual=c.n,
        seed_or_index=t.seed_or_index,
        log_aut_pos=d.log_aut_pos,
        log_aut_twin_canonical=d.log_aut_twin_canonical,
        log_aut_twin_local=d.log_aut_twin_local,
        log_aut_outcome=d.log_aut_outcome,
        delta_canonical_local=d.delta_canonical_local,
        delta_pos_twin_canonical=d.delta_pos_twin_canonical,
        delta_pos_twin_local=d.delta_pos_twin_local,
        delta_local_outcome=d.delta_local_outcome,
        twin_canonical_class_count=len(d.twin_canonical_class_sizes),
        twin_local_class_count=len(d.twin_local_class_sizes),
        outcome_class_count=len(d.outcome_class_sizes),
        count_L_shallow=hs["L"],
        count_R_shallow=hs["R"],
        count_eq_shallow=hs["="],
        count_fuzzy_shallow=hs["||"],
        count_L_recursive=hr["L"],
        count_R_recursive=hr["R"],
        count_eq_recursive=hr["="],
        count_fuzzy_recursive=hr["||"],
        recursive_skipped=rec_skip,
        time_s=dt,
        backend=spec.backend,
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
                ppid = -1
                for line in handle:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                        break
                if ppid >= 0:
                    ppid_by_pid[int(entry.name)] = ppid
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


def compute_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, int | float]]:
    """Per-generator summary statistics for JSON output."""
    summary: dict[str, dict[str, int | float]] = {}
    for gen_name in {r["generator"] for r in rows}:
        gen_rows = [r for r in rows if r["generator"] == gen_name]
        n_total = len(gen_rows)
        n_with_pos = sum(1 for r in gen_rows if r["log_aut_pos"] > 0)
        n_with_canon = sum(1 for r in gen_rows if r["log_aut_twin_canonical"] > 0)
        n_with_delta = sum(1 for r in gen_rows if r["delta_canonical_local"] > 0)
        if n_total:
            mpos = float(mean(r["log_aut_pos"] for r in gen_rows))
            mcan = float(mean(r["log_aut_twin_canonical"] for r in gen_rows))
            mloc = float(mean(r["log_aut_twin_local"] for r in gen_rows))
        else:
            mpos = 0.0
            mcan = 0.0
            mloc = 0.0
        summary[gen_name] = {
            "n_tasks": n_total,
            "n_with_pos": n_with_pos,
            "n_with_canon": n_with_canon,
            "n_with_delta": n_with_delta,
            "rate_with_pos": n_with_pos / n_total if n_total else 0.0,
            "rate_with_canon": n_with_canon / n_total if n_total else 0.0,
            "rate_with_delta": n_with_delta / n_total if n_total else 0.0,
            "mean_log_aut_pos": mpos,
            "mean_log_aut_canonical": mcan,
            "mean_log_aut_local": mloc,
        }
    return summary


def _row_to_dict(r: E8Row) -> dict[str, Any]:
    keys = E8Row.__dataclass_fields__.keys()
    return {k: getattr(r, k) for k in keys}


def run(
    *,
    n_workers: int = 4,
    seeds: int = DEFAULT_N_SEEDS_STOCHASTIC,
    max_n: int = 50,
    n_values: tuple[int, ...] = DEFAULT_N_VALUES_STOCHASTIC,
    log_queue: Any | None = None,
    task_timeout_s: float = DEFAULT_PER_TASK_TIMEOUT_S,
    recursive_max_n: int = DEFAULT_RECURSIVE_MAX_N,
    backend: str = "networkx",
) -> E8Result:
    n_vals = tuple(n for n in n_values if n <= max_n)
    tasks = [E8RunSpec(t, backend, recursive_max_n) for t in enumerate_tasks(n_vals, seeds)]
    logger.info(
        "scheduling %d tasks across %d workers | mp_context=spawn",
        len(tasks),
        n_workers,
    )
    rows: list[E8Row] = []
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
    future_to_task: dict[Any, E8RunSpec] = {}
    start_times: dict[str, float] = {}
    failed = 0
    timed_out = 0
    interrupted = False
    force_kill = False
    completed = 0
    try:
        try:
            for s in tasks:
                fut = pool.submit(_task, s)
                future_to_task[fut] = s
                start_times[_row_task_id(s.gtask)] = time.perf_counter()
            logger.info("all %d tasks submitted to pool", len(future_to_task))
            pending = set(future_to_task.keys())
            while pending:
                now = time.perf_counter()
                for fut in list(pending):
                    t = future_to_task[fut]
                    task_id = _row_task_id(t.gtask)
                    age = now - start_times[task_id]
                    if age > task_timeout_s:
                        pending.remove(fut)
                        timed_out += 1
                        force_kill = True
                        fut.cancel()
                        logger.error(
                            "[%s] task TIMEOUT after %.0fs",
                            task_id,
                            task_timeout_s,
                        )
                try:
                    for fut in as_completed(list(pending), timeout=0.5):
                        if fut not in pending:
                            continue
                        pending.remove(fut)
                        t = future_to_task[fut]
                        task_id = _row_task_id(t.gtask)
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
                                "progress | %d / %d (failed=%d, timed_out=%d)",
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
        if timed_out:
            logger.warning(
                "timed out tasks (%d) | forcing hard pool shutdown of workers",
                timed_out,
            )
    finally:
        _shutdown_pool_hard(pool, interrupted=interrupted or force_kill)
    if interrupted:
        _flush_logging_handlers()
        os._exit(130)

    def _sort_key(r: E8Row) -> tuple[str, int, int, str]:
        return (r.generator, r.n_target, r.seed_or_index, r.params)

    return E8Result(rows=tuple(sorted(rows, key=_sort_key)))


def write_outputs(result: E8Result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = [f.name for f in E8Row.__dataclass_fields__.values()]
    csv_path = output_dir / "e8_generator_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(cols)
        for row in result.rows:
            w.writerow([getattr(row, c) for c in cols])
    row_dicts: list[dict[str, Any]] = [_row_to_dict(r) for r in result.rows]
    summary = compute_summary(row_dicts)
    json_path = output_dir / "e8_generator_comparison.json"
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "n_rows": len(result.rows),
                "summary": summary,
            },
            jf,
            indent=2,
            allow_nan=True,
        )
