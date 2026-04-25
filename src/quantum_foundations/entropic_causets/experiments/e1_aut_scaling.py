"""E1: |Aut| scaling and full entropy decomposition (log–linear fit + CSV/JSON)."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from operator import attrgetter
from pathlib import Path
from typing import Any

from quantum_foundations.entropic_causets.causet import build_stratified_game_causet
from quantum_foundations.entropic_causets.entropy import entropy_decomposition
from quantum_foundations.entropic_causets.parallel import (
    TaskDiagnostic,
    iter_pool_unordered,
    spawn_task_seeds,
)

logger = logging.getLogger("quantum_foundations.entropic_causets.experiments.e1_aut_scaling")


@dataclass(frozen=True, slots=True)
class E1Task:
    """One worker unit: target size, draw seed, builder params, and automorphism backend."""

    n: int
    seed: int
    max_rank: int
    sample_rate: float
    backend: str


@dataclass(frozen=True, slots=True)
class E1Row:
    """One CSV row: sweep level, full entropy decomposition, wall time, backend."""

    n: int
    seed: int
    n_actual: int
    log_aut: float
    time_s: float
    backend: str
    log_aut_pos: float
    log_aut_twin: float
    log_rank_factorial: float
    log_n_factorial: float
    s_a_pos: float
    s_a_twin: float
    s_a_rank: float
    delta_max_pos: float
    delta_pos_twin: float
    twin_class_count: int
    twin_max_class: int


@dataclass(frozen=True, slots=True)
class E1LinearFit:
    """Linear least-squares of *y* vs *x* = ``N``; residual vs slope std errors split."""

    slope: float
    intercept: float
    r2: float
    stderr_residuals: float
    stderr_slope: float


@dataclass(frozen=True, slots=True)
class E1Result:
    """All per-seed rows, per-metric linear fits, per-*n* summaries, interpretation."""

    rows: tuple[E1Row, ...]
    fit: dict[str, E1LinearFit]
    summary_per_n: dict[str, dict[str, float]]
    interpretation: str
    interpretation_basis: str


def _linear_regression_metrics(xs: list[float], ys: list[float]) -> E1LinearFit:
    m = len(xs)
    if m < 2:
        return E1LinearFit(0.0, 0.0, 0.0, 0.0, 0.0)
    x_mean = sum(xs) / m
    y_mean = sum(ys) / m
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if sxx == 0.0:
        return E1LinearFit(0.0, y_mean, 1.0, 0.0, 0.0)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / sxx
    intercept = y_mean - slope * x_mean
    y_hat = [intercept + slope * x for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, y_hat, strict=True))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    dof = max(1, m - 2)
    stderr_res = math.sqrt(ss_res / dof)
    stderr_slope = stderr_res / math.sqrt(sxx)
    return E1LinearFit(
        slope=slope,
        intercept=intercept,
        r2=r2,
        stderr_residuals=stderr_res,
        stderr_slope=stderr_slope,
    )


def _e1_fit_by_variant(rows: list[E1Row]) -> dict[str, E1LinearFit]:
    xs = [float(r.n) for r in rows]
    out: dict[str, E1LinearFit] = {}
    for key, attr in (
        ("log_aut_pos", "log_aut_pos"),
        ("log_aut_twin", "log_aut_twin"),
        ("s_a_pos", "s_a_pos"),
        ("s_a_twin", "s_a_twin"),
        ("delta_pos_twin", "delta_pos_twin"),
    ):
        g = attrgetter(attr)
        ys = [float(g(r)) for r in rows]
        out[key] = _linear_regression_metrics(xs, ys)
    return out


def _e1_summary_per_n(rows: list[E1Row]) -> dict[str, dict[str, float]]:
    by_n: dict[int, list[E1Row]] = {}
    for r in rows:
        by_n.setdefault(r.n, []).append(r)
    summary: dict[str, dict[str, float]] = {}
    for n_key in sorted(by_n):
        rows_bucket = by_n[n_key]
        m_pos = [x.log_aut_pos for x in rows_bucket]
        m_tw = [x.log_aut_twin for x in rows_bucket]
        s_pos = [x.s_a_pos for x in rows_bucket]
        d_pt = [x.delta_pos_twin for x in rows_bucket]
        summary[str(n_key)] = {
            "mean_log_aut_pos": float(statistics.mean(m_pos)),
            "median_log_aut_pos": float(statistics.median(m_pos)),
            "std_log_aut_pos": float(statistics.stdev(m_pos)) if len(m_pos) > 1 else 0.0,
            "mean_log_aut_twin": float(statistics.mean(m_tw)),
            "median_log_aut_twin": float(statistics.median(m_tw)),
            "std_log_aut_twin": float(statistics.stdev(m_tw)) if len(m_tw) > 1 else 0.0,
            "mean_s_a_pos": float(statistics.mean(s_pos)),
            "mean_delta_pos_twin": float(statistics.mean(d_pt)),
        }
    return summary


def _e1_interpretation(fit: dict[str, E1LinearFit]) -> tuple[str, str]:
    f = fit.get("log_aut_pos")
    if f is None or (abs(f.slope) < 1e-15 and f.stderr_slope > 1e-9):
        return "inconclusive", "log_aut_pos fit unavailable or undefined slope"
    alpha = f.slope
    basis = f"log_aut_pos slope={alpha:.3f} ± {f.stderr_slope:.3f} (linear in N, semilog in |Aut|)"
    if abs(f.stderr_slope) > 0.5 * abs(alpha):
        return "inconclusive", basis + " (large slope uncertainty)"
    if alpha > 0.7:
        return "extensive", basis
    if alpha > 0.2:
        return "surface", basis
    if alpha > 0.0:
        return "logarithmic", basis
    return "inconclusive", basis


def _run_task(task: E1Task) -> tuple[E1Row, TaskDiagnostic]:
    task_id = f"e1:n={task.n}:s={task.seed}"
    t0 = time.perf_counter()
    causet = build_stratified_game_causet(
        n_target=task.n,
        max_rank=task.max_rank,
        sample_rate=task.sample_rate,
        seed=task.seed,
    )
    ed = entropy_decomposition(causet, backend=task.backend)
    elapsed = time.perf_counter() - t0
    log_aut = ed.log_aut_pos
    tcount = len(ed.twin_class_sizes)
    tmax = max(ed.twin_class_sizes) if ed.twin_class_sizes else 0
    row = E1Row(
        n=task.n,
        seed=task.seed,
        n_actual=causet.n,
        log_aut=log_aut,
        time_s=elapsed,
        backend=task.backend,
        log_aut_pos=ed.log_aut_pos,
        log_aut_twin=ed.log_aut_twin,
        log_rank_factorial=ed.log_rank_factorial,
        log_n_factorial=ed.log_n_factorial,
        s_a_pos=ed.s_a_pos,
        s_a_twin=ed.s_a_twin,
        s_a_rank=ed.s_a_rank,
        delta_max_pos=ed.delta_max_pos,
        delta_pos_twin=ed.delta_pos_twin,
        twin_class_count=tcount,
        twin_max_class=tmax,
    )
    msgs: tuple[tuple[int, str], ...] = (
        (
            logging.DEBUG,
            f"seed {task.seed} n_actual={row.n_actual} log_aut={log_aut:.4f} time_s={elapsed:.4f}",
        ),
    )
    return row, TaskDiagnostic(task_id, "ok", elapsed, msgs)


def run(
    *,
    n_workers: int = max(1, (os.cpu_count() or 1) - 1),
    seeds: int = 50,
    ns: tuple[int, ...] = (20, 30, 50, 70, 100),
    max_rank: int = 20,
    sample_rate: float = 0.2,
    global_seed: int = 20260425,
    backend: str = "auto",
    log_queue: Any | None = None,
) -> E1Result:
    """Run the E1 |Aut| sweep: one draw and one decomposition per (n, seed)."""
    total_tasks = len(ns) * seeds
    logger.info(
        "E1 start | n_workers=%d seeds=%d ns=%s max_rank=%d",
        n_workers,
        seeds,
        ns,
        max_rank,
    )
    seeds_spawned = spawn_task_seeds(global_seed, total_tasks)
    tasks: list[E1Task] = []
    idx = 0
    for n in ns:
        for _ in range(seeds):
            tasks.append(
                E1Task(
                    n=n,
                    seed=seeds_spawned[idx],
                    max_rank=max_rank,
                    sample_rate=sample_rate,
                    backend=backend,
                )
            )
            idx += 1
        logger.info("E1 checkpoint | scheduled all seeds for n=%d", n)

    row_list: list[E1Row] = []
    done = 0
    for row, diag in iter_pool_unordered(
        _run_task,
        tasks,
        n_workers=n_workers,
        log_queue=log_queue,
    ):
        for level, msg in diag.messages:
            logger.log(level, "[%s] %s", diag.task_id, msg)
        if diag.status == "skip":
            logger.warning("[%s] skipped after %.2fs", diag.task_id, diag.elapsed_s)
        elif diag.status == "error":
            logger.error("[%s] failed after %.2fs", diag.task_id, diag.elapsed_s)
        row_list.append(row)
        done += 1
        if done % max(1, total_tasks // 10) == 0 or done == total_tasks:
            logger.info("E1 progress | %d / %d tasks complete", done, total_tasks)

    rows_sorted = sorted(row_list, key=lambda r: (r.n, r.seed))
    rlist = list(rows_sorted)
    fit = _e1_fit_by_variant(rlist)
    summ = _e1_summary_per_n(rlist)
    interpretation, basis = _e1_interpretation(fit)

    by_n: dict[int, list[float]] = {}
    for r in rows_sorted:
        by_n.setdefault(r.n, []).append(r.log_aut)
    for n_val in sorted(by_n):
        bucket = by_n[n_val]
        m = float(statistics.mean(bucket))
        med = float(statistics.median(bucket))
        std = float(statistics.stdev(bucket)) if len(bucket) > 1 else 0.0
        logger.info(
            "E1 summary at n=%d | count=%d mean_log_aut=%.4f median=%.4f "
            "std=%.4f min=%.4f max=%.4f",
            n_val,
            len(bucket),
            m,
            med,
            std,
            min(bucket),
            max(bucket),
        )
        if len(bucket) >= 3 and min(abs(m), abs(med)) > 1e-9:
            ratio = max(m, med) / max(min(m, med), 1e-9)
            if ratio > 1.5:
                logger.warning(
                    "distribution at N=%d looks heavy-tailed (mean=%.4f median=%.4f ratio=%.2f)",
                    n_val,
                    m,
                    med,
                    ratio,
                )

    fp = fit.get("log_aut_pos")
    if fp:
        logger.info(
            "E1 fit (log_aut_pos) | slope=%.6f intercept=%.6f r2=%.6f "
            "stderr_residuals=%.6f stderr_slope=%.6f",
            fp.slope,
            fp.intercept,
            fp.r2,
            fp.stderr_residuals,
            fp.stderr_slope,
        )
        if abs(fp.slope) > 1e-9 and abs(fp.stderr_slope) > 0.5 * abs(fp.slope):
            logger.warning("fit unreliable: |stderr_slope| > 0.5|slope| for log_aut_pos")
    return E1Result(
        rows=tuple(rows_sorted),
        fit=fit,
        summary_per_n=summ,
        interpretation=interpretation,
        interpretation_basis=basis,
    )


def write_outputs(result: E1Result, output_dir: Path, *, n_workers: int) -> None:
    """Write ``e1_aut_scaling.{csv,json}`` to *output_dir* (creates the directory)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    header = [
        "n",
        "seed",
        "n_actual",
        "log_aut",
        "time_s",
        "backend",
        "log_aut_pos",
        "log_aut_twin",
        "log_rank_factorial",
        "log_n_factorial",
        "s_a_pos",
        "s_a_twin",
        "s_a_rank",
        "delta_max_pos",
        "delta_pos_twin",
        "twin_class_count",
        "twin_max_class",
    ]
    with (output_dir / "e1_aut_scaling.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in result.rows:
            writer.writerow(
                [
                    row.n,
                    row.seed,
                    row.n_actual,
                    row.log_aut,
                    row.time_s,
                    row.backend,
                    row.log_aut_pos,
                    row.log_aut_twin,
                    row.log_rank_factorial,
                    row.log_n_factorial,
                    row.s_a_pos,
                    row.s_a_twin,
                    row.s_a_rank,
                    row.delta_max_pos,
                    row.delta_pos_twin,
                    row.twin_class_count,
                    row.twin_max_class,
                ]
            )

    fit_json: dict[str, Any] = {k: asdict(v) for k, v in result.fit.items()}
    payload: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "n_workers": n_workers,
        "fit": fit_json,
        "summary_per_n": result.summary_per_n,
        "interpretation": result.interpretation,
        "interpretation_basis": result.interpretation_basis,
    }
    (output_dir / "e1_aut_scaling.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    e1r = run(n_workers=1, seeds=3, ns=(10, 15))
    write_outputs(e1r, Path("papers/entropic-causets/results"), n_workers=1)
