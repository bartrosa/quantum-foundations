"""E3: ΔS between mass-motif placement at combinatorial distance *r* vs “at infinity”."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from quantum_foundations.entropic_causets.causet import (
    GameCauset,
    build_stratified_game_causet,
    structural_diff,
)
from quantum_foundations.entropic_causets.entropy import entropy_decomposition
from quantum_foundations.entropic_causets.parallel import (
    TaskDiagnostic,
    iter_pool_unordered,
    spawn_task_seeds,
)

logger = logging.getLogger("quantum_foundations.entropic_causets.experiments.e3_verlinde")

_BG_MAX_RANK = 30
_BG_SAMPLE = 0.2


@dataclass(frozen=True, slots=True)
class E3Task:
    """One (r, seed, n_background) triple for a worker run."""

    r: int
    seed: int
    n_background: int


@dataclass(frozen=True, slots=True)
class E3Row:
    """E3 row: |Aut| and full decomposition at *r* vs at “infinity” (identical *n*)."""

    r: int
    seed: int
    log_aut_at_r: float
    log_aut_at_infinity: float
    delta_s: float
    log_aut_pos_at_r: float
    log_aut_pos_at_inf: float
    log_aut_twin_at_r: float
    log_aut_twin_at_inf: float
    log_n_factorial: float
    delta_s_b_pos: float
    delta_s_b_twin: float
    delta_s_a_pos: float
    delta_s_a_twin: float
    twin_classes_at_r: int
    twin_classes_at_inf: int


@dataclass(frozen=True, slots=True)
class E3Fit:
    """Linear least-squares of *y* vs ``log r``; residual vs slope stderr split."""

    slope: float
    intercept: float
    r2: float
    stderr_residuals: float
    stderr_slope: float


@dataclass(frozen=True, slots=True)
class E3Result:
    """Pooled E3 rows: per-metric log–log fits + qualitative *interpretation*."""

    rows: tuple[E3Row, ...]
    fit: dict[str, E3Fit]
    interpretation: str
    interpretation_basis: str


def _insert_mass(causet: GameCauset, anchor: int) -> GameCauset:
    """Attach a 5+1 “bundle” motif above *anchor* (5 covers + one top)."""
    base_n = causet.n
    rank = list(causet.rank)
    past = list(causet.past)
    base_rank = causet.rank[anchor]
    leaves: list[int] = []

    for _ in range(5):
        rank.append(base_rank + 1)
        past.append(frozenset({anchor}))
        leaves.append(len(rank) - 1)

    rank.append(base_rank + 2)
    past.append(frozenset(leaves))
    return GameCauset(n=base_n + 6, rank=tuple(rank), past=tuple(past))


def _pick_anchor(causet: GameCauset, target_rank: int) -> int | None:
    for idx, rk in enumerate(causet.rank):
        if rk == target_rank:
            return idx
    return None


def _find_dense_motif_anchor(causet: GameCauset, rng: np.random.Generator) -> int | None:
    """Return an index with ``0 < rank < max(rank)`` if possible, else any non-extremal."""
    if causet.n == 0:
        return None
    mr = max(causet.rank) if causet.rank else 0
    cands: list[int] = [i for i, rk in enumerate(causet.rank) if 0 < rk < mr]
    if cands:
        return int(rng.choice(np.array(cands, dtype=np.int64)))
    cands2 = [i for i, rk in enumerate(causet.rank) if rk == mr and mr > 0]
    if cands2:
        return int(rng.choice(np.array(cands2, dtype=np.int64)))
    if causet.n > 0:
        return 0
    return None


def _attempt_placement_at_distance(
    background: GameCauset,
    r: int,
    rng: np.random.Generator,
) -> tuple[GameCauset | None, list[tuple[int, str]]]:
    """Try two masses so the target rank offset between anchors is *r* (stratified proxy)."""
    msgs: list[tuple[int, str]] = []
    msgs.append((logging.DEBUG, f"placement_at_distance r={r} | causet_n={background.n}"))

    max_rank = max(background.rank) if background.rank else 0
    left_rank = max(0, max_rank // 4)
    target_right = left_rank + r
    if target_right > max_rank:
        msgs.append(
            (
                logging.WARNING,
                f"SKIP: no candidate at distance r={r} from m1 "
                f"(need rank {target_right} > max_rank {max_rank})",
            )
        )
        return None, msgs

    m1 = _pick_anchor(background, left_rank)
    if m1 is None:
        m1 = _find_dense_motif_anchor(background, rng)
    if m1 is None:
        msgs.append((logging.WARNING, f"placement_at_distance r={r} | SKIP: no anchor for m1"))
        return None, msgs
    msgs.append(
        (logging.DEBUG, f"  m1 anchored at i={m1} rank={background.rank[m1]} (host for first mass)")
    )

    m2 = _pick_anchor(background, target_right)
    if m2 is None:
        msgs.append(
            (
                logging.WARNING,
                f"SKIP: no candidate at distance r={r} from m1 (no vertex at rank {target_right}, "
                f"m1_rank={background.rank[m1]} max_rank={max_rank})",
            )
        )
        return None, msgs
    msgs.append(
        (
            logging.DEBUG,
            f"  m2 anchored at i={m2} rank={background.rank[m2]} target_right={target_right}",
        )
    )

    n0 = background.n
    c1 = _insert_mass(_insert_mass(background, m1), m2)
    msgs.append(
        (
            logging.DEBUG,
            f"  inserted at-r | n_before={n0} n_after={c1.n} hasse_edges={len(c1.hasse_edges())}",
        )
    )
    return c1, msgs


def _attempt_placement_at_infinity(
    background: GameCauset,
    rng: np.random.Generator,
) -> tuple[GameCauset | None, list[tuple[int, str]]]:
    """Place two masses on opposite strata (min rank and max rank) with disjoint-past heuristics."""
    msgs: list[tuple[int, str]] = []
    msgs.append((logging.DEBUG, f"placement_at_infinity | causet_n={background.n}"))

    max_rank = max(background.rank) if background.rank else 0
    i0 = _pick_anchor(background, 0)
    i1 = _pick_anchor(background, max(0, max_rank - 1)) if max_rank > 0 else i0
    if i0 is None:
        msgs.append((logging.WARNING, "SKIP: no anchor for m1 (rank-0 host missing)"))
        return None, msgs
    if i1 is None or (max_rank > 0 and i1 == i0):
        i2 = _find_dense_motif_anchor(background, rng)
        if i2 is None or i2 == i0:
            msgs.append(
                (
                    logging.WARNING,
                    "SKIP: at_infinity | could not separate two mass anchors on background",
                )
            )
            return None, msgs
        i1 = i2

    msgs.append(
        (
            logging.DEBUG,
            f"  infinity pair i_low={i0} rank0={background.rank[i0]} "
            f"i_high={i1} rank1={background.rank[i1]}",
        )
    )
    n0 = background.n
    out = _insert_mass(_insert_mass(background, i0), i1)
    msgs.append(
        (
            logging.DEBUG,
            f"  inserted at_infinity | n_before={n0} n_after={out.n} "
            f"hasse_edges={len(out.hasse_edges())}",
        )
    )
    return out, msgs


def _labeled_equal(c1: GameCauset, c2: GameCauset) -> bool:
    return c1.n == c2.n and c1.rank == c2.rank and c1.past == c2.past


def _e3_row_invalid(r: int, seed: int) -> E3Row:
    x = float("nan")
    return E3Row(
        r=r,
        seed=seed,
        log_aut_at_r=x,
        log_aut_at_infinity=x,
        delta_s=x,
        log_aut_pos_at_r=x,
        log_aut_pos_at_inf=x,
        log_aut_twin_at_r=x,
        log_aut_twin_at_inf=x,
        log_n_factorial=x,
        delta_s_b_pos=x,
        delta_s_b_twin=x,
        delta_s_a_pos=x,
        delta_s_a_twin=x,
        twin_classes_at_r=0,
        twin_classes_at_inf=0,
    )


def _e3_task(task: E3Task) -> tuple[E3Row, TaskDiagnostic]:
    task_id = f"e3:r={task.r}:s={task.seed}"
    t0 = time.perf_counter()
    msgs: list[tuple[int, str]] = []
    rng = np.random.default_rng(task.seed)

    bg = build_stratified_game_causet(
        n_target=task.n_background,
        max_rank=_BG_MAX_RANK,
        sample_rate=_BG_SAMPLE,
        seed=task.seed,
    )
    causet_r, m_r = _attempt_placement_at_distance(bg, task.r, rng)
    msgs.extend(m_r)
    if causet_r is None:
        elapsed = time.perf_counter() - t0
        return (
            _e3_row_invalid(task.r, task.seed),
            TaskDiagnostic(task_id, "skip", elapsed, tuple(msgs)),
        )

    causet_inf, m_i = _attempt_placement_at_infinity(bg, np.random.default_rng(task.seed + 1))
    msgs.extend(m_i)
    if causet_inf is None:
        elapsed = time.perf_counter() - t0
        return (
            _e3_row_invalid(task.r, task.seed),
            TaskDiagnostic(task_id, "skip", elapsed, tuple(msgs)),
        )

    diff = structural_diff(causet_r, causet_inf)
    msgs.append(
        (
            logging.DEBUG,
            f"  structural diff | n_diff={diff.n_diff} edges_diff={diff.edges_diff} "
            f"node_set_equal={diff.node_set_equal} rank_dist_equal={diff.rank_distribution_equal}",
        )
    )
    if _labeled_equal(causet_r, causet_inf):
        msgs.append(
            (
                logging.ERROR,
                "  STRUCTURAL DIFF EMPTY | labeled causets at r and at infinity are "
                "BIT-IDENTICAL — placement is BROKEN, |Aut| comparison is meaningless. "
                "Returning NaN.",
            )
        )
        elapsed = time.perf_counter() - t0
        return (
            _e3_row_invalid(task.r, task.seed),
            TaskDiagnostic(task_id, "error", elapsed, tuple(msgs)),
        )

    ed_r = entropy_decomposition(causet_r)
    ed_i = entropy_decomposition(causet_inf)
    log_r = ed_r.log_aut_pos
    log_i = ed_i.log_aut_pos
    if ed_r.n != ed_i.n or abs(ed_r.log_n_factorial - ed_i.log_n_factorial) > 1e-5:
        msgs.append(
            (
                logging.ERROR,
                "  n / ln n! mismatch between at-r and at-infinity — aborting row",
            )
        )
        elapsed = time.perf_counter() - t0
        return _e3_row_invalid(task.r, task.seed), TaskDiagnostic(
            task_id, "error", elapsed, tuple(msgs)
        )
    delta = log_r - log_i
    d_b_t = ed_r.log_aut_twin - ed_i.log_aut_twin
    d_a_p = ed_r.s_a_pos - ed_i.s_a_pos
    d_a_t = ed_r.s_a_twin - ed_i.s_a_twin
    tcr = len(ed_r.twin_class_sizes)
    tci = len(ed_i.twin_class_sizes)
    msgs.append(
        (
            logging.INFO,
            f"  measured | r={task.r} log_aut_r={log_r:.4f} log_aut_inf={log_i:.4f} "
            f"delta_s={delta:.4f} delta_s_b_twin={d_b_t:.4f}",
        )
    )
    if abs(delta) < 1e-9 and (diff.edges_diff > 0 or not diff.rank_distribution_equal):
        msgs.append(
            (
                logging.WARNING,
                "  SUSPICIOUS | delta_s ≈ 0 despite nontrivial structural diff — "
                "verify motif symmetry",
            )
        )

    elapsed = time.perf_counter() - t0
    return (
        E3Row(
            r=task.r,
            seed=task.seed,
            log_aut_at_r=log_r,
            log_aut_at_infinity=log_i,
            delta_s=delta,
            log_aut_pos_at_r=log_r,
            log_aut_pos_at_inf=log_i,
            log_aut_twin_at_r=ed_r.log_aut_twin,
            log_aut_twin_at_inf=ed_i.log_aut_twin,
            log_n_factorial=ed_r.log_n_factorial,
            delta_s_b_pos=delta,
            delta_s_b_twin=d_b_t,
            delta_s_a_pos=d_a_p,
            delta_s_a_twin=d_a_t,
            twin_classes_at_r=tcr,
            twin_classes_at_inf=tci,
        ),
        TaskDiagnostic(task_id, "ok", elapsed, tuple(msgs)),
    )


def _e3_get_y(row: E3Row, key: str) -> float:
    """Y for ``y`` vs ``log r`` fits (``NaN`` rows are excluded before fitting)."""
    return float(getattr(row, key))


def _fit_power(rows: list[E3Row], y_attr: str) -> E3Fit:
    valid = [row for row in rows if not math.isnan(_e3_get_y(row, y_attr))]
    m = len(valid)
    if m < 2:
        return E3Fit(0.0, 0.0, 0.0, 0.0, 0.0)
    xs = [math.log(float(row.r)) for row in valid]
    ys = [_e3_get_y(row, y_attr) for row in valid]
    x_mean = sum(xs) / m
    y_mean = sum(ys) / m
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if sxx == 0.0:
        return E3Fit(0.0, y_mean, 1.0, 0.0, 0.0)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    slope = num / sxx
    intercept = y_mean - slope * x_mean
    y_hat = [intercept + slope * x for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, y_hat, strict=True))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    dof = max(1, m - 2)
    stderr_res = math.sqrt(ss_res / dof)
    stderr_slope = stderr_res / math.sqrt(sxx)
    return E3Fit(
        slope=slope,
        intercept=intercept,
        r2=r2,
        stderr_residuals=stderr_res,
        stderr_slope=stderr_slope,
    )


def run(
    *,
    n_workers: int = max(1, (os.cpu_count() or 1) - 1),
    seeds: int = 20,
    rs: tuple[int, ...] = (2, 3, 4, 5, 7, 10, 15, 20),
    n_background: int = 20,
    global_seed: int = 20260425,
    log_queue: Any | None = None,
) -> E3Result:
    """Run E3 over all ``(r, seed)`` pairs; workers return structured diagnostics for logging.

    Example:
        ``run(n_workers=1, seeds=1, rs=(2, 3), n_background=40)`` for a local smoke.
    """
    n_tasks = len(rs) * seeds
    logger.info(
        "E3 start | n_workers=%d seeds=%d rs=%s n_background=%d",
        n_workers,
        seeds,
        rs,
        n_background,
    )
    child = spawn_task_seeds(global_seed, n_tasks)
    tasks: list[E3Task] = []
    idx = 0
    for r in rs:
        for _ in range(seeds):
            tasks.append(E3Task(r=r, seed=child[idx], n_background=n_background))
            idx += 1

    out_rows: list[E3Row] = []
    for row, diag in iter_pool_unordered(_e3_task, tasks, n_workers=n_workers, log_queue=log_queue):
        for level, msg in diag.messages:
            logger.log(level, "[%s] %s", diag.task_id, msg)
        if diag.status == "skip":
            logger.warning("[%s] skipped after %.2fs", diag.task_id, diag.elapsed_s)
        elif diag.status == "error":
            logger.error("[%s] failed/flagged after %.2fs", diag.task_id, diag.elapsed_s)
        out_rows.append(row)

    rows = sorted(out_rows, key=lambda z: (z.r, z.seed))
    rlist = list(rows)
    fit = {
        "delta_s_b_pos": _fit_power(rlist, "delta_s_b_pos"),
        "delta_s_b_twin": _fit_power(rlist, "delta_s_b_twin"),
        "delta_s_a_pos": _fit_power(rlist, "delta_s_a_pos"),
        "delta_s_a_twin": _fit_power(rlist, "delta_s_a_twin"),
    }
    f0 = fit["delta_s_b_pos"]
    interpretation = "inconclusive"
    if f0.slope < -1.5:
        interpretation = "compatible with steeply decaying entropy gradient"
    elif f0.slope < -0.5:
        interpretation = "compatible with moderate power-law decay"
    basis = (
        f"delta_s_b_pos (alias delta_s) slope={f0.slope:.4f} "
        f"± {f0.stderr_slope:.4f} | y=ΔS_B vs x=log r"
    )
    return E3Result(
        rows=tuple(rows),
        fit=fit,
        interpretation=interpretation,
        interpretation_basis=basis,
    )


def write_outputs(result: E3Result, output_dir: Path) -> None:
    """Write ``e3_verlinde.{csv,json}`` under *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "e3_verlinde.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "r",
                "seed",
                "log_aut_at_r",
                "log_aut_at_infinity",
                "delta_s",
                "log_aut_pos_at_r",
                "log_aut_pos_at_inf",
                "log_aut_twin_at_r",
                "log_aut_twin_at_inf",
                "log_n_factorial",
                "delta_s_b_pos",
                "delta_s_b_twin",
                "delta_s_a_pos",
                "delta_s_a_twin",
                "twin_classes_at_r",
                "twin_classes_at_inf",
            ]
        )
        for row in result.rows:
            writer.writerow(
                [
                    row.r,
                    row.seed,
                    row.log_aut_at_r,
                    row.log_aut_at_infinity,
                    row.delta_s,
                    row.log_aut_pos_at_r,
                    row.log_aut_pos_at_inf,
                    row.log_aut_twin_at_r,
                    row.log_aut_twin_at_inf,
                    row.log_n_factorial,
                    row.delta_s_b_pos,
                    row.delta_s_b_twin,
                    row.delta_s_a_pos,
                    row.delta_s_a_twin,
                    row.twin_classes_at_r,
                    row.twin_classes_at_inf,
                ]
            )

    fit_json: dict[str, Any] = {k: asdict(v) for k, v in result.fit.items()}
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "fit": fit_json,
        "interpretation": result.interpretation,
        "interpretation_basis": result.interpretation_basis,
    }
    (output_dir / "e3_verlinde.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    e3o = run(n_workers=1, seeds=2, rs=(2, 4), n_background=60)
    write_outputs(e3o, Path("papers/entropic-causets/results"))
