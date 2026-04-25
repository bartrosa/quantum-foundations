"""E2: discrete Gibbs check on |Aut| and S_A for disjoint union (iso vs non-iso)."""

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

from quantum_foundations.entropic_causets.automorphism import aut_order
from quantum_foundations.entropic_causets.causet import build_stratified_game_causet, disjoint_union
from quantum_foundations.entropic_causets.entropy import entropy_decomposition
from quantum_foundations.entropic_causets.parallel import (
    TaskDiagnostic,
    iter_pool_unordered,
    spawn_task_seeds,
)

logger = logging.getLogger("quantum_foundations.entropic_causets.experiments.e2_gibbs")

_GIBBS_EPS = 1e-9
_S_A_EPS = 1e-10


def _ln_binom(n: int, k: int) -> float:
    """``\\ln C(n,k)`` via ``lgamma`` (stable for large *n*)."""
    if k < 0 or k > n:
        return float("-inf")
    return float(
        math.lgamma(float(n) + 1.0) - math.lgamma(float(k) + 1.0) - math.lgamma(float(n - k) + 1.0)
    )


@dataclass(frozen=True, slots=True)
class E2Task:
    """One Gibbs trial: isomorphic (iso) or non-isomorphic (noniso) pair id + seeds."""

    trial: int
    kind: str
    n1: int
    n2: int
    seed1: int
    seed2: int


@dataclass(frozen=True, slots=True)
class E2Row:
    """|Aut| log-space test plus Boltzmann S_A disjoint-union excess."""

    trial: int
    kind: str
    n1: int
    n2: int
    log_aut1: float
    log_aut2: float
    log_aut_union: float
    ratio: float
    expected_ln: float
    deviation: float
    s_a_1: float
    s_a_2: float
    s_a_union: float
    s_a_extra: float
    s_a_extra_expected: float
    s_a_deviation: float


@dataclass(frozen=True, slots=True)
class E2Summary:
    """|Aut| and S_A pass/fail counts and worst absolute deviations."""

    pass_count: int
    fail_count: int
    max_abs_deviation: float
    s_a_pass_count: int
    s_a_fail_count: int
    s_a_max_abs_deviation: float


@dataclass(frozen=True, slots=True)
class E2Result:
    """All trials plus summaries used by the two gates."""

    rows: tuple[E2Row, ...]
    summary: E2Summary


def _task_to_row(task: E2Task) -> tuple[E2Row, TaskDiagnostic]:
    task_id = f"e2:trial={task.trial}"
    t0 = time.perf_counter()
    c1 = build_stratified_game_causet(
        n_target=task.n1,
        max_rank=8,
        sample_rate=0.25,
        seed=task.seed1,
    )
    if task.kind == "noniso":
        c2 = build_stratified_game_causet(
            n_target=task.n2,
            max_rank=8,
            sample_rate=0.25,
            seed=task.seed2,
        )
    else:
        c2 = c1
    union = disjoint_union(c1, c2)
    aut1 = aut_order(c1)
    aut2 = aut_order(c2)
    autu = aut_order(union)
    ed1 = entropy_decomposition(c1)
    ed2 = entropy_decomposition(c2)
    edu = entropy_decomposition(union)
    expected = 2.0 if task.kind == "iso" else 1.0
    ratio = autu / (aut1 * aut2)
    expected_ln = math.log(expected)
    deviation = math.log(autu) - (math.log(aut1) + math.log(aut2) + expected_ln)
    n1, n2 = task.n1, task.n2
    if task.kind == "iso":
        s_a_exp = _ln_binom(2 * n1, n1) - math.log(2.0)
    else:
        s_a_exp = _ln_binom(n1 + n2, n1)
    s_a_ex = edu.s_a_pos - ed1.s_a_pos - ed2.s_a_pos
    s_a_dev = abs(s_a_ex - s_a_exp)
    row = E2Row(
        trial=task.trial,
        kind=task.kind,
        n1=task.n1,
        n2=task.n2,
        log_aut1=math.log(aut1) if aut1 > 0 else 0.0,
        log_aut2=math.log(aut2) if aut2 > 0 else 0.0,
        log_aut_union=math.log(autu) if autu > 0 else 0.0,
        ratio=ratio,
        expected_ln=expected_ln,
        deviation=deviation,
        s_a_1=ed1.s_a_pos,
        s_a_2=ed2.s_a_pos,
        s_a_union=edu.s_a_pos,
        s_a_extra=s_a_ex,
        s_a_extra_expected=s_a_exp,
        s_a_deviation=s_a_dev,
    )
    elapsed = time.perf_counter() - t0
    return row, TaskDiagnostic(task_id, "ok", elapsed, ())


def run(
    *,
    n_workers: int = max(1, (os.cpu_count() or 1) - 1),
    seeds: int = 30,
    ns: tuple[int, ...] = (15, 20),
    global_seed: int = 20260425,
    log_queue: Any | None = None,
) -> E2Result:
    """Check disjoint-union |Aut| and S_A identities for iso / non-iso causet pairs."""
    needed = seeds * len(ns) * 3
    child = spawn_task_seeds(global_seed, needed)
    tasks: list[E2Task] = []
    idx = 0
    trial = 0
    logger.info("E2 start | n_workers=%d seeds=%d ns=%s", n_workers, seeds, ns)

    for n in ns:
        for _ in range(seeds):
            tasks.append(
                E2Task(
                    trial=trial,
                    kind="iso",
                    n1=n,
                    n2=n,
                    seed1=child[idx],
                    seed2=child[idx],
                )
            )
            idx += 1
            trial += 1
        for _ in range(seeds):
            tasks.append(
                E2Task(
                    trial=trial,
                    kind="noniso",
                    n1=n,
                    n2=n,
                    seed1=child[idx],
                    seed2=child[idx + 1],
                )
            )
            idx += 2
            trial += 1

    row_list: list[E2Row] = []
    for row, diag in iter_pool_unordered(
        _task_to_row,
        tasks,
        n_workers=n_workers,
        log_queue=log_queue,
    ):
        for level, msg in diag.messages:
            logger.log(level, "[%s] %s", diag.task_id, msg)
        logger.info(
            "trial %d (%s) | n1=%d n2=%d log_aut1=%.4f log_aut2=%.4f log_aut_union=%.4f "
            "ratio=%.4f expected_ratio=%.4f deviation=%.2e s_a_dev=%.2e",
            row.trial,
            row.kind,
            row.n1,
            row.n2,
            row.log_aut1,
            row.log_aut2,
            row.log_aut_union,
            row.ratio,
            math.exp(row.expected_ln),
            row.deviation,
            row.s_a_deviation,
        )
        if abs(row.deviation) > _GIBBS_EPS:
            logger.error(
                "GIBBS GATE FAILED | trial=%d kind=%s deviation=%.4e — "
                "aut_order or disjoint_union is buggy",
                row.trial,
                row.kind,
                row.deviation,
            )
        if row.s_a_deviation > _S_A_EPS:
            logger.error(
                "S_A GIBBS GATE FAILED | trial=%d kind=%s s_a_deviation=%.4e",
                row.trial,
                row.kind,
                row.s_a_deviation,
            )
        row_list.append(row)

    fail_count = sum(1 for row in row_list if abs(row.deviation) > _GIBBS_EPS)
    s_a_fail = sum(1 for row in row_list if row.s_a_deviation > _S_A_EPS)
    summary = E2Summary(
        pass_count=len(row_list) - fail_count,
        fail_count=fail_count,
        max_abs_deviation=max((abs(r.deviation) for r in row_list), default=0.0),
        s_a_pass_count=len(row_list) - s_a_fail,
        s_a_fail_count=s_a_fail,
        s_a_max_abs_deviation=max((r.s_a_deviation for r in row_list), default=0.0),
    )
    logger.info(
        "E2 summary | aut pass=%d fail=%d max_abs_deviation=%.4e | S_A pass=%d fail=%d max=%.4e",
        summary.pass_count,
        summary.fail_count,
        summary.max_abs_deviation,
        summary.s_a_pass_count,
        summary.s_a_fail_count,
        summary.s_a_max_abs_deviation,
    )
    return E2Result(
        rows=tuple(sorted(row_list, key=lambda r: r.trial)),
        summary=summary,
    )


def write_outputs(result: E2Result, output_dir: Path) -> None:
    """Write ``e2_gibbs.{csv,json}`` to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "e2_gibbs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "trial",
                "kind",
                "n1",
                "n2",
                "log_aut1",
                "log_aut2",
                "log_aut_union",
                "ratio",
                "expected_ln",
                "deviation",
                "s_a_1",
                "s_a_2",
                "s_a_union",
                "s_a_extra",
                "s_a_extra_expected",
                "s_a_deviation",
            ]
        )
        for row in result.rows:
            writer.writerow(
                [
                    row.trial,
                    row.kind,
                    row.n1,
                    row.n2,
                    row.log_aut1,
                    row.log_aut2,
                    row.log_aut_union,
                    row.ratio,
                    row.expected_ln,
                    row.deviation,
                    row.s_a_1,
                    row.s_a_2,
                    row.s_a_union,
                    row.s_a_extra,
                    row.s_a_extra_expected,
                    row.s_a_deviation,
                ]
            )
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "summary": asdict(result.summary),
    }
    (output_dir / "e2_gibbs.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    e2o = run(n_workers=1, seeds=2, ns=(10,))
    write_outputs(e2o, Path("papers/entropic-causets/results"))
