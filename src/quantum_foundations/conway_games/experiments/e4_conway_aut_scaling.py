"""E4: Conway |Aut| scaling with full decomposition reporting."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from quantum_foundations.conway_games.causet import build_random_conway_causet
from quantum_foundations.conway_games.entropy import entropy_decomposition_conway
from quantum_foundations.entropic_causets.parallel import (
    TaskDiagnostic,
    iter_pool_unordered,
    spawn_task_seeds,
)


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
    log_aut_twin: float
    log_aut_outcome: float
    log_rank_factorial: float
    log_n_factorial: float
    s_a_pos: float
    s_a_twin: float
    s_a_outcome: float
    s_a_rank: float
    delta_max_pos: float
    delta_pos_twin: float
    delta_twin_outcome: float
    twin_class_count: int
    outcome_class_count: int
    time_s: float
    backend: str


@dataclass(frozen=True, slots=True)
class E4Result:
    rows: tuple[E4Row, ...]


def _task(t: E4Task) -> tuple[E4Row, TaskDiagnostic]:
    t0 = time.perf_counter()
    c = build_random_conway_causet(n_target=t.n, max_rank=20, seed=t.seed)
    d = entropy_decomposition_conway(c, backend=t.backend)
    dt = time.perf_counter() - t0
    row = E4Row(
        n=t.n,
        seed=t.seed,
        n_actual=c.n,
        log_aut_pos=d.log_aut_pos,
        log_aut_twin=d.log_aut_twin,
        log_aut_outcome=d.log_aut_outcome,
        log_rank_factorial=d.log_rank_factorial,
        log_n_factorial=d.log_n_factorial,
        s_a_pos=d.s_a_pos,
        s_a_twin=d.s_a_twin,
        s_a_outcome=d.s_a_outcome,
        s_a_rank=d.s_a_rank,
        delta_max_pos=d.delta_max_pos,
        delta_pos_twin=d.delta_pos_twin,
        delta_twin_outcome=d.delta_twin_outcome,
        twin_class_count=len(d.twin_class_sizes),
        outcome_class_count=len(d.outcome_class_sizes),
        time_s=dt,
        backend=t.backend,
    )
    return row, TaskDiagnostic(f"e4:n={t.n}:s={t.seed}", "ok", dt, tuple())


def run(
    *,
    n_workers: int = max(1, (os.cpu_count() or 1) - 1),
    seeds: int = 30,
    ns: tuple[int, ...] = (15, 20, 30, 50, 70, 100),
    global_seed: int = 20260425,
    backend: str = "auto",
) -> E4Result:
    child = spawn_task_seeds(global_seed, len(ns) * seeds)
    tasks: list[E4Task] = []
    i = 0
    for n in ns:
        for _ in range(seeds):
            tasks.append(E4Task(n=n, seed=child[i], backend=backend))
            i += 1
    rows = [row for row, _ in iter_pool_unordered(_task, tasks, n_workers=n_workers)]
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
        "mean_log_aut_twin": (sum(r.log_aut_twin for r in valid) / len(valid)) if valid else 0.0,
    }
    (output_dir / "e4_conway_aut_scaling.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
