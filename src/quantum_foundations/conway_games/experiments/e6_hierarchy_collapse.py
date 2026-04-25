"""E6: hierarchy-collapse rates for Conway entropy decomposition."""

from __future__ import annotations

import csv
import json
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
class E6Task:
    n: int
    seed: int
    backend: str


@dataclass(frozen=True, slots=True)
class E6Row:
    n: int
    seed: int
    delta_max_pos: float
    delta_pos_twin: float
    delta_twin_outcome: float
    delta_twin_outcome_normalized: float
    log_n_factorial: float
    time_s: float


@dataclass(frozen=True, slots=True)
class E6Result:
    rows: tuple[E6Row, ...]


def _task(t: E6Task) -> tuple[E6Row, TaskDiagnostic]:
    t0 = time.perf_counter()
    c = build_random_conway_causet(n_target=t.n, max_rank=20, seed=t.seed)
    d = entropy_decomposition_conway(c, backend=t.backend)
    dt = time.perf_counter() - t0
    denom = d.log_n_factorial if abs(d.log_n_factorial) > 1e-12 else 1.0
    row = E6Row(
        n=t.n,
        seed=t.seed,
        delta_max_pos=d.delta_max_pos,
        delta_pos_twin=d.delta_pos_twin,
        delta_twin_outcome=d.delta_twin_outcome,
        delta_twin_outcome_normalized=d.delta_twin_outcome / denom,
        log_n_factorial=d.log_n_factorial,
        time_s=dt,
    )
    return row, TaskDiagnostic(f"e6:n={t.n}:s={t.seed}", "ok", dt, tuple())


def run(
    *,
    n_workers: int = max(1, (os.cpu_count() or 1) - 1),
    seeds: int = 30,
    ns: tuple[int, ...] = (20, 30, 50, 70, 100),
    global_seed: int = 20260425,
    backend: str = "auto",
) -> E6Result:
    child = spawn_task_seeds(global_seed, len(ns) * seeds)
    tasks: list[E6Task] = []
    i = 0
    for n in ns:
        for _ in range(seeds):
            tasks.append(E6Task(n=n, seed=child[i], backend=backend))
            i += 1
    rows = [row for row, _ in iter_pool_unordered(_task, tasks, n_workers=n_workers)]
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
