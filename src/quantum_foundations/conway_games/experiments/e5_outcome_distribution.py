"""E5: outcome distribution (shallow and optional recursive)."""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from quantum_foundations.conway_games.causet import build_random_conway_causet
from quantum_foundations.conway_games.outcome import outcome_histogram
from quantum_foundations.entropic_causets.parallel import (
    TaskDiagnostic,
    iter_pool_unordered,
    spawn_task_seeds,
)


@dataclass(frozen=True, slots=True)
class E5Task:
    n: int
    seed: int


@dataclass(frozen=True, slots=True)
class E5Row:
    n: int
    seed: int
    n_actual: int
    count_L_shallow: int
    count_R_shallow: int
    count_eq_shallow: int
    count_fuzzy_shallow: int
    count_L_recursive: int
    count_R_recursive: int
    count_eq_recursive: int
    count_fuzzy_recursive: int
    time_shallow_s: float
    time_recursive_s: float


@dataclass(frozen=True, slots=True)
class E5Result:
    rows: tuple[E5Row, ...]


def _task(t: E5Task) -> tuple[E5Row, TaskDiagnostic]:
    c = build_random_conway_causet(n_target=t.n, max_rank=20, seed=t.seed)
    t0 = time.perf_counter()
    hs = outcome_histogram(c, recursive=False)
    ts = time.perf_counter() - t0
    if t.n <= 50:
        t1 = time.perf_counter()
        hr = outcome_histogram(c, recursive=True)
        tr = time.perf_counter() - t1
    else:
        hr = {"L": 0, "R": 0, "=": 0, "||": 0}
        tr = 0.0
    row = E5Row(
        n=t.n,
        seed=t.seed,
        n_actual=c.n,
        count_L_shallow=hs["L"],
        count_R_shallow=hs["R"],
        count_eq_shallow=hs["="],
        count_fuzzy_shallow=hs["||"],
        count_L_recursive=hr["L"],
        count_R_recursive=hr["R"],
        count_eq_recursive=hr["="],
        count_fuzzy_recursive=hr["||"],
        time_shallow_s=ts,
        time_recursive_s=tr,
    )
    return row, TaskDiagnostic(f"e5:n={t.n}:s={t.seed}", "ok", ts + tr, tuple())


def run(
    *,
    n_workers: int = max(1, (os.cpu_count() or 1) - 1),
    seeds: int = 50,
    ns: tuple[int, ...] = (30, 50, 100, 200),
    global_seed: int = 20260425,
) -> E5Result:
    child = spawn_task_seeds(global_seed, len(ns) * seeds)
    tasks: list[E5Task] = []
    i = 0
    for n in ns:
        for _ in range(seeds):
            tasks.append(E5Task(n=n, seed=child[i]))
            i += 1
    rows = [row for row, _ in iter_pool_unordered(_task, tasks, n_workers=n_workers)]
    return E5Result(rows=tuple(sorted(rows, key=lambda r: (r.n, r.seed))))


def write_outputs(result: E5Result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = [f.name for f in E5Row.__dataclass_fields__.values()]
    with (output_dir / "e5_outcome_distribution.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(cols)
        for row in result.rows:
            w.writerow([getattr(row, c) for c in cols])
    payload = {"timestamp": datetime.now(UTC).isoformat(), "n_rows": len(result.rows)}
    (output_dir / "e5_outcome_distribution.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
