"""E7: Gibbs identities on disjoint unions for Conway causets."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from quantum_foundations.conway_games.causet import (
    build_random_conway_causet,
    disjoint_union_conway,
)
from quantum_foundations.conway_games.entropy import entropy_decomposition_conway
from quantum_foundations.entropic_causets.parallel import spawn_task_seeds


@dataclass(frozen=True, slots=True)
class E7Row:
    trial: int
    kind: str
    n1: int
    n2: int
    log_aut1_pos: float
    log_aut2_pos: float
    log_aut_union_pos: float
    log_aut1_twin: float
    log_aut2_twin: float
    log_aut_union_twin: float
    ratio_pos: float
    ratio_twin: float
    expected_ln: float
    deviation_pos: float
    deviation_twin: float
    s_a_1: float
    s_a_2: float
    s_a_union: float
    s_a_extra: float
    s_a_extra_expected: float
    s_a_deviation: float


@dataclass(frozen=True, slots=True)
class E7Summary:
    pass_count: int
    fail_count: int
    max_abs_deviation: float
    s_a_fail_count: int


@dataclass(frozen=True, slots=True)
class E7Result:
    rows: tuple[E7Row, ...]
    summary: E7Summary


def run(
    *,
    seeds: int = 30,
    ns: tuple[int, ...] = (10, 15, 20),
    global_seed: int = 20260425,
) -> E7Result:
    child = spawn_task_seeds(global_seed, len(ns) * seeds * 3)
    i = 0
    rows: list[E7Row] = []
    trial = 0
    for n in ns:
        for kind in ("iso", "noniso"):
            for _ in range(seeds):
                # Keep events-per-rank roughly stable across N during sweeps.
                c1 = build_random_conway_causet(
                    n_target=n, max_rank=max(2, n // 4), seed=child[i]
                )
                i += 1
                if kind == "iso":
                    c2 = c1
                else:
                    c2 = build_random_conway_causet(
                        n_target=n, max_rank=max(2, n // 4), seed=child[i]
                    )
                    i += 1
                u = disjoint_union_conway(c1, c2)
                d1 = entropy_decomposition_conway(c1)
                d2 = entropy_decomposition_conway(c2)
                du = entropy_decomposition_conway(u)
                exp = 2.0 if kind == "iso" else 1.0
                expected_ln = math.log(exp)
                dev_pos = du.log_aut_pos - (d1.log_aut_pos + d2.log_aut_pos + expected_ln)
                dev_twin = du.log_aut_twin_local - (
                    d1.log_aut_twin_local + d2.log_aut_twin_local + expected_ln
                )
                if kind == "iso":
                    s_a_exp = (
                        math.lgamma(float(2 * n) + 1.0)
                        - 2.0 * math.lgamma(float(n) + 1.0)
                        - math.log(2.0)
                    )
                else:
                    s_a_exp = (
                        math.lgamma(float(n + n) + 1.0)
                        - math.lgamma(float(n) + 1.0)
                        - math.lgamma(float(n) + 1.0)
                    )
                s_a_extra = du.s_a_pos - d1.s_a_pos - d2.s_a_pos
                rows.append(
                    E7Row(
                        trial=trial,
                        kind=kind,
                        n1=n,
                        n2=n,
                        log_aut1_pos=d1.log_aut_pos,
                        log_aut2_pos=d2.log_aut_pos,
                        log_aut_union_pos=du.log_aut_pos,
                        log_aut1_twin=d1.log_aut_twin_local,
                        log_aut2_twin=d2.log_aut_twin_local,
                        log_aut_union_twin=du.log_aut_twin_local,
                        ratio_pos=du.aut_order_pos / (d1.aut_order_pos * d2.aut_order_pos),
                        ratio_twin=du.aut_order_twin_local
                        / (d1.aut_order_twin_local * d2.aut_order_twin_local),
                        expected_ln=expected_ln,
                        deviation_pos=dev_pos,
                        deviation_twin=dev_twin,
                        s_a_1=d1.s_a_pos,
                        s_a_2=d2.s_a_pos,
                        s_a_union=du.s_a_pos,
                        s_a_extra=s_a_extra,
                        s_a_extra_expected=float(s_a_exp),
                        s_a_deviation=abs(s_a_extra - s_a_exp),
                    )
                )
                trial += 1
    fail = sum(1 for r in rows if abs(r.deviation_pos) > 1e-9 or abs(r.deviation_twin) > 1e-9)
    s_a_fail = sum(1 for r in rows if r.s_a_deviation > 1e-9)
    summary = E7Summary(
        pass_count=len(rows) - fail,
        fail_count=fail,
        max_abs_deviation=max((abs(r.deviation_pos) for r in rows), default=0.0),
        s_a_fail_count=s_a_fail,
    )
    return E7Result(rows=tuple(rows), summary=summary)


def write_outputs(result: E7Result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = [f.name for f in E7Row.__dataclass_fields__.values()]
    with (output_dir / "e7_conway_gibbs.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(cols)
        for row in result.rows:
            w.writerow([getattr(row, c) for c in cols])
    payload = {"timestamp": datetime.now(UTC).isoformat(), "summary": asdict(result.summary)}
    (output_dir / "e7_conway_gibbs.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
