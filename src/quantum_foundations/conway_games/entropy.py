"""Entropy decomposition for Conway causets across pos/twin/outcome/rank levels."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from quantum_foundations.conway_games.automorphism import aut_order_conway
from quantum_foundations.conway_games.causet import ConwayCauset
from quantum_foundations.conway_games.outcome import shallow_outcome


@dataclass(frozen=True, slots=True)
class ConwayEntropyDecomposition:
    """Entropy decomposition for ConwayCauset in S_A and S_B conventions."""

    aut_order_pos: int
    aut_order_twin: int
    aut_order_outcome: int
    rank_factorial_product: int
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
    twin_class_sizes: tuple[int, ...]
    outcome_class_sizes: tuple[int, ...]
    n: int


def twin_classes_conway(causet: ConwayCauset) -> tuple[tuple[int, ...], ...]:
    """Partition events by local colored-neighborhood indistinguishability."""
    ch_l: list[set[int]] = [set() for _ in range(causet.n)]
    ch_r: list[set[int]] = [set() for _ in range(causet.n)]
    for v in range(causet.n):
        for u in causet.past_left[v]:
            ch_l[u].add(v)
        for u in causet.past_right[v]:
            ch_r[u].add(v)
    groups: dict[
        tuple[int, frozenset[int], frozenset[int], frozenset[int], frozenset[int]],
        list[int],
    ] = defaultdict(list)
    for i in range(causet.n):
        groups[
            (
                causet.rank[i],
                causet.past_left[i],
                causet.past_right[i],
                frozenset(ch_l[i]),
                frozenset(ch_r[i]),
            )
        ].append(i)
    for xs in groups.values():
        xs.sort()
    return tuple(tuple(xs) for xs in sorted(groups.values(), key=lambda a: a[0]))


def twin_quotient_conway(causet: ConwayCauset) -> ConwayCauset:
    """Identify twins by ``(rank, P_L, P_R)`` and return quotient causet."""
    cls = twin_classes_conway(causet)
    idx = [0] * causet.n
    for q, members in enumerate(cls):
        for i in members:
            idx[i] = q
    rank: list[int] = []
    left: list[frozenset[int]] = []
    right: list[frozenset[int]] = []
    for members in cls:
        rep = members[0]
        rank.append(causet.rank[rep])
        left.append(frozenset(idx[p] for p in causet.past_left[rep]))
        right.append(frozenset(idx[p] for p in causet.past_right[rep]))
    return ConwayCauset(
        n=len(cls),
        rank=tuple(rank),
        past_left=tuple(left),
        past_right=tuple(right),
    )


def outcome_quotient_conway(causet: ConwayCauset) -> tuple[ConwayCauset, tuple[int, ...]]:
    """Quotient by identical ``(rank, shallow_outcome)`` classes."""
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for i in range(causet.n):
        groups[(causet.rank[i], shallow_outcome(i, causet))].append(i)
    classes = sorted(groups.values(), key=lambda x: x[0])
    idx = [0] * causet.n
    for q, members in enumerate(classes):
        for i in members:
            idx[i] = q
    rank: list[int] = []
    left: list[frozenset[int]] = []
    right: list[frozenset[int]] = []
    for members in classes:
        rep = members[0]
        rank.append(causet.rank[rep])
        left.append(frozenset(idx[p] for p in causet.past_left[rep]))
        right.append(frozenset(idx[p] for p in causet.past_right[rep]))
    sizes = tuple(sorted((len(m) for m in classes), reverse=True))
    out = ConwayCauset(
        n=len(classes),
        rank=tuple(rank),
        past_left=tuple(left),
        past_right=tuple(right),
    )
    return out, sizes


def _logf(n: int) -> float:
    return 0.0 if n <= 1 else float(math.lgamma(float(n) + 1.0))


def entropy_decomposition_conway(
    causet: ConwayCauset, *, backend: str = "auto"
) -> ConwayEntropyDecomposition:
    """Compute full decomposition across pos, twin, outcome, and rank levels."""
    o_pos = aut_order_conway(causet, backend=backend)
    twin_q = twin_quotient_conway(causet)
    o_twin = aut_order_conway(twin_q, backend=backend)
    out_q, out_sizes = outcome_quotient_conway(twin_q)
    o_out = aut_order_conway(out_q, backend=backend)

    by_rank = Counter(causet.rank)
    rank_prod = 1
    log_rank = 0.0
    for nr in by_rank.values():
        rank_prod *= math.factorial(nr)
        log_rank += _logf(nr)

    n = causet.n
    log_n = _logf(n)
    log_pos = 0.0 if o_pos <= 1 else math.log(float(o_pos))
    log_twin = 0.0 if o_twin <= 1 else math.log(float(o_twin))
    log_out = 0.0 if o_out <= 1 else math.log(float(o_out))
    s_a_pos = log_n - log_pos
    s_a_twin = log_n - log_twin
    s_a_out = log_n - log_out
    s_a_rank = log_n - log_rank
    twin_sizes = tuple(sorted((len(t) for t in twin_classes_conway(causet)), reverse=True))
    return ConwayEntropyDecomposition(
        aut_order_pos=o_pos,
        aut_order_twin=o_twin,
        aut_order_outcome=o_out,
        rank_factorial_product=rank_prod,
        log_aut_pos=log_pos,
        log_aut_twin=log_twin,
        log_aut_outcome=log_out,
        log_rank_factorial=log_rank,
        log_n_factorial=log_n,
        s_a_pos=s_a_pos,
        s_a_twin=s_a_twin,
        s_a_outcome=s_a_out,
        s_a_rank=s_a_rank,
        delta_max_pos=log_rank - log_pos,
        delta_pos_twin=log_pos - log_twin,
        delta_twin_outcome=log_twin - log_out,
        twin_class_sizes=twin_sizes,
        outcome_class_sizes=out_sizes,
        n=n,
    )
