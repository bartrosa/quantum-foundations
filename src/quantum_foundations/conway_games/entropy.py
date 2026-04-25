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
    """Entropy decomposition for Conway causet across coarse-graining levels.

    Twin equivalence is reported at two levels:
    - canonical: past-only (rank, P_L, P_R), tracked as an inequality level.
    - local: full neighborhood (rank, P_L, P_R, child_L, child_R), exact lemma level.
    """

    aut_order_pos: int
    aut_order_twin_canonical: int
    aut_order_twin_local: int
    aut_order_outcome: int
    rank_factorial_product: int
    log_aut_pos: float
    log_aut_twin_canonical: float
    log_aut_twin_local: float
    log_aut_outcome: float
    log_rank_factorial: float
    log_n_factorial: float
    s_a_pos: float
    s_a_twin_canonical: float
    s_a_twin_local: float
    s_a_outcome: float
    s_a_rank: float
    delta_max_pos: float
    delta_pos_twin_canonical: float
    delta_pos_twin_local: float
    delta_canonical_local: float
    delta_local_outcome: float
    twin_canonical_class_sizes: tuple[int, ...]
    twin_local_class_sizes: tuple[int, ...]
    outcome_class_sizes: tuple[int, ...]
    n: int


def twin_classes_local(causet: ConwayCauset) -> tuple[tuple[int, ...], ...]:
    """Partition events by full local colored neighborhood equivalence.

    Two events are local-twins iff (rank, P_L, P_R, child_L, child_R) match.
    This is a strict refinement of canonical twin equivalence (which uses
    only past structure). Local twins are guaranteed swappable in any
    Aut(C); the corresponding twin lemma holds with exact equality.
    """
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


def twin_classes_canonical(causet: ConwayCauset) -> tuple[tuple[int, ...], ...]:
    """Partition events by canonical (past-only) twin equivalence.

    Two events are canonical-twins iff (rank, P_L, P_R) match.
    """
    groups: dict[tuple[int, frozenset[int], frozenset[int]], list[int]] = defaultdict(list)
    for i in range(causet.n):
        groups[(causet.rank[i], causet.past_left[i], causet.past_right[i])].append(i)
    classes = sorted(groups.values(), key=lambda x: x[0])
    return tuple(tuple(c) for c in classes)


def _quotient_from_classes(
    causet: ConwayCauset, classes: tuple[tuple[int, ...], ...]
) -> tuple[ConwayCauset, tuple[int, ...]]:
    """Build quotient ConwayCauset from a partition of events."""
    cls = classes
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
    ), tuple(sorted((len(m) for m in cls), reverse=True))


def twin_quotient_local(causet: ConwayCauset) -> tuple[ConwayCauset, tuple[int, ...]]:
    """Quotient causet by local twin equivalence."""
    return _quotient_from_classes(causet, twin_classes_local(causet))


def twin_quotient_canonical(causet: ConwayCauset) -> tuple[ConwayCauset, tuple[int, ...]]:
    """Quotient causet by canonical (past-only) twin equivalence."""
    return _quotient_from_classes(causet, twin_classes_canonical(causet))


def twin_quotient_conway(causet: ConwayCauset) -> ConwayCauset:
    """Backward-compatible alias: local twin quotient causet only."""
    q, _sizes = twin_quotient_local(causet)
    return q


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
    twin_canonical_q, twin_canonical_sizes = twin_quotient_canonical(causet)
    o_twin_canonical = aut_order_conway(twin_canonical_q, backend=backend)
    twin_local_q, twin_local_sizes = twin_quotient_local(causet)
    o_twin_local = aut_order_conway(twin_local_q, backend=backend)
    out_q, out_sizes = outcome_quotient_conway(twin_local_q)
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
    log_twin_canonical = 0.0 if o_twin_canonical <= 1 else math.log(float(o_twin_canonical))
    log_twin_local = 0.0 if o_twin_local <= 1 else math.log(float(o_twin_local))
    log_out = 0.0 if o_out <= 1 else math.log(float(o_out))
    s_a_pos = log_n - log_pos
    s_a_twin_canonical = log_n - log_twin_canonical
    s_a_twin_local = log_n - log_twin_local
    s_a_out = log_n - log_out
    s_a_rank = log_n - log_rank
    return ConwayEntropyDecomposition(
        aut_order_pos=o_pos,
        aut_order_twin_canonical=o_twin_canonical,
        aut_order_twin_local=o_twin_local,
        aut_order_outcome=o_out,
        rank_factorial_product=rank_prod,
        log_aut_pos=log_pos,
        log_aut_twin_canonical=log_twin_canonical,
        log_aut_twin_local=log_twin_local,
        log_aut_outcome=log_out,
        log_rank_factorial=log_rank,
        log_n_factorial=log_n,
        s_a_pos=s_a_pos,
        s_a_twin_canonical=s_a_twin_canonical,
        s_a_twin_local=s_a_twin_local,
        s_a_outcome=s_a_out,
        s_a_rank=s_a_rank,
        delta_max_pos=log_rank - log_pos,
        delta_pos_twin_canonical=log_pos - log_twin_canonical,
        delta_pos_twin_local=log_pos - log_twin_local,
        delta_canonical_local=log_twin_canonical - log_twin_local,
        delta_local_outcome=log_twin_local - log_out,
        twin_canonical_class_sizes=twin_canonical_sizes,
        twin_local_class_sizes=twin_local_sizes,
        outcome_class_sizes=out_sizes,
        n=n,
    )
