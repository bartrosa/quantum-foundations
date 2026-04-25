"""Stratified game causets, builders, structural fingerprints, and test fixtures.

A :class:`GameCauset` stores the Hasse diagram of the cover relation: each event has a
non-negative *rank* and a set of *immediate* predecessors of strictly lower rank (no
transitive-closure table). Callers and experiments use this representation together with
:mod:`automorphism` and :mod:`entropy`.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StructuralDiff:
    """Labeled-structural comparison between two causets (E3 and debugging).

    This is a fingerprint, not graph isomorphism. If two labeled causets match on
    ``n``, rank tuple, and Hasse edge set, they are the same object as stored.

    Attributes:
        n_diff: ``abs(n1 - n2)``; zero iff vertex counts match.
        edges_diff: ``|E1 △ E2|`` (symmetric difference size) for Hasse edges.
        node_set_equal: True iff both have vertices ``0..n-1`` with the same *n*.
        rank_distribution_equal: True iff the multiset of ranks (sorted tuple) matches.
    """

    n_diff: int
    edges_diff: int
    node_set_equal: bool
    rank_distribution_equal: bool


@dataclass(frozen=True, slots=True)
class GameCauset:
    """Hasse (cover) representation: one rank and one set of past covers per event.

    Attributes:
        n: Number of events (vertices).
        rank: ``rank[i]`` is the stratum of event *i* (non-negative int).
        past: For each *i*, immediate predecessors in the cover graph (all have lower rank).
    """

    n: int
    rank: tuple[int, ...]
    past: tuple[frozenset[int], ...]

    def __post_init__(self) -> None:
        if self.n != len(self.rank) or self.n != len(self.past):
            raise ValueError("n must match rank and past lengths")
        for i in range(self.n):
            for p in self.past[i]:
                if p < 0 or p >= self.n:
                    raise ValueError("past indices must be in range")
                if self.rank[p] >= self.rank[i]:
                    raise ValueError("past edges must come from lower ranks")

    def hasse_edges(self) -> tuple[tuple[int, int], ...]:
        """Return all cover edges ``(i, j)`` with ``i ≺· j`` in sorted order per child."""
        edges: list[tuple[int, int]] = []
        for child, preds in enumerate(self.past):
            for pred in sorted(preds):
                edges.append((pred, child))
        return tuple(edges)

    def ancestors(self, i: int) -> frozenset[int]:
        """All strict ancestors of *i* (transitive predecessors)."""
        seen: set[int] = set()
        queue = deque(self.past[i])
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            queue.extend(self.past[node])
        return frozenset(seen)

    def descendants(self, i: int) -> frozenset[int]:
        """All strict descendants of *i* (transitive successors)."""
        children: list[list[int]] = [[] for _ in range(self.n)]
        for child, preds in enumerate(self.past):
            for pred in preds:
                children[pred].append(child)

        seen: set[int] = set()
        queue = deque(children[i])
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            queue.extend(children[node])
        return frozenset(seen)

    def is_chain(self) -> bool:
        """True iff the causet is a total order (one chain on all events)."""
        return all(
            len(self.ancestors(i)) + len(self.descendants(i)) == self.n - 1 for i in range(self.n)
        )


def _as_rng(seed: int | np.random.Generator) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _seed_repr(seed: int | np.random.Generator) -> str:
    if isinstance(seed, np.random.Generator):
        return "<Generator>"
    return str(seed)


def build_stratified_game_causet(
    *,
    n_target: int,
    max_rank: int,
    sample_rate: float,
    seed: int | np.random.Generator,
) -> GameCauset:
    """Build a random stratified game causet rank-by-rank with seeded RNG.

    Each new vertex gets a random rank in ``1..max_rank`` (or 0 for the first and when
    ``max_rank == 0``). Past is sampled from prior vertices of lower rank with a bounded
    support size and *sample_rate* gating, then falls back to a single minimal edge when
    the sample is rejected.

    Args:
        n_target: Target number of events ``n`` (the builder always returns exactly this).
        max_rank: Maximum rank (inclusive) for non-root draws.
        sample_rate: Probability to take the multi-predecessor branch before fallback.
        seed: ``int`` or ``numpy.random.Generator`` for reproducibility.

    Returns:
        A valid :class:`GameCauset` with ``n == n_target``.

    Raises:
        ValueError: If ``n_target``, ``max_rank``, or ``sample_rate`` is out of range.
    """
    if n_target < 0:
        raise ValueError("n_target must be non-negative")
    if max_rank < 0:
        raise ValueError("max_rank must be non-negative")
    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError("sample_rate must be in [0,1]")

    logger.debug(
        "build_stratified | n_target=%d max_rank=%d sample_rate=%.3f seed=%s",
        n_target,
        max_rank,
        sample_rate,
        _seed_repr(seed),
    )
    rng = _as_rng(seed)
    ranks: list[int] = []
    past: list[frozenset[int]] = []

    for node in range(n_target):
        if node == 0 or max_rank == 0:
            rank = 0
        else:
            rank = int(rng.integers(1, max_rank + 1))
        ranks.append(rank)

        candidates = [idx for idx in range(node) if ranks[idx] < rank]
        if rank == 0 or not candidates:
            past.append(frozenset())
            continue

        max_pick = min(3, len(candidates))
        support_size = int(rng.integers(1, max_pick + 1))
        if rng.random() > sample_rate:
            past.append(frozenset({candidates[0]}))
            continue

        choice = rng.choice(
            np.array(candidates, dtype=np.int64),
            size=support_size,
            replace=False,
        )
        past.append(frozenset(int(x) for x in choice.tolist()))

    causet = GameCauset(n=n_target, rank=tuple(ranks), past=tuple(past))
    n_edges = sum(len(p) for p in causet.past)
    max_rank_actual = max(causet.rank) if causet.rank else 0
    logger.debug(
        "built causet | n=%d max_rank_actual=%d edges=%d",
        causet.n,
        max_rank_actual,
        n_edges,
    )
    return causet


def disjoint_union(c1: GameCauset, c2: GameCauset) -> GameCauset:
    """Disjoint union of two causets, relabeling *c2*'s indices by ``+ n1``."""
    logger.debug("disjoint_union | n1=%d n2=%d → n=%d", c1.n, c2.n, c1.n + c2.n)
    offset = c1.n
    rank = c1.rank + c2.rank
    shifted = tuple(frozenset(p + offset for p in preds) for preds in c2.past)
    return GameCauset(n=c1.n + c2.n, rank=rank, past=c1.past + shifted)


def structural_diff(c1: GameCauset, c2: GameCauset) -> StructuralDiff:
    """Compare two labeled causets: vertex count, rank multiset, and Hasse edge set.

    If ``n`` differs, we still report an edge-symmetric-difference after comparing
    truncated/extended structures is meaningless; *edges_diff* is then the sum of
    sizes when viewed as set mismatch.

    Returns:
        A :class:`StructuralDiff` summarizing how far the two labeled DAGs are from
        identity on the same vertex set.

    Note:
        Two isomorphic but differently labeled graphs can differ on every field; E3
        uses this only to catch **bit-identical** placements (same *n*, *rank* tuple,
        and edge set) before comparing ``|Aut|``.
    """
    n_diff = abs(c1.n - c2.n)
    e1, e2 = set(c1.hasse_edges()), set(c2.hasse_edges())
    rank_distribution_equal: bool = sorted(c1.rank) == sorted(c2.rank) if c1.n == c2.n else False
    node_set_equal = c1.n == c2.n and c1.n > 0
    edges_diff = len(e1.symmetric_difference(e2))
    return StructuralDiff(
        n_diff=n_diff,
        edges_diff=edges_diff,
        node_set_equal=node_set_equal,
        rank_distribution_equal=rank_distribution_equal,
    )


def _immediate_child_set(causet: GameCauset, i: int) -> frozenset[int]:
    return frozenset(j for j in range(causet.n) if i in causet.past[j])


def twin_classes(causet: GameCauset) -> tuple[tuple[int, ...], ...]:
    """Partition events into twin classes.

    Two events are *twins* when they are indistinguishable in the 1-skeleton of
    the Hasse diagram: the same stratum, the same set of cover predecessors,
    and the same set of cover successors. This is the **coarsest** partition
    on which a within-class label permutation (when permitted by
    :func:`aut_order`) is graph-theoretically plausible, and is strictly finer
    than the relation “same *rank* and *past*” when those vertices have
    different children (where the twin-orbit product formula for ``|Aut|``
    can fail for the coarser relation).

    Classes are returned as sorted tuples of vertex indices; the outer tuple
    is ordered by the smallest index in each class; singletons are included.

    Args:
        causet: The input causet.

    Returns:
        Tuple of twin classes, each a sorted tuple of event indices.
    """
    groups: dict[tuple[int, frozenset[int], frozenset[int]], list[int]] = defaultdict(list)
    for i in range(causet.n):
        key = (causet.rank[i], causet.past[i], _immediate_child_set(causet, i))
        groups[key].append(i)
    for members in groups.values():
        members.sort()
    classes = sorted(groups.values(), key=lambda member: member[0])
    return tuple(tuple(member) for member in classes)


def twin_quotient(causet: GameCauset) -> GameCauset:
    """Identify twin events: one node per class of cover-indistinguishable events.

    Rank is preserved per class. A past edge ``p → t`` in the quotient exists if
    some original edge ``p' → t'`` has ``p'`` in the class of *p* and ``t'`` in
    the class of *t* (``twin_classes`` is well-defined as in that routine).

    Args:
        causet: The input causet.

    Returns:
        Quotient causet whose nodes are ``twin_classes`` of *causet*.
    """
    if causet.n == 0:
        return GameCauset(n=0, rank=tuple(), past=tuple())
    classes = twin_classes(causet)
    n_q = len(classes)
    old_to_new: list[int] = [0] * causet.n
    for new_id, cls in enumerate(classes):
        for v in cls:
            old_to_new[v] = new_id
    new_rank: list[int] = []
    new_past: list[frozenset[int]] = []
    for cls in classes:
        rep = cls[0]
        new_rank.append(causet.rank[rep])
        new_past.append(frozenset(old_to_new[p] for p in causet.past[rep]))
    return GameCauset(n=n_q, rank=tuple(new_rank), past=tuple(new_past))


def make_antichain(n: int) -> GameCauset:
    """Return *n* minimal elements with no relations (all rank 0, empty pasts)."""
    return GameCauset(
        n=n,
        rank=tuple(0 for _ in range(n)),
        past=tuple(frozenset() for _ in range(n)),
    )


def make_chain(n: int) -> GameCauset:
    """Return a single chain: rank ``0..n-1`` and each ``i>0`` covers ``i-1`` only."""
    if n == 0:
        return GameCauset(n=0, rank=tuple(), past=tuple())
    rank = tuple(range(n))
    past = tuple(frozenset({i - 1}) if i > 0 else frozenset() for i in range(n))
    return GameCauset(n=n, rank=rank, past=past)


def make_diamond() -> GameCauset:
    """Classic four-vertex ``N``/diamond (01 / \\02 /) Hasse poset for tests."""
    return GameCauset(
        n=4,
        rank=(0, 1, 1, 2),
        past=(frozenset(), frozenset({0}), frozenset({0}), frozenset({1, 2})),
    )
