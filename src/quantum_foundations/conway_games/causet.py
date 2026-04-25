"""Conway causets: stratified events with left/right option sets."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConwayCauset:
    """Stratified Conway causet with colored Hasse edges from L/R options.

    Attributes:
        n: Number of events.
        rank: ``rank[i]`` is rank of event ``i``.
        past_left: ``P_L(i)`` for each event.
        past_right: ``P_R(i)`` for each event.
    """

    n: int
    rank: tuple[int, ...]
    past_left: tuple[frozenset[int], ...]
    past_right: tuple[frozenset[int], ...]

    def __post_init__(self) -> None:
        if self.n != len(self.rank):
            raise ValueError("n must match rank length")
        if self.n != len(self.past_left) or self.n != len(self.past_right):
            raise ValueError("n must match past_left/past_right lengths")
        if not self.is_well_formed():
            raise ValueError("invalid ConwayCauset: options must point to strictly lower ranks")

    def is_well_formed(self) -> bool:
        """Return whether all options point to strictly lower-ranked events."""
        for i in range(self.n):
            for p in self.past_left[i] | self.past_right[i]:
                if p < 0 or p >= self.n:
                    return False
                if self.rank[p] >= self.rank[i]:
                    return False
        return True

    def past_union(self, i: int) -> frozenset[int]:
        """Return ``P_L(i) ∪ P_R(i)``."""
        return frozenset(self.past_left[i] | self.past_right[i])

    def edge_color(self, predecessor: int, successor: int) -> str | None:
        """Return edge color in ``{'L','R','LR'}`` or ``None`` if no edge."""
        in_l = predecessor in self.past_left[successor]
        in_r = predecessor in self.past_right[successor]
        if in_l and in_r:
            return "LR"
        if in_l:
            return "L"
        if in_r:
            return "R"
        return None

    def hasse_edges_with_color(self) -> tuple[tuple[int, int, str], ...]:
        """Return all colored edges ``(u, v, color)`` sorted lexicographically."""
        edges: list[tuple[int, int, str]] = []
        for v in range(self.n):
            for u in sorted(self.past_union(v)):
                color = self.edge_color(u, v)
                if color is not None:
                    edges.append((u, v, color))
        edges.sort()
        return tuple(edges)

    def ancestors(self, i: int) -> frozenset[int]:
        """Return transitive predecessors of ``i`` (color-agnostic)."""
        seen: set[int] = set()
        queue = deque(self.past_union(i))
        while queue:
            v = queue.popleft()
            if v in seen:
                continue
            seen.add(v)
            queue.extend(self.past_union(v))
        return frozenset(seen)

    def descendants(self, i: int) -> frozenset[int]:
        """Return transitive successors of ``i`` (color-agnostic)."""
        children: list[list[int]] = [[] for _ in range(self.n)]
        for v in range(self.n):
            for p in self.past_union(v):
                children[p].append(v)
        seen: set[int] = set()
        queue = deque(children[i])
        while queue:
            v = queue.popleft()
            if v in seen:
                continue
            seen.add(v)
            queue.extend(children[v])
        return frozenset(seen)


def _as_rng(seed: int | np.random.Generator) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def build_random_conway_causet(
    *,
    n_target: int,
    max_rank: int,
    sample_rate_left: float = 0.3,
    sample_rate_right: float = 0.3,
    overlap_rate: float | None = None,
    seed: int | np.random.Generator = 0,
) -> ConwayCauset:
    """Build a random short Conway causet by rank-by-rank independent sampling."""
    if n_target < 0:
        raise ValueError("n_target must be non-negative")
    if max_rank < 0:
        raise ValueError("max_rank must be non-negative")
    if not 0.0 <= sample_rate_left <= 1.0 or not 0.0 <= sample_rate_right <= 1.0:
        raise ValueError("sample rates must be in [0,1]")
    if overlap_rate is not None and not 0.0 <= overlap_rate <= 1.0:
        raise ValueError("overlap_rate must be in [0,1] or None")

    rng = _as_rng(seed)
    ranks: list[int] = []
    p_left: list[frozenset[int]] = []
    p_right: list[frozenset[int]] = []

    for i in range(n_target):
        rank_i = 0 if i == 0 or max_rank == 0 else int(rng.integers(1, max_rank + 1))
        ranks.append(rank_i)
        candidates = [j for j in range(i) if ranks[j] < rank_i]
        if rank_i == 0 or not candidates:
            p_left.append(frozenset())
            p_right.append(frozenset())
            continue

        if overlap_rate is None:
            left = {j for j in candidates if rng.random() < sample_rate_left}
            right = {j for j in candidates if rng.random() < sample_rate_right}
        else:
            base_rate = max(sample_rate_left, sample_rate_right)
            base = [j for j in candidates if rng.random() < base_rate]
            left = set()
            right = set()
            for j in base:
                if overlap_rate == 0.0:
                    if rng.random() < 0.5:
                        left.add(j)
                    else:
                        right.add(j)
                    continue
                if rng.random() < overlap_rate:
                    left.add(j)
                    right.add(j)
                elif rng.random() < 0.5:
                    left.add(j)
                else:
                    right.add(j)
        p_left.append(frozenset(left))
        p_right.append(frozenset(right))
    return ConwayCauset(
        n=n_target,
        rank=tuple(ranks),
        past_left=tuple(p_left),
        past_right=tuple(p_right),
    )


def make_chain_conway(n: int, color: str = "L") -> ConwayCauset:
    """Build chain of length ``n`` with each cover edge in selected color."""
    rank = tuple(range(n))
    left: list[frozenset[int]] = []
    right: list[frozenset[int]] = []
    for i in range(n):
        if i == 0:
            left.append(frozenset())
            right.append(frozenset())
            continue
        if color == "L":
            left.append(frozenset({i - 1}))
            right.append(frozenset())
        elif color == "R":
            left.append(frozenset())
            right.append(frozenset({i - 1}))
        else:
            left.append(frozenset({i - 1}))
            right.append(frozenset({i - 1}))
    return ConwayCauset(n=n, rank=rank, past_left=tuple(left), past_right=tuple(right))


def make_antichain_conway(n: int) -> ConwayCauset:
    """Build antichain of size ``n`` (all rank 0, no options)."""
    return ConwayCauset(
        n=n,
        rank=tuple(0 for _ in range(n)),
        past_left=tuple(frozenset() for _ in range(n)),
        past_right=tuple(frozenset() for _ in range(n)),
    )


def make_diamond_conway(color_left: str = "L", color_right: str = "R") -> ConwayCauset:
    """Build 4-node diamond with two middle nodes and colored edge options."""
    # 0 -> 1,2 -> 3 in Hasse shape.
    l0: frozenset[int] = frozenset()
    r0: frozenset[int] = frozenset()
    l1: frozenset[int]
    r1: frozenset[int]
    if color_left == "L":
        l1, r1 = frozenset({0}), frozenset()
    elif color_left == "R":
        l1, r1 = frozenset(), frozenset({0})
    else:
        l1, r1 = frozenset({0}), frozenset({0})
    l2: frozenset[int]
    r2: frozenset[int]
    if color_right == "L":
        l2, r2 = frozenset({0}), frozenset()
    elif color_right == "R":
        l2, r2 = frozenset(), frozenset({0})
    else:
        l2, r2 = frozenset({0}), frozenset({0})
    l3: frozenset[int] = frozenset({1, 2})
    r3: frozenset[int] = frozenset()
    return ConwayCauset(
        n=4,
        rank=(0, 1, 1, 2),
        past_left=(l0, l1, l2, l3),
        past_right=(r0, r1, r2, r3),
    )


def disjoint_union_conway(c1: ConwayCauset, c2: ConwayCauset) -> ConwayCauset:
    """Disjoint union with index shift on second causet."""
    off = c1.n
    rank = c1.rank + c2.rank
    l2 = tuple(frozenset(x + off for x in s) for s in c2.past_left)
    r2 = tuple(frozenset(x + off for x in s) for s in c2.past_right)
    return ConwayCauset(
        n=c1.n + c2.n,
        rank=rank,
        past_left=c1.past_left + l2,
        past_right=c1.past_right + r2,
    )
