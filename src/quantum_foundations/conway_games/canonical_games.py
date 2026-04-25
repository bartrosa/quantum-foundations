"""Deterministic Conway game constructors.

Builds canonical short Conway games (numeric, partisan, switches, nimbers)
as ConwayCauset instances. Used for sanity-checking the framework against
known game values from CGT literature, and as inputs to comparative generator
experiments.

Note on automorphism: ConwayCauset's ``aut_order_conway`` preserves rank
and edge color. For canonical short games encoded as game trees, ranks
typically uniquely identify events, so ``|Aut|`` is often 1 even for
abstractly symmetric games (e.g. ``*k``). This is a feature of game-tree
encoding, not a bug.
"""

from __future__ import annotations

from quantum_foundations.conway_games.causet import ConwayCauset

__all__ = [
    "make_zero",
    "make_star",
    "make_nimber",
    "make_integer",
    "make_negative_integer",
    "make_up",
    "make_down",
    "make_switch",
    "make_balanced_binary_tree",
]


def make_zero() -> ConwayCauset:
    """Game value ``0 = {|}``."""
    return ConwayCauset(
        n=1,
        rank=(0,),
        past_left=(frozenset(),),
        past_right=(frozenset(),),
    )


def make_star() -> ConwayCauset:
    """Game value ``* = {0|0}`` (two events: ``0`` then ``*`` with LR to ``0``)."""
    return ConwayCauset(
        n=2,
        rank=(0, 1),
        past_left=(frozenset(), frozenset({0})),
        past_right=(frozenset(), frozenset({0})),
    )


def make_nimber(k: int) -> ConwayCauset:
    """Game value ``*k`` as a chain: event ``j`` has ``P_L = P_R = {0,…,j-1}``.

    Total events ``k + 1``. Ranks are ``0,…,k`` so ``|Aut|`` is typically 1
    for this encoding (ranks distinguish events).
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    ranks = tuple(range(k + 1))
    left: list[frozenset[int]] = []
    right: list[frozenset[int]] = []
    for j in range(k + 1):
        if j == 0:
            left.append(frozenset())
            right.append(frozenset())
        else:
            opts = frozenset(range(j))
            left.append(opts)
            right.append(opts)
    return ConwayCauset(n=k + 1, rank=ranks, past_left=tuple(left), past_right=tuple(right))


def make_integer(n: int) -> ConwayCauset:
    """Game value ``n = {n-1|}`` for ``n >= 0`` (``n=0`` is ``make_zero()``)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return make_zero()
    ranks = tuple(range(n + 1))
    left: list[frozenset[int]] = []
    right: list[frozenset[int]] = []
    for j in range(n + 1):
        if j == 0:
            left.append(frozenset())
            right.append(frozenset())
        else:
            left.append(frozenset({j - 1}))
            right.append(frozenset())
    return ConwayCauset(n=n + 1, rank=ranks, past_left=tuple(left), past_right=tuple(right))


def make_negative_integer(n: int) -> ConwayCauset:
    """Game value ``-n = {| -(n-1)}`` for ``n >= 1``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    ranks = tuple(range(n + 1))
    left: list[frozenset[int]] = []
    right: list[frozenset[int]] = []
    for j in range(n + 1):
        if j == 0:
            left.append(frozenset())
            right.append(frozenset())
        else:
            left.append(frozenset())
            right.append(frozenset({j - 1}))
    return ConwayCauset(n=n + 1, rank=ranks, past_left=tuple(left), past_right=tuple(right))


def make_up() -> ConwayCauset:
    """Game value ``↑ = {0|*}`` (three events: ``0``, ``*``, ``↑``)."""
    return ConwayCauset(
        n=3,
        rank=(0, 1, 2),
        past_left=(frozenset(), frozenset({0}), frozenset({0})),
        past_right=(frozenset(), frozenset({0}), frozenset({1})),
    )


def make_down() -> ConwayCauset:
    """Game value ``↓ = {*|0}`` (three events: ``0``, ``*``, ``↓``)."""
    return ConwayCauset(
        n=3,
        rank=(0, 1, 2),
        past_left=(frozenset(), frozenset({0}), frozenset({1})),
        past_right=(frozenset(), frozenset({0}), frozenset({0})),
    )


def make_switch(a: int, b: int = 0) -> ConwayCauset:
    """Game value ``{a|b}`` for this iteration: only ``b == 0`` and ``a > 0``.

    Builds the integer chain ``0, 1, …, a`` then one top event at rank ``a + 1``
    with ``P_L = {a}``, ``P_R = {0}``.
    """
    if b != 0:
        raise ValueError("make_switch only supports b=0 in this version")
    if a <= 0:
        raise ValueError("a must be positive for {a|0}")
    ranks = tuple(range(a + 2))
    left: list[frozenset[int]] = []
    right: list[frozenset[int]] = []
    for j in range(a + 1):
        if j == 0:
            left.append(frozenset())
            right.append(frozenset())
        else:
            left.append(frozenset({j - 1}))
            right.append(frozenset())
    left.append(frozenset({a}))
    right.append(frozenset({0}))
    return ConwayCauset(
        n=a + 2,
        rank=ranks,
        past_left=tuple(left),
        past_right=tuple(right),
    )


def _heap_parent(i: int) -> int:
    return (i - 1) // 2


def make_balanced_binary_tree(depth: int) -> ConwayCauset:
    """Full balanced binary tree of given ``depth`` (root depth 0).

    ``depth == 0``: single root (same as ``make_zero()``).
    Otherwise ``n = 2**(depth + 1) - 1`` nodes in heap order: left child
    ``2p+1`` (L-edge from ``p``), right child ``2p+2`` (R-edge from ``p``).
    """
    if depth < 0:
        raise ValueError("depth must be non-negative")
    if depth == 0:
        return make_zero()
    n = (1 << (depth + 1)) - 1
    ranks = tuple(range(n))
    left: list[frozenset[int]] = [frozenset() for _ in range(n)]
    right: list[frozenset[int]] = [frozenset() for _ in range(n)]
    for i in range(1, n):
        p = _heap_parent(i)
        if i == 2 * p + 1:
            left[i] = frozenset({p})
        else:
            right[i] = frozenset({p})
    return ConwayCauset(n=n, rank=ranks, past_left=tuple(left), past_right=tuple(right))


def make_nimber_chain(k: int) -> ConwayCauset:
    """Alias for ``make_nimber(k)`` (recursive nimber tower in tree encoding)."""
    return make_nimber(k)


def make_dense_chain(n: int) -> ConwayCauset:
    """Alias for ``make_nimber(n)`` (dense LR chain over all lower events)."""
    return make_nimber(n)


if __name__ == "__main__":
    from quantum_foundations.conway_games.automorphism import aut_order_conway

    cases: list[tuple[str, ConwayCauset]] = [
        ("zero", make_zero()),
        ("star", make_star()),
        ("nimber(2)", make_nimber(2)),
        ("nimber(3)", make_nimber(3)),
        ("integer(5)", make_integer(5)),
        ("up", make_up()),
        ("down", make_down()),
        ("switch(1,0)", make_switch(1, 0)),
        ("switch(3,0)", make_switch(3, 0)),
        ("tree(2)", make_balanced_binary_tree(2)),
        ("tree(3)", make_balanced_binary_tree(3)),
    ]

    print(f"{'name':20s} {'n':>4s} {'|Aut|':>6s}")
    for name, c in cases:
        print(f"{name:20s} {c.n:>4d} {aut_order_conway(c):>6d}")
