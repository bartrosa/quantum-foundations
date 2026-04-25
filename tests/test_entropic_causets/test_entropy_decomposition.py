"""Tests for entropy decomposition, twin quotient, and S_A Gibbs identities."""
from __future__ import annotations

import math
import random
from collections import defaultdict

import pytest

from quantum_foundations.entropic_causets.automorphism import aut_order
from quantum_foundations.entropic_causets.causet import (
    GameCauset,
    build_stratified_game_causet,
    disjoint_union,
    make_antichain,
    twin_classes,
    twin_quotient,
)
from quantum_foundations.entropic_causets.entropy import entropy_decomposition


def _twin_classes_bruteforce(causet: GameCauset) -> tuple[tuple[int, ...], ...]:
    """O(n²) merge: same class iff same (rank, past). For N ≤ 8 cross-check."""
    if causet.n == 0:
        return tuple()
    parent = list(range(causet.n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(causet.n):
        for j in range(i + 1, causet.n):
            if causet.rank[i] != causet.rank[j] or causet.past[i] != causet.past[j]:
                continue
            chi = frozenset(k for k in range(causet.n) if i in causet.past[k])
            chj = frozenset(k for k in range(causet.n) if j in causet.past[k])
            if chi == chj:
                union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(causet.n):
        groups[find(i)].append(i)
    for vs in groups.values():
        vs.sort()
    classes = sorted(groups.values(), key=lambda t: t[0])
    return tuple(tuple(v) for v in classes)


def _ln_factorial_int(n: int) -> float:
    if n <= 1:
        return 0.0
    return float(math.lgamma(float(n) + 1.0))


def test_twin_factorization_lemma_random() -> None:
    rng = random.Random(42)
    for _ in range(20):
        n = rng.randint(1, 30)
        seed = rng.randint(0, 10_000)
        c = build_stratified_game_causet(n_target=n, max_rank=10, sample_rate=0.25, seed=seed)
        q = twin_quotient(c)
        sizes = [len(t) for t in twin_classes(c)]
        prod_k = 1
        for k in sizes:
            for t in range(2, k + 1):
                prod_k *= t
        lhs = aut_order(c)
        rhs = aut_order(q) * prod_k
        assert lhs == rhs, f"n={n} seed={seed} |Aut| mismatch {lhs} != {rhs}"


def test_entropy_inequalities_random() -> None:
    rng = random.Random(7)
    for _ in range(50):
        n = rng.randint(0, 25)
        if n == 0:
            c = GameCauset(n=0, rank=tuple(), past=tuple())
        else:
            c = build_stratified_game_causet(
                n_target=n, max_rank=9, sample_rate=0.22, seed=rng.randint(0, 50000)
            )
        ed = entropy_decomposition(c)
        assert ed.log_rank_factorial + 1e-9 >= ed.log_aut_pos >= ed.log_aut_twin - 1e-9
        assert ed.log_aut_twin >= -1e-9


def test_s_a_plus_s_b_anchor() -> None:
    rng = random.Random(0)
    for _ in range(30):
        n = rng.randint(1, 22)
        c = build_stratified_game_causet(
            n_target=n, max_rank=8, sample_rate=0.2, seed=rng.randrange(10**6)
        )
        ed = entropy_decomposition(c)
        assert abs(ed.s_a_pos + ed.log_aut_pos - ed.log_n_factorial) < 1e-11


def test_gibbs_s_a_isomorphic_union() -> None:
    rng = random.Random(123)
    for n in (10, 15, 20):
        for _ in range(5):
            c = build_stratified_game_causet(
                n_target=n, max_rank=8, sample_rate=0.25, seed=rng.randrange(10**6)
            )
            u = disjoint_union(c, c)
            edc = entropy_decomposition(c)
            edu = entropy_decomposition(u)
            extra = edu.s_a_pos - 2 * edc.s_a_pos
            expected = float(
                math.lgamma(float(2 * n) + 1.0)
                - 2.0 * math.lgamma(float(n) + 1.0)
                - math.log(2.0)
            )
            assert abs(extra - expected) < 1e-9


def test_gibbs_s_a_nonisomorphic_union() -> None:
    rng = random.Random(99)
    for _ in range(5):
        n1 = rng.randint(5, 14)
        n2 = rng.randint(5, 14)
        c1 = build_stratified_game_causet(
            n_target=n1, max_rank=8, sample_rate=0.25, seed=rng.randrange(10**6)
        )
        c2 = build_stratified_game_causet(
            n_target=n2, max_rank=8, sample_rate=0.25, seed=rng.randrange(10**6)
        )
        u = disjoint_union(c1, c2)
        e1, e2, eu = entropy_decomposition(c1), entropy_decomposition(c2), entropy_decomposition(u)
        extra = eu.s_a_pos - e1.s_a_pos - e2.s_a_pos
        expected = float(
            math.lgamma(float(n1 + n2) + 1.0)
            - math.lgamma(float(n1) + 1.0)
            - math.lgamma(float(n2) + 1.0)
        )
        assert abs(extra - expected) < 1e-9


def test_singleton_causet() -> None:
    c = make_antichain(1)
    ed = entropy_decomposition(c)
    assert ed.n == 1
    assert ed.log_aut_pos == ed.log_aut_twin == ed.log_rank_factorial == 0.0
    assert ed.s_a_pos == pytest.approx(0.0)


def test_empty_causet() -> None:
    c = GameCauset(n=0, rank=tuple(), past=tuple())
    ed = entropy_decomposition(c)
    assert ed.n == 0
    assert ed.aut_order_pos == ed.aut_order_twin == 1
    assert ed.log_n_factorial == 0.0
    assert ed.s_a_pos == ed.s_a_twin == ed.s_a_rank == 0.0


def test_antichain_five_twin_quotient() -> None:
    c = make_antichain(5)
    q = twin_quotient(c)
    assert q.n == 1
    assert aut_order(c) == math.factorial(5)
    assert aut_order(q) == 1
    ed = entropy_decomposition(c)
    assert ed.aut_order_pos == 120
    assert ed.aut_order_twin == 1
    assert ed.delta_pos_twin == pytest.approx(_ln_factorial_int(5))


def test_twin_classes_match_bruteforce_small() -> None:
    rng = random.Random(3)
    for n in range(1, 9):
        for _ in range(15):
            c = build_stratified_game_causet(
                n_target=n, max_rank=5, sample_rate=0.3, seed=rng.randrange(10**5)
            )
            assert twin_classes(c) == _twin_classes_bruteforce(c)


def test_delta_pos_twin_equals_log_twin_class_factorials() -> None:
    rng = random.Random(11)
    for _ in range(15):
        n = rng.randint(2, 22)
        c = build_stratified_game_causet(
            n_target=n, max_rank=9, sample_rate=0.25, seed=rng.randrange(10**6)
        )
        ed = entropy_decomposition(c)
        expected = sum(float(math.lgamma(float(k) + 1.0)) for k in ed.twin_class_sizes)
        assert abs(ed.delta_pos_twin - expected) < 1e-8
