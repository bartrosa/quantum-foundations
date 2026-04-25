from __future__ import annotations

from quantum_foundations.entropic_causets.causet import (
    build_stratified_game_causet,
    disjoint_union,
)


def test_build_is_deterministic_for_fixed_seed() -> None:
    c1 = build_stratified_game_causet(n_target=20, max_rank=7, sample_rate=0.2, seed=1234)
    c2 = build_stratified_game_causet(n_target=20, max_rank=7, sample_rate=0.2, seed=1234)
    assert c1 == c2


def test_past_is_well_formed() -> None:
    causet = build_stratified_game_causet(n_target=25, max_rank=8, sample_rate=0.3, seed=123)
    for i in range(causet.n):
        for p in causet.past[i]:
            assert causet.rank[p] < causet.rank[i]


def test_disjoint_union_counts_and_no_cross_edges() -> None:
    c1 = build_stratified_game_causet(n_target=10, max_rank=5, sample_rate=0.2, seed=3)
    c2 = build_stratified_game_causet(n_target=12, max_rank=5, sample_rate=0.2, seed=4)
    merged = disjoint_union(c1, c2)
    assert merged.n == c1.n + c2.n
    for i in range(c1.n, merged.n):
        assert all(p >= c1.n for p in merged.past[i])
