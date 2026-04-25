from __future__ import annotations

from quantum_foundations.conway_games.causet import (
    ConwayCauset,
    build_random_conway_causet,
    disjoint_union_conway,
    make_diamond_conway,
)


def test_builder_deterministic() -> None:
    c1 = build_random_conway_causet(n_target=20, max_rank=8, seed=123)
    c2 = build_random_conway_causet(n_target=20, max_rank=8, seed=123)
    assert c1 == c2


def test_options_lower_rank() -> None:
    c = build_random_conway_causet(n_target=25, max_rank=10, seed=7)
    assert c.is_well_formed()


def test_disjoint_union_size() -> None:
    c1 = build_random_conway_causet(n_target=10, max_rank=6, seed=1)
    c2 = build_random_conway_causet(n_target=9, max_rank=6, seed=2)
    u = disjoint_union_conway(c1, c2)
    assert u.n == 19


def test_make_diamond() -> None:
    d = make_diamond_conway()
    assert d.n == 4
    assert d.rank == (0, 1, 1, 2)


def test_manual_conway_well_formed() -> None:
    c = ConwayCauset(
        n=2,
        rank=(0, 1),
        past_left=(frozenset(), frozenset({0})),
        past_right=(frozenset(), frozenset()),
    )
    assert c.is_well_formed()
