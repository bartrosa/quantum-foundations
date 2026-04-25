from __future__ import annotations

from quantum_foundations.conway_games.causet import ConwayCauset, make_chain_conway
from quantum_foundations.conway_games.outcome import recursive_outcome, shallow_outcome


def test_empty_like_zero() -> None:
    c = ConwayCauset(n=1, rank=(0,), past_left=(frozenset(),), past_right=(frozenset(),))
    assert shallow_outcome(0, c) == "="


def test_chain_l_outcome() -> None:
    c = make_chain_conway(4, color="L")
    assert shallow_outcome(3, c) == "L"


def test_fuzzy_star() -> None:
    c = ConwayCauset(
        n=2,
        rank=(0, 1),
        past_left=(frozenset(), frozenset({0})),
        past_right=(frozenset(), frozenset({0})),
    )
    assert recursive_outcome(1, c) == "||"


def test_depth_zero_falls_back_shallow() -> None:
    c = make_chain_conway(3, color="L")
    assert recursive_outcome(2, c, max_depth=0) == shallow_outcome(2, c)
