from __future__ import annotations

import pytest

from quantum_foundations.conway_games.automorphism import aut_order_conway, pynauty_available_conway
from quantum_foundations.conway_games.causet import (
    ConwayCauset,
    build_random_conway_causet,
    make_chain_conway,
)
from quantum_foundations.entropic_causets.automorphism import aut_order
from quantum_foundations.entropic_causets.causet import GameCauset


def _to_legacy(c: ConwayCauset) -> GameCauset:
    return GameCauset(n=c.n, rank=c.rank, past=tuple(c.past_left))


def test_empty_aut() -> None:
    c = ConwayCauset(n=0, rank=tuple(), past_left=tuple(), past_right=tuple())
    assert aut_order_conway(c) == 1


def test_chain_aut_one() -> None:
    assert aut_order_conway(make_chain_conway(3, color="L")) == 1


def test_crosscheck_backends_small() -> None:
    if not pynauty_available_conway():
        pytest.skip("pynauty missing")
    for seed in range(20):
        c = build_random_conway_causet(n_target=12, max_rank=6, seed=seed)
        assert aut_order_conway(c, backend="networkx") == aut_order_conway(c, backend="pynauty")


def test_backward_compat_when_right_empty() -> None:
    for seed in range(10):
        c = build_random_conway_causet(n_target=14, max_rank=7, sample_rate_right=0.0, seed=seed)
        got = aut_order_conway(c, backend="networkx")
        expect = aut_order(_to_legacy(c), backend="networkx")
        assert got == expect
