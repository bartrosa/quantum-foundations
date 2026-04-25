from __future__ import annotations

import pytest

from quantum_foundations.entropic_causets.automorphism import aut_order, pynauty_available
from quantum_foundations.entropic_causets.causet import (
    GameCauset,
    build_stratified_game_causet,
    make_antichain,
    make_chain,
    make_diamond,
)


def test_empty_causet_has_unit_automorphism_group() -> None:
    causet = GameCauset(n=0, rank=tuple(), past=tuple())
    assert aut_order(causet, backend="networkx") == 1


def test_chain_is_rigid() -> None:
    assert aut_order(make_chain(3), backend="networkx") == 1


def test_antichain_order_three_is_six() -> None:
    assert aut_order(make_antichain(3), backend="networkx") == 6


def test_diamond_has_two_automorphisms() -> None:
    assert aut_order(make_diamond(), backend="networkx") == 2


@pytest.mark.skipif(not pynauty_available(), reason="pynauty not installed")
@pytest.mark.parametrize("seed", list(range(20)))
def test_backends_match_on_random_small_causets(seed: int) -> None:
    # Keep n modest: NetworkX exact VF2 is worst-case heavy at n≈30.
    causet = build_stratified_game_causet(n_target=20, max_rank=8, sample_rate=0.25, seed=seed)
    assert aut_order(causet, backend="networkx") == aut_order(causet, backend="pynauty")
