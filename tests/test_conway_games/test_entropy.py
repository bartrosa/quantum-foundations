from __future__ import annotations

import math

from quantum_foundations.conway_games.automorphism import aut_order_conway
from quantum_foundations.conway_games.causet import build_random_conway_causet
from quantum_foundations.conway_games.entropy import (
    entropy_decomposition_conway,
    twin_quotient_conway,
)
from quantum_foundations.entropic_causets.causet import GameCauset
from quantum_foundations.entropic_causets.entropy import entropy_decomposition


def test_twin_lemma_conway_exact() -> None:
    for seed in range(5):
        c = build_random_conway_causet(n_target=10, max_rank=6, seed=seed)
        d = entropy_decomposition_conway(c, backend="networkx")
        prod = 1
        for k in d.twin_class_sizes:
            prod *= math.factorial(k)
        assert d.aut_order_pos == d.aut_order_twin * prod


def test_inequalities_conway() -> None:
    for seed in range(10):
        c = build_random_conway_causet(n_target=10, max_rank=6, seed=seed)
        d = entropy_decomposition_conway(c, backend="networkx")
        assert d.log_rank_factorial >= d.log_aut_pos >= d.log_aut_twin >= d.log_aut_outcome >= 0.0


def test_backward_compat_entropy_bare() -> None:
    for seed in range(6):
        c = build_random_conway_causet(n_target=12, max_rank=6, sample_rate_right=0.0, seed=seed)
        legacy = GameCauset(n=c.n, rank=c.rank, past=tuple(c.past_left))
        dc = entropy_decomposition_conway(c, backend="networkx")
        dl = entropy_decomposition(legacy)
        assert abs(dc.log_aut_pos - dl.log_aut_pos) < 1e-12


def test_twin_quotient_aut_nonincreasing() -> None:
    c = build_random_conway_causet(n_target=15, max_rank=8, seed=88)
    assert aut_order_conway(twin_quotient_conway(c)) <= aut_order_conway(c)
