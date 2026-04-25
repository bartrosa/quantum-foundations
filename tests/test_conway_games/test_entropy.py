from __future__ import annotations

import math

from quantum_foundations.conway_games.automorphism import aut_order_conway
from quantum_foundations.conway_games.causet import build_random_conway_causet
from quantum_foundations.conway_games.entropy import (
    entropy_decomposition_conway,
    twin_classes_canonical,
    twin_classes_local,
    twin_quotient_conway,
)
from quantum_foundations.entropic_causets.causet import GameCauset
from quantum_foundations.entropic_causets.entropy import entropy_decomposition


def test_twin_lemma_local_exact() -> None:
    for seed in range(5):
        c = build_random_conway_causet(n_target=10, max_rank=6, seed=seed)
        d = entropy_decomposition_conway(c, backend="networkx")
        prod = 1
        for k in d.twin_local_class_sizes:
            prod *= math.factorial(k)
        assert d.aut_order_pos == d.aut_order_twin_local * prod


def test_twin_lemma_canonical_inequality() -> None:
    for seed in range(10):
        c = build_random_conway_causet(n_target=10, max_rank=6, seed=seed)
        d = entropy_decomposition_conway(c, backend="networkx")
        prod = 1
        for k in d.twin_canonical_class_sizes:
            prod *= math.factorial(k)
        assert d.aut_order_pos <= d.aut_order_twin_canonical * prod
        assert d.delta_canonical_local >= -1e-12


def test_twin_local_refines_canonical() -> None:
    for seed in range(10):
        c = build_random_conway_causet(n_target=12, max_rank=6, seed=seed)
        local = twin_classes_local(c)
        canonical = twin_classes_canonical(c)
        canon_class_of: dict[int, int] = {}
        for ci, cls in enumerate(canonical):
            for ev in cls:
                canon_class_of[ev] = ci
        for cls in local:
            ci_set = {canon_class_of[ev] for ev in cls}
            assert len(ci_set) == 1


def test_bare_causet_canonical_equals_local() -> None:
    for seed in range(10):
        c = build_random_conway_causet(
            n_target=12,
            max_rank=6,
            sample_rate_left=0.0,
            sample_rate_right=0.0,
            seed=seed,
        )
        local = twin_classes_local(c)
        canonical = twin_classes_canonical(c)
        local_sets = frozenset(frozenset(cls) for cls in local)
        canon_sets = frozenset(frozenset(cls) for cls in canonical)
        assert local_sets == canon_sets


def test_inequalities_conway() -> None:
    for seed in range(10):
        c = build_random_conway_causet(n_target=10, max_rank=6, seed=seed)
        d = entropy_decomposition_conway(c, backend="networkx")
        assert (
            d.log_rank_factorial
            >= d.log_aut_pos
            >= d.log_aut_twin_canonical
            >= d.log_aut_twin_local
            >= d.log_aut_outcome
            >= 0.0
        )


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
