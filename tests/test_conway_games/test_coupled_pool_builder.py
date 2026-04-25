from __future__ import annotations

import pytest

from quantum_foundations.conway_games.causet import build_coupled_pool_conway_causet
from quantum_foundations.conway_games.entropy import entropy_decomposition_conway


def test_coupled_pool_smoke() -> None:
    c = build_coupled_pool_conway_causet(n_target=20, max_rank=4, seed=0)
    assert c.n == 20
    assert max(c.rank) <= 4


def test_coupled_pool_pool_overlap_zero() -> None:
    c = build_coupled_pool_conway_causet(n_target=30, max_rank=6, pool_overlap=0.0, seed=42)
    for i in range(c.n):
        assert c.past_left[i].isdisjoint(c.past_right[i]), (
            f"Event {i} has shared predecessor with pool_overlap=0"
        )


def test_coupled_pool_pool_overlap_one() -> None:
    c = build_coupled_pool_conway_causet(
        n_target=50, max_rank=10, pool_overlap=1.0, sample_rate=0.5, seed=42
    )
    lr_events = sum(1 for i in range(c.n) if c.past_left[i] & c.past_right[i])
    assert lr_events >= 5, f"Expected ≥5 LR-events, got {lr_events}"


def test_coupled_pool_determinism() -> None:
    c1 = build_coupled_pool_conway_causet(n_target=20, max_rank=5, pool_overlap=0.5, seed=123)
    c2 = build_coupled_pool_conway_causet(n_target=20, max_rank=5, pool_overlap=0.5, seed=123)
    assert c1.rank == c2.rank
    assert c1.past_left == c2.past_left
    assert c1.past_right == c2.past_right


def test_coupled_pool_validation() -> None:
    with pytest.raises(ValueError):
        build_coupled_pool_conway_causet(10, max_rank=2, pool_overlap=-0.1)
    with pytest.raises(ValueError):
        build_coupled_pool_conway_causet(10, max_rank=2, pool_overlap=1.5)
    with pytest.raises(ValueError):
        build_coupled_pool_conway_causet(10, max_rank=2, sample_rate=-0.1)
    with pytest.raises(ValueError):
        build_coupled_pool_conway_causet(0, max_rank=2)
    with pytest.raises(ValueError):
        build_coupled_pool_conway_causet(10, max_rank=-1)


def test_coupled_pool_overlap_monotonicity() -> None:
    n_seeds = 30

    def mean_lr_fraction(pool_overlap: float) -> float:
        total = 0.0
        for seed in range(n_seeds):
            c = build_coupled_pool_conway_causet(
                n_target=30,
                max_rank=6,
                pool_overlap=pool_overlap,
                sample_rate=0.4,
                seed=seed,
            )
            lr_count = sum(len(c.past_left[i] & c.past_right[i]) for i in range(c.n))
            total += lr_count
        return total / n_seeds

    f_zero = mean_lr_fraction(0.0)
    f_half = mean_lr_fraction(0.5)
    f_full = mean_lr_fraction(1.0)

    assert f_zero < f_half, f"Expected f(0) < f(0.5), got {f_zero} vs {f_half}"
    assert f_half < f_full, f"Expected f(0.5) < f(1), got {f_half} vs {f_full}"
    assert f_zero == 0.0, f"Expected zero LR with pool_overlap=0, got {f_zero}"


def test_coupled_pool_entropy_decomposition() -> None:
    c = build_coupled_pool_conway_causet(n_target=20, max_rank=4, pool_overlap=0.5, seed=42)
    d = entropy_decomposition_conway(c)
    assert d.log_aut_pos >= 0
    assert d.log_aut_twin_canonical >= 0
    assert d.log_aut_twin_local >= 0
