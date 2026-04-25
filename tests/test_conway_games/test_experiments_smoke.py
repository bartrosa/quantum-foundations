from __future__ import annotations

from quantum_foundations.conway_games.experiments import (
    e4_conway_aut_scaling,
    e5_outcome_distribution,
    e6_hierarchy_collapse,
    e7_conway_gibbs,
)


def test_e4_smoke() -> None:
    result = e4_conway_aut_scaling.run(n_workers=2, seeds=1, ns=(10,))
    assert len(result.rows) == 1


def test_e5_smoke() -> None:
    result = e5_outcome_distribution.run(n_workers=2, seeds=1, ns=(10,))
    assert len(result.rows) == 1


def test_e6_smoke() -> None:
    result = e6_hierarchy_collapse.run(n_workers=2, seeds=1, ns=(10,))
    assert len(result.rows) == 1


def test_e7_smoke() -> None:
    result = e7_conway_gibbs.run(seeds=1, ns=(8,))
    assert len(result.rows) == 2
