from __future__ import annotations

from quantum_foundations.entropic_causets.experiments import (
    e1_aut_scaling,
    e2_gibbs,
    e3_verlinde,
)


def test_e1_smoke() -> None:
    result = e1_aut_scaling.run(n_workers=1, seeds=3, ns=(10,))
    assert len(result.rows) == 3


def test_e2_smoke() -> None:
    result = e2_gibbs.run(n_workers=1, seeds=1, ns=(10,))
    assert len(result.rows) == 2


def test_e3_smoke() -> None:
    result = e3_verlinde.run(n_workers=1, seeds=1, rs=(2,), n_background=10)
    assert len(result.rows) == 1
